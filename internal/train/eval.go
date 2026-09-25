package train

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"

	"github.com/jmwri/decide/internal/data"
	"github.com/jmwri/decide/internal/nn"
	"github.com/jmwri/decide/internal/tokenizer"
)

// Metrics summarises predictions on a set of examples.
type Metrics struct {
	N   int     `json:"n"`
	Acc float64 `json:"acc"`
	NLL float64 `json:"nll"`
	// ECE is the expected calibration error of the top-option confidence (15 bins).
	ECE float64 `json:"ece"`
	// MAE is the mean absolute error of the expected level on ordinal examples.
	MAE float64 `json:"mae,omitempty"`
	// Order-invariance and other diagnostics are filled by callers when relevant.
	nMAE int
}

// Report groups metrics.
type Report struct {
	Overall Metrics             `json:"overall"`
	ByKind  map[string]*Metrics `json:"by_kind"`
	ByTask  map[string]*Metrics `json:"by_task"`
}

// Softmax returns the temperature-scaled softmax of logits.
func Softmax(logits []float32, temp float64) []float64 {
	if temp <= 0 {
		temp = 1
	}
	maxv := float64(logits[0])
	for _, v := range logits {
		maxv = math.Max(maxv, float64(v))
	}
	out := make([]float64, len(logits))
	var sum float64
	for i, v := range logits {
		out[i] = math.Exp((float64(v) - maxv) / temp)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// Predict runs the model over items in token-budgeted batches and returns the
// logits of every item.
func Predict(m *nn.Model, items []Item, tokenBudget int) ([][]float32, error) {
	out := make([][]float32, len(items))
	c := nn.NewCache(m)
	for start := 0; start < len(items); {
		end, tokens := start, 0
		for end < len(items) && (end == start || tokens+len(items[end].Seq.IDs) <= tokenBudget) {
			tokens += len(items[end].Seq.IDs)
			end++
		}
		batch := make([]nn.Sequence, 0, end-start)
		for i := start; i < end; i++ {
			batch = append(batch, items[i].Seq)
		}
		logits, err := m.Forward(c, batch, 1<<20)
		if err != nil {
			return nil, err
		}
		off := 0
		for i := start; i < end; i++ {
			k := len(items[i].Seq.MaskPos)
			out[i] = append([]float32(nil), logits[off:off+k]...)
			off += k
		}
		start = end
	}
	return out, nil
}

// Score computes metrics from logits at a given temperature.
func Score(items []Item, logits [][]float32, temp float64) *Report {
	return ScoreWith(items, logits, func(string) float64 { return temp })
}

// ScoreWith computes metrics using a per-kind temperature.
func ScoreWith(items []Item, logits [][]float32, tempFor func(kind string) float64) *Report {
	rep := &Report{ByKind: map[string]*Metrics{}, ByTask: map[string]*Metrics{}}
	type acc struct {
		m       *Metrics
		correct float64
		nll     float64
		mae     float64
		conf    [15]float64
		hit     [15]float64
		cnt     [15]int
	}
	accs := map[*Metrics]*acc{}
	get := func(mm map[string]*Metrics, k string) *acc {
		x, ok := mm[k]
		if !ok {
			x = &Metrics{}
			mm[k] = x
		}
		a, ok := accs[x]
		if !ok {
			a = &acc{m: x}
			accs[x] = a
		}
		return a
	}
	overall := &acc{m: &rep.Overall}
	accs[&rep.Overall] = overall
	for i, it := range items {
		p := Softmax(logits[i], tempFor(it.Kind))
		best := 0
		for j, v := range p {
			if v > p[best] {
				best = j
			}
		}
		hit := 0.0
		if best == it.Gold {
			hit = 1
		}
		nll := -math.Log(math.Max(p[it.Gold], 1e-12))
		bin := min(int(p[best]*15), 14)
		for _, a := range []*acc{overall, get(rep.ByKind, it.Kind), get(rep.ByTask, it.Task)} {
			a.m.N++
			a.correct += hit
			a.nll += nll
			a.conf[bin] += p[best]
			a.hit[bin] += hit
			a.cnt[bin]++
			if it.Ordinal {
				var e float64
				for j, v := range p {
					e += float64(j) * v
				}
				a.mae += math.Abs(e - float64(it.Gold))
				a.m.nMAE++
			}
		}
	}
	for _, a := range accs {
		n := float64(max(a.m.N, 1))
		a.m.Acc, a.m.NLL = a.correct/n, a.nll/n
		for b := range a.cnt {
			if a.cnt[b] > 0 {
				a.m.ECE += float64(a.cnt[b]) / n * math.Abs(a.hit[b]/float64(a.cnt[b])-a.conf[b]/float64(a.cnt[b]))
			}
		}
		if a.m.nMAE > 0 {
			a.m.MAE = a.mae / float64(a.m.nMAE)
		}
	}
	return rep
}

// String renders a report as a table.
func (r *Report) String() string {
	var b strings.Builder
	row := func(name string, m *Metrics) {
		mae := ""
		if m.nMAE > 0 {
			mae = fmt.Sprintf("  mae %.3f", m.MAE)
		}
		fmt.Fprintf(&b, "  %-18s n=%-5d acc %5.1f%%  nll %.3f  ece %.3f%s\n", name, m.N, 100*m.Acc, m.NLL, m.ECE, mae)
	}
	row("OVERALL", &r.Overall)
	for _, k := range sortedKeys(r.ByKind) {
		row("kind:"+k, r.ByKind[k])
	}
	for _, k := range sortedKeys(r.ByTask) {
		row(k, r.ByTask[k])
	}
	return b.String()
}

func sortedKeys(m map[string]*Metrics) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// PrepareEval tokenises examples for evaluation. Large option pools are
// subsampled deterministically (per example index) to at most kMax options.
// perTask > 0 keeps only the first perTask examples of each task.
func PrepareEval(tok *tokenizer.Tokenizer, exs []data.Example, perTask, kMax, maxLen int, independent bool) []Item {
	var items []Item
	count := map[string]int{}
	for i := range exs {
		e := &exs[i]
		if perTask > 0 && count[e.Task] >= perTask {
			continue
		}
		rng := rand.New(rand.NewSource(int64(i)*7919 + 13))
		r := Realize(e, rng, kMax)
		it, ok := Tokenize(tok, &r, maxLen, independent)
		if !ok {
			continue
		}
		count[e.Task]++
		items = append(items, it)
	}
	return items
}
