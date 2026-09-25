package nn

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/jmwri/decide/internal/safetensors"
)

type refMeta struct {
	Config struct {
		Hidden, Intermediate, Heads, Layers, Vocab int
		HalfWindow                                 int     `json:"half_window"`
		GlobalTheta                                float64 `json:"global_theta"`
		LocalTheta                                 float64 `json:"local_theta"`
	} `json:"config"`
	MaskID int32 `json:"mask_id"`
	Seqs   []struct {
		IDs         []int32 `json:"ids"`
		Independent bool    `json:"independent"`
		Target      int     `json:"target"`
	} `json:"seqs"`
}

func loadRef(t *testing.T) (*Model, *safetensors.File, refMeta) {
	t.Helper()
	f, err := safetensors.Open("testdata/tiny_ref.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	var meta refMeta
	if err := json.Unmarshal([]byte(f.Metadata["meta"]), &meta); err != nil {
		t.Fatal(err)
	}
	c := meta.Config
	cfg := Config{
		Hidden: c.Hidden, Intermediate: c.Intermediate, Heads: c.Heads, Layers: c.Layers, Vocab: c.Vocab,
		Eps: 1e-5, HalfWindow: c.HalfWindow, GlobalTheta: c.GlobalTheta, LocalTheta: c.LocalTheta,
		LayerTypes: []string{"full_attention", "sliding_attention", "sliding_attention", "full_attention"},
	}
	m := New(cfg)
	for _, nt := range m.Tensors() {
		data, _, err := f.Float32("w." + nt.Name)
		if err != nil {
			t.Fatal(err)
		}
		copy(nt.Data, data)
	}
	return m, f, meta
}

func batchOf(meta refMeta) []Sequence {
	var b []Sequence
	for _, s := range meta.Seqs {
		b = append(b, BuildSequence(s.IDs, meta.MaskID, s.Independent))
	}
	return b
}

// lossAndGrad returns the mean cross-entropy over sequences and dLoss/dlogits.
func lossAndGrad(logits []float32, batch []Sequence, targets []int) (float64, []float32) {
	d := make([]float32, len(logits))
	var loss float64
	off := 0
	for si, s := range batch {
		k := len(s.MaskPos)
		l := logits[off : off+k]
		maxv := l[0]
		for _, v := range l {
			maxv = max(maxv, v)
		}
		var sum float64
		for _, v := range l {
			sum += math.Exp(float64(v - maxv))
		}
		for i, v := range l {
			p := math.Exp(float64(v-maxv)) / sum
			d[off+i] = float32(p) / float32(len(batch))
			if i == targets[si] {
				loss -= math.Log(p) / float64(len(batch))
				d[off+i] -= 1 / float32(len(batch))
			}
		}
		off += k
	}
	return loss, d
}

// relErr is ||got-want|| / ||want||, with a 1e-4 floor on the denominator so
// gradients that are analytically zero (e.g. biases under softmax
// cross-entropy) are compared in absolute terms.
func relErr(got, want []float32) float64 {
	var num, den float64
	for i := range got {
		d := float64(got[i]) - float64(want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	return math.Sqrt(num) / (math.Sqrt(den) + 1e-4)
}

func TestForwardBackwardMatchesPyTorch(t *testing.T) {
	m, f, meta := loadRef(t)
	batch := batchOf(meta)
	targets := []int{}
	for _, s := range meta.Seqs {
		targets = append(targets, s.Target)
	}
	wantLogits, _, _ := f.Float32("logits")
	wantLoss, _, _ := f.Float32("loss")

	for _, trainFrom := range []int{0, 2} {
		c := NewCache(m)
		logits, err := m.Forward(c, batch, trainFrom)
		if err != nil {
			t.Fatal(err)
		}
		if e := relErr(logits, wantLogits); e > 1e-4 {
			t.Fatalf("trainFrom=%d logits rel err %g\n got %v\nwant %v", trainFrom, e, logits, wantLogits)
		}
		loss, dl := lossAndGrad(logits, batch, targets)
		if math.Abs(loss-float64(wantLoss[0])) > 1e-4 {
			t.Fatalf("loss %v want %v", loss, wantLoss[0])
		}
		g := NewLike(m)
		m.Backward(c, dl, g, true)
		gts := map[string]NamedTensor{}
		for _, nt := range g.Tensors() {
			gts[nt.Name] = nt
		}
		for _, nt := range m.Tensors() {
			want, _, err := f.Float32("g." + nt.Name)
			if err != nil {
				t.Fatal(err)
			}
			got := gts[nt.Name].Data
			frozen := false
			for l := 0; l < trainFrom; l++ {
				if hasPrefix(nt.Name, "model.layers."+string(rune('0'+l))+".") {
					frozen = true
				}
			}
			if trainFrom > 0 && (frozen || nt.Name == "model.embeddings.tok_embeddings.weight" || nt.Name == "model.embeddings.norm.weight") {
				for _, v := range got {
					if v != 0 {
						t.Fatalf("trainFrom=%d: frozen tensor %s received gradient", trainFrom, nt.Name)
					}
				}
				continue
			}
			if e := relErr(got, want); e > 2e-3 {
				t.Errorf("trainFrom=%d grad %s rel err %g", trainFrom, nt.Name, e)
			}
		}
	}
}

func hasPrefix(s, p string) bool { return len(s) >= len(p) && s[:len(p)] == p }

// Permuting options in an independent-options sequence must permute the logits.
func TestOrderInvariance(t *testing.T) {
	m, _, meta := loadRef(t)
	base := meta.Seqs[0].IDs
	// spans: prefix [0, mask0), options start at each MASK token
	var starts []int
	for i, id := range base {
		if id == meta.MaskID {
			starts = append(starts, i)
		}
	}
	last := len(base) - 1
	spans := [][]int32{}
	for k, s := range starts {
		e := last
		if k+1 < len(starts) {
			e = starts[k+1]
		}
		spans = append(spans, base[s:e])
	}
	build := func(order []int) []int32 {
		ids := append([]int32(nil), base[:starts[0]]...)
		for _, k := range order {
			ids = append(ids, spans[k]...)
		}
		return append(ids, base[last])
	}
	run := func(ids []int32) []float32 {
		c := NewCache(m)
		out, err := m.Forward(c, []Sequence{BuildSequence(ids, meta.MaskID, true)}, 99)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), out...)
	}
	orig := run(build([]int{0, 1, 2}))
	perm := []int{2, 0, 1}
	got := run(build(perm))
	for slot, k := range perm {
		if d := math.Abs(float64(got[slot] - orig[k])); d > 1e-5 {
			t.Fatalf("option %d moved to slot %d: %v vs %v", k, slot, orig[k], got[slot])
		}
	}
}
