package decide

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/jmwri/decide/internal/bundle"
	"github.com/jmwri/decide/internal/nn"
)

// Version is the model release served by default.
const Version = "0.1"

// ModelID is the identifier stamped on responses when a bundle names none.
const ModelID = "decide-0.1.0"

// ModelAliases are the names that resolve to the current model.
var ModelAliases = []string{"decide-0.1", "0.1", "decide", "default", "latest", "decide-latest"}

// maxTokens bounds the packed sequence.
const maxTokens = 8192

// Local runs the model in-process on the CPU, in pure Go: no Python, cgo or
// network dependency at inference time. Weights load lazily on first use.
//
// Local is safe for concurrent use; inference calls are serialised because
// each one already saturates every core.
type Local struct {
	dir string

	mu       sync.Mutex
	loaded   *loadedModel
	Progress func(msg string) // optional; reports download/load progress
}

type loadedModel struct {
	b     *bundle.Bundle
	fmu   sync.Mutex // guards cache
	cache *nn.Cache
}

// NewLocal returns a Local backend reading the model bundle in modelDir. If
// modelDir is empty, $DECIDE_MODEL_DIR, ./models/decide and finally the
// download cache are tried in turn (see DownloadModel).
func NewLocal(modelDir string) *Local { return &Local{dir: modelDir} }

// Load eagerly loads the weights. It is called implicitly by the first query.
func (l *Local) Load(ctx context.Context) error {
	_, err := l.get(ctx)
	return err
}

func (l *Local) progress(format string, args ...any) {
	if l.Progress != nil {
		l.Progress(fmt.Sprintf(format, args...))
	}
}

func (l *Local) get(ctx context.Context) (*loadedModel, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.loaded != nil {
		return l.loaded, nil
	}
	dir, err := l.resolveDir(ctx)
	if err != nil {
		return nil, err
	}
	l.progress("loading model from %s", dir)
	b, err := bundle.Load(dir)
	if err != nil {
		return nil, err
	}
	l.loaded = &loadedModel{b: b, cache: nn.NewCache(b.Model)}
	l.progress("%s ready (%d parameters, independent_options=%v)", b.Meta.ModelID, b.Model.NumParams(), b.Meta.IndependentOptions)
	return l.loaded, nil
}

func (l *Local) resolveDir(ctx context.Context) (string, error) {
	has := func(d string) bool {
		_, err := os.Stat(filepath.Join(d, "decide.json"))
		return err == nil
	}
	if l.dir != "" {
		if !has(l.dir) {
			return "", fmt.Errorf("decide: no decide.json in model dir %q", l.dir)
		}
		return l.dir, nil
	}
	if d := os.Getenv("DECIDE_MODEL_DIR"); d != "" {
		if !has(d) {
			return "", fmt.Errorf("decide: no decide.json in $DECIDE_MODEL_DIR %q", d)
		}
		return d, nil
	}
	if has("models/decide") {
		return "models/decide", nil
	}
	if d := cachedModelDir(); has(d) {
		return d, nil
	}
	return DownloadModel(ctx, "", "", l.progress)
}

// SystemOne implements Evaluator.
func (l *Local) SystemOne(ctx context.Context, state any, questions Questions) (*Response, error) {
	return l.Evaluate(ctx, state, questions)
}

// Evaluate answers every question against state.
func (l *Local) Evaluate(ctx context.Context, state any, questions Questions) (*Response, error) {
	lm, err := l.get(ctx)
	if err != nil {
		return nil, err
	}
	stateText := formatState(state)
	resp := &Response{Model: lm.b.Meta.ModelID, Answers: map[string]Answer{}}
	qChars := 0
	for _, nq := range questions {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if _, dup := resp.Answers[nq.ID]; dup {
			return nil, fmt.Errorf("decide: duplicate question id %q", nq.ID)
		}
		var ans Answer
		switch q := asValue(nq.Question).(type) {
		case Choice:
			ans, err = lm.evalChoice(stateText, q)
			qChars += utf8.RuneCountInString(q.Instructions)
		case Noul:
			ans, err = lm.evalNoul(stateText, q)
			qChars += utf8.RuneCountInString(q.Instructions)
		case Score:
			ans, err = lm.evalScore(stateText, q)
			qChars += utf8.RuneCountInString(q.Instructions)
		default:
			err = fmt.Errorf("unsupported question type %T", nq.Question)
		}
		if err != nil {
			return nil, fmt.Errorf("decide: question %q: %w", nq.ID, err)
		}
		resp.Answers[nq.ID] = ans
		resp.Order = append(resp.Order, nq.ID)
	}
	resp.Usage = Usage{
		InputTokens:  max(1, utf8.RuneCountInString(stateText)/4) + max(1, qChars/4),
		OutputTokens: len(resp.Answers),
	}
	return resp, nil
}

// formatState renders a state as the model's input text: strings pass
// through, objects become "key: value" lines (values rendered as Python's
// str() would), anything else is its string form.
func formatState(state any) string {
	if s, ok := state.(string); ok {
		return s
	}
	n := normalize(state)
	if obj, ok := n.(Object); ok {
		lines := make([]string, len(obj))
		for i, kv := range obj {
			lines[i] = kv.Key + ": " + pyStr(kv.Value)
		}
		return strings.Join(lines, "\n")
	}
	return pyStr(n)
}

// ---------------------------------------------------------------------------
// scoring

// rawLogits scores every option in one encoder pass.
func (lm *loadedModel) rawLogits(state, question string, options []string) ([]float64, error) {
	b := lm.b
	packed := bundle.PackText(state, question, options)
	ids := b.Tok.Encode(packed, true)
	if len(ids) > maxTokens {
		return nil, fmt.Errorf("input is %d tokens; the limit is %d", len(ids), maxTokens)
	}
	seq := nn.BuildSequence(ids, b.Tok.MaskID(), b.Meta.IndependentOptions)
	if len(seq.MaskPos) != len(options) {
		return nil, errors.New("input contains the reserved [MASK] token")
	}
	lm.fmu.Lock()
	defer lm.fmu.Unlock()
	out, err := b.Model.Forward(lm.cache, []nn.Sequence{seq}, 1<<20)
	if err != nil {
		return nil, err
	}
	logits := make([]float64, len(out))
	for i, v := range out {
		logits[i] = float64(v)
	}
	return logits, nil
}

func softmax(x []float64) []float64 {
	maxv := math.Inf(-1)
	for _, v := range x {
		maxv = math.Max(maxv, v)
	}
	out := make([]float64, len(x))
	var sum float64
	for i, v := range x {
		out[i] = math.Exp(v - maxv)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// calibrated turns logits into probabilities using the bundle's fitted
// temperature for this kind of question.
func (lm *loadedModel) calibrated(logits []float64, kind string) []float64 {
	t := math.Max(lm.b.Meta.Calibration.TemperatureFor(kind), 1e-4)
	scaled := make([]float64, len(logits))
	for i, v := range logits {
		scaled[i] = v / t
	}
	return softmax(scaled)
}

func round(x float64, places int) float64 {
	p := math.Pow(10, float64(places))
	return math.RoundToEven(x*p) / p
}

// marginConfidence is (n*p_max - 1)/(n - 1): 0 for a uniform distribution and
// 1 for a certain one, anchored to the n-way chance baseline.
func marginConfidence(probs []float64) float64 {
	n := len(probs)
	if n <= 1 {
		return 1.0
	}
	pmax := 0.0
	for _, p := range probs {
		pmax = math.Max(pmax, p)
	}
	c := (float64(n)*pmax - 1) / float64(n-1)
	return round(math.Max(0, math.Min(1, c)), 3)
}

func argmax(x []float64) int {
	best := 0
	for i, v := range x {
		if v > x[best] {
			best = i
		}
	}
	return best
}

func (lm *loadedModel) evalChoice(state string, q Choice) (Answer, error) {
	n := len(q.Criteria)
	if n == 0 {
		return &ChoiceAnswer{Probabilities: map[string]float64{}}, nil
	}
	seen := make(map[string]bool, n)
	descs := make([]string, n)
	for i, o := range q.Criteria {
		if seen[o.Key] {
			return nil, fmt.Errorf("duplicate choice %q", o.Key)
		}
		seen[o.Key] = true
		if d := strings.TrimSpace(o.Description); d != "" {
			descs[i] = d
		} else {
			descs[i] = strings.TrimSpace(o.Key)
		}
	}
	if n == 1 { // nothing to decide; skip the forward pass
		return &ChoiceAnswer{Choice: q.Criteria[0].Key, Probabilities: map[string]float64{q.Criteria[0].Key: 1}, Confidence: 1}, nil
	}
	logits, err := lm.rawLogits(state, q.Instructions, descs)
	if err != nil {
		return nil, err
	}
	probs := lm.calibrated(logits, "choice")
	out := &ChoiceAnswer{
		Choice:        q.Criteria[argmax(logits)].Key,
		Probabilities: make(map[string]float64, n),
		Confidence:    marginConfidence(probs),
	}
	for i, o := range q.Criteria {
		out.Probabilities[o.Key] = round(probs[i], 4)
	}
	return out, nil
}

func (lm *loadedModel) evalScore(state string, q Score) (Answer, error) {
	n := len(q.Criteria)
	if n == 0 {
		return &ScoreAnswer{Legend: map[string]string{}, Probabilities: map[string]float64{}}, nil
	}
	legend := make(map[string]string, n)
	descs := make([]string, n)
	for i, lv := range q.Criteria {
		descs[i] = lv.Text()
		legend[fmt.Sprint(i)] = descs[i]
	}
	var probs []float64
	if n == 1 {
		probs = []float64{1}
	} else {
		logits, err := lm.rawLogits(state, q.Instructions, descs)
		if err != nil {
			return nil, err
		}
		probs = lm.calibrated(logits, "score")
	}
	out := &ScoreAnswer{Legend: legend, Probabilities: make(map[string]float64, n)}
	expect := 0.0
	for i, p := range probs {
		out.Probabilities[fmt.Sprint(i)] = round(p, 4)
		expect += float64(i) * p
	}
	out.Score = round(expect, 2)
	out.Confidence = marginConfidence(probs)
	return out, nil
}

// Default option texts for a Noul without explicit criteria; the model was
// trained with these as its most common yes/no pair.
const (
	defaultYes = "Yes, condition holds true."
	defaultNo  = "No, condition is false."
)

func (lm *loadedModel) evalNoul(state string, q Noul) (Answer, error) {
	yes, no := defaultYes, defaultNo
	if q.Criteria != nil {
		if s := strings.TrimSpace(q.Criteria.True); s != "" {
			yes = s
		}
		if s := strings.TrimSpace(q.Criteria.False); s != "" {
			no = s
		}
	}
	logits, err := lm.rawLogits(state, q.Instructions, []string{yes, no})
	if err != nil {
		return nil, err
	}
	probs := lm.calibrated(logits, "noul")
	return &NoulAnswer{Noul: round(math.Max(0, math.Min(1, probs[0])), 4)}, nil
}
