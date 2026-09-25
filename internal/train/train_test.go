package train

import (
	"context"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jmwri/decide/internal/bundle"
	"github.com/jmwri/decide/internal/cuda"
	"github.com/jmwri/decide/internal/data"
	"github.com/jmwri/decide/internal/nn"
	"github.com/jmwri/decide/internal/tokenizer"
)

const tinyConfig = `{"hidden_size":32,"intermediate_size":48,"num_attention_heads":4,"num_hidden_layers":3,
"vocab_size":300,"norm_eps":1e-5,"local_attention":8,"global_attn_every_n_layers":3,
"global_rope_theta":160000.0,"local_rope_theta":10000.0}`

// tinyBase writes a random tiny "base model" directory (config, tokenizer, weights).
func tinyBase(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	must := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	must(os.WriteFile(filepath.Join(dir, "config.json"), []byte(tinyConfig), 0o644))
	must(os.WriteFile(filepath.Join(dir, "tokenizer.json"), tokenizer.MinimalJSON(), 0o644))
	cfg, err := nn.LoadConfig(filepath.Join(dir, "config.json"))
	must(err)
	m := nn.New(cfg)
	m.InitRandom(rand.New(rand.NewSource(1)), 0.1)
	must(m.Save(filepath.Join(dir, "model.safetensors"), nil))
	return dir
}

func minimalTok(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	tok, err := tokenizer.Parse(tokenizer.MinimalJSON())
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func TestPackTextLayout(t *testing.T) {
	got := PackText("the state", "Which?", []string{" a ", "b"})
	if got != "Which? the state [SEP] [MASK] a [MASK] b" {
		t.Fatalf("%q", got)
	}
	if got := PackText("only state", "", []string{"x", "y"}); got != "only state [SEP] [MASK] x [MASK] y" {
		t.Fatalf("%q", got)
	}
	if PackText("s", "i", []string{"a"}) != bundle.PackText("s", "i", []string{"a"}) {
		t.Fatal("train and inference must share one layout")
	}
}

func TestRealizeKeepsGoldAndBounds(t *testing.T) {
	opts := make([]string, 40)
	for i := range opts {
		opts[i] = "option " + string(rune('A'+i%26)) + string(rune('a'+i/26))
	}
	e := data.Example{Kind: data.KindChoice, Options: opts, Gold: 17, Subsample: true, Task: "t"}
	rng := rand.New(rand.NewSource(1))
	sizes := map[int]bool{}
	for i := 0; i < 300; i++ {
		r := Realize(&e, rng, 10)
		if len(r.Options) < 2 || len(r.Options) > 10 || r.Options[r.Gold] != opts[17] || r.Subsample {
			t.Fatalf("bad realisation: %+v", r)
		}
		seen := map[string]bool{}
		for _, o := range r.Options {
			if seen[o] {
				t.Fatalf("duplicate option in %v", r.Options)
			}
			seen[o] = true
		}
		sizes[len(r.Options)] = true
	}
	if len(sizes) < 6 {
		t.Fatalf("option count barely varies: %v", sizes)
	}
	small := data.Example{Options: []string{"a", "b", "c"}, Gold: 2, Subsample: true}
	if r := Realize(&small, rng, 10); len(r.Options) != 3 || r.Gold != 2 {
		t.Fatalf("small pools must be untouched: %+v", r)
	}
	// ordinal scales are never subsampled
	ord := data.Example{Options: opts, Gold: 3, Ordinal: true}
	if r := Realize(&ord, rng, 10); len(r.Options) != 40 {
		t.Fatal("ordinal example was subsampled")
	}
}

func TestTokenizeItem(t *testing.T) {
	tok := minimalTok(t)
	e := data.Example{Task: "t", Kind: data.KindChoice, Instructions: "pick", State: "some state", Options: []string{"aa", "bb", "cc"}, Gold: 1}
	it, ok := Tokenize(tok, &e, 200, true)
	if !ok || len(it.Seq.MaskPos) != 3 || it.Target[1] != 1 || len(it.Seq.OptID) != len(it.Seq.IDs) {
		t.Fatalf("%+v ok=%v", it, ok)
	}
	// over-long states are trimmed to fit
	long := e
	long.State = strings.Repeat("word ", 500)
	it, ok = Tokenize(tok, &long, 120, true)
	if !ok || len(it.Seq.IDs) > 120 {
		t.Fatalf("expected the state to be trimmed: ok=%v len=%d", ok, len(it.Seq.IDs))
	}
	// an option that contains the reserved token is unusable
	bad := e
	bad.Options = []string{"aa", "b[MASK]b", "cc"}
	if _, ok := Tokenize(tok, &bad, 200, true); ok {
		t.Fatal("reserved token in an option must be rejected")
	}
	// soft targets pass through
	soft := e
	soft.Soft = []float32{0.1, 0.8, 0.1}
	if it, _ := Tokenize(tok, &soft, 200, true); it.Target[0] != 0.1 {
		t.Fatal("soft target lost")
	}
}

func TestLRSchedule(t *testing.T) {
	c := Config{LR: 1e-3, HeadLR: 1e-2, Warmup: 0.1, MinLRFrac: 0.1}
	if lr, _ := lrAt(&c, 0, 100); lr > 2e-4 {
		t.Fatalf("first step should be inside warmup: %v", lr)
	}
	peak, head := lrAt(&c, 9, 100)
	if math.Abs(peak-1e-3) > 1e-9 || math.Abs(head-1e-2) > 1e-9 {
		t.Fatalf("peak %v %v", peak, head)
	}
	end, _ := lrAt(&c, 99, 100)
	if end < 1e-4 || end > 1.5e-4 {
		t.Fatalf("final lr %v should approach 10%% of peak", end)
	}
	prev := peak
	for s := 10; s < 100; s++ {
		lr, _ := lrAt(&c, s, 100)
		if lr > prev {
			t.Fatalf("lr must decay monotonically after warmup (step %d)", s)
		}
		prev = lr
	}
}

func TestSoftmaxAndScore(t *testing.T) {
	p := Softmax([]float32{0, 0, 0}, 1)
	if math.Abs(p[0]-1.0/3) > 1e-9 {
		t.Fatal(p)
	}
	hot := Softmax([]float32{10, 0}, 1)
	cool := Softmax([]float32{10, 0}, 5)
	if !(hot[0] > cool[0] && cool[0] > 0.5) {
		t.Fatal("higher temperature must flatten the distribution")
	}
	items := []Item{
		{Gold: 0, Task: "a", Kind: "choice", Target: []float32{1, 0}},
		{Gold: 1, Task: "a", Kind: "choice", Target: []float32{0, 1}},
		{Gold: 0, Task: "b", Kind: "score", Ordinal: true, Target: []float32{1, 0, 0}},
	}
	logits := [][]float32{{5, 0}, {5, 0}, {0, 0, 0}}
	rep := Score(items, logits, 1)
	if rep.Overall.N != 3 || math.Abs(rep.Overall.Acc-2.0/3) > 1e-9 {
		t.Fatalf("%+v", rep.Overall)
	}
	if rep.ByTask["a"].Acc != 0.5 || rep.ByKind["score"].N != 1 || rep.ByTask["b"].MAE < 0.99 || rep.ByTask["b"].MAE > 1.01 {
		t.Fatalf("by-group metrics: a=%+v b=%+v", rep.ByTask["a"], rep.ByTask["b"])
	}
	if !strings.Contains(rep.String(), "OVERALL") {
		t.Fatal("report text")
	}
}

// A model that is 100% sure but only right 70% of the time must be flattened.
func TestCalibrationFitsTemperature(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	var items []Item
	var logits [][]float32
	for i := 0; i < 4000; i++ {
		gold := rng.Intn(2)
		pred := gold
		if rng.Float64() > 0.7 {
			pred = 1 - gold
		}
		l := []float32{0, 0}
		l[pred] = 6
		items = append(items, Item{Gold: gold, Kind: "choice", Task: "x", Target: []float32{1, 0}})
		logits = append(logits, l)
	}
	cal := FitTemperatures(items, logits)
	temp := cal.TemperatureFor("choice")
	if temp < 3 || temp > 8 { // ideal T: 6/ln(0.7/0.3) ~ 7.1
		t.Fatalf("temperature %v", temp)
	}
	if cal.ECEAfter["choice"] > 0.03 || cal.ECEAfter["choice"] >= cal.ECEBefore["choice"] {
		t.Fatalf("ECE before %.3f after %.3f", cal.ECEBefore["choice"], cal.ECEAfter["choice"])
	}
	if cal.TemperatureFor("unknown-kind") != cal.Default {
		t.Fatal("unknown kinds fall back to the default temperature")
	}
}

func TestAdamWUpdatesOnlyTrainable(t *testing.T) {
	cfg, err := nn.LoadConfig(writeTemp(t, tinyConfig))
	if err != nil {
		t.Fatal(err)
	}
	m := nn.New(cfg)
	m.InitRandom(rand.New(rand.NewSource(3)), 0.1)
	g := nn.NewLike(m)
	for _, tt := range g.Tensors() {
		for i := range tt.Data {
			tt.Data[i] = 0.5
		}
	}
	before := map[string]float32{}
	for _, tt := range m.Tensors() {
		before[tt.Name] = tt.Data[0]
	}
	opt := NewAdamW(m, g, 0.01, 2, false, false) // freeze layers 0-1 and embeddings
	opt.Update(1e-2, 1e-2, 1)
	after := map[string]float32{}
	for _, tt := range m.Tensors() {
		after[tt.Name] = tt.Data[0]
	}
	for _, name := range []string{"model.embeddings.tok_embeddings.weight", "model.layers.0.attn.Wqkv.weight", "model.layers.1.mlp.Wo.weight"} {
		if before[name] != after[name] {
			t.Errorf("%s is frozen but changed", name)
		}
	}
	for _, name := range []string{"model.layers.2.attn.Wqkv.weight", "model.final_norm.weight", "scorer.dense.weight", "scorer.out_proj.bias"} {
		if before[name] == after[name] {
			t.Errorf("%s should have been updated", name)
		}
	}
	// Adam's first step moves each weight by about lr, whatever the gradient scale.
	if d := math.Abs(float64(after["model.layers.2.attn.Wqkv.weight"] - before["model.layers.2.attn.Wqkv.weight"])); d < 0.008 || d > 0.0125 {
		t.Fatalf("first Adam step moved a weight by %v, want ~0.01", d)
	}
	if gn := opt.GradNorm(); gn <= 0 {
		t.Fatal("grad norm")
	}
}

func writeTemp(t *testing.T, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// toyCorpus writes a trivially learnable corpus: the state is the word of the
// option that is correct.
func toyCorpus(t *testing.T, n int) string {
	t.Helper()
	dir := t.TempDir()
	words := []string{"red", "blue", "green", "gold"}[:3]
	rng := rand.New(rand.NewSource(4))
	var train, val []data.Example
	for i := 0; i < n+40; i++ {
		perm := rng.Perm(len(words))
		opts := make([]string, len(words))
		gold := 0
		target := words[rng.Intn(len(words))]
		for slot, k := range perm {
			opts[slot] = words[k]
			if words[k] == target {
				gold = slot
			}
		}
		e := data.Example{Task: "toy", Kind: data.KindChoice, Instructions: "Which colour is named?", State: "the answer is " + target, Options: opts, Gold: gold}
		if i < n {
			train = append(train, e)
		} else {
			val = append(val, e)
		}
	}
	if err := data.WriteJSONL(filepath.Join(dir, "train.jsonl"), train); err != nil {
		t.Fatal(err)
	}
	if err := data.WriteJSONL(filepath.Join(dir, "val.jsonl"), val); err != nil {
		t.Fatal(err)
	}
	return dir
}

// gpuModes lists the engines to test: always the CPU, plus the GPU when present.
func gpuModes(t *testing.T) []string {
	modes := []string{GPUOff}
	if d, err := cuda.Open(); err == nil {
		d.Close()
		modes = append(modes, GPUOn)
	}
	return modes
}

func TestRunLearnsAndResumes(t *testing.T) {
	for _, mode := range gpuModes(t) {
		t.Run("gpu="+mode, func(t *testing.T) { runLearnsAndResumes(t, mode) })
	}
}

func runLearnsAndResumes(t *testing.T, mode string) {
	base, corpus, out := tinyBase(t), toyCorpus(t, 1920), t.TempDir()
	var lines []string
	cfg := Config{
		DataDir: corpus, BaseDir: base, OutDir: out, Epochs: 1, BatchExamples: 16, TokenBudget: 800, MaxLen: 128, KMax: 4,
		LR: 2e-3, HeadLR: 2e-3, EvalEvery: 1000, EvalPerTask: 40, SaveEvery: 40, LogEvery: 1, TrainEmb: true, Independent: true, Seed: 3, GPU: mode,
		Log: func(f string, a ...any) { lines = append(lines, f) },
	}
	// First half of the run, then resume for the rest.
	half := cfg
	half.Epochs = 0.5
	if err := Run(context.Background(), half); err != nil {
		t.Fatal(err)
	}
	if latestCheckpoint(out) == "" {
		t.Fatal("no checkpoint written")
	}
	cfg.Resume = true
	if err := Run(context.Background(), cfg); err != nil {
		t.Fatal(err)
	}

	// The final model must beat chance (33%) on held-out toy examples.
	bundleDir := filepath.Join(t.TempDir(), "b")
	mcfg, _ := nn.LoadConfig(filepath.Join(base, "config.json"))
	model := nn.New(mcfg)
	if err := model.LoadWeights(filepath.Join(out, "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	tok := minimalTok(t)
	valExs, _ := data.ReadJSONL(filepath.Join(corpus, "val.jsonl"))
	items := PrepareEval(tok, valExs, 0, 4, 128, true)
	logits, err := Predict(model, items, 1000)
	if err != nil {
		t.Fatal(err)
	}
	rep := Score(items, logits, 1)
	t.Logf("toy accuracy %.2f", rep.Overall.Acc)
	if rep.Overall.Acc < 0.5 {
		t.Fatalf("toy task not learned: acc %.2f nll %.3f", rep.Overall.Acc, rep.Overall.NLL)
	}

	// Export produces a loadable bundle whose predictions match.
	if err := Export(ExportConfig{ModelPath: filepath.Join(out, "model.safetensors"), BaseDir: base, DataDir: corpus, OutDir: bundleDir, ModelID: "decide-test"}); err != nil {
		t.Fatal(err)
	}
	b, err := bundle.Load(bundleDir)
	if err != nil {
		t.Fatal(err)
	}
	if b.Meta.ModelID != "decide-test" || !b.Meta.IndependentOptions || b.Meta.Calibration == nil {
		t.Fatalf("%+v", b.Meta)
	}
	again, _ := Predict(b.Model, items[:5], 1000)
	for i := range again {
		for j := range again[i] {
			if math.Abs(float64(again[i][j]-logits[i][j])) > 1e-5 {
				t.Fatalf("bundle predictions differ from the trained model")
			}
		}
	}
	if dev, err := OrderInvariance(model, tok, valExs, 10, 4); err != nil || dev > 1e-4 {
		t.Fatalf("order invariance deviation %v (%v)", dev, err)
	}
}

func TestModelCard(t *testing.T) {
	rep := func(acc float64) map[string]any {
		return map[string]any{
			"overall": map[string]any{"n": 100, "acc": acc, "nll": 0.5, "ece": 0.03},
			"by_kind": map[string]any{"choice": map[string]any{"n": 60, "acc": acc, "nll": 0.4, "ece": 0.02}},
			"by_task": map[string]any{"mnli": map[string]any{"n": 30, "acc": 0.9, "nll": 0.3, "ece": 0.01}, "sst5": map[string]any{"n": 20, "acc": 0.5, "nll": 1.1, "ece": 0.04, "mae": 0.61}},
		}
	}
	meta := bundle.Meta{ModelID: "decide-9.9.9", Training: map[string]any{
		"reports": map[string]any{"val": rep(0.87), "ood": rep(0.61), "order_invariance_max_logit_delta": 3e-7},
		"corpus": map[string]any{"train": 1234, "tasks": []any{
			map[string]any{"name": "mnli", "license": "cc-by-3.0", "train": 1000},
			map[string]any{"name": "mmlu", "license": "mit", "ood": 50},
			map[string]any{"name": "flaky", "license": "x", "skipped": "download failed"},
		}},
	}}
	card := ModelCard(meta, "me/decide")
	for _, want := range []string{"license: apache-2.0", "# decide-9.9.9", "3.0e-07", "87.0%", "61.0%", "MAE 0.61", "| mnli | 1000 | cc-by-3.0 |", "mmlu (held out, evaluation only)", "me/decide", "## Limitations"} {
		if !strings.Contains(card, want) {
			t.Errorf("model card is missing %q", want)
		}
	}
	if strings.Contains(card, "flaky") {
		t.Error("skipped tasks must not appear in the training data table")
	}
	if !strings.HasPrefix(card, "---\n") {
		t.Error("model card needs YAML front matter")
	}
}

// --init must start from a previously trained model rather than a fresh head.
func TestRunInitialisesFromTrainedModel(t *testing.T) {
	for _, mode := range gpuModes(t) {
		t.Run("gpu="+mode, func(t *testing.T) { runInitialises(t, mode) })
	}
}

func runInitialises(t *testing.T, mode string) {
	base, corpus, out1, out2 := tinyBase(t), toyCorpus(t, 640), t.TempDir(), t.TempDir()
	cfg := Config{
		DataDir: corpus, BaseDir: base, OutDir: out1, Epochs: 1, BatchExamples: 16, TokenBudget: 800, MaxLen: 128, KMax: 4,
		LR: 2e-3, HeadLR: 2e-3, EvalEvery: 1000, EvalPerTask: 40, SaveEvery: 1000, LogEvery: 1000, TrainEmb: true, Independent: true, Seed: 3, GPU: mode,
	}
	if err := Run(context.Background(), cfg); err != nil {
		t.Fatal(err)
	}
	// Continue for zero effective learning rate: the result must equal the init model.
	next := cfg
	next.OutDir, next.InitModel, next.LR, next.HeadLR, next.MaxExamples, next.Epochs = out2, filepath.Join(out1, "model.safetensors"), 1e-12, 1e-12, 64, 1
	if err := Run(context.Background(), next); err != nil {
		t.Fatal(err)
	}
	a, b := loadTiny(t, base, filepath.Join(out1, "model.safetensors")), loadTiny(t, base, filepath.Join(out2, "model.safetensors"))
	fresh := loadTiny(t, base, "")
	if d := maxDiff(a, b); d > 1e-4 {
		t.Fatalf("continued model drifted from its init by %g", d)
	}
	if d := maxDiff(a, fresh); d < 1e-3 {
		t.Fatal("trained model should differ from the untrained base")
	}
}

func loadTiny(t *testing.T, base, weights string) *nn.Model {
	t.Helper()
	cfg, err := nn.LoadConfig(filepath.Join(base, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	m := nn.New(cfg)
	if weights == "" {
		m, err = LoadBase(base, 3)
		if err != nil {
			t.Fatal(err)
		}
		return m
	}
	if err := m.LoadWeights(weights); err != nil {
		t.Fatal(err)
	}
	return m
}

func maxDiff(a, b *nn.Model) float64 {
	var worst float64
	bt := b.Tensors()
	for i, ta := range a.Tensors() {
		for j := range ta.Data {
			worst = math.Max(worst, math.Abs(float64(ta.Data[j]-bt[i].Data[j])))
		}
	}
	return worst
}
