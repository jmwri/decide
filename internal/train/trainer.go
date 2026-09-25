package train

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/jmwri/decide/internal/data"
	"github.com/jmwri/decide/internal/nn"
	"github.com/jmwri/decide/internal/tokenizer"
)

// Config controls a training run.
type Config struct {
	DataDir string // corpus from `decide-train data`
	BaseDir string // ModernBERT-base: config.json, model.safetensors, tokenizer.json
	OutDir  string // run directory: checkpoints and logs

	Epochs        float64 // passes over the (possibly capped) training set
	MaxExamples   int     // cap on training examples per epoch (0 = all)
	BatchExamples int     // examples per optimizer step
	TokenBudget   int     // tokens per forward/backward micro-batch
	MaxLen        int     // longest packed sequence
	KMax          int     // most options shown per example

	LR, HeadLR  float64
	WeightDecay float64
	Warmup      float64 // fraction of steps
	MinLRFrac   float64 // final LR as a fraction of LR
	ClipNorm    float64

	TrainFrom int // freeze encoder layers below this index
	TrainEmb  bool

	EvalEvery   int
	EvalPerTask int
	SaveEvery   int
	LogEvery    int
	KeepCkpts   int

	Independent bool // order-invariant option attention
	Seed        int64
	Resume      bool
	Exclude     []string // tasks left out of training
	Log         func(format string, args ...any)
}

// Defaults fills unset fields.
func (c *Config) Defaults() {
	setI := func(p *int, v int) {
		if *p == 0 {
			*p = v
		}
	}
	setF := func(p *float64, v float64) {
		if *p == 0 {
			*p = v
		}
	}
	setF(&c.Epochs, 1)
	setI(&c.BatchExamples, 32)
	setI(&c.TokenBudget, 1200)
	setI(&c.MaxLen, 384)
	setI(&c.KMax, 10)
	setF(&c.LR, 4e-5)
	setF(&c.HeadLR, 3e-4)
	setF(&c.WeightDecay, 0.01)
	setF(&c.Warmup, 0.05)
	setF(&c.MinLRFrac, 0.05)
	setF(&c.ClipNorm, 1.0)
	setI(&c.EvalEvery, 250)
	setI(&c.EvalPerTask, 30)
	setI(&c.SaveEvery, 250)
	setI(&c.LogEvery, 5)
	setI(&c.KeepCkpts, 2)
	if c.Seed == 0 {
		c.Seed = 1
	}
}

func (c *Config) logf(format string, args ...any) {
	if c.Log != nil {
		c.Log(format, args...)
	}
}

// State is what a checkpoint records about progress.
type State struct {
	Step     int     `json:"step"`
	Examples int     `json:"examples"` // training examples consumed
	LossEMA  float64 `json:"loss_ema"`
	AccEMA   float64 `json:"acc_ema"`
	Elapsed  float64 `json:"elapsed_sec"`
	Config   struct {
		Seed        int64   `json:"seed"`
		Epochs      float64 `json:"epochs"`
		MaxExamples int     `json:"max_examples"`
	} `json:"config"`
}

func lrAt(c *Config, step, total int) (lr, headLR float64) {
	warm := int(math.Ceil(c.Warmup * float64(total)))
	f := 1.0
	switch {
	case step < warm:
		f = float64(step+1) / float64(max(warm, 1))
	default:
		p := float64(step-warm) / float64(max(total-warm, 1))
		f = c.MinLRFrac + (1-c.MinLRFrac)*(1-p)
	}
	return c.LR * f, c.HeadLR * f
}

// LoadBase builds a fresh model from ModernBERT-base plus a new scorer head.
func LoadBase(dir string, seed int64) (*nn.Model, error) {
	cfg, err := nn.LoadConfig(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	m := nn.New(cfg)
	if err := m.LoadBase(filepath.Join(dir, "model.safetensors")); err != nil {
		return nil, err
	}
	m.InitHead(rand.New(rand.NewSource(seed)))
	return m, nil
}

// Run trains until the step budget is spent or ctx is cancelled (saving a
// checkpoint in either case).
func Run(ctx context.Context, cfg Config) error {
	cfg.Defaults()
	if err := os.MkdirAll(cfg.OutDir, 0o755); err != nil {
		return err
	}
	tok, err := tokenizer.Load(filepath.Join(cfg.BaseDir, "tokenizer.json"))
	if err != nil {
		return err
	}
	all, err := data.ReadJSONL(filepath.Join(cfg.DataDir, "train.jsonl"))
	if err != nil {
		return err
	}
	if len(cfg.Exclude) > 0 {
		kept := all[:0]
		for _, e := range all {
			skip := false
			for _, x := range cfg.Exclude {
				skip = skip || e.Task == x
			}
			if !skip {
				kept = append(kept, e)
			}
		}
		all = kept
	}
	valExs, err := data.ReadJSONL(filepath.Join(cfg.DataDir, "val.jsonl"))
	if err != nil {
		return err
	}
	if cfg.MaxExamples > 0 && cfg.MaxExamples < len(all) {
		all = all[:cfg.MaxExamples]
	}
	perEpoch := len(all)
	totalExamples := int(float64(perEpoch) * cfg.Epochs)
	totalSteps := (totalExamples + cfg.BatchExamples - 1) / cfg.BatchExamples
	cfg.logf("%d training examples/epoch, %.2f epochs -> %d steps of %d examples", perEpoch, cfg.Epochs, totalSteps, cfg.BatchExamples)

	model, err := LoadBase(cfg.BaseDir, cfg.Seed)
	if err != nil {
		return err
	}
	grads := nn.NewLike(model)
	opt := NewAdamW(model, grads, float32(cfg.WeightDecay), cfg.TrainFrom, cfg.TrainEmb)
	cfg.logf("model: %d parameters, %d layers; training from layer %d (embeddings: %v)", model.NumParams(), model.Cfg.Layers, cfg.TrainFrom, cfg.TrainEmb)

	st := State{}
	st.Config.Seed, st.Config.Epochs, st.Config.MaxExamples = cfg.Seed, cfg.Epochs, cfg.MaxExamples
	if cfg.Resume {
		if dir := latestCheckpoint(cfg.OutDir); dir != "" {
			if err := loadCheckpoint(dir, model, opt, &st); err != nil {
				return fmt.Errorf("resuming from %s: %w", dir, err)
			}
			cfg.logf("resumed from %s at step %d (%d examples)", dir, st.Step, st.Examples)
		}
	}

	evalItems := PrepareEval(tok, valExs, cfg.EvalPerTask, cfg.KMax, cfg.MaxLen, cfg.Independent)
	cfg.logf("validation subset: %d examples", len(evalItems))

	cache := nn.NewCache(model)
	start := time.Now()
	offset := st.Elapsed
	var order []int
	orderEpoch := -1
	rngStep := rand.New(rand.NewSource(cfg.Seed))

	save := func() error {
		st.Elapsed = offset + time.Since(start).Seconds()
		dir := filepath.Join(cfg.OutDir, fmt.Sprintf("ckpt-%06d", st.Step))
		if err := saveCheckpoint(dir, model, opt, &st); err != nil {
			return err
		}
		pruneCheckpoints(cfg.OutDir, cfg.KeepCkpts)
		cfg.logf("saved %s", dir)
		return nil
	}
	evaluate := func() {
		if len(evalItems) == 0 {
			return
		}
		t0 := time.Now()
		logits, err := Predict(model, evalItems, 2000)
		if err != nil {
			cfg.logf("eval failed: %v", err)
			return
		}
		rep := Score(evalItems, logits, 1)
		cfg.logf("step %d eval (%.0fs): acc %.1f%% nll %.3f ece %.3f", st.Step, time.Since(t0).Seconds(), 100*rep.Overall.Acc, rep.Overall.NLL, rep.Overall.ECE)
		if b, err := json.Marshal(map[string]any{"step": st.Step, "report": rep}); err == nil {
			appendLine(filepath.Join(cfg.OutDir, "eval.jsonl"), b)
		}
	}

	for st.Step < totalSteps {
		if ctx.Err() != nil {
			cfg.logf("interrupted; saving checkpoint")
			return save()
		}
		stepStart := time.Now()
		// Draw this step's examples from the epoch-shuffled stream.
		var items []Item
		for n := 0; n < cfg.BatchExamples; n++ {
			epoch, pos := st.Examples/perEpoch, st.Examples%perEpoch
			if epoch != orderEpoch {
				order = rand.New(rand.NewSource(cfg.Seed*1000003 + int64(epoch))).Perm(perEpoch)
				orderEpoch = epoch
			}
			st.Examples++
			e := &all[order[pos]]
			r := Realize(e, rngStep, cfg.KMax)
			if it, ok := Tokenize(tok, &r, cfg.MaxLen, cfg.Independent); ok {
				items = append(items, it)
			}
		}
		if len(items) == 0 {
			continue
		}

		var lossSum, correct float64
		tokens := 0
		for lo := 0; lo < len(items); {
			hi, tk := lo, 0
			for hi < len(items) && (hi == lo || tk+len(items[hi].Seq.IDs) <= cfg.TokenBudget) {
				tk += len(items[hi].Seq.IDs)
				hi++
			}
			micro := items[lo:hi]
			batch := make([]nn.Sequence, len(micro))
			for i := range micro {
				batch[i] = micro[i].Seq
			}
			logits, err := model.Forward(cache, batch, cfg.TrainFrom)
			if err != nil {
				return err
			}
			dl := make([]float32, len(logits))
			off := 0
			for i := range micro {
				k := len(micro[i].Target)
				p := Softmax(logits[off:off+k], 1)
				best := 0
				for j := range p {
					if p[j] > p[best] {
						best = j
					}
					lossSum -= float64(micro[i].Target[j]) * math.Log(math.Max(p[j], 1e-12))
					dl[off+j] = float32((p[j] - float64(micro[i].Target[j])) / float64(len(items)))
				}
				if best == micro[i].Gold {
					correct++
				}
				off += k
			}
			model.Backward(cache, dl, grads, cfg.TrainEmb)
			tokens += tk
			lo = hi
		}
		gn := opt.GradNorm()
		scale := float32(1)
		if cfg.ClipNorm > 0 && gn > cfg.ClipNorm {
			scale = float32(cfg.ClipNorm / (gn + 1e-6))
		}
		lr, headLR := lrAt(&cfg, st.Step, totalSteps)
		opt.Update(float32(lr), float32(headLR), scale)
		grads.Zero()
		st.Step++

		loss, acc := lossSum/float64(len(items)), correct/float64(len(items))
		if st.Step == 1 || st.LossEMA == 0 {
			st.LossEMA, st.AccEMA = loss, acc
		} else {
			st.LossEMA = 0.95*st.LossEMA + 0.05*loss
			st.AccEMA = 0.95*st.AccEMA + 0.05*acc
		}
		if st.Step%cfg.LogEvery == 0 || st.Step == totalSteps {
			eta := time.Duration(float64(totalSteps-st.Step) * time.Since(stepStart).Seconds() * float64(time.Second))
			cfg.logf("step %d/%d loss %.4f (ema %.4f) acc %.2f (ema %.2f) lr %.2e gnorm %.2f  %.0f tok/s  eta %s",
				st.Step, totalSteps, loss, st.LossEMA, acc, st.AccEMA, lr, gn, float64(tokens)/time.Since(stepStart).Seconds(), eta.Round(time.Minute))
		}
		if st.Step%cfg.EvalEvery == 0 {
			evaluate()
		}
		if st.Step%cfg.SaveEvery == 0 && st.Step < totalSteps {
			if err := save(); err != nil {
				return err
			}
		}
	}
	if st.Step%cfg.EvalEvery != 0 {
		evaluate()
	}
	if err := save(); err != nil {
		return err
	}
	final := filepath.Join(cfg.OutDir, "model.safetensors")
	if err := model.Save(final, map[string]string{"format": "decide", "steps": fmt.Sprint(st.Step)}); err != nil {
		return err
	}
	cfg.logf("training finished: %s", final)
	return nil
}

func appendLine(path string, b []byte) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	f.Write(append(b, '\n'))
}

func saveCheckpoint(dir string, model *nn.Model, opt *AdamW, st *State) error {
	tmp := dir + ".tmp"
	if err := os.MkdirAll(tmp, 0o755); err != nil {
		return err
	}
	if err := model.Save(filepath.Join(tmp, "model.safetensors"), nil); err != nil {
		return err
	}
	if err := opt.M.Save(filepath.Join(tmp, "adam_m.safetensors"), nil); err != nil {
		return err
	}
	if err := opt.V.Save(filepath.Join(tmp, "adam_v.safetensors"), nil); err != nil {
		return err
	}
	sb, _ := json.MarshalIndent(struct {
		*State
		AdamStep int `json:"adam_step"`
	}{st, opt.Step}, "", "  ")
	if err := os.WriteFile(filepath.Join(tmp, "state.json"), sb, 0o644); err != nil {
		return err
	}
	os.RemoveAll(dir)
	return os.Rename(tmp, dir)
}

func loadCheckpoint(dir string, model *nn.Model, opt *AdamW, st *State) error {
	if err := model.LoadWeights(filepath.Join(dir, "model.safetensors")); err != nil {
		return err
	}
	if err := opt.M.LoadWeights(filepath.Join(dir, "adam_m.safetensors")); err != nil {
		return err
	}
	if err := opt.V.LoadWeights(filepath.Join(dir, "adam_v.safetensors")); err != nil {
		return err
	}
	raw, err := os.ReadFile(filepath.Join(dir, "state.json"))
	if err != nil {
		return err
	}
	var s struct {
		State
		AdamStep int `json:"adam_step"`
	}
	if err := json.Unmarshal(raw, &s); err != nil {
		return err
	}
	*st = s.State
	opt.Step = s.AdamStep
	return nil
}

func latestCheckpoint(out string) string {
	dirs, _ := filepath.Glob(filepath.Join(out, "ckpt-*"))
	var ok []string
	for _, d := range dirs {
		if !strings.HasSuffix(d, ".tmp") {
			if _, err := os.Stat(filepath.Join(d, "state.json")); err == nil {
				ok = append(ok, d)
			}
		}
	}
	sort.Strings(ok)
	if len(ok) == 0 {
		return ""
	}
	return ok[len(ok)-1]
}

func pruneCheckpoints(out string, keep int) {
	dirs, _ := filepath.Glob(filepath.Join(out, "ckpt-*"))
	sort.Strings(dirs)
	for len(dirs) > keep {
		os.RemoveAll(dirs[0])
		dirs = dirs[1:]
	}
}
