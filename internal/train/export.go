package train

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/jmwri/decide/internal/bundle"
	"github.com/jmwri/decide/internal/data"
	"github.com/jmwri/decide/internal/nn"
	"github.com/jmwri/decide/internal/tokenizer"
)

// ExportConfig describes a bundle to produce from a trained model.
type ExportConfig struct {
	ModelPath string // model.safetensors from training
	BaseDir   string // ModernBERT-base directory (config.json, tokenizer.json)
	DataDir   string // corpus (val.jsonl for calibration, ood.jsonl for reporting)
	OutDir    string
	ModelID   string
	KMax      int
	GPU       string // "auto" (default), "on" or "off"
	Log       func(format string, args ...any)
}

// Export calibrates a trained model on held-out data and writes a bundle.
func Export(cfg ExportConfig) error {
	logf := func(f string, a ...any) {
		if cfg.Log != nil {
			cfg.Log(f, a...)
		}
	}
	if cfg.KMax == 0 {
		cfg.KMax = 10
	}
	mcfg, err := nn.LoadConfig(filepath.Join(cfg.BaseDir, "config.json"))
	if err != nil {
		return err
	}
	model := nn.New(mcfg)
	if err := model.LoadWeights(cfg.ModelPath); err != nil {
		return err
	}
	tok, err := tokenizer.Load(filepath.Join(cfg.BaseDir, "tokenizer.json"))
	if err != nil {
		return err
	}
	valExs, err := data.ReadJSONL(filepath.Join(cfg.DataDir, "val.jsonl"))
	if err != nil {
		return err
	}
	fwd, closeFn, where, err := NewInferer(model, cfg.GPU, logf)
	if err != nil {
		return err
	}
	defer closeFn()
	logf("scoring on %s", where)
	val := PrepareEval(tok, valExs, 0, cfg.KMax, 384, true)
	logf("calibrating on %d validation examples", len(val))
	valLogits, err := PredictWith(fwd, val, 4000)
	if err != nil {
		return err
	}
	cal := FitTemperatures(val, valLogits)
	logf("temperatures: %v (default %.3f)", cal.Temperature, cal.Default)

	valRep := ScoreWith(val, valLogits, cal.TemperatureFor)
	logf("validation (calibrated):\n%s", valRep.String())

	reports := map[string]any{"val": valRep, "val_uncalibrated": Score(val, valLogits, 1)}
	if oodExs, err := data.ReadJSONL(filepath.Join(cfg.DataDir, "ood.jsonl")); err == nil && len(oodExs) > 0 {
		ood := PrepareEval(tok, oodExs, 0, cfg.KMax, 384, true)
		oodLogits, err := PredictWith(fwd, ood, 4000)
		if err != nil {
			return err
		}
		rep := ScoreWith(ood, oodLogits, cal.TemperatureFor)
		reports["ood"] = rep
		logf("held-out tasks (never trained on):\n%s", rep.String())
	}
	dev, err := OrderInvariance(model, tok, valExs, 60, cfg.KMax)
	if err != nil {
		return err
	}
	logf("order invariance: max |Δlogit| under option permutation = %.2e", dev)
	reports["order_invariance_max_logit_delta"] = dev

	training := map[string]any{"reports": reports}
	if raw, err := os.ReadFile(filepath.Join(cfg.DataDir, "manifest.json")); err == nil {
		var m any
		if json.Unmarshal(raw, &m) == nil {
			training["corpus"] = m
		}
	}
	meta := bundle.Meta{
		Name: "decide", ModelID: cfg.ModelID, IndependentOptions: true, MaxTokens: 8192,
		Calibration: cal, Base: "answerdotai/ModernBERT-base", Training: training,
	}
	if err := bundle.Save(cfg.OutDir, model, filepath.Join(cfg.BaseDir, "config.json"), filepath.Join(cfg.BaseDir, "tokenizer.json"), meta); err != nil {
		return err
	}
	rb, _ := json.MarshalIndent(reports, "", "  ")
	if err := os.WriteFile(filepath.Join(cfg.OutDir, "eval.json"), rb, 0o644); err != nil {
		return err
	}
	logf("wrote bundle %s", cfg.OutDir)
	return nil
}

// OrderInvariance measures how much option logits move when the options of an
// example are presented in a different order. The independent-options
// attention makes this exactly zero up to floating-point noise.
func OrderInvariance(model *nn.Model, tok *tokenizer.Tokenizer, exs []data.Example, n, kMax int) (float64, error) {
	rng := rand.New(rand.NewSource(5))
	var worst float64
	done := 0
	for i := range exs {
		if done >= n {
			break
		}
		e := exs[i]
		if e.Ordinal || len(e.Options) < 3 || len(e.Options) > kMax {
			continue
		}
		perm := rng.Perm(len(e.Options))
		if isIdentity(perm) {
			continue
		}
		p := e
		p.Options = make([]string, len(perm))
		for slot, k := range perm {
			p.Options[slot] = e.Options[k]
		}
		a, ok1 := Tokenize(tok, &e, 384, true)
		b, ok2 := Tokenize(tok, &p, 384, true)
		if !ok1 || !ok2 {
			continue
		}
		out, err := Predict(model, []Item{a, b}, 1<<20)
		if err != nil {
			return 0, err
		}
		for slot, k := range perm {
			worst = math.Max(worst, math.Abs(float64(out[1][slot]-out[0][k])))
		}
		done++
	}
	return worst, nil
}

func isIdentity(p []int) bool {
	for i, v := range p {
		if i != v {
			return false
		}
	}
	return true
}
