// Package bundle defines the on-disk model format shared by training (which
// writes it) and inference (which reads it): a directory holding
//
//	model.safetensors  encoder + scorer weights (see nn.Model.Tensors for names)
//	config.json        the ModernBERT architecture (HuggingFace format)
//	tokenizer.json     the tokenizer (HuggingFace format)
//	decide.json        metadata: model id, attention mode, calibration, provenance
package bundle

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/jmwri/decide/internal/nn"
	"github.com/jmwri/decide/internal/tokenizer"
)

// FormatVersion identifies the packing layout and scoring conventions. A
// bundle with a different version cannot be run by this code.
const FormatVersion = 1

// Calibration holds softmax temperatures fitted on held-out data. Temperature
// changes how confident a probability claims to be, never which option wins.
type Calibration struct {
	// Temperature by question kind ("choice", "noul", "score").
	Temperature map[string]float64 `json:"temperature"`
	// Default applies to kinds with no fitted value.
	Default float64 `json:"default"`
	// Examples is how many held-out examples each fit used.
	Examples map[string]int `json:"examples,omitempty"`
	// ECEBefore / ECEAfter are the expected calibration errors on the fit set.
	ECEBefore map[string]float64 `json:"ece_before,omitempty"`
	ECEAfter  map[string]float64 `json:"ece_after,omitempty"`
}

// TemperatureFor returns the temperature to use for a question kind.
func (c *Calibration) TemperatureFor(kind string) float64 {
	if c != nil {
		if t, ok := c.Temperature[kind]; ok && t > 0 {
			return t
		}
		if c.Default > 0 {
			return c.Default
		}
	}
	return 1
}

// Meta is decide.json.
type Meta struct {
	Name    string `json:"name"`
	ModelID string `json:"model_id"`
	Format  int    `json:"format"`
	// IndependentOptions: each option attends only to the premise and itself,
	// with position ids restarting after the premise, so scores are provably
	// invariant to option order.
	IndependentOptions bool         `json:"independent_options"`
	MaxTokens          int          `json:"max_tokens"`
	Calibration        *Calibration `json:"calibration,omitempty"`
	Base               string       `json:"base_model,omitempty"`
	Training           any          `json:"training,omitempty"`
}

// Bundle is a loaded model.
type Bundle struct {
	Dir   string
	Meta  Meta
	Model *nn.Model
	Tok   *tokenizer.Tokenizer
}

// PackText lays out a premise and its candidate options as
//
//	<instructions> <state> [SEP] [MASK] option0 [MASK] option1 ...
//
// The encoder reads it in one pass and the scorer reads the hidden state at
// every [MASK]. Training and inference must use this exact layout.
func PackText(state, instructions string, options []string) string {
	var prefix string
	if instructions != "" {
		prefix = strings.TrimSpace(instructions + " " + state)
	} else {
		prefix = strings.TrimSpace(state)
	}
	parts := make([]string, len(options))
	for i, o := range options {
		parts[i] = "[MASK] " + strings.TrimSpace(o)
	}
	return prefix + " [SEP] " + strings.Join(parts, " ")
}

// Load reads a bundle directory.
func Load(dir string) (*Bundle, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "decide.json"))
	if err != nil {
		return nil, fmt.Errorf("bundle: %w", err)
	}
	var meta Meta
	if err := json.Unmarshal(raw, &meta); err != nil {
		return nil, fmt.Errorf("bundle: decide.json: %w", err)
	}
	if meta.Format != FormatVersion {
		return nil, fmt.Errorf("bundle: %s has format %d, this build reads format %d", dir, meta.Format, FormatVersion)
	}
	cfg, err := nn.LoadConfig(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	model := nn.New(cfg)
	if err := model.LoadWeights(filepath.Join(dir, "model.safetensors")); err != nil {
		return nil, err
	}
	tok, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	return &Bundle{Dir: dir, Meta: meta, Model: model, Tok: tok}, nil
}

// Save writes a bundle. configPath and tokenizerPath are copied verbatim
// (the ModernBERT config and tokenizer the model was trained with).
func Save(dir string, model *nn.Model, configPath, tokenizerPath string, meta Meta) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	meta.Format = FormatVersion
	if err := model.Save(filepath.Join(dir, "model.safetensors"), map[string]string{"format": "pt"}); err != nil {
		return err
	}
	for src, name := range map[string]string{configPath: "config.json", tokenizerPath: "tokenizer.json"} {
		b, err := os.ReadFile(src)
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
			return err
		}
	}
	mb, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "decide.json"), mb, 0o644)
}
