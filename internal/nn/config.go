// Package nn is a from-scratch ModernBERT encoder with a hand-written
// backward pass, used to fine-tune the Decide option-scoring model on the CPU.
package nn

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config holds the ModernBERT hyper-parameters.
type Config struct {
	Hidden       int
	Intermediate int
	Heads        int
	Layers       int
	Vocab        int
	Eps          float32
	// HalfWindow is the one-sided sliding-window radius (local_attention / 2).
	HalfWindow  int
	LayerTypes  []string // "full_attention" | "sliding_attention" per layer
	GlobalTheta float64
	LocalTheta  float64
}

// HeadDim is the per-head width.
func (c Config) HeadDim() int { return c.Hidden / c.Heads }

// Sliding reports whether layer i uses local attention.
func (c Config) Sliding(i int) bool { return c.LayerTypes[i] == "sliding_attention" }

// LoadConfig reads a HuggingFace ModernBERT config.json.
func LoadConfig(filename string) (Config, error) {
	raw, err := os.ReadFile(filename)
	if err != nil {
		return Config{}, err
	}
	var f struct {
		Hidden         int      `json:"hidden_size"`
		Intermediate   int      `json:"intermediate_size"`
		Heads          int      `json:"num_attention_heads"`
		Layers         int      `json:"num_hidden_layers"`
		Vocab          int      `json:"vocab_size"`
		NormEps        float64  `json:"norm_eps"`
		LocalAttention int      `json:"local_attention"`
		GlobalEvery    int      `json:"global_attn_every_n_layers"`
		LayerTypes     []string `json:"layer_types"`
		RopeParameters map[string]struct {
			Theta float64 `json:"rope_theta"`
		} `json:"rope_parameters"`
		GlobalRopeTheta float64 `json:"global_rope_theta"`
		LocalRopeTheta  float64 `json:"local_rope_theta"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		return Config{}, fmt.Errorf("nn: parsing %s: %w", filename, err)
	}
	c := Config{
		Hidden: f.Hidden, Intermediate: f.Intermediate, Heads: f.Heads, Layers: f.Layers, Vocab: f.Vocab,
		Eps: float32(f.NormEps), HalfWindow: f.LocalAttention / 2, LayerTypes: f.LayerTypes,
		GlobalTheta: 160000, LocalTheta: 10000,
	}
	if c.Eps == 0 {
		c.Eps = 1e-5
	}
	if c.HalfWindow == 0 {
		c.HalfWindow = 64
	}
	if p, ok := f.RopeParameters["full_attention"]; ok && p.Theta > 0 {
		c.GlobalTheta = p.Theta
	} else if f.GlobalRopeTheta > 0 {
		c.GlobalTheta = f.GlobalRopeTheta
	}
	if p, ok := f.RopeParameters["sliding_attention"]; ok && p.Theta > 0 {
		c.LocalTheta = p.Theta
	} else if f.LocalRopeTheta > 0 {
		c.LocalTheta = f.LocalRopeTheta
	}
	if len(c.LayerTypes) == 0 {
		every := f.GlobalEvery
		if every == 0 {
			every = 3
		}
		for i := 0; i < c.Layers; i++ {
			if i%every == 0 {
				c.LayerTypes = append(c.LayerTypes, "full_attention")
			} else {
				c.LayerTypes = append(c.LayerTypes, "sliding_attention")
			}
		}
	}
	if c.Hidden == 0 || c.Heads == 0 || c.Layers == 0 || len(c.LayerTypes) != c.Layers || c.Hidden%c.Heads != 0 || c.Hidden%2 != 0 {
		return Config{}, fmt.Errorf("nn: incomplete or inconsistent config in %s", filename)
	}
	return c, nil
}
