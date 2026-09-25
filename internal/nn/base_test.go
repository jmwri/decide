package nn

import (
	"encoding/json"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func loadBase(t testing.TB) *Model {
	dir := os.Getenv("DECIDE_BASE_DIR")
	if dir == "" {
		t.Skip("DECIDE_BASE_DIR not set (directory with ModernBERT-base config.json + model.safetensors)")
	}
	cfg, err := LoadConfig(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	m := New(cfg)
	if err := m.LoadBase(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	m.InitHead(rand.New(rand.NewSource(1)))
	return m
}

// TestBaseStepSpeed times a full forward+backward step on real ModernBERT-base.
func TestBaseStepSpeed(t *testing.T) {
	m := loadBase(t)
	t.Logf("%d parameters, %d layers", m.NumParams(), m.Cfg.Layers)
	rng := rand.New(rand.NewSource(2))
	const maskID = 50284
	mk := func(prefix int, opts []int) Sequence {
		ids := []int32{50281}
		for i := 0; i < prefix; i++ {
			ids = append(ids, int32(rng.Intn(40000)+100))
		}
		ids = append(ids, 50282)
		for _, n := range opts {
			ids = append(ids, maskID)
			for i := 0; i < n; i++ {
				ids = append(ids, int32(rng.Intn(40000)+100))
			}
		}
		ids = append(ids, 50282)
		return BuildSequence(ids, maskID, true)
	}
	var batch []Sequence
	for i := 0; i < 8; i++ {
		batch = append(batch, mk(60, []int{10, 12, 9, 11}))
	}
	tokens := 0
	for _, s := range batch {
		tokens += len(s.IDs)
	}
	c := NewCache(m)
	g := NewLike(m)
	for _, trainFrom := range []int{0, 12} {
		for rep := 0; rep < 2; rep++ {
			start := time.Now()
			logits, err := m.Forward(c, batch, trainFrom)
			if err != nil {
				t.Fatal(err)
			}
			fwd := time.Since(start)
			dl := make([]float32, len(logits))
			for i := range dl {
				dl[i] = 0.01
			}
			start = time.Now()
			m.Backward(c, dl, g, true)
			t.Logf("trainFrom=%d tokens=%d forward %v backward %v (%.0f tok/s)", trainFrom, tokens, fwd, time.Since(start), float64(tokens)/(fwd+time.Since(start)).Seconds())
		}
	}
}

// TestBaseMatchesHuggingFace compares encoder output at the [MASK] rows with
// HuggingFace transformers on the real ModernBERT-base weights. The reference
// is produced by DECIDE_BASE_REF (a JSON file written by a small transformers
// script); it exercises both attention modes and the 128-token local window.
func TestBaseMatchesHuggingFace(t *testing.T) {
	m := loadBase(t)
	refPath := os.Getenv("DECIDE_BASE_REF")
	if refPath == "" {
		t.Skip("DECIDE_BASE_REF not set")
	}
	raw, err := os.ReadFile(refPath)
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		MaskID int32 `json:"mask_id"`
		Cases  []struct {
			IDs         []int32     `json:"ids"`
			Independent bool        `json:"independent"`
			Rows        [][]float32 `json:"rows"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	H := m.Cfg.Hidden
	for i, cs := range ref.Cases {
		c := NewCache(m)
		if _, err := m.Forward(c, []Sequence{BuildSequence(cs.IDs, ref.MaskID, cs.Independent)}, 99); err != nil {
			t.Fatal(err)
		}
		var got, want []float32
		for r, row := range cs.Rows {
			got = append(got, c.fN[r*H:(r+1)*H]...)
			want = append(want, row...)
		}
		e := relErr(got, want)
		t.Logf("case %d (%d tokens, independent=%v): rel err %g", i, len(cs.IDs), cs.Independent, e)
		if e > 1e-3 {
			t.Errorf("case %d differs from HuggingFace: rel err %g", i, e)
		}
	}
}

// TestBaseInferenceLatency times single-sequence forward passes (the shape of
// one inference request) on real ModernBERT-base.
func TestBaseInferenceLatency(t *testing.T) {
	m := loadBase(t)
	rng := rand.New(rand.NewSource(3))
	const maskID = 50284
	for _, prefix := range []int{30, 100, 300} {
		ids := []int32{50281}
		for i := 0; i < prefix; i++ {
			ids = append(ids, int32(rng.Intn(40000)+100))
		}
		ids = append(ids, 50282)
		for k := 0; k < 4; k++ {
			ids = append(ids, maskID)
			for i := 0; i < 10; i++ {
				ids = append(ids, int32(rng.Intn(40000)+100))
			}
		}
		ids = append(ids, 50282)
		seq := BuildSequence(ids, maskID, true)
		c := NewCache(m)
		var best time.Duration
		for rep := 0; rep < 4; rep++ {
			start := time.Now()
			if _, err := m.Forward(c, []Sequence{seq}, 99); err != nil {
				t.Fatal(err)
			}
			if d := time.Since(start); rep == 0 || d < best {
				best = d
			}
		}
		t.Logf("%d tokens: %v", len(ids), best)
	}
}
