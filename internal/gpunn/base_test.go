package gpunn

import (
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/jmwri/decide/internal/nn"
)

func loadBase(t testing.TB) *nn.Model {
	dir := os.Getenv("DECIDE_BASE_DIR")
	if dir == "" {
		t.Skip("DECIDE_BASE_DIR not set")
	}
	cfg, err := nn.LoadConfig(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	m := nn.New(cfg)
	if err := m.LoadBase(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	m.InitHead(rand.New(rand.NewSource(1)))
	return m
}

func randBatch(rng *rand.Rand, n, prefix int) []nn.Sequence {
	const maskID = 50284
	var batch []nn.Sequence
	for i := 0; i < n; i++ {
		ids := []int32{50281}
		for j := 0; j < prefix+rng.Intn(40); j++ {
			ids = append(ids, int32(rng.Intn(40000)+100))
		}
		ids = append(ids, 50282)
		for k := 0; k < 3+rng.Intn(3); k++ {
			ids = append(ids, maskID)
			for j := 0; j < 6+rng.Intn(10); j++ {
				ids = append(ids, int32(rng.Intn(40000)+100))
			}
		}
		ids = append(ids, 50282)
		batch = append(batch, nn.BuildSequence(ids, maskID, i%4 != 3))
	}
	return batch
}

// TestBaseMatchesCPU compares the GPU with the CPU on real ModernBERT-base weights.
func TestBaseMatchesCPU(t *testing.T) {
	dev := openOrSkip(t)
	cpu := loadBase(t)
	g, err := New(dev, clone(t, cpu))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()
	batch := randBatch(rand.New(rand.NewSource(5)), 6, 80)
	want, err := cpu.Forward(nn.NewCache(cpu), batch, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	got, err := g.Forward(batch)
	if err != nil {
		t.Fatal(err)
	}
	e := relErr(got, want)
	t.Logf("forward on real weights: rel err %g", e)
	if e > 1e-3 {
		t.Fatalf("logits differ: rel err %g", e)
	}
}

func TestBaseStepSpeed(t *testing.T) {
	dev := openOrSkip(t)
	host := loadBase(t)
	g, err := New(dev, host)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()
	grads := nn.NewLike(host)
	rng := rand.New(rand.NewSource(6))
	for _, cfg := range []struct{ seqs, prefix int }{{8, 60}, {20, 80}, {40, 80}} {
		batch := randBatch(rng, cfg.seqs, cfg.prefix)
		tokens := 0
		for _, s := range batch {
			tokens += len(s.IDs)
		}
		for rep := 0; rep < 3; rep++ {
			t0 := time.Now()
			logits, err := g.Forward(batch)
			if err != nil {
				t.Fatal(err)
			}
			tf := time.Since(t0)
			dl := make([]float32, len(logits))
			for i := range dl {
				dl[i] = 0.01
			}
			t1 := time.Now()
			if err := g.Backward(dl, grads); err != nil {
				t.Fatal(err)
			}
			tb := time.Since(t1)
			if rep == 2 {
				t.Logf("%d seqs, %d tokens: forward %v backward %v  => %.0f tok/s", cfg.seqs, tokens, tf, tb, float64(tokens)/(tf+tb).Seconds())
			}
		}
	}
	free, total := uint64(0), uint64(0)
	dev.Do(func() { free, total = dev.MemInfo() })
	t.Logf("GPU memory: %d MB used of %d MB", (total-free)>>20, total>>20)
}
