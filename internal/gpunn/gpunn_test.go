package gpunn

import (
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/jmwri/decide/internal/cuda"
	"github.com/jmwri/decide/internal/nn"
	"github.com/jmwri/decide/internal/safetensors"
)

func openOrSkip(t *testing.T) *cuda.Device {
	t.Helper()
	d, err := cuda.Open()
	if err != nil {
		t.Skipf("no GPU: %v", err)
	}
	t.Cleanup(d.Close)
	return d
}

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

// loadTiny loads the small random model and batch that internal/nn checks
// against PyTorch (same file), so the GPU is compared with a verified CPU path.
func loadTiny(t *testing.T) (*nn.Model, []nn.Sequence, []int) {
	t.Helper()
	f, err := safetensors.Open("../nn/testdata/tiny_ref.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var meta refMeta
	if err := json.Unmarshal([]byte(f.Metadata["meta"]), &meta); err != nil {
		t.Fatal(err)
	}
	c := meta.Config
	m := nn.New(nn.Config{
		Hidden: c.Hidden, Intermediate: c.Intermediate, Heads: c.Heads, Layers: c.Layers, Vocab: c.Vocab, Eps: 1e-5,
		HalfWindow: c.HalfWindow, GlobalTheta: c.GlobalTheta, LocalTheta: c.LocalTheta,
		LayerTypes: []string{"full_attention", "sliding_attention", "sliding_attention", "full_attention"},
	})
	for _, nt := range m.Tensors() {
		data, _, err := f.Float32("w." + nt.Name)
		if err != nil {
			t.Fatal(err)
		}
		copy(nt.Data, data)
	}
	var batch []nn.Sequence
	var targets []int
	for _, s := range meta.Seqs {
		batch = append(batch, nn.BuildSequence(s.IDs, meta.MaskID, s.Independent))
		targets = append(targets, s.Target)
	}
	return m, batch, targets
}

func clone(t *testing.T, m *nn.Model) *nn.Model {
	c := nn.New(m.Cfg)
	src := m.Tensors()
	for i, nt := range c.Tensors() {
		copy(nt.Data, src[i].Data)
	}
	return c
}

func softmaxCEGrad(logits []float32, batch []nn.Sequence, targets []int) []float32 {
	d := make([]float32, len(logits))
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
				d[off+i] -= 1 / float32(len(batch))
			}
		}
		off += k
	}
	return d
}

func relErr(got, want []float32) float64 {
	var num, den float64
	for i := range got {
		d := float64(got[i]) - float64(want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	return math.Sqrt(num) / (math.Sqrt(den) + 1e-4)
}

func TestForwardBackwardMatchesCPU(t *testing.T) {
	dev := openOrSkip(t)
	for _, trainFrom := range []int{0, 2} {
		cpu, batch, targets := loadTiny(t)
		gpuHost := clone(t, cpu)
		g, err := New(dev, gpuHost)
		if err != nil {
			t.Fatal(err)
		}
		g.SetTrainable(trainFrom, true)

		c := nn.NewCache(cpu)
		want, err := cpu.Forward(c, batch, trainFrom)
		if err != nil {
			t.Fatal(err)
		}
		got, err := g.Forward(batch)
		if err != nil {
			t.Fatal(err)
		}
		if e := relErr(got, want); e > 1e-4 {
			t.Fatalf("trainFrom=%d logits rel err %g\n got %v\nwant %v", trainFrom, e, got, want)
		}

		dl := softmaxCEGrad(want, batch, targets)
		gCPU := nn.NewLike(cpu)
		cpu.Backward(c, dl, gCPU, true)
		gGPU := nn.NewLike(gpuHost)
		if err := g.Backward(dl, gGPU); err != nil {
			t.Fatal(err)
		}
		if err := g.DownloadGrads(gGPU); err != nil {
			t.Fatal(err)
		}
		wt := map[string][]float32{}
		for _, nt := range gCPU.Tensors() {
			wt[nt.Name] = nt.Data
		}
		for _, nt := range gGPU.Tensors() {
			if e := relErr(nt.Data, wt[nt.Name]); e > 2e-3 {
				t.Errorf("trainFrom=%d grad %s rel err %g", trainFrom, nt.Name, e)
			}
		}
		g.Free()
	}
}

// One optimizer step on the device must match the CPU AdamW implementation.
func TestAdamWStepMatchesCPU(t *testing.T) {
	dev := openOrSkip(t)
	cpu, _, _ := loadTiny(t)
	host := clone(t, cpu)
	g, err := New(dev, host)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()
	g.SetTrainable(1, false) // freeze layer 0 and embeddings

	// Give every encoder parameter a deterministic gradient.
	gradHost := nn.NewLike(host)
	for _, nt := range gradHost.Tensors() {
		for i := range nt.Data {
			nt.Data[i] = float32(math.Sin(float64(i)*0.37)) * 0.1
		}
	}
	dev.Do(func() {
		byName := map[string][]float32{}
		for _, nt := range gradHost.Tensors() {
			byName[nt.Name] = nt.Data
		}
		for _, p := range g.params {
			if err := p.g.Upload(0, byName[p.name]); err != nil {
				t.Fatal(err)
			}
		}
	})
	if ss := g.GradSumSq(); ss <= 0 {
		t.Fatal("grad norm should be positive")
	}
	const lr, wd = 1e-2, 0.01
	g.Update(lr, wd, 0.5)
	if err := g.SyncToHost(); err != nil {
		t.Fatal(err)
	}
	after := map[string][]float32{}
	for _, nt := range host.Tensors() {
		after[nt.Name] = nt.Data
	}
	for _, nt := range cpu.Tensors() {
		got := after[nt.Name]
		encoder := strings.HasPrefix(nt.Name, "model.embeddings.") || strings.HasPrefix(nt.Name, "model.layers.")
		if len(got) == 0 || !encoder { // the final norm and scorer head live on the CPU
			continue
		}
		frozen := nt.Name == "model.embeddings.tok_embeddings.weight" || nt.Name == "model.embeddings.norm.weight" ||
			len(nt.Name) > 15 && nt.Name[:15] == "model.layers.0."
		if frozen {
			if relErr(got, nt.Data) != 0 {
				t.Errorf("%s is frozen but changed", nt.Name)
			}
			continue
		}
		// reference: AdamW step 1 with grad*0.5
		wdeff := float32(0)
		if len(nt.Shape) == 2 {
			wdeff = wd * lr
		}
		gt := map[string][]float32{}
		for _, x := range gradHost.Tensors() {
			gt[x.Name] = x.Data
		}
		want := make([]float32, len(nt.Data))
		for i, w0 := range nt.Data {
			gr := gt[nt.Name][i] * 0.5
			m := (1 - 0.9) * gr
			v := (1 - 0.999) * gr * gr
			den := float32(math.Sqrt(float64(v)))/float32(math.Sqrt(1-0.999)) + 1e-8
			want[i] = w0 - float32(lr)*m/(1-0.9)/den - wdeff*w0
		}
		if e := relErr(got, want); e > 1e-4 {
			t.Errorf("%s: adam step rel err %g", nt.Name, e)
		}
	}
}
