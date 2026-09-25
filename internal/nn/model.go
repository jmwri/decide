package nn

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/jmwri/decide/internal/safetensors"
)

// Layer holds one encoder layer's weights (or gradients).
type Layer struct {
	AttnNorm []float32 // nil in layer 0 (identity)
	Wqkv     []float32 // (3H, H)
	Wo       []float32 // (H, H)
	MlpNorm  []float32 // (H)
	Wi       []float32 // (2I, H): rows [0,I) are the value half, [I,2I) the gate half
	MlpWo    []float32 // (H, I)
}

// Scorer is the option-scoring head: LN -> Linear(H,H/2) -> GELU -> LN -> Linear(H/2,1).
type Scorer struct {
	InW, InB     []float32 // (H)
	DenseW       []float32 // (H/2, H)
	DenseB       []float32 // (H/2)
	NormW, NormB []float32 // (H/2)
	OutW         []float32 // (H/2)
	OutB         []float32 // (1)
}

// Model is an encoder plus scorer. The same type doubles as a gradient
// container (see NewLike).
type Model struct {
	Cfg       Config
	Tok       []float32 // (Vocab, H)
	EmbNorm   []float32 // (H)
	Layers    []Layer
	FinalNorm []float32 // (H)
	Head      Scorer
}

// NamedTensor is a flat view of one parameter.
type NamedTensor struct {
	Name  string
	Shape []int
	Data  []float32
}

// Tensors lists every parameter in a stable order. Names follow the
// HuggingFace ModernBERT layout ("model.*") plus "scorer.*" for the head.
func (m *Model) Tensors() []NamedTensor {
	H, I := m.Cfg.Hidden, m.Cfg.Intermediate
	t := []NamedTensor{
		{"model.embeddings.tok_embeddings.weight", []int{m.Cfg.Vocab, H}, m.Tok},
		{"model.embeddings.norm.weight", []int{H}, m.EmbNorm},
	}
	for i := range m.Layers {
		l := &m.Layers[i]
		p := fmt.Sprintf("model.layers.%d.", i)
		if l.AttnNorm != nil {
			t = append(t, NamedTensor{p + "attn_norm.weight", []int{H}, l.AttnNorm})
		}
		t = append(t,
			NamedTensor{p + "attn.Wqkv.weight", []int{3 * H, H}, l.Wqkv},
			NamedTensor{p + "attn.Wo.weight", []int{H, H}, l.Wo},
			NamedTensor{p + "mlp_norm.weight", []int{H}, l.MlpNorm},
			NamedTensor{p + "mlp.Wi.weight", []int{2 * I, H}, l.Wi},
			NamedTensor{p + "mlp.Wo.weight", []int{H, I}, l.MlpWo},
		)
	}
	h := &m.Head
	t = append(t,
		NamedTensor{"model.final_norm.weight", []int{H}, m.FinalNorm},
		NamedTensor{"scorer.input_norm.weight", []int{H}, h.InW},
		NamedTensor{"scorer.input_norm.bias", []int{H}, h.InB},
		NamedTensor{"scorer.dense.weight", []int{H / 2, H}, h.DenseW},
		NamedTensor{"scorer.dense.bias", []int{H / 2}, h.DenseB},
		NamedTensor{"scorer.norm.weight", []int{H / 2}, h.NormW},
		NamedTensor{"scorer.norm.bias", []int{H / 2}, h.NormB},
		NamedTensor{"scorer.out_proj.weight", []int{1, H / 2}, h.OutW},
		NamedTensor{"scorer.out_proj.bias", []int{1}, h.OutB},
	)
	return t
}

// NumParams counts parameters.
func (m *Model) NumParams() int {
	n := 0
	for _, t := range m.Tensors() {
		n += len(t.Data)
	}
	return n
}

// New allocates a zeroed model.
func New(cfg Config) *Model {
	H, I := cfg.Hidden, cfg.Intermediate
	m := &Model{Cfg: cfg,
		Tok:       make([]float32, cfg.Vocab*H),
		EmbNorm:   make([]float32, H),
		FinalNorm: make([]float32, H),
		Head: Scorer{
			InW: make([]float32, H), InB: make([]float32, H),
			DenseW: make([]float32, H/2*H), DenseB: make([]float32, H/2),
			NormW: make([]float32, H/2), NormB: make([]float32, H/2),
			OutW: make([]float32, H/2), OutB: make([]float32, 1),
		},
	}
	for i := 0; i < cfg.Layers; i++ {
		l := Layer{
			Wqkv: make([]float32, 3*H*H), Wo: make([]float32, H*H), MlpNorm: make([]float32, H),
			Wi: make([]float32, 2*I*H), MlpWo: make([]float32, H*I),
		}
		if i > 0 {
			l.AttnNorm = make([]float32, H)
		}
		m.Layers = append(m.Layers, l)
	}
	return m
}

// NewLike allocates a zeroed model with the same shapes (for gradients and optimizer state).
func NewLike(m *Model) *Model { return New(m.Cfg) }

// Zero clears every value.
func (m *Model) Zero() {
	for _, t := range m.Tensors() {
		clear(t.Data)
	}
}

// InitHead initialises the scorer head with small random weights.
func (m *Model) InitHead(rng *rand.Rand) {
	h := &m.Head
	for i := range h.InW {
		h.InW[i], h.InB[i] = 1, 0
	}
	for i := range h.NormW {
		h.NormW[i], h.NormB[i] = 1, 0
	}
	for i := range h.DenseW {
		h.DenseW[i] = float32(rng.NormFloat64() * 0.02)
	}
	for i := range h.OutW {
		h.OutW[i] = float32(rng.NormFloat64() * 0.02)
	}
	clear(h.DenseB)
	clear(h.OutB)
}

// InitRandom fills every weight randomly (tests and from-scratch runs).
func (m *Model) InitRandom(rng *rand.Rand, std float64) {
	fill := func(x []float32) {
		for i := range x {
			x[i] = float32(rng.NormFloat64() * std)
		}
	}
	fill(m.Tok)
	for i := range m.EmbNorm {
		m.EmbNorm[i], m.FinalNorm[i] = 1, 1
	}
	for i := range m.Layers {
		l := &m.Layers[i]
		fill(l.Wqkv)
		fill(l.Wo)
		fill(l.Wi)
		fill(l.MlpWo)
		for j := range l.MlpNorm {
			l.MlpNorm[j] = 1
		}
		for j := range l.AttnNorm {
			l.AttnNorm[j] = 1
		}
	}
	m.InitHead(rng)
}

// LoadBase loads encoder weights from a HuggingFace ModernBERT safetensors
// file (tensor names prefixed "model."), leaving the scorer head untouched.
func (m *Model) LoadBase(path string) error {
	f, err := safetensors.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	byName := map[string]NamedTensor{}
	for _, t := range m.Tensors() {
		byName[t.Name] = t
	}
	for name, t := range byName {
		if len(name) < 6 || name[:6] != "model." {
			continue
		}
		data, shape, err := f.Float32(name)
		if err != nil {
			return err
		}
		if len(data) != len(t.Data) {
			return fmt.Errorf("nn: %s has shape %v, model expects %v", name, shape, t.Shape)
		}
		copy(t.Data, data)
	}
	return nil
}

// Save writes every parameter to a safetensors file.
func (m *Model) Save(path string, meta map[string]string) error {
	ts := m.Tensors()
	out := make([]safetensors.Tensor, len(ts))
	for i, t := range ts {
		out[i] = safetensors.Tensor{Name: t.Name, Shape: t.Shape, Data: t.Data}
	}
	return safetensors.Write(path, out, meta)
}

// LoadWeights reads a file written by Save into an existing model.
func (m *Model) LoadWeights(path string) error {
	f, err := safetensors.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	for _, t := range m.Tensors() {
		data, shape, err := f.Float32(t.Name)
		if err != nil {
			return err
		}
		if len(data) != len(t.Data) {
			return fmt.Errorf("nn: %s has shape %v, model expects %v", t.Name, shape, t.Shape)
		}
		copy(t.Data, data)
	}
	return nil
}

func ropeTables(pos []int32, headDim int, theta float64) (cos, sin []float32) {
	half := headDim / 2
	inv := make([]float64, half)
	for i := range inv {
		inv[i] = 1.0 / math.Pow(theta, float64(2*i)/float64(headDim))
	}
	cos = make([]float32, len(pos)*half)
	sin = make([]float32, len(pos)*half)
	for s, p := range pos {
		for i := 0; i < half; i++ {
			f := float64(float32(float64(p) * inv[i]))
			cos[s*half+i] = float32(math.Cos(f))
			sin[s*half+i] = float32(math.Sin(f))
		}
	}
	return
}
