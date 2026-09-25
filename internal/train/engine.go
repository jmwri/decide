package train

import (
	"errors"
	"fmt"
	"math"

	"github.com/jmwri/decide/internal/cuda"
	"github.com/jmwri/decide/internal/gpunn"
	"github.com/jmwri/decide/internal/nn"
)

// Engine runs one training step's forward and backward passes and the
// optimizer, on the CPU or on an NVIDIA GPU.
type Engine interface {
	// Forward runs a training forward pass and returns one logit per option.
	Forward(batch []nn.Sequence) ([]float32, error)
	// Infer runs a forward pass with no intention of a backward pass.
	Infer(batch []nn.Sequence) ([]float32, error)
	// Backward accumulates gradients for the loss derivative w.r.t. the logits.
	Backward(dlogits []float32) error
	// GradNorm is the global L2 norm of the accumulated trainable gradients.
	GradNorm() float64
	// Update applies one AdamW step, scaling gradients by scale first.
	Update(lr, headLR, scale float32)
	ZeroGrad()
	// PrepareSave makes the host model and optimizer moments current
	// (a no-op on the CPU; copies device state back on the GPU).
	PrepareSave() error
	Close()
	Name() string
}

// GPU selection modes.
const (
	GPUAuto = "auto" // use a GPU when one is available
	GPUOn   = "on"   // require a GPU
	GPUOff  = "off"  // always use the CPU
)

// openDevice returns the GPU device according to mode, or nil for the CPU.
func openDevice(mode string, logf func(string, ...any)) (*cuda.Device, error) {
	switch mode {
	case GPUOff:
		return nil, nil
	case "", GPUAuto, GPUOn:
	default:
		return nil, fmt.Errorf("unknown --gpu mode %q (want auto, on or off)", mode)
	}
	dev, err := cuda.Open()
	if err != nil {
		if mode == GPUOn {
			return nil, err
		}
		if !errors.Is(err, cuda.ErrNoGPU) {
			logf("GPU unavailable (%v); using the CPU", err)
		}
		return nil, nil
	}
	return dev, nil
}

// ---- CPU engine

type cpuEngine struct {
	model     *nn.Model
	grads     *nn.Model
	opt       *AdamW
	cache     *nn.Cache
	trainFrom int
	trainEmb  bool
}

func newCPUEngine(model, grads *nn.Model, opt *AdamW, cfg *Config) *cpuEngine {
	return &cpuEngine{model: model, grads: grads, opt: opt, cache: nn.NewCache(model), trainFrom: cfg.TrainFrom, trainEmb: cfg.TrainEmb}
}

func (e *cpuEngine) Forward(batch []nn.Sequence) ([]float32, error) {
	return e.model.Forward(e.cache, batch, e.trainFrom)
}
func (e *cpuEngine) Infer(batch []nn.Sequence) ([]float32, error) {
	return e.model.Forward(e.cache, batch, 1<<20)
}
func (e *cpuEngine) Backward(dl []float32) error {
	e.model.Backward(e.cache, dl, e.grads, e.trainEmb)
	return nil
}
func (e *cpuEngine) GradNorm() float64 { return e.opt.GradNorm() }
func (e *cpuEngine) Update(lr, headLR, scale float32) {
	e.opt.Update(lr, headLR, scale)
}
func (e *cpuEngine) ZeroGrad()          { e.grads.Zero() }
func (e *cpuEngine) PrepareSave() error { return nil }
func (e *cpuEngine) Close()             {}
func (e *cpuEngine) Name() string       { return "CPU" }

// ---- GPU engine

type gpuEngine struct {
	dev   *cuda.Device
	gm    *gpunn.Model
	grads *nn.Model
	opt   *AdamW // head-only: the encoder is optimised on the device
	wd    float32
}

func newGPUEngine(dev *cuda.Device, model, grads *nn.Model, opt *AdamW, cfg *Config) (*gpuEngine, error) {
	gm, err := gpunn.New(dev, model)
	if err != nil {
		return nil, err
	}
	gm.SetTrainable(cfg.TrainFrom, cfg.TrainEmb)
	if opt.Step > 0 { // resuming: restore the encoder's Adam moments
		if err := gm.UploadAdam(opt.M, opt.V, opt.Step); err != nil {
			return nil, err
		}
	}
	return &gpuEngine{dev: dev, gm: gm, grads: grads, opt: opt, wd: float32(cfg.WeightDecay)}, nil
}

func (e *gpuEngine) Forward(batch []nn.Sequence) ([]float32, error) { return e.gm.Forward(batch) }
func (e *gpuEngine) Infer(batch []nn.Sequence) ([]float32, error)   { return e.gm.Forward(batch) }
func (e *gpuEngine) Backward(dl []float32) error                    { return e.gm.Backward(dl, e.grads) }
func (e *gpuEngine) GradNorm() float64 {
	h := e.opt.GradNorm()
	return math.Sqrt(e.gm.GradSumSq() + h*h)
}
func (e *gpuEngine) Update(lr, headLR, scale float32) {
	e.gm.Update(lr, e.wd, scale)
	e.opt.Update(lr, headLR, scale)
}
func (e *gpuEngine) ZeroGrad() {
	e.gm.ZeroGrad()
	e.grads.Zero()
}
func (e *gpuEngine) PrepareSave() error {
	if err := e.gm.SyncToHost(); err != nil {
		return err
	}
	return e.gm.DownloadAdam(e.opt.M, e.opt.V)
}
func (e *gpuEngine) Close() {
	e.gm.Free()
	e.dev.Close()
}
func (e *gpuEngine) Name() string {
	return fmt.Sprintf("GPU (%s, %d SMs)", e.dev.Name, e.dev.SMs)
}

// NewInferer returns a forward function for evaluation and export, on the GPU
// when mode allows and one is present, plus a close function and a label.
func NewInferer(model *nn.Model, mode string, logf func(string, ...any)) (fwd func([]nn.Sequence) ([]float32, error), closeFn func(), name string, err error) {
	dev, err := openDevice(mode, logf)
	if err != nil {
		return nil, nil, "", err
	}
	if dev == nil {
		cache := nn.NewCache(model)
		return func(b []nn.Sequence) ([]float32, error) { return model.Forward(cache, b, 1<<20) }, func() {}, "CPU", nil
	}
	gm, err := gpunn.New(dev, model)
	if err != nil {
		dev.Close()
		return nil, nil, "", err
	}
	return gm.Forward, func() { gm.Free(); dev.Close() }, fmt.Sprintf("GPU (%s)", dev.Name), nil
}
