package train

import (
	"math"
	"runtime"
	"strings"
	"sync"

	"github.com/jmwri/decide/internal/nn"
)

// AdamW is decoupled-weight-decay Adam over a nn.Model's parameters.
type AdamW struct {
	Beta1, Beta2, Eps float32
	WeightDecay       float32

	params []nn.NamedTensor
	grads  []nn.NamedTensor
	M, V   *nn.Model
	m, v   []nn.NamedTensor
	Step   int

	// Frozen reports tensors that must not be updated.
	frozen []bool
	head   []bool
	decay  []bool
}

// NewAdamW builds an optimizer. trainFrom freezes encoder layers below it and
// trainEmb controls the token embeddings.
func NewAdamW(model, grads *nn.Model, wd float32, trainFrom int, trainEmb bool) *AdamW {
	o := &AdamW{Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, WeightDecay: wd}
	o.M, o.V = nn.NewLike(model), nn.NewLike(model)
	o.params, o.grads, o.m, o.v = model.Tensors(), grads.Tensors(), o.M.Tensors(), o.V.Tensors()
	for _, t := range o.params {
		frozen, head, decay := false, strings.HasPrefix(t.Name, "scorer."), len(t.Shape) == 2
		switch {
		case t.Name == "model.embeddings.tok_embeddings.weight":
			frozen = !trainEmb || trainFrom > 0
			decay = false
		case t.Name == "model.embeddings.norm.weight":
			frozen = trainFrom > 0
		case strings.HasPrefix(t.Name, "model.layers."):
			var l int
			rest := strings.TrimPrefix(t.Name, "model.layers.")
			for _, ch := range rest {
				if ch < '0' || ch > '9' {
					break
				}
				l = l*10 + int(ch-'0')
			}
			frozen = l < trainFrom
		}
		o.frozen = append(o.frozen, frozen)
		o.head = append(o.head, head)
		o.decay = append(o.decay, decay)
	}
	return o
}

// GradNorm returns the global L2 norm of the gradients of trainable tensors.
func (o *AdamW) GradNorm() float64 {
	var mu sync.Mutex
	var total float64
	var wg sync.WaitGroup
	for i, g := range o.grads {
		if o.frozen[i] {
			continue
		}
		wg.Add(1)
		go func(d []float32) {
			defer wg.Done()
			var s float64
			for _, v := range d {
				s += float64(v) * float64(v)
			}
			mu.Lock()
			total += s
			mu.Unlock()
		}(g.Data)
	}
	wg.Wait()
	return math.Sqrt(total)
}

// Update applies one AdamW step with the given learning rates; the gradients
// are first multiplied by gradScale (for clipping).
func (o *AdamW) Update(lr, headLR float32, gradScale float32) {
	o.Step++
	b1c := 1 - math.Pow(float64(o.Beta1), float64(o.Step))
	b2c := 1 - math.Pow(float64(o.Beta2), float64(o.Step))
	stepSize := float32(1 / b1c)
	invSqrtB2c := float32(1 / math.Sqrt(b2c))

	type job struct {
		i      int
		lo, hi int
	}
	var jobs []job
	const chunk = 1 << 18
	for i, p := range o.params {
		if o.frozen[i] {
			continue
		}
		for lo := 0; lo < len(p.Data); lo += chunk {
			jobs = append(jobs, job{i, lo, min(lo+chunk, len(p.Data))})
		}
	}
	next := make(chan job, len(jobs))
	for _, j := range jobs {
		next <- j
	}
	close(next)
	var wg sync.WaitGroup
	for w := 0; w < runtime.GOMAXPROCS(0); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range next {
				rate := lr
				if o.head[j.i] {
					rate = headLR
				}
				wd := float32(0)
				if o.decay[j.i] {
					wd = o.WeightDecay * rate
				}
				p, g := o.params[j.i].Data[j.lo:j.hi], o.grads[j.i].Data[j.lo:j.hi]
				m, v := o.m[j.i].Data[j.lo:j.hi], o.v[j.i].Data[j.lo:j.hi]
				for k := range p {
					gr := g[k] * gradScale
					m[k] = o.Beta1*m[k] + (1-o.Beta1)*gr
					v[k] = o.Beta2*v[k] + (1-o.Beta2)*gr*gr
					den := float32(math.Sqrt(float64(v[k])))*invSqrtB2c + o.Eps
					p[k] -= rate*stepSize*m[k]/den + wd*p[k]
				}
			}
		}()
	}
	wg.Wait()
}
