package train

import (
	"math"
	"sort"

	"github.com/jmwri/decide/internal/bundle"
)

func meanNLL(items []Item, logits [][]float32, idx []int, temp float64) float64 {
	var s float64
	for _, i := range idx {
		p := Softmax(logits[i], temp)
		s -= math.Log(math.Max(p[items[i].Gold], 1e-12))
	}
	return s / float64(len(idx))
}

// FitTemperatures fits one temperature per kind by minimising the negative
// log-likelihood of the gold option on held-out logits (a 1-D convex problem
// in 1/T, solved by golden-section search).
func FitTemperatures(items []Item, logits [][]float32) *bundle.Calibration {
	cal := &bundle.Calibration{
		Temperature: map[string]float64{}, Examples: map[string]int{},
		ECEBefore: map[string]float64{}, ECEAfter: map[string]float64{},
	}
	byKind := map[string][]int{}
	var all []int
	for i, it := range items {
		byKind[it.Kind] = append(byKind[it.Kind], i)
		all = append(all, i)
	}
	fit := func(idx []int) float64 {
		// search over log T in [ln 0.25, ln 16]
		lo, hi := math.Log(0.25), math.Log(16)
		const phi = 0.6180339887498949
		a, b := hi-phi*(hi-lo), lo+phi*(hi-lo)
		fa, fb := meanNLL(items, logits, idx, math.Exp(a)), meanNLL(items, logits, idx, math.Exp(b))
		for it := 0; it < 60; it++ {
			if fa < fb {
				hi, b, fb = b, a, fa
				a = hi - phi*(hi-lo)
				fa = meanNLL(items, logits, idx, math.Exp(a))
			} else {
				lo, a, fa = a, b, fb
				b = lo + phi*(hi-lo)
				fb = meanNLL(items, logits, idx, math.Exp(b))
			}
		}
		return math.Exp((lo + hi) / 2)
	}
	cal.Default = fit(all)
	kinds := make([]string, 0, len(byKind))
	for k := range byKind {
		kinds = append(kinds, k)
	}
	sort.Strings(kinds)
	for _, k := range kinds {
		idx := byKind[k]
		cal.Examples[k] = len(idx)
		if len(idx) < 50 { // too few to fit reliably
			continue
		}
		cal.Temperature[k] = fit(idx)
	}
	// ECE before/after, by kind.
	for _, k := range kinds {
		sub := make([]Item, len(byKind[k]))
		sl := make([][]float32, len(byKind[k]))
		for j, i := range byKind[k] {
			sub[j], sl[j] = items[i], logits[i]
		}
		cal.ECEBefore[k] = Score(sub, sl, 1).Overall.ECE
		cal.ECEAfter[k] = Score(sub, sl, cal.TemperatureFor(k)).Overall.ECE
	}
	return cal
}
