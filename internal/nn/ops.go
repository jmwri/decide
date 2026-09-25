package nn

import (
	"math"
	"runtime"
	"sync"
)

// parallelFor splits [0,n) into contiguous chunks, one goroutine each.
func parallelFor(n int, fn func(lo, hi int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		if n > 0 {
			fn(0, n)
		}
		return
	}
	var wg sync.WaitGroup
	chunk := (n + workers - 1) / workers
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// grow returns a slice of length n reusing b's storage when possible. The
// contents are unspecified.
func grow(b []float32, n int) []float32 {
	if cap(b) < n {
		return make([]float32, n)
	}
	return b[:n]
}

func growI(b []int32, n int) []int32 {
	if cap(b) < n {
		return make([]int32, n)
	}
	return b[:n]
}

// layerNormFwd computes y = (x-mean)*rstd*w (+b) per row, recording mean and rstd.
func layerNormFwd(y, x []float32, rows, dim int, w, b []float32, eps float32, mean, rstd []float32) {
	parallelFor(rows, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row := x[r*dim : (r+1)*dim]
			var sum float64
			for _, v := range row {
				sum += float64(v)
			}
			mu := sum / float64(dim)
			var vs float64
			for _, v := range row {
				d := float64(v) - mu
				vs += d * d
			}
			rs := float32(1.0 / math.Sqrt(vs/float64(dim)+float64(eps)))
			m := float32(mu)
			mean[r], rstd[r] = m, rs
			o := y[r*dim : (r+1)*dim]
			if b != nil {
				for i, v := range row {
					o[i] = (v-m)*rs*w[i] + b[i]
				}
			} else {
				for i, v := range row {
					o[i] = (v - m) * rs * w[i]
				}
			}
		}
	})
}

// layerNormBwd back-propagates through layerNormFwd. It ADDS the input
// gradient into dx (so a residual gradient can be pre-loaded there) and
// accumulates dw (and db when non-nil).
func layerNormBwd(dx, dy, x []float32, rows, dim int, w []float32, mean, rstd []float32, dw, db []float32) {
	var mu sync.Mutex
	parallelFor(rows, func(lo, hi int) {
		pw := make([]float32, dim)
		var pb []float32
		if db != nil {
			pb = make([]float32, dim)
		}
		xh := make([]float32, dim)
		for r := lo; r < hi; r++ {
			row := x[r*dim : (r+1)*dim]
			g := dy[r*dim : (r+1)*dim]
			m, rs := mean[r], rstd[r]
			var s1, s2 float64
			for i := 0; i < dim; i++ {
				xh[i] = (row[i] - m) * rs
				gw := g[i] * w[i]
				s1 += float64(gw)
				s2 += float64(gw * xh[i])
				pw[i] += g[i] * xh[i]
				if pb != nil {
					pb[i] += g[i]
				}
			}
			a, b := float32(s1/float64(dim)), float32(s2/float64(dim))
			o := dx[r*dim : (r+1)*dim]
			for i := 0; i < dim; i++ {
				o[i] += rs * (g[i]*w[i] - a - xh[i]*b)
			}
		}
		mu.Lock()
		for i := range pw {
			dw[i] += pw[i]
		}
		if pb != nil {
			for i := range pb {
				db[i] += pb[i]
			}
		}
		mu.Unlock()
	})
}

func gelu(x float32) float32 {
	return 0.5 * x * (1 + float32(math.Erf(float64(x)*math.Sqrt2/2)))
}

// geluGrad is d gelu(x) / dx.
func geluGrad(x float32) float32 {
	xf := float64(x)
	cdf := 0.5 * (1 + math.Erf(xf*math.Sqrt2/2))
	pdf := math.Exp(-0.5*xf*xf) / math.Sqrt(2*math.Pi)
	return float32(cdf + xf*pdf)
}

// applyRope rotates the q and k slices of every (token, head) in place.
// qkv rows are laid out [q | k | v], each H wide, heads of width D.
// With inverse, the rotation is transposed (the backward pass).
func applyRope(qkv []float32, T, H, heads, D int, cos, sin []float32, inverse bool) {
	half := D / 2
	stride := 3 * H
	parallelFor(T, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			for hd := 0; hd < heads; hd++ {
				for _, off := range [2]int{hd * D, H + hd*D} {
					v := qkv[t*stride+off : t*stride+off+D]
					for i := 0; i < half; i++ {
						c, s := cos[t*half+i], sin[t*half+i]
						if inverse {
							s = -s
						}
						a, b := v[i], v[i+half]
						v[i] = a*c - b*s
						v[i+half] = b*c + a*s
					}
				}
			}
		}
	})
}
