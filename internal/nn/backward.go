package nn

import (
	"math"

	"github.com/jmwri/decide/internal/blas"
)

func zeros(b []float32, n int) []float32 {
	b = grow(b, n)
	clear(b)
	return b
}

// Backward accumulates into g the gradient of a loss whose derivative with
// respect to the option logits is dlogits (one entry per [MASK] row of the
// preceding Forward). Encoder layers below the Forward's trainFrom receive no
// gradient; the token embeddings are updated only when trainEmb is set and
// trainFrom is 0.
func (m *Model) Backward(c *Cache, dlogits []float32, g *Model, trainEmb bool) {
	cfg := m.Cfg
	H, I, D, L := cfg.Hidden, cfg.Intermediate, cfg.HeadDim(), cfg.Layers
	T, R := c.T, c.R
	hL := c.hs[L]
	rows := make([]float32, R*H)
	for r, t := range c.rowTok {
		copy(rows[r*H:(r+1)*H], hL[t*H:(t+1)*H])
	}
	dRows := m.HeadBackward(c, rows, dlogits, g)
	c.dcur = zeros(c.dcur, T*H)
	for r, t := range c.rowTok {
		copy(c.dcur[t*H:(t+1)*H], dRows[r*H:(r+1)*H])
	}

	// ---- encoder layers
	c.dh1 = grow(c.dh1, T*H)
	c.dact = grow(c.dact, T*I)
	c.dwi = grow(c.dwi, T*2*I)
	c.dn2 = zeros(c.dn2, T*H)
	c.dctx = grow(c.dctx, T*H)
	c.dqkv = grow(c.dqkv, T*3*H)
	c.dn1 = grow(c.dn1, T*H)
	c.dprev = grow(c.dprev, T*H)

	for l := L - 1; l >= c.trainFrom && l >= 0; l-- {
		ly, gl := &m.Layers[l], &g.Layers[l]
		lc := &c.lc[l]
		in := c.hs[l]
		dout := c.dcur

		// MLP: out = h1 + Wo(act)
		blas.Sgemm(false, false, T, I, H, dout, H, ly.MlpWo, I, c.dact, I, false)
		blas.Sgemm(true, false, H, I, T, dout, H, lc.act, I, gl.MlpWo, I, true)
		parallelFor(T, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				row := lc.wi[t*2*I : (t+1)*2*I]
				da := c.dact[t*I : (t+1)*I]
				dw := c.dwi[t*2*I : (t+1)*2*I]
				for i := 0; i < I; i++ {
					a, gt := row[i], row[I+i]
					dw[i] = da[i] * gt * geluGrad(a)
					dw[I+i] = da[i] * gelu(a)
				}
			}
		})
		blas.Sgemm(true, false, 2*I, H, T, c.dwi, 2*I, lc.n2, H, gl.Wi, H, true)
		blas.Sgemm(false, false, T, H, 2*I, c.dwi, 2*I, ly.Wi, H, c.dn2, H, false)
		copy(c.dh1, dout) // residual path
		layerNormBwd(c.dh1, c.dn2, lc.h1, T, H, ly.MlpNorm, lc.mean2, lc.rstd2, gl.MlpNorm, nil)

		// Attention: h1 = in + Wo(ctx)
		blas.Sgemm(false, false, T, H, H, c.dh1, H, ly.Wo, H, c.dctx, H, false)
		blas.Sgemm(true, false, H, H, T, c.dh1, H, lc.ctx, H, gl.Wo, H, true)
		c.attnBwd(c.dqkv, c.dctx, lc.qkv, c.pkeep[l])
		cos, sin := c.cosG, c.sinG
		if cfg.Sliding(l) {
			cos, sin = c.cosL, c.sinL
		}
		applyRope(c.dqkv, T, H, cfg.Heads, D, cos, sin, true)
		n1 := in
		if ly.AttnNorm != nil {
			n1 = lc.n1
		}
		blas.Sgemm(true, false, 3*H, H, T, c.dqkv, 3*H, n1, H, gl.Wqkv, H, true)
		blas.Sgemm(false, false, T, H, 3*H, c.dqkv, 3*H, ly.Wqkv, H, c.dn1, H, false)
		copy(c.dprev, c.dh1) // residual path
		if ly.AttnNorm != nil {
			layerNormBwd(c.dprev, c.dn1, in, T, H, ly.AttnNorm, lc.mean1, lc.rstd1, gl.AttnNorm, nil)
		} else {
			parallelFor(T*H, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					c.dprev[i] += c.dn1[i]
				}
			})
		}
		c.dcur, c.dprev = c.dprev, c.dcur
		clear(c.dn2) // layerNormBwd adds into its dx; dn2 is overwritten by Sgemm each layer, kept clear for safety
	}

	// ---- embeddings
	if trainEmb && c.trainFrom == 0 {
		dE := zeros(c.dprev, T*H)
		layerNormBwd(dE, c.dcur, c.embX, T, H, m.EmbNorm, c.embMean, c.embRstd, g.EmbNorm, nil)
		for t := 0; t < T; t++ {
			id := int(c.ids[t])
			dst := g.Tok[id*H : (id+1)*H]
			src := dE[t*H : (t+1)*H]
			for i, v := range src {
				dst[i] += v
			}
		}
		c.dprev = dE
	}
}

// attnBwd back-propagates through the attention of one layer into dqkv
// (before the RoPE inverse).
func (c *Cache) attnBwd(dqkv, dctx, qkv, P []float32) {
	cfg := c.m.Cfg
	H, D, heads := cfg.Hidden, cfg.HeadDim(), cfg.Heads
	stride := 3 * H
	scale := float32(1 / math.Sqrt(float64(D)))
	items := len(c.seqs) * heads
	parallelFor(items, func(lo, hi int) {
		var dP []float32
		for it := lo; it < hi; it++ {
			si, hd := it/heads, it%heads
			s := &c.seqs[si]
			n := s.n
			p := P[s.pOff+hd*n*n : s.pOff+(hd+1)*n*n]
			dP = grow(dP, n*n)
			q := qkv[s.off*stride+hd*D:]
			k := qkv[s.off*stride+H+hd*D:]
			v := qkv[s.off*stride+2*H+hd*D:]
			dc := dctx[s.off*H+hd*D:]
			// dP = dctx · Vᵀ
			blas.Sgemm1(false, true, n, n, D, dc, H, v, stride, dP, n, false)
			// softmax backward, folding in the score scale: dS = P ⊙ (dP − Σ P dP) · scale
			for i := 0; i < n; i++ {
				pr, dr := p[i*n:(i+1)*n], dP[i*n:(i+1)*n]
				var dot float32
				for j := range pr {
					dot += pr[j] * dr[j]
				}
				for j := range pr {
					dr[j] = pr[j] * (dr[j] - dot) * scale
				}
			}
			// dV = Pᵀ · dctx ; dQ = dS · K ; dK = dSᵀ · Q
			blas.Sgemm1(true, false, n, D, n, p, n, dc, H, dqkv[s.off*stride+2*H+hd*D:], stride, false)
			blas.Sgemm1(false, false, n, D, n, dP, n, k, stride, dqkv[s.off*stride+hd*D:], stride, false)
			blas.Sgemm1(true, false, n, D, n, dP, n, q, stride, dqkv[s.off*stride+H+hd*D:], stride, false)
		}
	})
}

// HeadBackward back-propagates dlogits through the scorer head and the final
// norm, accumulating parameter gradients into g. rows are the same R x H
// encoder rows passed to HeadForward; the result is the gradient with respect
// to them.
func (m *Model) HeadBackward(c *Cache, rows, dlogits []float32, g *Model) []float32 {
	cfg := m.Cfg
	H := cfg.Hidden
	R := c.R
	half := H / 2
	h, gh := &m.Head, &g.Head

	ds2 := make([]float32, R*half)
	for r := 0; r < R; r++ {
		dl := dlogits[r]
		gh.OutB[0] += dl
		row := c.s2[r*half : (r+1)*half]
		for i := 0; i < half; i++ {
			gh.OutW[i] += dl * row[i]
			ds2[r*half+i] = dl * h.OutW[i]
		}
	}
	dge := make([]float32, R*half)
	layerNormBwd(dge, ds2, c.ge, R, half, h.NormW, c.m2, c.r2, gh.NormW, gh.NormB)
	dd := make([]float32, R*half)
	for r := 0; r < R; r++ {
		for i := 0; i < half; i++ {
			v := dge[r*half+i] * geluGrad(c.d[r*half+i])
			dd[r*half+i] = v
			gh.DenseB[i] += v
		}
	}
	blas.Sgemm(true, false, half, H, R, dd, half, c.s1, H, gh.DenseW, H, true)
	ds1 := make([]float32, R*H)
	blas.Sgemm(false, false, R, H, half, dd, half, h.DenseW, H, ds1, H, false)
	dfN := make([]float32, R*H)
	layerNormBwd(dfN, ds1, c.fN, R, H, h.InW, c.m1, c.r1, gh.InW, gh.InB)
	dRows := make([]float32, R*H)
	layerNormBwd(dRows, dfN, rows, R, H, m.FinalNorm, c.fMean, c.fRstd, g.FinalNorm, nil)
	return dRows
}
