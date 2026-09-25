package nn

import (
	"fmt"
	"math"

	"github.com/jmwri/decide/internal/blas"
)

// Sequence is one packed input: a premise followed by option spans, each
// introduced by a [MASK] token.
type Sequence struct {
	IDs []int32
	// Pos are RoPE position ids (and the basis of the sliding window).
	Pos []int32
	// OptID is -1 for prefix tokens and k for tokens of option k; nil means
	// plain bidirectional attention.
	OptID []int32
	// MaskPos are the indices of the option-start tokens, one per option.
	MaskPos []int
}

// BuildSequence derives positions, option ids and mask positions from token
// ids. With independent set, every option attends only to the shared prefix
// and to itself, and each option's position ids restart right after the
// prefix, so an option's score depends on (premise, that option) alone and
// permuting options provably permutes the scores.
func BuildSequence(ids []int32, maskID int32, independent bool) Sequence {
	s := Sequence{IDs: ids, Pos: make([]int32, len(ids))}
	for i, id := range ids {
		s.Pos[i] = int32(i)
		if id == maskID {
			s.MaskPos = append(s.MaskPos, i)
		}
	}
	if !independent || len(s.MaskPos) == 0 {
		return s
	}
	last := len(ids) - 1 // the trailing [SEP] belongs to no option
	s.OptID = make([]int32, len(ids))
	for i := range s.OptID {
		s.OptID[i] = -1
	}
	prefix := s.MaskPos[0]
	for k, start := range s.MaskPos {
		end := last
		if k+1 < len(s.MaskPos) {
			end = s.MaskPos[k+1]
		}
		for i := start; i < end; i++ {
			s.OptID[i] = int32(k)
			s.Pos[i] = int32(prefix + i - start)
		}
	}
	return s
}

type seqInfo struct {
	off, n int
	pOff   int     // offset of this sequence's attention probabilities
	full   []uint8 // n*n allowed matrix for global layers; nil = all allowed
	slide  []uint8 // n*n allowed matrix for local layers
}

type layerCache struct {
	n1, mean1, rstd1 []float32
	qkv              []float32
	ctx              []float32
	h1               []float32
	n2, mean2, rstd2 []float32
	wi, act          []float32
	tmp              []float32
}

// Cache holds one forward pass's activations and reusable work buffers. It
// can be reused across steps; buffers only grow.
type Cache struct {
	m         *Model
	trainFrom int
	T, R      int
	seqs      []seqInfo
	ids       []int32

	cosG, sinG, cosL, sinL []float32

	embX, embMean, embRstd []float32
	hs                     [][]float32 // hs[l] = input of layer l; hs[L] = encoder output
	ping, pong             []float32
	lc                     []layerCache
	frozen                 layerCache
	pbuf                   []float32 // kept attention probabilities for trainable layers, per layer
	pkeep                  [][]float32

	rowTok []int
	// scorer
	fN, fMean, fRstd    []float32
	s1, m1, r1          []float32
	d, ge               []float32
	s2, m2, r2          []float32
	logits              []float32
	dlogits             []float32
	dcur, dh1, dact     []float32
	dwi, dn2, dctx      []float32
	dqkv, dn1, dprev    []float32
	dRows, dTmpA, dTmpB []float32
	dPscratch           [][]float32
}

// NewCache allocates an empty cache for m.
func NewCache(m *Model) *Cache {
	c := &Cache{m: m}
	c.lc = make([]layerCache, m.Cfg.Layers)
	c.pkeep = make([][]float32, m.Cfg.Layers)
	c.hs = make([][]float32, m.Cfg.Layers+1)
	return c
}

// Logits returns the option logits of the last Forward, one per [MASK] row in
// batch order.
func (c *Cache) Logits() []float32 { return c.logits }

func linearFwd(y, x []float32, T, K int, w []float32, N int) {
	blas.Sgemm(false, true, T, N, K, x, K, w, K, y, N, false)
}

// Forward runs the encoder and scorer over a batch. Layers below trainFrom
// keep no activations (they are frozen); pass trainFrom = 0 to train all
// layers, or a value >= Layers to train the scorer head only.
func (m *Model) Forward(c *Cache, batch []Sequence, trainFrom int) ([]float32, error) {
	cfg := m.Cfg
	H, I, D, L := cfg.Hidden, cfg.Intermediate, cfg.HeadDim(), cfg.Layers
	if c.m != m {
		return nil, fmt.Errorf("nn: cache belongs to a different model")
	}
	c.trainFrom = min(max(trainFrom, 0), L+1)

	// Lay out the batch.
	c.seqs = c.seqs[:0]
	c.ids = c.ids[:0]
	var pos []int32
	T, pTotal, R := 0, 0, 0
	for si := range batch {
		s := &batch[si]
		n := len(s.IDs)
		if n == 0 || len(s.Pos) != n || (s.OptID != nil && len(s.OptID) != n) {
			return nil, fmt.Errorf("nn: malformed sequence %d", si)
		}
		if len(s.MaskPos) == 0 {
			return nil, fmt.Errorf("nn: sequence %d has no options", si)
		}
		info := seqInfo{off: T, n: n, pOff: pTotal}
		if s.OptID != nil {
			info.full = make([]uint8, n*n)
			for i := 0; i < n; i++ {
				oi := s.OptID[i]
				for j := 0; j < n; j++ {
					oj := s.OptID[j]
					if (oi == -1 && oj == -1) || (oi != -1 && (oj == -1 || oi == oj)) {
						info.full[i*n+j] = 1
					}
				}
			}
		}
		info.slide = make([]uint8, n*n)
		w := int32(cfg.HalfWindow)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				d := s.Pos[i] - s.Pos[j]
				if d < 0 {
					d = -d
				}
				if d <= w && (info.full == nil || info.full[i*n+j] == 1) {
					info.slide[i*n+j] = 1
				}
			}
		}
		c.seqs = append(c.seqs, info)
		c.ids = append(c.ids, s.IDs...)
		pos = append(pos, s.Pos...)
		for _, mp := range s.MaskPos {
			if mp < 0 || mp >= n {
				return nil, fmt.Errorf("nn: sequence %d mask position %d out of range", si, mp)
			}
			c.rowTok = append(c.rowTok[:R], T+mp)
			R++
		}
		T += n
		pTotal += n * n * cfg.Heads
	}
	c.T, c.R = T, R
	c.rowTok = c.rowTok[:R]
	for _, id := range c.ids {
		if id < 0 || int(id) >= cfg.Vocab {
			return nil, fmt.Errorf("nn: token id %d out of range", id)
		}
	}

	c.cosG, c.sinG = ropeTables(pos, D, cfg.GlobalTheta)
	c.cosL, c.sinL = ropeTables(pos, D, cfg.LocalTheta)

	// Embeddings.
	c.embX = grow(c.embX, T*H)
	c.embMean, c.embRstd = grow(c.embMean, T), grow(c.embRstd, T)
	parallelFor(T, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			id := int(c.ids[t])
			copy(c.embX[t*H:(t+1)*H], m.Tok[id*H:(id+1)*H])
		}
	})
	for l := 0; l <= L; l++ {
		if l >= c.trainFrom || l == L {
			c.hs[l] = grow(c.hs[l], T*H)
		}
	}
	c.ping, c.pong = grow(c.ping, T*H), grow(c.pong, T*H)
	cur := c.ping
	if c.trainFrom == 0 {
		cur = c.hs[0]
	}
	layerNormFwd(cur, c.embX, T, H, m.EmbNorm, nil, cfg.Eps, c.embMean, c.embRstd)

	for l := 0; l < L; l++ {
		var out []float32
		if l+1 >= c.trainFrom || l+1 == L {
			out = c.hs[l+1]
		} else if &cur[0] == &c.ping[0] {
			out = c.pong
		} else {
			out = c.ping
		}
		lc := &c.frozen
		keep := l >= c.trainFrom
		if keep {
			lc = &c.lc[l]
		}
		m.layerFwd(c, l, cur, out, lc, keep, T, H, I, D)
		cur = out
	}
	hL := c.hs[L]
	rows := make([]float32, R*H)
	for r, t := range c.rowTok {
		copy(rows[r*H:(r+1)*H], hL[t*H:(t+1)*H])
	}
	return m.HeadForward(c, rows, R), nil
}

// HeadForward applies the final norm and the scorer head to R encoder rows
// (the hidden states at the [MASK] positions, R x H) and returns one logit
// per row. It records what HeadBackward needs in c.
func (m *Model) HeadForward(c *Cache, rows []float32, R int) []float32 {
	cfg := m.Cfg
	H := cfg.Hidden
	c.R = R
	c.fN = grow(c.fN, R*H)
	c.fMean, c.fRstd = grow(c.fMean, R), grow(c.fRstd, R)
	layerNormFwd(c.fN, rows, R, H, m.FinalNorm, nil, cfg.Eps, c.fMean, c.fRstd)
	h := &m.Head
	half := H / 2
	c.s1, c.m1, c.r1 = grow(c.s1, R*H), grow(c.m1, R), grow(c.r1, R)
	layerNormFwd(c.s1, c.fN, R, H, h.InW, h.InB, 1e-5, c.m1, c.r1)
	c.d = grow(c.d, R*half)
	linearFwd(c.d, c.s1, R, H, h.DenseW, half)
	c.ge = grow(c.ge, R*half)
	for r := 0; r < R; r++ {
		for i := 0; i < half; i++ {
			c.d[r*half+i] += h.DenseB[i]
			c.ge[r*half+i] = gelu(c.d[r*half+i])
		}
	}
	c.s2, c.m2, c.r2 = grow(c.s2, R*half), grow(c.m2, R), grow(c.r2, R)
	layerNormFwd(c.s2, c.ge, R, half, h.NormW, h.NormB, 1e-5, c.m2, c.r2)
	c.logits = grow(c.logits, R)
	for r := 0; r < R; r++ {
		var s float32
		row := c.s2[r*half : (r+1)*half]
		for i, v := range row {
			s += v * h.OutW[i]
		}
		c.logits[r] = s + h.OutB[0]
	}
	return c.logits
}

func (m *Model) layerFwd(c *Cache, l int, in, out []float32, lc *layerCache, keep bool, T, H, I, D int) {
	cfg := m.Cfg
	ly := &m.Layers[l]
	sliding := cfg.Sliding(l)

	n1 := in
	if ly.AttnNorm != nil {
		lc.n1, lc.mean1, lc.rstd1 = grow(lc.n1, T*H), grow(lc.mean1, T), grow(lc.rstd1, T)
		layerNormFwd(lc.n1, in, T, H, ly.AttnNorm, nil, cfg.Eps, lc.mean1, lc.rstd1)
		n1 = lc.n1
	}
	lc.qkv = grow(lc.qkv, T*3*H)
	linearFwd(lc.qkv, n1, T, H, ly.Wqkv, 3*H)
	cos, sin := c.cosG, c.sinG
	if sliding {
		cos, sin = c.cosL, c.sinL
	}
	applyRope(lc.qkv, T, H, cfg.Heads, D, cos, sin, false)

	lc.ctx = grow(lc.ctx, T*H)
	var pkeep []float32
	if keep {
		total := 0
		for _, s := range c.seqs {
			total += s.n * s.n * cfg.Heads
		}
		c.pkeep[l] = grow(c.pkeep[l], total)
		pkeep = c.pkeep[l]
	}
	c.attnFwd(lc.ctx, lc.qkv, pkeep, sliding)

	lc.tmp = grow(lc.tmp, T*H)
	linearFwd(lc.tmp, lc.ctx, T, H, ly.Wo, H)
	lc.h1 = grow(lc.h1, T*H)
	parallelFor(T*H, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			lc.h1[i] = in[i] + lc.tmp[i]
		}
	})
	lc.n2, lc.mean2, lc.rstd2 = grow(lc.n2, T*H), grow(lc.mean2, T), grow(lc.rstd2, T)
	layerNormFwd(lc.n2, lc.h1, T, H, ly.MlpNorm, nil, cfg.Eps, lc.mean2, lc.rstd2)
	lc.wi = grow(lc.wi, T*2*I)
	linearFwd(lc.wi, lc.n2, T, H, ly.Wi, 2*I)
	lc.act = grow(lc.act, T*I)
	parallelFor(T, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			row := lc.wi[t*2*I : (t+1)*2*I]
			a := lc.act[t*I : (t+1)*I]
			for i := 0; i < I; i++ {
				a[i] = gelu(row[i]) * row[I+i]
			}
		}
	})
	linearFwd(lc.tmp, lc.act, T, I, ly.MlpWo, H)
	parallelFor(T*H, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			out[i] = lc.h1[i] + lc.tmp[i]
		}
	})
}

// attnFwd computes masked multi-head attention. When pkeep is non-nil the
// probabilities are stored there for the backward pass.
func (c *Cache) attnFwd(ctx, qkv, pkeep []float32, sliding bool) {
	cfg := c.m.Cfg
	H, D, heads := cfg.Hidden, cfg.HeadDim(), cfg.Heads
	stride := 3 * H
	scale := float32(1 / math.Sqrt(float64(D)))
	items := len(c.seqs) * heads
	parallelFor(items, func(lo, hi int) {
		var scratch []float32
		for it := lo; it < hi; it++ {
			si, hd := it/heads, it%heads
			s := &c.seqs[si]
			n := s.n
			var P []float32
			if pkeep != nil {
				P = pkeep[s.pOff+hd*n*n : s.pOff+(hd+1)*n*n]
			} else {
				scratch = grow(scratch, n*n)
				P = scratch
			}
			q := qkv[s.off*stride+hd*D:]
			k := qkv[s.off*stride+H+hd*D:]
			v := qkv[s.off*stride+2*H+hd*D:]
			blas.Sgemm1(false, true, n, n, D, q, stride, k, stride, P, n, false)
			mask := s.full
			if sliding {
				mask = s.slide
			}
			softmaxMasked(P, n, scale, mask)
			blas.Sgemm1(false, false, n, D, n, P, n, v, stride, ctx[s.off*H+hd*D:], H, false)
		}
	})
}

// softmaxMasked applies scale, masks and a row-wise softmax in place. Masked
// entries become exactly zero.
func softmaxMasked(P []float32, n int, scale float32, mask []uint8) {
	for i := 0; i < n; i++ {
		row := P[i*n : (i+1)*n]
		maxv := float32(math.Inf(-1))
		for j, v := range row {
			if mask != nil && mask[i*n+j] == 0 {
				continue
			}
			if v = v * scale; v > maxv {
				maxv = v
			}
		}
		var sum float64
		for j, v := range row {
			if mask != nil && mask[i*n+j] == 0 {
				row[j] = 0
				continue
			}
			e := float32(math.Exp(float64(v*scale - maxv)))
			row[j] = e
			sum += float64(e)
		}
		inv := float32(1 / sum)
		for j := range row {
			row[j] *= inv
		}
	}
}
