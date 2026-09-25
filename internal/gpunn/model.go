// Package gpunn runs the ModernBERT encoder forward and backward passes and
// AdamW on an NVIDIA GPU through internal/cuda (driver API only, no cgo). It
// mirrors internal/nn step for step so the two can be compared tensor by
// tensor. The final norm, scorer head and loss stay on the CPU (they touch
// only a few rows per example); weights, gradients and optimizer state of the
// encoder live on the device.
package gpunn

import (
	"fmt"
	"math"
	"strings"

	"github.com/jmwri/decide/internal/cuda"
	"github.com/jmwri/decide/internal/nn"
)

type param struct {
	name          string
	n             int
	w, g, m, v    *cuda.Buf
	host          []float32
	decay, frozen bool
}

type wbuf struct {
	b   *cuda.Buf
	cap int
}

type layerBuf struct {
	n1, mean1, rstd1, qkv, P, ctx, h1, n2, mean2, rstd2, wi, act, tmp wbuf
}

type layerParams struct {
	attnNorm, wqkv, wo, mlpNorm, wi, mlpWo *param
}

// Model is a device-resident encoder.
type Model struct {
	Dev  *cuda.Device
	Host *nn.Model // head and final norm live here; encoder tensors are synced on demand
	cfg  nn.Config
	ks   *cuda.Kernels

	params   []*param
	tok      *param
	embNorm  *param
	layers   []layerParams
	trainFrm int
	trainEmb bool
	AdamStep int

	// per-batch layout
	T, R    int
	nSeq    int
	maxS    int
	pTotal  int // floats of attention probabilities per layer
	hostIDs []int32

	ids, pos, opt, rowTok     wbuf
	cosG, sinG, cosL, sinL    wbuf
	tables                    wbuf
	tScores, tPV, tDP, tDV    int // element offsets of the gemm item tables
	tDQ, tDK, tSM             int
	embX, embMean, embRstd    wbuf
	hs                        []wbuf
	lc                        []layerBuf
	dcur, dprev, dh1, dact    wbuf
	dwi, dn2, dctx, dqkv, dn1 wbuf
	dP, dE, rows, dRows       wbuf
	scalar                    *cuda.Buf
	headCache                 *nn.Cache
	hostRows                  []float32
	rowTokHost                []int32
}

// New uploads the encoder weights of host to the device. The scorer head and
// final norm stay on the host model.
func New(dev *cuda.Device, host *nn.Model) (*Model, error) {
	g := &Model{Dev: dev, Host: host, cfg: host.Cfg, headCache: nn.NewCache(host), hs: make([]wbuf, host.Cfg.Layers+1), lc: make([]layerBuf, host.Cfg.Layers)}
	var err error
	dev.Do(func() {
		if g.ks, err = dev.LoadKernels(); err != nil {
			return
		}
		byName := map[string]*param{}
		for _, t := range host.Tensors() {
			if !strings.HasPrefix(t.Name, "model.embeddings.") && !strings.HasPrefix(t.Name, "model.layers.") {
				continue
			}
			p := &param{name: t.Name, n: len(t.Data), host: t.Data, decay: len(t.Shape) == 2 && t.Name != "model.embeddings.tok_embeddings.weight"}
			for _, dst := range []**cuda.Buf{&p.w, &p.g, &p.m, &p.v} {
				if *dst, err = dev.Alloc(p.n); err != nil {
					return
				}
			}
			p.g.Zero()
			p.m.Zero()
			p.v.Zero()
			if err = p.w.Upload(0, t.Data); err != nil {
				return
			}
			g.params = append(g.params, p)
			byName[t.Name] = p
		}
		g.tok, g.embNorm = byName["model.embeddings.tok_embeddings.weight"], byName["model.embeddings.norm.weight"]
		for i := 0; i < host.Cfg.Layers; i++ {
			pre := fmt.Sprintf("model.layers.%d.", i)
			g.layers = append(g.layers, layerParams{
				attnNorm: byName[pre+"attn_norm.weight"], wqkv: byName[pre+"attn.Wqkv.weight"], wo: byName[pre+"attn.Wo.weight"],
				mlpNorm: byName[pre+"mlp_norm.weight"], wi: byName[pre+"mlp.Wi.weight"], mlpWo: byName[pre+"mlp.Wo.weight"],
			})
		}
		g.scalar, err = dev.Alloc(1)
	})
	if err != nil {
		return nil, err
	}
	g.SetTrainable(0, true)
	return g, nil
}

// Free releases all device memory.
func (g *Model) Free() {
	g.Dev.Do(func() {
		for _, p := range g.params {
			p.w.Free()
			p.g.Free()
			p.m.Free()
			p.v.Free()
		}
		for _, w := range g.allWork() {
			w.b.Free()
		}
		g.scalar.Free()
	})
}

func (g *Model) allWork() []*wbuf {
	ws := []*wbuf{&g.ids, &g.pos, &g.opt, &g.rowTok, &g.cosG, &g.sinG, &g.cosL, &g.sinL, &g.tables, &g.embX, &g.embMean, &g.embRstd,
		&g.dcur, &g.dprev, &g.dh1, &g.dact, &g.dwi, &g.dn2, &g.dctx, &g.dqkv, &g.dn1, &g.dP, &g.dE, &g.rows, &g.dRows}
	for i := range g.hs {
		ws = append(ws, &g.hs[i])
	}
	for i := range g.lc {
		l := &g.lc[i]
		ws = append(ws, &l.n1, &l.mean1, &l.rstd1, &l.qkv, &l.P, &l.ctx, &l.h1, &l.n2, &l.mean2, &l.rstd2, &l.wi, &l.act, &l.tmp)
	}
	return ws
}

// SetTrainable freezes encoder layers below trainFrom and, unless trainEmb,
// the token embeddings (they are always frozen when trainFrom > 0).
func (g *Model) SetTrainable(trainFrom int, trainEmb bool) {
	g.trainFrm, g.trainEmb = trainFrom, trainEmb
	for _, p := range g.params {
		p.frozen = false
		switch {
		case p.name == "model.embeddings.tok_embeddings.weight":
			p.frozen = !trainEmb || trainFrom > 0
		case p.name == "model.embeddings.norm.weight":
			p.frozen = trainFrom > 0
		default:
			var l int
			fmt.Sscanf(strings.TrimPrefix(p.name, "model.layers."), "%d", &l)
			p.frozen = l < trainFrom
		}
	}
}

// get returns a device buffer of at least n elements, reallocating when needed.
func (g *Model) get(w *wbuf, n int) *cuda.Buf {
	if n < 1 {
		n = 1
	}
	if w.b == nil || w.cap < n {
		w.b.Free()
		b, err := g.Dev.Alloc(n + n/8)
		if err != nil {
			panic(err)
		}
		w.b, w.cap = b, n+n/8
	}
	return w.b
}

type seqLayout struct{ off, n, pOff int }

func (g *Model) chk(err error) {
	if err != nil {
		panic(err)
	}
}

func ew(n int) [3]int { return [3]int{(n + 255) / 256, 1, 1} }

var blk256 = [3]int{256, 1, 1}

// ---------------------------------------------------------------------------

// Forward runs the encoder on the device and the head on the host, returning
// one logit per [MASK] row (in batch order).
func (g *Model) Forward(batch []nn.Sequence) (logits []float32, err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(error); ok {
				err = e
				return
			}
			panic(r)
		}
	}()
	g.Dev.Do(func() { logits = g.forward(batch) })
	return logits, nil
}

func (g *Model) forward(batch []nn.Sequence) []float32 {
	cfg := g.cfg
	H, I, D, L, heads := cfg.Hidden, cfg.Intermediate, cfg.HeadDim(), cfg.Layers, cfg.Heads
	half := D / 2

	// ---- layout
	var ids, pos, opt, rowTok []int32
	var seqs []seqLayout
	T, pTotal, maxS := 0, 0, 0
	for si := range batch {
		s := &batch[si]
		n := len(s.IDs)
		if n == 0 || len(s.Pos) != n || len(s.MaskPos) == 0 || (s.OptID != nil && len(s.OptID) != n) {
			panic(fmt.Errorf("gpunn: malformed sequence %d", si))
		}
		seqs = append(seqs, seqLayout{off: T, n: n, pOff: pTotal})
		ids = append(ids, s.IDs...)
		pos = append(pos, s.Pos...)
		if s.OptID != nil {
			opt = append(opt, s.OptID...)
		} else {
			for i := 0; i < n; i++ {
				opt = append(opt, -1) // plain attention == everything is prefix
			}
		}
		for _, mp := range s.MaskPos {
			if mp < 0 || mp >= n {
				panic(fmt.Errorf("gpunn: sequence %d mask position %d out of range", si, mp))
			}
			rowTok = append(rowTok, int32(T+mp))
		}
		for _, id := range s.IDs {
			if id < 0 || int(id) >= cfg.Vocab {
				panic(fmt.Errorf("gpunn: token id %d out of range", id))
			}
		}
		T += n
		pTotal += n * n * heads
		maxS = max(maxS, n)
	}
	g.T, g.R, g.nSeq, g.maxS, g.pTotal, g.hostIDs, g.rowTokHost = T, len(rowTok), len(batch), maxS, pTotal, ids, rowTok
	R := g.R

	g.chk(g.get(&g.ids, T).UploadInt32(0, ids))
	g.chk(g.get(&g.pos, T).UploadInt32(0, pos))
	g.chk(g.get(&g.opt, T).UploadInt32(0, opt))
	g.chk(g.get(&g.rowTok, R).UploadInt32(0, rowTok))
	cg, sg := nn.RopeTables(pos, D, cfg.GlobalTheta)
	cl, sl := nn.RopeTables(pos, D, cfg.LocalTheta)
	g.chk(g.get(&g.cosG, T*half).Upload(0, cg))
	g.chk(g.get(&g.sinG, T*half).Upload(0, sg))
	g.chk(g.get(&g.cosL, T*half).Upload(0, cl))
	g.chk(g.get(&g.sinL, T*half).Upload(0, sl))
	g.buildTables(seqs)

	// ---- embeddings
	embX := g.get(&g.embX, T*H)
	g.chk(g.ks.Launch("embed_gather", [3]int{T, 1, 1}, blk256, 0, embX, g.tok.w, g.ids.b, int32(H)))
	for l := 0; l <= L; l++ {
		g.get(&g.hs[l], T*H)
	}
	g.chk(g.ks.Launch("ln_fwd", [3]int{T, 1, 1}, blk256, 0, g.hs[0].b, embX, g.embNorm.w, (*cuda.Buf)(nil),
		g.get(&g.embMean, T), g.get(&g.embRstd, T), int32(H), cfg.Eps))

	// ---- layers
	scale := float32(1 / math.Sqrt(float64(D)))
	items := g.tables.b
	for l := 0; l < L; l++ {
		lp, lc := &g.layers[l], &g.lc[l]
		in, out := g.hs[l].b, g.hs[l+1].b
		sliding := int32(0)
		cosb, sinb := g.cosG.b, g.sinG.b
		if cfg.Sliding(l) {
			sliding = 1
			cosb, sinb = g.cosL.b, g.sinL.b
		}
		n1 := in
		if lp.attnNorm != nil {
			n1 = g.get(&lc.n1, T*H)
			g.chk(g.ks.Launch("ln_fwd", [3]int{T, 1, 1}, blk256, 0, n1, in, lp.attnNorm.w, (*cuda.Buf)(nil),
				g.get(&lc.mean1, T), g.get(&lc.rstd1, T), int32(H), cfg.Eps))
		}
		qkv := g.get(&lc.qkv, T*3*H)
		g.chk(g.ks.Gemm(false, true, T, 3*H, H, n1, H, lp.wqkv.w, H, qkv, 3*H, false))
		g.chk(g.ks.Launch("rope", ew(T*heads*half), blk256, 0, qkv, cosb, sinb, int32(T), int32(H), int32(heads), int32(D), int32(0)))
		P := g.get(&lc.P, pTotal)
		g.chk(g.ks.GemmBatched(false, true, items.Slice(g.tScores, g.nSeq*heads*9), g.nSeq*heads, maxS, maxS, qkv, qkv, P, false))
		g.chk(g.ks.Launch("softmax_masked", [3]int{(maxS + 7) / 8, g.nSeq * heads, 1}, blk256, 0, P, items.Slice(g.tSM, g.nSeq*4), g.opt.b, g.pos.b,
			int32(heads), sliding, int32(cfg.HalfWindow), scale))
		ctx := g.get(&lc.ctx, T*H)
		g.chk(g.ks.GemmBatched(false, false, items.Slice(g.tPV, g.nSeq*heads*9), g.nSeq*heads, maxS, D, P, qkv, ctx, false))
		tmp := g.get(&lc.tmp, T*H)
		g.chk(g.ks.Gemm(false, true, T, H, H, ctx, H, lp.wo.w, H, tmp, H, false))
		h1 := g.get(&lc.h1, T*H)
		g.chk(g.ks.Launch("add2", ew(T*H), blk256, 0, h1, in, tmp, int32(T*H)))
		n2 := g.get(&lc.n2, T*H)
		g.chk(g.ks.Launch("ln_fwd", [3]int{T, 1, 1}, blk256, 0, n2, h1, lp.mlpNorm.w, (*cuda.Buf)(nil),
			g.get(&lc.mean2, T), g.get(&lc.rstd2, T), int32(H), cfg.Eps))
		wi := g.get(&lc.wi, T*2*I)
		g.chk(g.ks.Gemm(false, true, T, 2*I, H, n2, H, lp.wi.w, H, wi, 2*I, false))
		act := g.get(&lc.act, T*I)
		g.chk(g.ks.Launch("geglu_fwd", ew(T*I), blk256, 0, act, wi, int32(T), int32(I)))
		g.chk(g.ks.Gemm(false, true, T, H, I, act, I, lp.mlpWo.w, I, tmp, H, false))
		g.chk(g.ks.Launch("add2", ew(T*H), blk256, 0, out, h1, tmp, int32(T*H)))
	}

	// ---- gather the [MASK] rows, run the head on the host
	rows := g.get(&g.rows, R*H)
	g.chk(g.ks.Launch("gather_rows", [3]int{R, 1, 1}, blk256, 0, rows, g.hs[L].b, g.rowTok.b, int32(H)))
	g.hostRows = growF(g.hostRows, R*H)
	g.chk(rows.Download(0, g.hostRows[:R*H]))
	return g.Host.HeadForward(g.headCache, g.hostRows[:R*H], R)
}

func growF(b []float32, n int) []float32 {
	if cap(b) < n {
		return make([]float32, n)
	}
	return b[:n]
}

// buildTables uploads the per-sequence attention product descriptions.
func (g *Model) buildTables(seqs []seqLayout) {
	cfg := g.cfg
	H, D, heads := cfg.Hidden, cfg.HeadDim(), cfg.Heads
	stride := 3 * H
	var scores, pv, dp, dv, dq, dk []cuda.GemmItem
	var sm []int32
	for _, s := range seqs {
		S := s.n
		sm = append(sm, int32(S), int32(s.pOff), int32(s.off), 0)
		for hd := 0; hd < heads; hd++ {
			pOff := s.pOff + hd*S*S
			q := s.off*stride + hd*D
			k := s.off*stride + H + hd*D
			v := s.off*stride + 2*H + hd*D
			c := s.off*H + hd*D
			scores = append(scores, cuda.GemmItem{M: S, N: S, K: D, OffA: q, OffB: k, OffC: pOff, LDA: stride, LDB: stride, LDC: S})
			pv = append(pv, cuda.GemmItem{M: S, N: D, K: S, OffA: pOff, OffB: v, OffC: c, LDA: S, LDB: stride, LDC: H})
			dp = append(dp, cuda.GemmItem{M: S, N: S, K: D, OffA: c, OffB: v, OffC: pOff, LDA: H, LDB: stride, LDC: S})
			dv = append(dv, cuda.GemmItem{M: S, N: D, K: S, OffA: pOff, OffB: c, OffC: v, LDA: S, LDB: H, LDC: stride})
			dq = append(dq, cuda.GemmItem{M: S, N: D, K: S, OffA: pOff, OffB: k, OffC: q, LDA: S, LDB: stride, LDC: stride})
			dk = append(dk, cuda.GemmItem{M: S, N: D, K: S, OffA: pOff, OffB: q, OffC: k, LDA: S, LDB: stride, LDC: stride})
		}
	}
	var all []int32
	add := func(off *int, items []cuda.GemmItem) {
		*off = len(all)
		all = append(all, cuda.PackItems(items)...)
	}
	add(&g.tScores, scores)
	add(&g.tPV, pv)
	add(&g.tDP, dp)
	add(&g.tDV, dv)
	add(&g.tDQ, dq)
	add(&g.tDK, dk)
	g.tSM = len(all)
	all = append(all, sm...)
	g.chk(g.get(&g.tables, len(all)).UploadInt32(0, all))
}

// ---------------------------------------------------------------------------

// Backward accumulates gradients for the dlogits of the preceding Forward.
// Head parameter gradients go into hostGrads (a model shaped like Host);
// encoder gradients accumulate on the device.
func (g *Model) Backward(dlogits []float32, hostGrads *nn.Model) (err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(error); ok {
				err = e
				return
			}
			panic(r)
		}
	}()
	g.Dev.Do(func() { g.backward(dlogits, hostGrads) })
	return nil
}

func (g *Model) backward(dlogits []float32, hostGrads *nn.Model) {
	cfg := g.cfg
	H, I, D, L, heads := cfg.Hidden, cfg.Intermediate, cfg.HeadDim(), cfg.Layers, cfg.Heads
	half := D / 2
	T, R := g.T, g.R
	items := g.tables.b

	dRows := g.Host.HeadBackward(g.headCache, g.hostRows[:R*H], dlogits, hostGrads)
	g.chk(g.get(&g.dRows, R*H).Upload(0, dRows))
	dcur := g.get(&g.dcur, T*H)
	g.chk(dcur.Zero())
	g.chk(g.ks.Launch("scatter_rows", [3]int{R, 1, 1}, blk256, 0, dcur, g.dRows.b, g.rowTok.b, int32(H)))

	dprev := g.get(&g.dprev, T*H)
	dh1 := g.get(&g.dh1, T*H)
	dact := g.get(&g.dact, T*I)
	dwi := g.get(&g.dwi, T*2*I)
	dn2 := g.get(&g.dn2, T*H)
	dctx := g.get(&g.dctx, T*H)
	dqkv := g.get(&g.dqkv, T*3*H)
	dn1 := g.get(&g.dn1, T*H)
	dP := g.get(&g.dP, g.pTotal)
	scale := float32(1 / math.Sqrt(float64(D)))
	lnGrid := [3]int{min(T, 168), 1, 1}

	for l := L - 1; l >= g.trainFrm && l >= 0; l-- {
		lp, lc := &g.layers[l], &g.lc[l]
		in := g.hs[l].b
		dout := dcur
		cosb, sinb := g.cosG.b, g.sinG.b
		if cfg.Sliding(l) {
			cosb, sinb = g.cosL.b, g.sinL.b
		}

		// MLP: out = h1 + Wo(act)
		g.chk(g.ks.Gemm(false, false, T, I, H, dout, H, lp.mlpWo.w, I, dact, I, false))
		g.chk(g.ks.Gemm(true, false, H, I, T, dout, H, lc.act.b, I, lp.mlpWo.g, I, true))
		g.chk(g.ks.Launch("geglu_bwd", ew(T*I), blk256, 0, dwi, dact, lc.wi.b, int32(T), int32(I)))
		g.chk(g.ks.Gemm(true, false, 2*I, H, T, dwi, 2*I, lc.n2.b, H, lp.wi.g, H, true))
		g.chk(g.ks.Gemm(false, false, T, H, 2*I, dwi, 2*I, lp.wi.w, H, dn2, H, false))
		g.chk(dh1.CopyFrom(0, dout, 0, T*H))
		g.chk(g.ks.Launch("ln_bwd", lnGrid, blk256, 0, dh1, dn2, lc.h1.b, lp.mlpNorm.w, lc.mean2.b, lc.rstd2.b, lp.mlpNorm.g, (*cuda.Buf)(nil), int32(T), int32(H)))

		// Attention: h1 = in + Wo(ctx)
		g.chk(g.ks.Gemm(false, false, T, H, H, dh1, H, lp.wo.w, H, dctx, H, false))
		g.chk(g.ks.Gemm(true, false, H, H, T, dh1, H, lc.ctx.b, H, lp.wo.g, H, true))
		nb := g.nSeq * heads
		g.chk(g.ks.GemmBatched(false, true, items.Slice(g.tDP, nb*9), nb, g.maxS, g.maxS, dctx, lc.qkv.b, dP, false))
		g.chk(g.ks.Launch("softmax_bwd", [3]int{(g.maxS + 7) / 8, nb, 1}, blk256, 0, dP, lc.P.b, items.Slice(g.tSM, g.nSeq*4), int32(heads), scale))
		g.chk(g.ks.GemmBatched(true, false, items.Slice(g.tDV, nb*9), nb, g.maxS, D, lc.P.b, dctx, dqkv, false))
		g.chk(g.ks.GemmBatched(false, false, items.Slice(g.tDQ, nb*9), nb, g.maxS, D, dP, lc.qkv.b, dqkv, false))
		g.chk(g.ks.GemmBatched(true, false, items.Slice(g.tDK, nb*9), nb, g.maxS, D, dP, lc.qkv.b, dqkv, false))
		g.chk(g.ks.Launch("rope", ew(T*heads*half), blk256, 0, dqkv, cosb, sinb, int32(T), int32(H), int32(heads), int32(D), int32(1)))
		n1 := in
		if lp.attnNorm != nil {
			n1 = lc.n1.b
		}
		g.chk(g.ks.Gemm(true, false, 3*H, H, T, dqkv, 3*H, n1, H, lp.wqkv.g, H, true))
		g.chk(g.ks.Gemm(false, false, T, H, 3*H, dqkv, 3*H, lp.wqkv.w, H, dn1, H, false))
		g.chk(dprev.CopyFrom(0, dh1, 0, T*H))
		if lp.attnNorm != nil {
			g.chk(g.ks.Launch("ln_bwd", lnGrid, blk256, 0, dprev, dn1, in, lp.attnNorm.w, lc.mean1.b, lc.rstd1.b, lp.attnNorm.g, (*cuda.Buf)(nil), int32(T), int32(H)))
		} else {
			g.chk(g.ks.Launch("add_inplace", ew(T*H), blk256, 0, dprev, dn1, int32(T*H)))
		}
		dcur, dprev = dprev, dcur
	}

	if g.trainEmb && g.trainFrm == 0 {
		dE := g.get(&g.dE, T*H)
		g.chk(dE.Zero())
		g.chk(g.ks.Launch("ln_bwd", lnGrid, blk256, 0, dE, dcur, g.embX.b, g.embNorm.w, g.embMean.b, g.embRstd.b, g.embNorm.g, (*cuda.Buf)(nil), int32(T), int32(H)))
		g.chk(g.ks.Launch("embed_scatter_add", [3]int{T, 1, 1}, blk256, 0, g.tok.g, dE, g.ids.b, int32(H)))
	}
	g.chk(g.Dev.Sync())
}

// ---------------------------------------------------------------------------
// optimizer and state transfer

// ZeroGrad clears all device gradients.
func (g *Model) ZeroGrad() {
	g.Dev.Do(func() {
		for _, p := range g.params {
			g.chk(p.g.Zero())
		}
	})
}

// GradSumSq returns the sum of squares of all trainable device gradients.
func (g *Model) GradSumSq() (s float64) {
	g.Dev.Do(func() {
		g.chk(g.scalar.Zero())
		for _, p := range g.params {
			if p.frozen {
				continue
			}
			g.chk(g.ks.Launch("sumsq", [3]int{min((p.n+255)/256, 2048), 1, 1}, blk256, 0, p.g, int64(p.n), g.scalar))
		}
		var out [1]float32
		g.chk(g.scalar.Download(0, out[:]))
		s = float64(out[0])
	})
	return s
}

// Update applies one AdamW step to the encoder with the given learning rate;
// gradients are first scaled by gradScale. Call after every gradient is
// accumulated. AdamStep is advanced.
func (g *Model) Update(lr, weightDecay, gradScale float32) {
	g.AdamStep++
	const b1, b2, eps = 0.9, 0.999, 1e-8
	b1c := 1 - math.Pow(b1, float64(g.AdamStep))
	b2c := 1 - math.Pow(b2, float64(g.AdamStep))
	stepSize := float32(1 / b1c)
	invSqrt := float32(1 / math.Sqrt(b2c))
	g.Dev.Do(func() {
		for _, p := range g.params {
			if p.frozen {
				continue
			}
			wd := float32(0)
			if p.decay {
				wd = weightDecay * lr
			}
			g.chk(g.ks.Launch("adamw", [3]int{min((p.n+255)/256, 4096), 1, 1}, blk256, 0, p.w, p.g, p.m, p.v, int64(p.n),
				lr, wd, float32(b1), float32(b2), float32(eps), gradScale, stepSize, invSqrt))
		}
		g.chk(g.Dev.Sync())
	})
}

// SyncToHost copies the device encoder weights into the host model.
func (g *Model) SyncToHost() error {
	var err error
	g.Dev.Do(func() {
		for _, p := range g.params {
			if err = p.w.Download(0, p.host); err != nil {
				return
			}
		}
	})
	return err
}

// LoadFromHost uploads the host model's encoder weights to the device.
func (g *Model) LoadFromHost() error {
	var err error
	g.Dev.Do(func() {
		for _, p := range g.params {
			if err = p.w.Upload(0, p.host); err != nil {
				return
			}
		}
	})
	return err
}

func (g *Model) hostTensors(m *nn.Model) map[string][]float32 {
	out := map[string][]float32{}
	for _, t := range m.Tensors() {
		out[t.Name] = t.Data
	}
	return out
}

// DownloadGrads copies the device gradients into the encoder tensors of dst.
func (g *Model) DownloadGrads(dst *nn.Model) error {
	return g.download(dst, func(p *param) *cuda.Buf { return p.g })
}

// DownloadAdam copies the Adam moments into the encoder tensors of m and v.
func (g *Model) DownloadAdam(m, v *nn.Model) error {
	if err := g.download(m, func(p *param) *cuda.Buf { return p.m }); err != nil {
		return err
	}
	return g.download(v, func(p *param) *cuda.Buf { return p.v })
}

// UploadAdam restores the Adam moments and step count.
func (g *Model) UploadAdam(m, v *nn.Model, step int) error {
	g.AdamStep = step
	var err error
	g.Dev.Do(func() {
		mt, vt := g.hostTensors(m), g.hostTensors(v)
		for _, p := range g.params {
			if err = p.m.Upload(0, mt[p.name]); err != nil {
				return
			}
			if err = p.v.Upload(0, vt[p.name]); err != nil {
				return
			}
		}
	})
	return err
}

func (g *Model) download(dst *nn.Model, pick func(*param) *cuda.Buf) error {
	var err error
	g.Dev.Do(func() {
		ht := g.hostTensors(dst)
		for _, p := range g.params {
			if err = pick(p).Download(0, ht[p.name]); err != nil {
				return
			}
		}
	})
	return err
}
