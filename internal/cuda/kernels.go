package cuda

import (
	_ "embed"
	"fmt"
)

//go:embed kernels.ptx
var kernelPTX string

// Kernels holds the compiled encoder kernels of one device.
type Kernels struct {
	dev *Device
	mod *Module
	fn  map[string]*Kernel
}

var kernelNames = []string{
	"gemm_nn", "gemm_nt", "gemm_tn", "gemm_nn_b", "gemm_nt_b", "gemm_tn_b",
	"ln_fwd", "ln_bwd", "embed_gather", "embed_scatter_add", "gather_rows", "scatter_rows",
	"rope", "softmax_masked", "softmax_bwd", "geglu_fwd", "geglu_bwd", "add2", "add_inplace", "sumsq", "adamw",
}

// LoadKernels JIT-compiles the embedded PTX for the device (call inside Do).
func (d *Device) LoadKernels() (*Kernels, error) {
	mod, err := d.LoadPTX(kernelPTX)
	if err != nil {
		return nil, err
	}
	ks := &Kernels{dev: d, mod: mod, fn: map[string]*Kernel{}}
	for _, n := range kernelNames {
		k, err := mod.Func(n)
		if err != nil {
			return nil, err
		}
		ks.fn[n] = k
	}
	return ks, nil
}

// Launch runs a named kernel asynchronously (see Kernel.Launch for argument types).
func (ks *Kernels) Launch(name string, grid, block [3]int, shared int, args ...any) error {
	return ks.fn[name].Launch(grid, block, shared, args...)
}

func ceilDiv(a, b int) int { return (a + b - 1) / b }

const gemmTile = 128

// Gemm computes C (M x N) (+)= op(A) (M x K) * op(B) (K x N) with the same
// conventions as blas.Sgemm.
func (ks *Kernels) Gemm(transA, transB bool, M, N, K int, A *Buf, lda int, B *Buf, ldb int, C *Buf, ldc int, accumulate bool) error {
	if M == 0 || N == 0 {
		return nil
	}
	name := "gemm_nn"
	switch {
	case transA && !transB:
		name = "gemm_tn"
	case !transA && transB:
		name = "gemm_nt"
	case transA && transB:
		return fmt.Errorf("cuda: gemm with both operands transposed is not implemented")
	}
	acc := int32(0)
	if accumulate {
		acc = 1
	}
	return ks.Launch(name, [3]int{ceilDiv(N, gemmTile), ceilDiv(M, gemmTile), 1}, [3]int{256, 1, 1}, 0,
		int32(M), int32(N), int32(K), A, int32(lda), B, int32(ldb), C, int32(ldc), acc)
}

// GemmItem describes one product of a batched GEMM.
type GemmItem struct {
	M, N, K          int
	OffA, OffB, OffC int // element offsets from the base buffers
	LDA, LDB, LDC    int
}

// GemmBatched runs many independent products (possibly of different sizes) in
// one launch. items must live in device memory (see PackItems).
func (ks *Kernels) GemmBatched(transA, transB bool, items *Buf, count, maxM, maxN int, A, B, C *Buf, accumulate bool) error {
	name := "gemm_nn_b"
	switch {
	case transA && !transB:
		name = "gemm_tn_b"
	case !transA && transB:
		name = "gemm_nt_b"
	case transA && transB:
		return fmt.Errorf("cuda: gemm with both operands transposed is not implemented")
	}
	acc := int32(0)
	if accumulate {
		acc = 1
	}
	return ks.Launch(name, [3]int{ceilDiv(maxN, gemmTile), ceilDiv(maxM, gemmTile), count}, [3]int{256, 1, 1}, 0, items, A, B, C, acc)
}

// PackItems flattens GEMM items for upload as int32s.
func PackItems(items []GemmItem) []int32 {
	out := make([]int32, 0, len(items)*9)
	for _, it := range items {
		out = append(out, int32(it.M), int32(it.N), int32(it.K), int32(it.OffA), int32(it.OffB), int32(it.OffC), int32(it.LDA), int32(it.LDB), int32(it.LDC))
	}
	return out
}
