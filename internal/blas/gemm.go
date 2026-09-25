// Package blas provides a cache-blocked, multi-threaded single-precision
// matrix multiply with an AVX2/FMA micro-kernel on amd64 and a portable
// fallback elsewhere.
package blas

import (
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

const (
	mr = 6
	nr = 16
	kc = 256
	mc = 96 // multiple of mr
)

// kern6x16 computes C[6x16] += A_panel * B_panel. Assembly-backed on amd64.
var kern6x16 = kern6x16Generic

func kern6x16Generic(k int, a, b, c *float32, ldc int) {
	as := unsafe.Slice(a, k*mr)
	bs := unsafe.Slice(b, k*nr)
	cs := unsafe.Slice(c, (mr-1)*ldc+nr)
	var acc [mr][nr]float32
	for p := 0; p < k; p++ {
		for i := 0; i < mr; i++ {
			av := as[p*mr+i]
			for j := 0; j < nr; j++ {
				acc[i][j] += av * bs[p*nr+j]
			}
		}
	}
	for i := 0; i < mr; i++ {
		for j := 0; j < nr; j++ {
			cs[i*ldc+j] += acc[i][j]
		}
	}
}

// Workers bounds the number of goroutines a single Sgemm call uses (default GOMAXPROCS).
var Workers = runtime.GOMAXPROCS(0)

type scratch struct {
	a, b []float32
	tmp  [mr * nr]float32
}

var scratchPool = sync.Pool{New: func() any { return &scratch{} }}

func roundUp(x, m int) int { return (x + m - 1) / m * m }

// Sgemm computes C (M x N) = op(A) (M x K) * op(B) (K x N), adding into C
// when accumulate is set and overwriting it otherwise.
//
// Matrices are row-major with leading dimensions lda/ldb/ldc. With transA,
// A is stored K x M (element (i,k) is A[k*lda+i]); with transB, B is stored
// N x K (element (k,j) is B[j*ldb+k]).
func Sgemm(transA, transB bool, M, N, K int, A []float32, lda int, B []float32, ldb int, C []float32, ldc int, accumulate bool) {
	sgemm(Workers, transA, transB, M, N, K, A, lda, B, ldb, C, ldc, accumulate)
}

// Sgemm1 is Sgemm on the calling goroutine only, for callers that already
// parallelise over many small products.
func Sgemm1(transA, transB bool, M, N, K int, A []float32, lda int, B []float32, ldb int, C []float32, ldc int, accumulate bool) {
	sgemm(1, transA, transB, M, N, K, A, lda, B, ldb, C, ldc, accumulate)
}

func sgemm(workers int, transA, transB bool, M, N, K int, A []float32, lda int, B []float32, ldb int, C []float32, ldc int, accumulate bool) {
	if M == 0 || N == 0 {
		return
	}
	if !accumulate {
		for i := 0; i < M; i++ {
			row := C[i*ldc : i*ldc+N]
			for j := range row {
				row[j] = 0
			}
		}
	}
	if K == 0 {
		return
	}
	if M*N*K < 1<<15 {
		workers = 1
	}
	// Block sizes: enough jobs to keep every worker busy.
	mBlk := min(mc, roundUp(M, mr))
	mBlocks := (M + mBlk - 1) / mBlk
	nBlk := roundUp(N, nr)
	if want := (2*workers + mBlocks - 1) / mBlocks; want > 1 {
		nBlk = min(nBlk, max(nr, roundUp((N+want-1)/want, nr)))
	}
	nBlk = min(nBlk, 256)
	nBlocks := (N + nBlk - 1) / nBlk
	jobs := mBlocks * nBlocks

	run := func(job int, s *scratch) {
		ic := (job % mBlocks) * mBlk
		jc := (job / mBlocks) * nBlk
		mb := min(mBlk, M-ic)
		nb := min(nBlk, N-jc)
		for pc := 0; pc < K; pc += kc {
			kb := min(kc, K-pc)
			packA(s, transA, A, lda, ic, pc, mb, kb)
			packB(s, transB, B, ldb, pc, jc, kb, nb)
			macro(s, C[ic*ldc+jc:], ldc, mb, nb, kb)
		}
	}
	if workers == 1 || jobs == 1 {
		s := scratchPool.Get().(*scratch)
		for j := 0; j < jobs; j++ {
			run(j, s)
		}
		scratchPool.Put(s)
		return
	}
	var next atomic.Int64
	var wg sync.WaitGroup
	for w := 0; w < min(workers, jobs); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s := scratchPool.Get().(*scratch)
			for {
				j := int(next.Add(1)) - 1
				if j >= jobs {
					break
				}
				run(j, s)
			}
			scratchPool.Put(s)
		}()
	}
	wg.Wait()
}

func grow(b []float32, n int) []float32 {
	if cap(b) < n {
		return make([]float32, n)
	}
	return b[:n]
}

// packA packs an mb x kb block of op(A) into row panels of height mr, k-major, zero padded.
func packA(s *scratch, trans bool, A []float32, lda, i0, p0, mb, kb int) {
	panels := (mb + mr - 1) / mr
	s.a = grow(s.a, panels*mr*kb)
	dst := s.a
	for pi := 0; pi < panels; pi++ {
		base := pi * mr * kb
		rows := min(mr, mb-pi*mr)
		for p := 0; p < kb; p++ {
			d := dst[base+p*mr : base+p*mr+mr]
			if !trans {
				for i := 0; i < rows; i++ {
					d[i] = A[(i0+pi*mr+i)*lda+p0+p]
				}
			} else {
				src := A[(p0+p)*lda+i0+pi*mr:]
				copy(d[:rows], src[:rows])
			}
			for i := rows; i < mr; i++ {
				d[i] = 0
			}
		}
	}
}

// packB packs a kb x nb block of op(B) into column panels of width nr, k-major, zero padded.
func packB(s *scratch, trans bool, B []float32, ldb, p0, j0, kb, nb int) {
	panels := (nb + nr - 1) / nr
	s.b = grow(s.b, panels*nr*kb)
	dst := s.b
	for pj := 0; pj < panels; pj++ {
		base := pj * nr * kb
		cols := min(nr, nb-pj*nr)
		for p := 0; p < kb; p++ {
			d := dst[base+p*nr : base+p*nr+nr]
			if !trans {
				src := B[(p0+p)*ldb+j0+pj*nr:]
				copy(d[:cols], src[:cols])
			} else {
				for j := 0; j < cols; j++ {
					d[j] = B[(j0+pj*nr+j)*ldb+p0+p]
				}
			}
			for j := cols; j < nr; j++ {
				d[j] = 0
			}
		}
	}
}

func macro(s *scratch, C []float32, ldc, mb, nb, kb int) {
	for jr := 0; jr < nb; jr += nr {
		bp := &s.b[(jr/nr)*nr*kb]
		cols := min(nr, nb-jr)
		for ir := 0; ir < mb; ir += mr {
			ap := &s.a[(ir/mr)*mr*kb]
			rows := min(mr, mb-ir)
			if rows == mr && cols == nr {
				kern6x16(kb, ap, bp, &C[ir*ldc+jr], ldc)
				continue
			}
			for i := range s.tmp {
				s.tmp[i] = 0
			}
			kern6x16(kb, ap, bp, &s.tmp[0], nr)
			for i := 0; i < rows; i++ {
				dst := C[(ir+i)*ldc+jr : (ir+i)*ldc+jr+cols]
				for j := range dst {
					dst[j] += s.tmp[i*nr+j]
				}
			}
		}
	}
}
