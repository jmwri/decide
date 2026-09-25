package blas

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

func naive(transA, transB bool, M, N, K int, A []float32, lda int, B []float32, ldb int) []float64 {
	out := make([]float64, M*N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var s float64
			for k := 0; k < K; k++ {
				var a, b float32
				if transA {
					a = A[k*lda+i]
				} else {
					a = A[i*lda+k]
				}
				if transB {
					b = B[j*ldb+k]
				} else {
					b = B[k*ldb+j]
				}
				s += float64(a) * float64(b)
			}
			out[i*N+j] = s
		}
	}
	return out
}

func randMat(rng *rand.Rand, n int) []float32 {
	m := make([]float32, n)
	for i := range m {
		m[i] = float32(rng.NormFloat64())
	}
	return m
}

func TestSgemm(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	shapes := [][3]int{{1, 1, 1}, {6, 16, 8}, {7, 17, 9}, {13, 33, 300}, {100, 250, 257}, {5, 800, 64}, {200, 5, 512}, {97, 768, 768}, {300, 50, 1}}
	for _, sh := range shapes {
		M, N, K := sh[0], sh[1], sh[2]
		for _, ta := range []bool{false, true} {
			for _, tb := range []bool{false, true} {
				lda, ldb := K, N
				if ta {
					lda = M
				}
				if tb {
					ldb = K
				}
				A, B := randMat(rng, M*K), randMat(rng, K*N)
				want := naive(ta, tb, M, N, K, A, lda, B, ldb)
				for _, acc := range []bool{false, true} {
					C := randMat(rng, M*N)
					init := append([]float32(nil), C...)
					Sgemm(ta, tb, M, N, K, A, lda, B, ldb, C, N, acc)
					for i := range C {
						w := want[i]
						if acc {
							w += float64(init[i])
						}
						if d := math.Abs(float64(C[i]) - w); d > 1e-3*math.Sqrt(float64(K)+1) {
							t.Fatalf("shape %v ta=%v tb=%v acc=%v idx %d: got %v want %v", sh, ta, tb, acc, i, C[i], w)
						}
					}
				}
			}
		}
	}
}

func TestSgemmLeadingDims(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	M, N, K := 20, 30, 40
	lda, ldb, ldc := K+3, N+5, N+7
	A, B := randMat(rng, M*lda), randMat(rng, K*ldb)
	C := make([]float32, M*ldc)
	Sgemm(false, false, M, N, K, A, lda, B, ldb, C, ldc, false)
	want := naive(false, false, M, N, K, A, lda, B, ldb)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			if math.Abs(float64(C[i*ldc+j])-want[i*N+j]) > 1e-3 {
				t.Fatalf("(%d,%d) got %v want %v", i, j, C[i*ldc+j], want[i*N+j])
			}
		}
		for j := N; j < ldc; j++ {
			if C[i*ldc+j] != 0 {
				t.Fatal("wrote outside the N columns")
			}
		}
	}
}

func TestBenchGFLOPS(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, sh := range [][3]int{{800, 2304, 768}, {800, 768, 1152}, {96, 768, 768}} {
		M, N, K := sh[0], sh[1], sh[2]
		A, B, C := randMat(rng, M*K), randMat(rng, K*N), make([]float32, M*N)
		for _, tr := range []string{"NN", "TN(dW)", "NT(fwd)"} {
			ta, tb := tr == "TN(dW)", tr == "NT(fwd)"
			start := time.Now()
			n := 0
			for time.Since(start) < 500*time.Millisecond {
				lda, ldb := K, N
				if ta {
					lda = M
				}
				if tb {
					ldb = K
				}
				Sgemm(ta, tb, M, N, K, A, lda, B, ldb, C, N, false)
				n++
			}
			el := time.Since(start).Seconds()
			t.Logf("%dx%dx%d %s: %.1f GFLOPS", M, N, K, tr, 2*float64(M)*float64(N)*float64(K)*float64(n)/el/1e9)
		}
	}
}
