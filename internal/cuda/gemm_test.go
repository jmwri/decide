package cuda

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/jmwri/decide/internal/blas"
)

func randFloats(rng *rand.Rand, n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	return x
}

func loadKernelsOrSkip(t *testing.T) (*Device, *Kernels) {
	t.Helper()
	d := openOrSkip(t)
	var ks *Kernels
	var err error
	d.Do(func() { ks, err = d.LoadKernels() })
	if err != nil {
		d.Close()
		t.Fatal(err)
	}
	return d, ks
}

func TestGemmMatchesCPU(t *testing.T) {
	d, ks := loadKernelsOrSkip(t)
	defer d.Close()
	rng := rand.New(rand.NewSource(1))
	shapes := [][3]int{{1, 1, 1}, {6, 16, 8}, {7, 17, 9}, {13, 33, 300}, {129, 130, 257}, {5, 800, 64}, {200, 5, 512}, {97, 768, 768}, {300, 50, 1}}
	d.Do(func() {
		for _, sh := range shapes {
			M, N, K := sh[0], sh[1], sh[2]
			for _, tr := range [][2]bool{{false, false}, {false, true}, {true, false}} {
				ta, tb := tr[0], tr[1]
				lda, ldb, ldc := K+2, N+3, N+5
				if ta {
					lda = M + 2
				}
				if tb {
					ldb = K + 3
				}
				rowsA, rowsB := M, K
				if ta {
					rowsA = K
				}
				if tb {
					rowsB = N
				}
				A, B := randFloats(rng, rowsA*lda), randFloats(rng, rowsB*ldb)
				for _, acc := range []bool{false, true} {
					C0 := randFloats(rng, M*ldc)
					want := append([]float32(nil), C0...)
					blas.Sgemm(ta, tb, M, N, K, A, lda, B, ldb, want, ldc, acc)
					dA, _ := d.Alloc(len(A))
					dB, _ := d.Alloc(len(B))
					dC, _ := d.Alloc(len(C0))
					dA.Upload(0, A)
					dB.Upload(0, B)
					dC.Upload(0, C0)
					if err := ks.Gemm(ta, tb, M, N, K, dA, lda, dB, ldb, dC, ldc, acc); err != nil {
						t.Fatal(err)
					}
					if err := d.Sync(); err != nil {
						t.Fatal(err)
					}
					got := make([]float32, len(C0))
					dC.Download(0, got)
					for i := 0; i < M; i++ {
						for j := 0; j < ldc; j++ {
							g, w := got[i*ldc+j], want[i*ldc+j]
							if j >= N { // padding columns must be untouched
								w = C0[i*ldc+j]
							}
							if math.Abs(float64(g-w)) > 1e-3*math.Sqrt(float64(K)+1) {
								t.Fatalf("shape %v ta=%v tb=%v acc=%v (%d,%d): got %v want %v", sh, ta, tb, acc, i, j, g, w)
							}
						}
					}
					dA.Free()
					dB.Free()
					dC.Free()
				}
			}
		}
	})
}

func TestGemmBatchedMatchesCPU(t *testing.T) {
	d, ks := loadKernelsOrSkip(t)
	defer d.Close()
	rng := rand.New(rand.NewSource(2))
	// three items of different sizes sharing base buffers, like per-sequence attention
	type shape struct{ M, N, K int }
	shapes := []shape{{30, 30, 64}, {150, 150, 64}, {7, 7, 64}}
	var items []GemmItem
	var A, B, C []float32
	for _, s := range shapes {
		it := GemmItem{M: s.M, N: s.N, K: s.K, OffA: len(A), OffB: len(B), OffC: len(C), LDA: s.K, LDB: s.K, LDC: s.N}
		items = append(items, it)
		A = append(A, randFloats(rng, s.M*s.K)...)
		B = append(B, randFloats(rng, s.N*s.K)...)
		C = append(C, make([]float32, s.M*s.N)...)
	}
	d.Do(func() {
		dA, _ := d.Alloc(len(A))
		dB, _ := d.Alloc(len(B))
		dC, _ := d.Alloc(len(C))
		dI, _ := d.Alloc(len(items) * 9)
		dA.Upload(0, A)
		dB.Upload(0, B)
		dC.Upload(0, C)
		dI.UploadInt32(0, PackItems(items))
		if err := ks.GemmBatched(false, true, dI, len(items), 150, 150, dA, dB, dC, false); err != nil {
			t.Fatal(err)
		}
		if err := d.Sync(); err != nil {
			t.Fatal(err)
		}
		got := make([]float32, len(C))
		dC.Download(0, got)
		for _, it := range items {
			want := make([]float32, it.M*it.N)
			blas.Sgemm(false, true, it.M, it.N, it.K, A[it.OffA:], it.LDA, B[it.OffB:], it.LDB, want, it.LDC, false)
			for i := range want {
				if math.Abs(float64(got[it.OffC+i]-want[i])) > 1e-3 {
					t.Fatalf("item %+v idx %d: got %v want %v", it, i, got[it.OffC+i], want[i])
				}
			}
		}
	})
}

func TestGemmSpeed(t *testing.T) {
	d, ks := loadKernelsOrSkip(t)
	defer d.Close()
	rng := rand.New(rand.NewSource(3))
	d.Do(func() {
		for _, sh := range [][3]int{{1200, 2304, 768}, {1200, 768, 1152}, {2304, 768, 1200}} {
			M, N, K := sh[0], sh[1], sh[2]
			for _, tr := range []struct {
				name   string
				ta, tb bool
			}{{"NN", false, false}, {"NT(fwd)", false, true}, {"TN(dW)", true, false}} {
				A, B := randFloats(rng, M*K), randFloats(rng, K*N)
				dA, _ := d.Alloc(len(A))
				dB, _ := d.Alloc(len(B))
				dC, _ := d.Alloc(M * N)
				dA.Upload(0, A)
				dB.Upload(0, B)
				lda, ldb := K, N
				if tr.ta {
					lda = M
				}
				if tr.tb {
					ldb = K
				}
				ks.Gemm(tr.ta, tr.tb, M, N, K, dA, lda, dB, ldb, dC, N, false)
				d.Sync()
				const reps = 20
				start := time.Now()
				for i := 0; i < reps; i++ {
					ks.Gemm(tr.ta, tr.tb, M, N, K, dA, lda, dB, ldb, dC, N, false)
				}
				d.Sync()
				el := time.Since(start).Seconds()
				t.Logf("%dx%dx%d %-8s %.2f TFLOPS", M, N, K, tr.name, 2*float64(M)*float64(N)*float64(K)*reps/el/1e12)
				dA.Free()
				dB.Free()
				dC.Free()
			}
		}
	})
}
