package cuda

import (
	"math"
	"os"
	"testing"
)

const vecAdd = `
extern "C" __global__ void vadd(const float* a, const float* b, float* c, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + s * b[i];
}`

func openOrSkip(t *testing.T) *Device {
	t.Helper()
	d, err := Open()
	if err != nil {
		t.Skipf("no GPU: %v", err)
	}
	return d
}

// TestDriverEndToEnd compiles a kernel with NVRTC, loads the PTX through the
// driver, launches it and reads the result back.
func TestDriverEndToEnd(t *testing.T) {
	d := openOrSkip(t)
	defer d.Close()
	if os.Getenv("NVRTC_DLL") == "" {
		t.Skip("NVRTC_DLL not set")
	}
	cc, err := OpenCompiler("")
	if err != nil {
		t.Fatal(err)
	}
	ptx, err := cc.PTX(vecAdd, "vadd.cu", "compute_80")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%s (sm_%d%d, %d SMs, %d MB)", d.Name, d.Major, d.Minor, d.SMs, d.TotalMem>>20)
	const n = 100000
	a, b := make([]float32, n), make([]float32, n)
	for i := range a {
		a[i], b[i] = float32(i), float32(2*i)
	}
	out := make([]float32, n)
	d.Do(func() {
		m, err := d.LoadPTX(ptx)
		if err != nil {
			t.Fatal(err)
		}
		k, err := m.Func("vadd")
		if err != nil {
			t.Fatal(err)
		}
		A, _ := d.Alloc(n)
		B, _ := d.Alloc(n)
		C, _ := d.Alloc(n)
		defer A.Free()
		defer B.Free()
		defer C.Free()
		if err := A.Upload(0, a); err != nil {
			t.Fatal(err)
		}
		B.Upload(0, b)
		if err := k.Launch([3]int{(n + 255) / 256, 1, 1}, [3]int{256, 1, 1}, 0, A, B, C, int32(n), float32(0.5)); err != nil {
			t.Fatal(err)
		}
		if err := d.Sync(); err != nil {
			t.Fatal(err)
		}
		if err := C.Download(0, out); err != nil {
			t.Fatal(err)
		}
	})
	for i := range out {
		if want := a[i] + 0.5*b[i]; math.Abs(float64(out[i]-want)) > 1e-3 {
			t.Fatalf("out[%d] = %v want %v", i, out[i], want)
		}
	}
}
