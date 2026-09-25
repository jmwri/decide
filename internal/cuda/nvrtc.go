package cuda

import (
	"fmt"
	"os"
	"path/filepath"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Compiler compiles CUDA C to PTX with NVRTC. It is a development-time tool
// (used by cmd/ptxgen to produce the embedded kernels.ptx); training and
// inference only need the GPU driver.
type Compiler struct {
	create  func(prog *uintptr, src *byte, name *byte, nh int32, hdr **byte, inc **byte) int32
	compile func(prog uintptr, n int32, opts **byte) int32
	ptxSize func(prog uintptr, n *uintptr) int32
	ptx     func(prog uintptr, out *byte) int32
	logSize func(prog uintptr, n *uintptr) int32
	log     func(prog uintptr, out *byte) int32
	destroy func(prog *uintptr) int32
}

// OpenCompiler loads the NVRTC library at dll (e.g. nvrtc64_120_0.dll from
// the nvidia-cuda-nvrtc-cu12 wheel, or $NVRTC_DLL when dll is empty).
func OpenCompiler(dll string) (*Compiler, error) {
	if dll == "" {
		dll = os.Getenv("NVRTC_DLL")
	}
	if dll == "" {
		return nil, fmt.Errorf("cuda: set NVRTC_DLL to the path of nvrtc64_*.dll / libnvrtc.so")
	}
	// nvrtc loads its builtins library by bare name, so its directory must be searchable.
	if dir := filepath.Dir(dll); dir != "." {
		_ = os.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	}
	lib, err := openLib(dll)
	if err != nil {
		return nil, fmt.Errorf("cuda: loading %s: %w", dll, err)
	}
	c := &Compiler{}
	purego.RegisterLibFunc(&c.create, lib, "nvrtcCreateProgram")
	purego.RegisterLibFunc(&c.compile, lib, "nvrtcCompileProgram")
	purego.RegisterLibFunc(&c.ptxSize, lib, "nvrtcGetPTXSize")
	purego.RegisterLibFunc(&c.ptx, lib, "nvrtcGetPTX")
	purego.RegisterLibFunc(&c.logSize, lib, "nvrtcGetProgramLogSize")
	purego.RegisterLibFunc(&c.log, lib, "nvrtcGetProgramLog")
	purego.RegisterLibFunc(&c.destroy, lib, "nvrtcDestroyProgram")
	return c, nil
}

// PTX compiles CUDA C source for a virtual architecture such as "compute_80".
func (c *Compiler) PTX(src, name, arch string, extra ...string) (string, error) {
	sb, nb := append([]byte(src), 0), append([]byte(name), 0)
	var prog uintptr
	if r := c.create(&prog, &sb[0], &nb[0], 0, nil, nil); r != 0 {
		return "", fmt.Errorf("nvrtcCreateProgram: code %d", r)
	}
	defer c.destroy(&prog)
	opts := append([]string{"--gpu-architecture=" + arch, "--std=c++17", "-default-device"}, extra...)
	bufs := make([][]byte, len(opts))
	ptrs := make([]*byte, len(opts))
	for i, o := range opts {
		bufs[i] = append([]byte(o), 0)
		ptrs[i] = &bufs[i][0]
	}
	r := c.compile(prog, int32(len(opts)), &ptrs[0])
	var ln uintptr
	c.logSize(prog, &ln)
	logText := ""
	if ln > 1 {
		lb := make([]byte, ln)
		c.log(prog, &lb[0])
		logText = string(lb[:ln-1])
	}
	if r != 0 {
		return "", fmt.Errorf("nvrtc compile failed (code %d):\n%s", r, logText)
	}
	var n uintptr
	c.ptxSize(prog, &n)
	out := make([]byte, n)
	c.ptx(prog, &out[0])
	_ = unsafe.Pointer(nil)
	return string(out[:n-1]), nil
}
