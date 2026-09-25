// Package cuda drives an NVIDIA GPU through the CUDA driver API using only the
// driver library that ships with the GPU driver (nvcuda.dll / libcuda.so):
// no CUDA Toolkit, no cgo. Kernels are PTX text (see kernels/*.cu and
// cmd/ptxgen), JIT-compiled to the installed GPU by the driver at load time.
package cuda

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// ErrNoGPU is returned when no usable CUDA device or driver is present.
var ErrNoGPU = errors.New("cuda: no usable NVIDIA GPU / driver found")

type api struct {
	cuInit                   func(flags uint32) int32
	cuDeviceGetCount         func(count *int32) int32
	cuDeviceGet              func(dev *int32, ordinal int32) int32
	cuDeviceGetName          func(name *byte, n int32, dev int32) int32
	cuDeviceGetAttribute     func(v *int32, attr int32, dev int32) int32
	cuDeviceTotalMem         func(bytes *uintptr, dev int32) int32
	cuDevicePrimaryCtxRetain func(ctx *uintptr, dev int32) int32
	cuCtxSetCurrent          func(ctx uintptr) int32
	cuCtxSynchronize         func() int32
	cuMemAlloc               func(ptr *uintptr, size uintptr) int32
	cuMemFree                func(ptr uintptr) int32
	cuMemGetInfo             func(free, total *uintptr) int32
	cuMemcpyHtoD             func(dst uintptr, src unsafe.Pointer, n uintptr) int32
	cuMemcpyDtoH             func(dst unsafe.Pointer, src uintptr, n uintptr) int32
	cuMemcpyDtoD             func(dst, src uintptr, n uintptr) int32
	cuMemsetD32              func(dst uintptr, v uint32, n uintptr) int32
	cuModuleLoadDataEx       func(mod *uintptr, image unsafe.Pointer, nopt uint32, opts *int32, vals *unsafe.Pointer) int32
	cuModuleGetFunction      func(f *uintptr, mod uintptr, name *byte) int32
	cuFuncSetAttribute       func(f uintptr, attr int32, v int32) int32
	cuLaunchKernel           func(f uintptr, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, params *unsafe.Pointer, extra *unsafe.Pointer) int32
	cuGetErrorName           func(code int32, s **byte) int32
	cuGetErrorString         func(code int32, s **byte) int32
}

var (
	loadOnce sync.Once
	drv      api
	loadErr  error
)

func libName() string {
	switch runtime.GOOS {
	case "windows":
		return "nvcuda.dll"
	case "darwin":
		return "" // no CUDA on macOS
	}
	return "libcuda.so.1"
}

func load() error {
	loadOnce.Do(func() {
		name := libName()
		if name == "" {
			loadErr = ErrNoGPU
			return
		}
		lib, err := openLib(name)
		if err != nil {
			loadErr = fmt.Errorf("%w (%v)", ErrNoGPU, err)
			return
		}
		reg := func(fn any, sym string) {
			defer func() { _ = recover() }() // a missing optional symbol leaves the func nil
			purego.RegisterLibFunc(fn, lib, sym)
		}
		reg(&drv.cuInit, "cuInit")
		reg(&drv.cuDeviceGetCount, "cuDeviceGetCount")
		reg(&drv.cuDeviceGet, "cuDeviceGet")
		reg(&drv.cuDeviceGetName, "cuDeviceGetName")
		reg(&drv.cuDeviceGetAttribute, "cuDeviceGetAttribute")
		reg(&drv.cuDeviceTotalMem, "cuDeviceTotalMem_v2")
		reg(&drv.cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain")
		reg(&drv.cuCtxSetCurrent, "cuCtxSetCurrent")
		reg(&drv.cuCtxSynchronize, "cuCtxSynchronize")
		reg(&drv.cuMemAlloc, "cuMemAlloc_v2")
		reg(&drv.cuMemFree, "cuMemFree_v2")
		reg(&drv.cuMemGetInfo, "cuMemGetInfo_v2")
		reg(&drv.cuMemcpyHtoD, "cuMemcpyHtoD_v2")
		reg(&drv.cuMemcpyDtoH, "cuMemcpyDtoH_v2")
		reg(&drv.cuMemcpyDtoD, "cuMemcpyDtoD_v2")
		reg(&drv.cuMemsetD32, "cuMemsetD32_v2")
		reg(&drv.cuModuleLoadDataEx, "cuModuleLoadDataEx")
		reg(&drv.cuModuleGetFunction, "cuModuleGetFunction")
		reg(&drv.cuFuncSetAttribute, "cuFuncSetAttribute")
		reg(&drv.cuLaunchKernel, "cuLaunchKernel")
		reg(&drv.cuGetErrorName, "cuGetErrorName")
		reg(&drv.cuGetErrorString, "cuGetErrorString")
		if drv.cuInit == nil || drv.cuLaunchKernel == nil {
			loadErr = fmt.Errorf("%w (driver library is missing required entry points)", ErrNoGPU)
		}
	})
	return loadErr
}

func cstr(p *byte) string {
	if p == nil {
		return ""
	}
	var b []byte
	for i := 0; ; i++ {
		c := *(*byte)(unsafe.Add(unsafe.Pointer(p), i))
		if c == 0 {
			break
		}
		b = append(b, c)
	}
	return string(b)
}

// Error is a CUDA driver error.
type Error struct {
	Code int32
	Op   string
}

func (e *Error) Error() string {
	var name, desc *byte
	if drv.cuGetErrorName != nil {
		drv.cuGetErrorName(e.Code, &name)
		drv.cuGetErrorString(e.Code, &desc)
	}
	return fmt.Sprintf("cuda: %s failed: %s (%s) [code %d]", e.Op, cstr(name), cstr(desc), e.Code)
}

func check(op string, code int32) error {
	if code != 0 {
		return &Error{Code: code, Op: op}
	}
	return nil
}

// Device is an initialised GPU with its primary context. All calls that touch
// the GPU must run on the device's locked OS thread; use Do.
type Device struct {
	Name         string
	Major, Minor int
	TotalMem     uint64
	SMs          int
	ctx          uintptr
	jobs         chan func()
	done         chan struct{}
}

// Open initialises the driver and the first GPU.
func Open() (*Device, error) {
	if err := load(); err != nil {
		return nil, err
	}
	if err := check("cuInit", drv.cuInit(0)); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrNoGPU, err)
	}
	var n int32
	if err := check("cuDeviceGetCount", drv.cuDeviceGetCount(&n)); err != nil || n == 0 {
		return nil, ErrNoGPU
	}
	var dev int32
	if err := check("cuDeviceGet", drv.cuDeviceGet(&dev, 0)); err != nil {
		return nil, err
	}
	d := &Device{jobs: make(chan func()), done: make(chan struct{})}
	name := make([]byte, 128)
	drv.cuDeviceGetName(&name[0], int32(len(name)), dev)
	d.Name = cstr(&name[0])
	var v int32
	drv.cuDeviceGetAttribute(&v, 75, dev) // COMPUTE_CAPABILITY_MAJOR
	d.Major = int(v)
	drv.cuDeviceGetAttribute(&v, 76, dev) // COMPUTE_CAPABILITY_MINOR
	d.Minor = int(v)
	drv.cuDeviceGetAttribute(&v, 16, dev) // MULTIPROCESSOR_COUNT
	d.SMs = int(v)
	var tot uintptr
	drv.cuDeviceTotalMem(&tot, dev)
	d.TotalMem = uint64(tot)
	if err := check("cuDevicePrimaryCtxRetain", drv.cuDevicePrimaryCtxRetain(&d.ctx, dev)); err != nil {
		return nil, err
	}
	go d.loop()
	return d, nil
}

// loop is the device's dedicated OS thread: the CUDA context is thread-bound,
// and Go goroutines migrate, so every GPU call runs here.
func (d *Device) loop() {
	runtime.LockOSThread()
	drv.cuCtxSetCurrent(d.ctx)
	for f := range d.jobs {
		f()
	}
	close(d.done)
}

// Do runs fn on the device thread and waits for it. Panics propagate.
func (d *Device) Do(fn func()) {
	var pv any
	fin := make(chan struct{})
	d.jobs <- func() {
		defer close(fin)
		defer func() { pv = recover() }()
		fn()
	}
	<-fin
	if pv != nil {
		panic(pv)
	}
}

// Close stops the device thread. Buffers must be freed first.
func (d *Device) Close() {
	close(d.jobs)
	<-d.done
}

// Sync waits for all queued kernels (call inside Do).
func (d *Device) Sync() error { return check("cuCtxSynchronize", drv.cuCtxSynchronize()) }

// MemInfo reports free and total device memory in bytes (call inside Do).
func (d *Device) MemInfo() (free, total uint64) {
	var f, t uintptr
	drv.cuMemGetInfo(&f, &t)
	return uint64(f), uint64(t)
}

// Buf is device memory holding float32 (or int32) values.
type Buf struct {
	ptr uintptr
	n   int // elements
}

// Alloc allocates n 4-byte elements (call inside Do).
func (d *Device) Alloc(n int) (*Buf, error) {
	if n <= 0 {
		n = 1
	}
	var p uintptr
	if err := check("cuMemAlloc", drv.cuMemAlloc(&p, uintptr(n)*4)); err != nil {
		return nil, fmt.Errorf("allocating %d MB: %w", n*4>>20, err)
	}
	return &Buf{ptr: p, n: n}, nil
}

// Free releases the buffer (call inside Do).
func (b *Buf) Free() {
	if b != nil && b.ptr != 0 {
		drv.cuMemFree(b.ptr)
		b.ptr = 0
	}
}

// Len is the number of elements.
func (b *Buf) Len() int { return b.n }

// Ptr is the raw device address.
func (b *Buf) Ptr() uintptr { return b.ptr }

// Slice returns a view at element offset off (length n) sharing storage.
func (b *Buf) Slice(off, n int) *Buf { return &Buf{ptr: b.ptr + uintptr(off)*4, n: n} }

// Upload copies host floats to the buffer starting at element off.
func (b *Buf) Upload(off int, src []float32) error {
	if len(src) == 0 {
		return nil
	}
	if off+len(src) > b.n {
		return fmt.Errorf("cuda: upload of %d floats at %d overflows a %d-element buffer", len(src), off, b.n)
	}
	return check("cuMemcpyHtoD", drv.cuMemcpyHtoD(b.ptr+uintptr(off)*4, unsafe.Pointer(&src[0]), uintptr(len(src))*4))
}

// UploadInt32 copies host int32s.
func (b *Buf) UploadInt32(off int, src []int32) error {
	if len(src) == 0 {
		return nil
	}
	if off+len(src) > b.n {
		return fmt.Errorf("cuda: upload of %d ints at %d overflows a %d-element buffer", len(src), off, b.n)
	}
	return check("cuMemcpyHtoD", drv.cuMemcpyHtoD(b.ptr+uintptr(off)*4, unsafe.Pointer(&src[0]), uintptr(len(src))*4))
}

// Download copies floats from the buffer (starting at element off) into dst.
func (b *Buf) Download(off int, dst []float32) error {
	if len(dst) == 0 {
		return nil
	}
	if off+len(dst) > b.n {
		return fmt.Errorf("cuda: download of %d floats at %d overflows a %d-element buffer", len(dst), off, b.n)
	}
	return check("cuMemcpyDtoH", drv.cuMemcpyDtoH(unsafe.Pointer(&dst[0]), b.ptr+uintptr(off)*4, uintptr(len(dst))*4))
}

// CopyFrom copies n elements device-to-device.
func (b *Buf) CopyFrom(dstOff int, src *Buf, srcOff, n int) error {
	return check("cuMemcpyDtoD", drv.cuMemcpyDtoD(b.ptr+uintptr(dstOff)*4, src.ptr+uintptr(srcOff)*4, uintptr(n)*4))
}

// Zero sets every element to 0.
func (b *Buf) Zero() error { return check("cuMemsetD32", drv.cuMemsetD32(b.ptr, 0, uintptr(b.n))) }

// Module is a loaded PTX module.
type Module struct{ h uintptr }

// LoadPTX JIT-compiles PTX text for the current GPU.
func (d *Device) LoadPTX(ptx string) (*Module, error) {
	img := append([]byte(ptx), 0)
	var m uintptr
	const (
		optErrorLogBuffer = 5
		optErrorLogSize   = 6
	)
	logBuf := make([]byte, 16<<10)
	opts := []int32{optErrorLogBuffer, optErrorLogSize}
	vals := []uintptr{uintptr(unsafe.Pointer(&logBuf[0])), uintptr(len(logBuf))}
	code := drv.cuModuleLoadDataEx(&m, unsafe.Pointer(&img[0]), 2, &opts[0], (*unsafe.Pointer)(unsafe.Pointer(&vals[0])))
	if code != 0 {
		return nil, fmt.Errorf("%w\n%s", &Error{Code: code, Op: "cuModuleLoadDataEx"}, cstr(&logBuf[0]))
	}
	return &Module{h: m}, nil
}

// Kernel is one entry point of a module.
type Kernel struct {
	h    uintptr
	Name string
}

// Func looks up a kernel by name.
func (m *Module) Func(name string) (*Kernel, error) {
	var f uintptr
	nb := append([]byte(name), 0)
	if err := check("cuModuleGetFunction("+name+")", drv.cuModuleGetFunction(&f, m.h, &nb[0])); err != nil {
		return nil, err
	}
	return &Kernel{h: f, Name: name}, nil
}

// SetMaxDynamicShared raises the dynamic shared-memory limit for the kernel.
func (k *Kernel) SetMaxDynamicShared(bytes int) error {
	return check("cuFuncSetAttribute", drv.cuFuncSetAttribute(k.h, 8, int32(bytes)))
}

// Launch runs the kernel asynchronously. Arguments must match the kernel's
// parameters: *Buf (device pointer), int32, uint32, int64 or float32.
func (k *Kernel) Launch(grid, block [3]int, shared int, args ...any) error {
	vals := make([]uint64, len(args))
	ptrs := make([]unsafe.Pointer, len(args))
	for i, a := range args {
		switch v := a.(type) {
		case *Buf:
			if v != nil { // a nil buffer passes a null pointer (optional kernel arguments)
				vals[i] = uint64(v.ptr)
			}
		case int32:
			vals[i] = uint64(uint32(v))
		case uint32:
			vals[i] = uint64(v)
		case int:
			vals[i] = uint64(uint32(int32(v)))
		case int64:
			vals[i] = uint64(v)
		case float32:
			vals[i] = uint64(math.Float32bits(v))
		default:
			return fmt.Errorf("cuda: unsupported kernel argument %d of type %T", i, a)
		}
		ptrs[i] = unsafe.Pointer(&vals[i])
	}
	var pp *unsafe.Pointer
	if len(ptrs) > 0 {
		pp = &ptrs[0]
	}
	code := drv.cuLaunchKernel(k.h, uint32(grid[0]), uint32(grid[1]), uint32(grid[2]), uint32(block[0]), uint32(block[1]), uint32(block[2]), uint32(shared), 0, pp, nil)
	runtime.KeepAlive(vals)
	runtime.KeepAlive(ptrs)
	return check("cuLaunchKernel("+k.Name+")", code)
}
