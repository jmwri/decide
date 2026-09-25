// Command ptxgen compiles internal/cuda/kernels/kernels.cu to PTX with NVRTC
// and writes internal/cuda/kernels.ptx (embedded into the binary). It is only
// needed when the kernels change:
//
//	NVRTC_DLL=/path/to/nvrtc64_120_0.dll go run ./cmd/ptxgen
//
// The PTX targets the virtual architecture compute_80; the driver JIT-compiles
// it for the installed GPU (sm_80 and newer).
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/jmwri/decide/internal/cuda"
)

func main() {
	src := flag.String("src", "internal/cuda/kernels/kernels.cu", "CUDA C source")
	out := flag.String("out", "internal/cuda/kernels.ptx", "PTX output")
	arch := flag.String("arch", "compute_80", "virtual architecture")
	dll := flag.String("nvrtc", "", "path to the NVRTC library (default $NVRTC_DLL)")
	flag.Parse()
	code, err := os.ReadFile(*src)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	cc, err := cuda.OpenCompiler(*dll)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	ptx, err := cc.PTX(string(code), "kernels.cu", *arch, "--fmad=true")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := os.WriteFile(*out, []byte(ptx), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s (%d bytes)\n", *out, len(ptx))
}
