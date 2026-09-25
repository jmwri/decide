package blas

import "golang.org/x/sys/cpu"

//go:noescape
func kern6x16asm(kc int, a, b, c *float32, ldc int)

func init() {
	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		kern6x16 = kern6x16asm
	}
}
