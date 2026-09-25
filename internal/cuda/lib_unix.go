//go:build !windows

package cuda

import "github.com/ebitengine/purego"

func openLib(name string) (uintptr, error) {
	return purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}
