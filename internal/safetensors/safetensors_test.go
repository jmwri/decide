package safetensors

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.safetensors")
	in := []Tensor{
		{"b.weight", []int{2, 3}, []float32{1, 2, 3, 4, 5, 6}},
		{"a.bias", []int{3}, []float32{-1, 0.5, 2}},
		{"empty", []int{0}, nil},
	}
	if err := Write(path, in, map[string]string{"format": "pt"}); err != nil {
		t.Fatal(err)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if !reflect.DeepEqual(f.Names(), []string{"a.bias", "b.weight", "empty"}) || f.Metadata["format"] != "pt" {
		t.Fatalf("names/metadata: %v %v", f.Names(), f.Metadata)
	}
	for _, want := range in {
		got, shape, err := f.Float32(want.Name)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(shape, want.Shape) || (len(want.Data) > 0 && !reflect.DeepEqual(got, want.Data)) {
			t.Fatalf("%s: %v %v", want.Name, got, shape)
		}
	}
	if _, _, err := f.Float32("nope"); err == nil {
		t.Fatal("missing tensor must error")
	}
}

func TestReadBaseModel(t *testing.T) {
	path := os.Getenv("DECIDE_BASE_MODEL")
	if path == "" {
		t.Skip("DECIDE_BASE_MODEL not set")
	}
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	w, shape, err := f.Float32("model.layers.0.attn.Wqkv.weight")
	if err != nil || !reflect.DeepEqual(shape, []int{2304, 768}) || len(w) != 2304*768 {
		t.Fatal(err, shape)
	}
}
