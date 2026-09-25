// Package safetensors reads and writes the safetensors tensor container
// (https://github.com/huggingface/safetensors) for float tensors.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"unsafe"
)

// Info describes one tensor in a file.
type Info struct {
	DType   string   `json:"dtype"`
	Shape   []int    `json:"shape"`
	Offsets [2]int64 `json:"data_offsets"`
}

// File is an opened safetensors file.
type File struct {
	f        *os.File
	dataBase int64
	Tensors  map[string]Info
	Metadata map[string]string
}

// Close closes the file.
func (f *File) Close() error { return f.f.Close() }

// Names returns tensor names, sorted.
func (f *File) Names() []string {
	out := make([]string, 0, len(f.Tensors))
	for k := range f.Tensors {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Open reads the header of a safetensors file.
func Open(path string) (*File, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var n [8]byte
	if _, err := io.ReadFull(fh, n[:]); err != nil {
		fh.Close()
		return nil, err
	}
	hlen := binary.LittleEndian.Uint64(n[:])
	if hlen > 1<<28 {
		fh.Close()
		return nil, errors.New("safetensors: implausible header size")
	}
	hb := make([]byte, hlen)
	if _, err := io.ReadFull(fh, hb); err != nil {
		fh.Close()
		return nil, err
	}
	raw := map[string]json.RawMessage{}
	if err := json.Unmarshal(hb, &raw); err != nil {
		fh.Close()
		return nil, fmt.Errorf("safetensors: bad header: %w", err)
	}
	f := &File{f: fh, dataBase: 8 + int64(hlen), Tensors: map[string]Info{}}
	for k, v := range raw {
		if k == "__metadata__" {
			_ = json.Unmarshal(v, &f.Metadata)
			continue
		}
		var info Info
		if err := json.Unmarshal(v, &info); err != nil {
			fh.Close()
			return nil, fmt.Errorf("safetensors: tensor %s: %w", k, err)
		}
		f.Tensors[k] = info
	}
	return f, nil
}

// Float32 reads a tensor, converting F16/BF16/F64 to float32.
func (f *File) Float32(name string) ([]float32, []int, error) {
	info, ok := f.Tensors[name]
	if !ok {
		return nil, nil, fmt.Errorf("safetensors: tensor %q not found", name)
	}
	n := 1
	for _, d := range info.Shape {
		n *= d
	}
	size := info.Offsets[1] - info.Offsets[0]
	out := make([]float32, n)
	switch info.DType {
	case "F32":
		if size != int64(n)*4 {
			return nil, nil, fmt.Errorf("safetensors: %s has %d bytes for %d floats", name, size, n)
		}
		buf := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(out))), n*4)
		if _, err := f.f.ReadAt(buf, f.dataBase+info.Offsets[0]); err != nil {
			return nil, nil, err
		}
		if !littleEndian() {
			for i := range out {
				out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
			}
		}
	case "F16", "BF16":
		buf := make([]byte, size)
		if _, err := f.f.ReadAt(buf, f.dataBase+info.Offsets[0]); err != nil {
			return nil, nil, err
		}
		for i := 0; i < n; i++ {
			h := binary.LittleEndian.Uint16(buf[i*2:])
			if info.DType == "BF16" {
				out[i] = math.Float32frombits(uint32(h) << 16)
			} else {
				out[i] = halfToFloat(h)
			}
		}
	case "F64":
		buf := make([]byte, size)
		if _, err := f.f.ReadAt(buf, f.dataBase+info.Offsets[0]); err != nil {
			return nil, nil, err
		}
		for i := 0; i < n; i++ {
			out[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(buf[i*8:])))
		}
	default:
		return nil, nil, fmt.Errorf("safetensors: %s has unsupported dtype %s", name, info.DType)
	}
	return out, append([]int(nil), info.Shape...), nil
}

func littleEndian() bool {
	x := uint16(1)
	return *(*byte)(unsafe.Pointer(&x)) == 1
}

func halfToFloat(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h) & 0x3ff
	switch {
	case exp == 0:
		if frac == 0 {
			return math.Float32frombits(sign << 31)
		}
		e := uint32(127 - 15 + 1)
		for frac&0x400 == 0 {
			frac <<= 1
			e--
		}
		frac &= 0x3ff
		return math.Float32frombits(sign<<31 | e<<23 | frac<<13)
	case exp == 0x1f:
		return math.Float32frombits(sign<<31 | 0xff<<23 | frac<<13)
	}
	return math.Float32frombits(sign<<31 | (exp+127-15)<<23 | frac<<13)
}

// Tensor is a named float32 tensor to write.
type Tensor struct {
	Name  string
	Shape []int
	Data  []float32
}

// Write stores tensors as F32 in a safetensors file (atomically, via a temp
// file), with optional string metadata.
func Write(path string, tensors []Tensor, metadata map[string]string) error {
	sorted := append([]Tensor(nil), tensors...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Name < sorted[j].Name })
	header := map[string]any{}
	if len(metadata) > 0 {
		header["__metadata__"] = metadata
	}
	var off int64
	for _, t := range sorted {
		n := 1
		for _, d := range t.Shape {
			n *= d
		}
		if n != len(t.Data) {
			return fmt.Errorf("safetensors: %s shape %v does not match %d values", t.Name, t.Shape, len(t.Data))
		}
		header[t.Name] = Info{DType: "F32", Shape: t.Shape, Offsets: [2]int64{off, off + int64(n)*4}}
		off += int64(n) * 4
	}
	hb, err := json.Marshal(header)
	if err != nil {
		return err
	}
	// pad header to 8-byte alignment with spaces
	for (8+len(hb))%8 != 0 {
		hb = append(hb, ' ')
	}
	tmp := path + ".tmp"
	fh, err := os.Create(tmp)
	if err != nil {
		return err
	}
	var n [8]byte
	binary.LittleEndian.PutUint64(n[:], uint64(len(hb)))
	if _, err := fh.Write(n[:]); err == nil {
		_, err = fh.Write(hb)
	}
	if err != nil {
		fh.Close()
		return err
	}
	for _, t := range sorted {
		if len(t.Data) == 0 {
			continue
		}
		var werr error
		if littleEndian() {
			_, werr = fh.Write(unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(t.Data))), len(t.Data)*4))
		} else {
			buf := make([]byte, len(t.Data)*4)
			for i, v := range t.Data {
				binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
			}
			_, werr = fh.Write(buf)
		}
		if werr != nil {
			fh.Close()
			return werr
		}
	}
	if err := fh.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
