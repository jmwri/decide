package decide

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// KV is one key/value pair of an Object.
type KV struct {
	Key   string
	Value any
}

// Object is an insertion-ordered JSON object. Decide renders a state object into
// the model's input text in key order, so order is significant; Go maps have
// none. Plain map[string]any states are accepted and rendered in sorted key
// order.
type Object []KV

// Get returns the value stored under key.
func (o Object) Get(key string) (any, bool) {
	for _, kv := range o {
		if kv.Key == key {
			return kv.Value, true
		}
	}
	return nil, false
}

// MarshalJSON writes the object with keys in order.
func (o Object) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, kv := range o {
		if i > 0 {
			b.WriteByte(',')
		}
		k, _ := json.Marshal(kv.Key)
		b.Write(k)
		b.WriteByte(':')
		v, err := json.Marshal(kv.Value)
		if err != nil {
			return nil, err
		}
		b.Write(v)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

// UnmarshalJSON reads an object, preserving key order.
func (o *Object) UnmarshalJSON(data []byte) error {
	v, err := decodeOrdered(data)
	if err != nil {
		return err
	}
	obj, ok := v.(Object)
	if !ok {
		return fmt.Errorf("decide: expected JSON object, got %T", v)
	}
	*o = obj
	return nil
}

// decodeOrdered parses JSON into: Object, []any, string, json.Number, bool, nil.
func decodeOrdered(data []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	v, err := decodeValue(dec)
	if err != nil {
		return nil, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, fmt.Errorf("decide: trailing data after JSON value")
	}
	return v, nil
}

func decodeValue(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	switch t := tok.(type) {
	case json.Delim:
		switch t {
		case '{':
			obj := Object{}
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return nil, err
				}
				key, _ := kt.(string)
				val, err := decodeValue(dec)
				if err != nil {
					return nil, err
				}
				replaced := false
				for i := range obj {
					if obj[i].Key == key { // duplicate keys: last wins, first position (Python dict semantics)
						obj[i].Value = val
						replaced = true
						break
					}
				}
				if !replaced {
					obj = append(obj, KV{key, val})
				}
			}
			if _, err := dec.Token(); err != nil {
				return nil, err
			}
			return obj, nil
		case '[':
			arr := []any{}
			for dec.More() {
				val, err := decodeValue(dec)
				if err != nil {
					return nil, err
				}
				arr = append(arr, val)
			}
			if _, err := dec.Token(); err != nil {
				return nil, err
			}
			return arr, nil
		}
	}
	return tok, nil // string, json.Number, bool, nil
}

// normalize converts arbitrary Go values (structs, maps, slices) into the
// ordered-JSON model by round-tripping through encoding/json.
func normalize(v any) any {
	switch x := v.(type) {
	case nil, string, bool, json.Number, Object, []any:
		return v
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		o := make(Object, 0, len(keys))
		for _, k := range keys {
			o = append(o, KV{k, normalize(x[k])})
		}
		return o
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return json.Number(strconv.FormatInt(rv.Int(), 10))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return json.Number(strconv.FormatUint(rv.Uint(), 10))
	case reflect.Float32, reflect.Float64:
		return rv.Float()
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprint(v)
	}
	out, err := decodeOrdered(raw)
	if err != nil {
		return fmt.Sprint(v)
	}
	return out
}

// ---------------------------------------------------------------------------
// Python-compatible text rendering. The model was trained on inputs produced
// by Python's f-strings, so nested values are rendered the way str()/repr()
// would.

func pyStr(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return pyRepr(v)
}

func pyRepr(v any) string {
	v = normalize(v)
	switch x := v.(type) {
	case nil:
		return "None"
	case bool:
		if x {
			return "True"
		}
		return "False"
	case string:
		return pyStringRepr(x)
	case float64:
		return pyFloat(x)
	case json.Number:
		s := x.String()
		if !strings.ContainsAny(s, ".eE") {
			return s
		}
		f, err := x.Float64()
		if err != nil {
			return s
		}
		return pyFloat(f)
	case []any:
		parts := make([]string, len(x))
		for i, e := range x {
			parts[i] = pyRepr(e)
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case Object:
		parts := make([]string, len(x))
		for i, kv := range x {
			parts[i] = pyStringRepr(kv.Key) + ": " + pyRepr(kv.Value)
		}
		return "{" + strings.Join(parts, ", ") + "}"
	}
	return fmt.Sprint(v)
}

func pyStringRepr(s string) string {
	quote := byte('\'')
	if strings.Contains(s, "'") && !strings.Contains(s, "\"") {
		quote = '"'
	}
	var b strings.Builder
	b.WriteByte(quote)
	for _, r := range s {
		switch {
		case r == rune(quote) || r == '\\':
			b.WriteByte('\\')
			b.WriteRune(r)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r == '\t':
			b.WriteString(`\t`)
		case r < 0x20 || r == 0x7f:
			fmt.Fprintf(&b, `\x%02x`, r)
		case r < 0x7f:
			b.WriteRune(r)
		case unicode.IsPrint(r):
			b.WriteRune(r)
		case r <= 0xff:
			fmt.Fprintf(&b, `\x%02x`, r)
		case r <= 0xffff:
			fmt.Fprintf(&b, `\u%04x`, r)
		default:
			fmt.Fprintf(&b, `\U%08x`, r)
		}
	}
	b.WriteByte(quote)
	return b.String()
}

// pyFloat formats like Python's repr(float).
func pyFloat(f float64) string {
	switch {
	case math.IsNaN(f):
		return "nan"
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	case f == 0:
		if math.Signbit(f) {
			return "-0.0"
		}
		return "0.0"
	}
	e := strconv.FormatFloat(f, 'e', -1, 64) // d.ddde±XX
	neg := strings.HasPrefix(e, "-")
	e = strings.TrimPrefix(e, "-")
	mant, expS, _ := strings.Cut(e, "e")
	exp, _ := strconv.Atoi(expS)
	digits := strings.Replace(mant, ".", "", 1)
	decpt := exp + 1
	var out string
	switch {
	case -4 < decpt && decpt <= 16:
		switch {
		case decpt <= 0:
			out = "0." + strings.Repeat("0", -decpt) + digits
		case decpt >= len(digits):
			out = digits + strings.Repeat("0", decpt-len(digits)) + ".0"
		default:
			out = digits[:decpt] + "." + digits[decpt:]
		}
	default:
		m := digits[:1]
		if len(digits) > 1 {
			m += "." + digits[1:]
		}
		sign := "+"
		if exp < 0 {
			sign = "-"
			exp = -exp
		}
		out = fmt.Sprintf("%se%s%02d", m, sign, exp)
	}
	if neg {
		out = "-" + out
	}
	return out
}

// pyJSONDumps mimics Python's json.dumps (default separators, ensure_ascii).
// Structured instructions are serialised this way so the model sees the same
// text regardless of which SDK produced it.
func pyJSONDumps(v any, sortKeys bool) string {
	var b strings.Builder
	pyDump(&b, normalize(v), sortKeys)
	return b.String()
}

func pyDump(b *strings.Builder, v any, sortKeys bool) {
	switch x := v.(type) {
	case nil:
		b.WriteString("null")
	case bool:
		if x {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case string:
		pyDumpString(b, x)
	case float64:
		b.WriteString(pyFloat(x))
	case json.Number:
		s := x.String()
		if strings.ContainsAny(s, ".eE") {
			if f, err := x.Float64(); err == nil {
				s = pyFloat(f)
			}
		}
		b.WriteString(s)
	case []any:
		b.WriteByte('[')
		for i, e := range x {
			if i > 0 {
				b.WriteString(", ")
			}
			pyDump(b, normalize(e), sortKeys)
		}
		b.WriteByte(']')
	case Object:
		kvs := append(Object(nil), x...)
		if sortKeys {
			sort.SliceStable(kvs, func(i, j int) bool { return codepointLess(kvs[i].Key, kvs[j].Key) })
		}
		b.WriteByte('{')
		for i, kv := range kvs {
			if i > 0 {
				b.WriteString(", ")
			}
			pyDumpString(b, kv.Key)
			b.WriteString(": ")
			pyDump(b, normalize(kv.Value), sortKeys)
		}
		b.WriteByte('}')
	default:
		pyDumpString(b, fmt.Sprint(v))
	}
}

// codepointLess orders strings by Unicode code point, like Python's sort.
func codepointLess(a, b string) bool {
	for len(a) > 0 && len(b) > 0 {
		ra, sa := utf8.DecodeRuneInString(a)
		rb, sb := utf8.DecodeRuneInString(b)
		if ra != rb {
			return ra < rb
		}
		a, b = a[sa:], b[sb:]
	}
	return len(a) < len(b)
}

func pyDumpString(b *strings.Builder, s string) {
	b.WriteByte('"')
	for _, r := range s {
		switch {
		case r == '"':
			b.WriteString(`\"`)
		case r == '\\':
			b.WriteString(`\\`)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r == '\t':
			b.WriteString(`\t`)
		case r == '\b':
			b.WriteString(`\b`)
		case r == '\f':
			b.WriteString(`\f`)
		case r < 0x20:
			fmt.Fprintf(b, `\u%04x`, r)
		case r < 0x7f || r == 0x7f:
			b.WriteRune(r)
		case r <= 0xffff:
			fmt.Fprintf(b, `\u%04x`, r)
		default:
			r -= 0x10000
			fmt.Fprintf(b, `\u%04x\u%04x`, 0xd800+(r>>10), 0xdc00+(r&0x3ff))
		}
	}
	b.WriteByte('"')
}
