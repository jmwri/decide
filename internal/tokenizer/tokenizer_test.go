package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestGolden(t *testing.T) {
	dir := os.Getenv("DECIDE_BASE_DIR")
	if dir == "" {
		t.Skip("DECIDE_BASE_DIR not set (ModernBERT-base directory with tokenizer.json)")
	}
	tok, err := Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("testdata/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Text       string  `json:"text"`
		IDs        []int32 `json:"ids"`
		IDsSpecial []int32 `json:"ids_special"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		if got := tok.Encode(c.Text, false); !reflect.DeepEqual(nilToEmpty(got), nilToEmpty(c.IDs)) {
			t.Errorf("Encode(%q)\n got %v\nwant %v", c.Text, got, c.IDs)
		}
		if got := tok.Encode(c.Text, true); !reflect.DeepEqual(got, c.IDsSpecial) {
			t.Errorf("Encode(%q, special)\n got %v\nwant %v", c.Text, got, c.IDsSpecial)
		}
	}
}

func nilToEmpty(s []int32) []int32 {
	if s == nil {
		return []int32{}
	}
	return s
}

func TestPreTokenize(t *testing.T) {
	cases := map[string][]string{
		"Hello world": {"Hello", " world"},
		"a  b":        {"a", " ", " b"},
		"it's":        {"it", "'s"},
		"x  ":         {"x", "  "},
		"foo\n\nbar":  {"foo", "\n", "\n", "bar"},
		"a1 b2":       {"a", "1", " b", "2"},
		"end.\n":      {"end", ".", "\n"},
		" ,":          {" ,"},
	}
	for in, want := range cases {
		if got := preTokenize(in); !reflect.DeepEqual(got, want) {
			t.Errorf("preTokenize(%q) = %q, want %q", in, got, want)
		}
	}
}
