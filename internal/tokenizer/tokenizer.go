// Package tokenizer implements the byte-level BPE tokenizer used by
// ModernBERT (a HuggingFace tokenizers.json with an NFC normalizer, a
// GPT-2 style ByteLevel pre-tokenizer, added tokens and a [CLS] $A [SEP]
// post-processor).
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

type addedToken struct {
	ID         int32
	Content    string
	Special    bool
	Normalized bool
	LStrip     bool
	RStrip     bool
}

type pair struct{ a, b string }

// Tokenizer encodes text into ModernBERT token ids. It is safe for concurrent use.
type Tokenizer struct {
	vocab   map[string]int32
	ranks   map[pair]int
	special []addedToken // special, matched on the raw text
	normal  []addedToken // matched after NFC normalisation

	clsID, sepID, maskID, padID int32

	cacheMu sync.RWMutex
	cache   map[string][]int32
}

// MaskID is the id of the [MASK] token.
func (t *Tokenizer) MaskID() int32 { return t.maskID }

// SepID is the id of the [SEP] token.
func (t *Tokenizer) SepID() int32 { return t.sepID }

// ClsID is the id of the [CLS] token.
func (t *Tokenizer) ClsID() int32 { return t.clsID }

// Load reads a tokenizer.json file.
func Load(filename string) (*Tokenizer, error) {
	raw, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return Parse(raw)
}

// Parse parses tokenizer.json contents.
func Parse(raw []byte) (*Tokenizer, error) {
	var f struct {
		AddedTokens []struct {
			ID         int32  `json:"id"`
			Content    string `json:"content"`
			Special    bool   `json:"special"`
			Normalized bool   `json:"normalized"`
			LStrip     bool   `json:"lstrip"`
			RStrip     bool   `json:"rstrip"`
		} `json:"added_tokens"`
		Model struct {
			Type   string            `json:"type"`
			Vocab  map[string]int32  `json:"vocab"`
			Merges []json.RawMessage `json:"merges"`
		} `json:"model"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, fmt.Errorf("tokenizer: %w", err)
	}
	if f.Model.Type != "BPE" {
		return nil, fmt.Errorf("tokenizer: unsupported model type %q", f.Model.Type)
	}
	t := &Tokenizer{
		vocab: f.Model.Vocab,
		ranks: make(map[pair]int, len(f.Model.Merges)),
		cache: map[string][]int32{},
	}
	for i, m := range f.Model.Merges {
		var p pair
		var arr []string
		if err := json.Unmarshal(m, &arr); err == nil && len(arr) == 2 {
			p = pair{arr[0], arr[1]}
		} else {
			var s string
			if err := json.Unmarshal(m, &s); err != nil {
				return nil, fmt.Errorf("tokenizer: bad merge %d", i)
			}
			a, b, ok := strings.Cut(s, " ")
			if !ok {
				return nil, fmt.Errorf("tokenizer: bad merge %q", s)
			}
			p = pair{a, b}
		}
		if _, dup := t.ranks[p]; !dup {
			t.ranks[p] = i
		}
	}
	for _, a := range f.AddedTokens {
		at := addedToken{a.ID, a.Content, a.Special, a.Normalized, a.LStrip, a.RStrip}
		if a.Special && !a.Normalized {
			t.special = append(t.special, at)
		} else {
			t.normal = append(t.normal, at)
		}
		switch a.Content {
		case "[CLS]":
			t.clsID = a.ID
		case "[SEP]":
			t.sepID = a.ID
		case "[MASK]":
			t.maskID = a.ID
		case "[PAD]":
			t.padID = a.ID
		}
	}
	if t.maskID == 0 || t.sepID == 0 || t.clsID == 0 {
		return nil, fmt.Errorf("tokenizer: missing [CLS]/[SEP]/[MASK] added tokens")
	}
	return t, nil
}

// Encode tokenizes text. With addSpecial it wraps the ids in [CLS] ... [SEP].
func (t *Tokenizer) Encode(text string, addSpecial bool) []int32 {
	var ids []int32
	if addSpecial {
		ids = append(ids, t.clsID)
	}
	// 1. Split on non-normalized (special) added tokens of the raw text.
	for _, seg := range splitOnTokens(text, t.special, false) {
		if seg.tok != nil {
			ids = append(ids, seg.tok.ID)
			continue
		}
		// 2. Normalise, then split on normalized added tokens.
		s := norm.NFC.String(seg.text)
		for _, seg2 := range splitOnTokens(s, t.normal, true) {
			if seg2.tok != nil {
				ids = append(ids, seg2.tok.ID)
				continue
			}
			for _, w := range preTokenize(seg2.text) {
				ids = append(ids, t.bpe(byteLevel(w))...)
			}
		}
	}
	if addSpecial {
		ids = append(ids, t.sepID)
	}
	return ids
}

// Count returns the number of tokens in text, without special tokens.
func (t *Tokenizer) Count(text string) int { return len(t.Encode(text, false)) }

type segment struct {
	text string
	tok  *addedToken
}

// splitOnTokens performs leftmost-longest extraction of added tokens,
// honouring lstrip/rstrip for the special ones.
func splitOnTokens(s string, toks []addedToken, _ bool) []segment {
	if len(toks) == 0 || s == "" {
		if s == "" {
			return nil
		}
		return []segment{{text: s}}
	}
	var out []segment
	rest := s
	for len(rest) > 0 {
		best, bestIdx := -1, -1
		for i := range toks {
			idx := strings.Index(rest, toks[i].Content)
			if idx < 0 {
				continue
			}
			if bestIdx < 0 || idx < bestIdx || (idx == bestIdx && len(toks[i].Content) > len(toks[best].Content)) {
				best, bestIdx = i, idx
			}
		}
		if best < 0 {
			out = append(out, segment{text: rest})
			break
		}
		tk := &toks[best]
		before := rest[:bestIdx]
		if tk.LStrip {
			before = strings.TrimRightFunc(before, unicode.IsSpace)
		}
		if before != "" {
			out = append(out, segment{text: before})
		}
		out = append(out, segment{tok: tk})
		rest = rest[bestIdx+len(tk.Content):]
		if tk.RStrip {
			rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
		}
	}
	return out
}

// preTokenize implements the GPT-2 split regex
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
func preTokenize(s string) []string {
	var out []string
	rs := []rune(s)
	n := len(rs)
	isL := unicode.IsLetter
	isN := func(r rune) bool { return unicode.IsNumber(r) }
	isS := unicode.IsSpace
	isO := func(r rune) bool { return !isS(r) && !isL(r) && !isN(r) }
	i := 0
	for i < n {
		start := i
		r := rs[i]
		// contractions
		if r == '\'' && i+1 < n {
			if l := contraction(rs[i+1:]); l > 0 {
				i += 1 + l
				out = append(out, string(rs[start:i]))
				continue
			}
		}
		// optional single leading space
		j := i
		if r == ' ' && i+1 < n {
			j = i + 1
		}
		if j < n {
			var pred func(rune) bool
			switch c := rs[j]; {
			case isL(c):
				pred = isL
			case isN(c):
				pred = isN
			case isO(c):
				pred = isO
			}
			if pred != nil {
				k := j
				for k < n && pred(rs[k]) {
					k++
				}
				out = append(out, string(rs[start:k]))
				i = k
				continue
			}
		}
		// whitespace run
		k := i
		for k < n && isS(rs[k]) {
			k++
		}
		switch {
		case k >= n: // \s+ at end of input
		case k-i > 1:
			k-- // \s+(?!\S) leaves the last space for the next word
		}
		out = append(out, string(rs[start:k]))
		i = k
	}
	return out
}

func contraction(rs []rune) int {
	switch {
	case len(rs) >= 2 && rs[0] == 'l' && rs[1] == 'l', len(rs) >= 2 && rs[0] == 'v' && rs[1] == 'e', len(rs) >= 2 && rs[0] == 'r' && rs[1] == 'e':
		return 2
	case len(rs) >= 1 && (rs[0] == 's' || rs[0] == 't' || rs[0] == 'm' || rs[0] == 'd'):
		return 1
	}
	return 0
}

// byteLevel maps raw bytes to the printable unicode alphabet of GPT-2.
var byteToRune = func() [256]rune {
	var m [256]rune
	var bs []int
	for b := '!'; b <= '~'; b++ {
		bs = append(bs, int(b))
	}
	for b := 0xA1; b <= 0xAC; b++ {
		bs = append(bs, b)
	}
	for b := 0xAE; b <= 0xFF; b++ {
		bs = append(bs, b)
	}
	have := map[int]bool{}
	for _, b := range bs {
		m[b] = rune(b)
		have[b] = true
	}
	n := 0
	for b := 0; b < 256; b++ {
		if !have[b] {
			m[b] = rune(256 + n)
			n++
		}
	}
	return m
}()

func byteLevel(s string) []string {
	syms := make([]string, 0, len(s))
	for i := 0; i < len(s); i++ {
		syms = append(syms, string(byteToRune[s[i]]))
	}
	return syms
}

func (t *Tokenizer) bpe(syms []string) []int32 {
	key := strings.Join(syms, "\x00")
	t.cacheMu.RLock()
	if ids, ok := t.cache[key]; ok {
		t.cacheMu.RUnlock()
		return ids
	}
	t.cacheMu.RUnlock()

	parts := append([]string(nil), syms...)
	for len(parts) > 1 {
		bestRank, bestAt := int(^uint(0)>>1), -1
		for i := 0; i+1 < len(parts); i++ {
			if r, ok := t.ranks[pair{parts[i], parts[i+1]}]; ok && r < bestRank {
				bestRank, bestAt = r, i
			}
		}
		if bestAt < 0 {
			break
		}
		merged := parts[bestAt] + parts[bestAt+1]
		next := make([]string, 0, len(parts)-1)
		next = append(next, parts[:bestAt]...)
		next = append(next, merged)
		next = append(next, parts[bestAt+2:]...)
		parts = next
	}
	ids := make([]int32, 0, len(parts))
	for _, p := range parts {
		if id, ok := t.vocab[p]; ok {
			ids = append(ids, id)
			continue
		}
		// No byte-fallback / unk token in this model: emit constituent symbols if known.
		for _, r := range p {
			if id, ok := t.vocab[string(r)]; ok {
				ids = append(ids, id)
			}
		}
	}
	t.cacheMu.Lock()
	if len(t.cache) > 50000 {
		t.cache = map[string][]int32{}
	}
	t.cache[key] = ids
	t.cacheMu.Unlock()
	return ids
}

// MinimalJSON returns a tiny tokenizer.json with a byte-level vocabulary (ids
// 0-255, no merges) and the ModernBERT special tokens at ids 256-260. It is
// meant for tests that need a working tokenizer without real vocabulary files.
func MinimalJSON() []byte {
	vocab := make(map[string]int32, 256)
	for b := 0; b < 256; b++ {
		vocab[string(byteToRune[b])] = int32(b)
	}
	type added struct {
		ID         int32  `json:"id"`
		Content    string `json:"content"`
		Special    bool   `json:"special"`
		Normalized bool   `json:"normalized"`
		LStrip     bool   `json:"lstrip"`
	}
	out, _ := json.Marshal(map[string]any{
		"added_tokens": []added{
			{256, "[CLS]", true, false, false}, {257, "[SEP]", true, false, false},
			{258, "[MASK]", true, false, true}, {259, "[PAD]", true, false, false}, {260, "[UNK]", true, false, false},
		},
		"model": map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{}},
	})
	return out
}
