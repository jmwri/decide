// Package train fine-tunes the option-scoring model: it packs examples into
// token sequences, runs mini-batch AdamW, evaluates and checkpoints.
package train

import (
	"fmt"
	"math/rand"
	"strings"

	"github.com/jmwri/decide/internal/bundle"
	"github.com/jmwri/decide/internal/data"
	"github.com/jmwri/decide/internal/nn"
	"github.com/jmwri/decide/internal/tokenizer"
)

// PackText lays out a premise and options; see bundle.PackText.
func PackText(state, instructions string, options []string) string {
	return bundle.PackText(state, instructions, options)
}

// Item is a tokenised training or evaluation example.
type Item struct {
	Seq     nn.Sequence
	Target  []float32 // distribution over the options
	Gold    int
	Task    string
	Kind    string
	Ordinal bool
}

// Realize chooses the options an example shows this time (a random subset,
// always containing the gold option, when the example carries a large pool)
// and returns the resulting example.
func Realize(e *data.Example, rng *rand.Rand, kMax int) data.Example {
	if !e.Subsample || len(e.Options) <= kMax {
		return *e
	}
	k := 2 + rng.Intn(kMax-1)
	idx := make([]int, 0, k)
	idx = append(idx, e.Gold)
	for _, j := range rng.Perm(len(e.Options)) {
		if len(idx) == k {
			break
		}
		if j != e.Gold {
			idx = append(idx, j)
		}
	}
	rng.Shuffle(len(idx), func(a, b int) { idx[a], idx[b] = idx[b], idx[a] })
	out := *e
	out.Options = make([]string, len(idx))
	out.Subsample = false
	for i, j := range idx {
		out.Options[i] = e.Options[j]
		if j == e.Gold {
			out.Gold = i
		}
	}
	out.Soft = nil
	return out
}

// Tokenize packs and tokenises an example. Over-long states are trimmed until
// the sequence fits maxLen; if it still does not fit, ok is false.
func Tokenize(tok *tokenizer.Tokenizer, e *data.Example, maxLen int, independent bool) (Item, bool) {
	state := e.State
	var ids []int32
	for attempt := 0; attempt < 6; attempt++ {
		ids = tok.Encode(PackText(state, e.Instructions, e.Options), true)
		if len(ids) <= maxLen {
			break
		}
		state = shorten(state, 0.75*float64(maxLen)/float64(len(ids)))
	}
	if len(ids) > maxLen {
		return Item{}, false
	}
	seq := nn.BuildSequence(ids, tok.MaskID(), independent)
	if len(seq.MaskPos) != len(e.Options) {
		return Item{}, false // an option text contained the reserved [MASK] token
	}
	target := make([]float32, len(e.Options))
	if e.Soft != nil && len(e.Soft) == len(e.Options) {
		copy(target, e.Soft)
	} else {
		target[e.Gold] = 1
	}
	return Item{Seq: seq, Target: target, Gold: e.Gold, Task: e.Task, Kind: e.Kind, Ordinal: e.Ordinal}, true
}

func shorten(s string, frac float64) string {
	n := int(float64(len(s)) * frac)
	if n >= len(s) {
		n = len(s) - 1
	}
	if n < 20 {
		return s[:min(len(s), 20)]
	}
	s = s[:n]
	if i := strings.LastIndexByte(s, ' '); i > n/2 {
		s = s[:i]
	}
	return s
}

// Validate checks a tokenised item is usable.
func (it *Item) Validate() error {
	if len(it.Seq.MaskPos) != len(it.Target) {
		return fmt.Errorf("item has %d options but %d targets", len(it.Seq.MaskPos), len(it.Target))
	}
	return nil
}
