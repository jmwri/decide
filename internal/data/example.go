// Package data builds the training corpus for the Decide model from public
// datasets: it downloads them from the Hugging Face hub, converts each task to
// "premise + instructions + options" examples, and writes JSONL splits.
package data

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
)

// Kinds of question.
const (
	KindChoice = "choice"
	KindNoul   = "noul"
	KindScore  = "score"
)

// Example is one decision problem: pick the right one of Options for State
// under Instructions.
type Example struct {
	Task         string   `json:"task"`
	Kind         string   `json:"kind"`
	Instructions string   `json:"instructions"`
	State        string   `json:"state"`
	Options      []string `json:"options"`
	// Gold indexes Options. For noul examples Options is [yes-text, no-text].
	Gold int `json:"gold"`
	// Soft optionally replaces the one-hot target (ordinal tasks).
	Soft []float32 `json:"soft,omitempty"`
	// Subsample marks Options as a pool from which a subset that includes Gold
	// may be drawn at training time (large label spaces).
	Subsample bool `json:"subsample,omitempty"`
	// Ordinal marks Options as an ordered scale (Score).
	Ordinal bool `json:"ordinal,omitempty"`
}

// Validate reports structural problems.
func (e *Example) Validate() error {
	switch {
	case e.Instructions == "" && e.State == "":
		return fmt.Errorf("empty example")
	case len(e.Options) < 2:
		return fmt.Errorf("%s: fewer than two options", e.Task)
	case e.Gold < 0 || e.Gold >= len(e.Options):
		return fmt.Errorf("%s: gold %d out of range", e.Task, e.Gold)
	case e.Soft != nil && len(e.Soft) != len(e.Options):
		return fmt.Errorf("%s: soft target has %d entries for %d options", e.Task, len(e.Soft), len(e.Options))
	}
	for _, o := range e.Options {
		if o == "" {
			return fmt.Errorf("%s: empty option", e.Task)
		}
	}
	return nil
}

// WriteJSONL writes examples one per line.
func WriteJSONL(path string, exs []Example) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	w := bufio.NewWriterSize(f, 1<<20)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	for i := range exs {
		if err := enc.Encode(&exs[i]); err != nil {
			f.Close()
			return err
		}
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// ReadJSONL loads a JSONL file written by WriteJSONL.
func ReadJSONL(path string) ([]Example, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []Example
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for line := 1; sc.Scan(); line++ {
		var e Example
		if err := json.Unmarshal(sc.Bytes(), &e); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, line, err)
		}
		out = append(out, e)
	}
	return out, sc.Err()
}
