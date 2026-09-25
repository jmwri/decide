package data

import (
	"math/rand"
	"strings"
	"unicode"
	"unicode/utf8"
)

func pick[T any](r *rand.Rand, xs []T) T { return xs[r.Intn(len(xs))] }

// oneLine collapses all whitespace runs to single spaces.
func oneLine(s string) string { return strings.Join(strings.Fields(s), " ") }

// clip shortens s to at most n bytes at a word boundary.
func clip(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	s = s[:n]
	for !utf8.ValidString(s) {
		s = s[:len(s)-1]
	}
	if i := strings.LastIndexFunc(s, unicode.IsSpace); i > n/2 {
		s = s[:i]
	}
	return strings.TrimSpace(s) + " ..."
}

func capitalize(s string) string {
	if s == "" {
		return s
	}
	r, n := utf8.DecodeRuneInString(s)
	return string(unicode.ToUpper(r)) + s[n:]
}

func humanize(name string) string {
	return strings.TrimSpace(strings.NewReplacer("_", " ", "-", " ", "/", " / ").Replace(name))
}

// fields lays out labelled fields the way a structured state renders, either
// one per line ("key: value") or, less often, run together on one line.
func fields(r *rand.Rand, kv ...string) string {
	var b strings.Builder
	multi := r.Float64() < 0.7
	for i := 0; i+1 < len(kv); i += 2 {
		if i > 0 {
			if multi {
				b.WriteByte('\n')
			} else {
				b.WriteByte(' ')
			}
		}
		k := kv[i]
		if !multi {
			k = capitalize(k)
		}
		b.WriteString(k + ": " + oneLine(kv[i+1]))
	}
	return b.String()
}

// yesNoPairs are the (true, false) option texts used for yes/no questions.
// The first pair matches the model's default Noul criteria and is the most common.
var yesNoPairs = [][2]string{
	{"Yes, condition holds true.", "No, condition is false."},
	{"Yes, condition holds true.", "No, condition is false."},
	{"Yes, condition holds true.", "No, condition is false."},
	{"Yes", "No"},
	{"True", "False"},
	{"Yes, this is correct.", "No, this is not correct."},
}

// noulExample builds a yes/no example. desc, when set, replaces the generic
// option texts with explicit criteria ("true": ..., "false": ...).
func noulExample(r *rand.Rand, task, state, question string, truth bool, trueDesc, falseDesc string) Example {
	yes, no := trueDesc, falseDesc
	if yes == "" || no == "" {
		p := pick(r, yesNoPairs)
		yes, no = p[0], p[1]
	}
	gold := 0
	if !truth {
		gold = 1
	}
	return Example{Task: task, Kind: KindNoul, Instructions: question, State: state, Options: []string{yes, no}, Gold: gold}
}

// Label is one class of a classification task. Verb holds alternative
// option texts; an example uses the same style index for every label.
type Label struct {
	Name string
	Verb []string
}

// ClassTask turns (state, gold label) pairs into choice examples, and
// sometimes into yes/no "is the label X?" verification questions.
type ClassTask struct {
	Name         string
	Labels       []Label
	Instructions []string // choice-question phrasings
	YesNo        []string // verification phrasings containing {label}
	NoulProb     float64  // probability of emitting a verification question (default 0.2)
	// Explicit gives noul verification questions explicit true/false criteria some of the time.
	Explicit bool
}

func (c *ClassTask) styles() int {
	n := 1 << 30
	for _, l := range c.Labels {
		n = min(n, len(l.Verb))
	}
	return n
}

func (c *ClassTask) optionText(l int, style int) string {
	return c.Labels[l].Verb[style%len(c.Labels[l].Verb)]
}

// Make builds one example for a state whose gold label is gold.
func (c *ClassTask) Make(r *rand.Rand, state string, gold int) Example {
	p := c.NoulProb
	if p == 0 {
		p = 0.2
	}
	style := r.Intn(c.styles())
	if len(c.YesNo) > 0 && r.Float64() < p {
		l := gold
		truth := true
		if r.Float64() < 0.5 && len(c.Labels) > 1 {
			for l == gold {
				l = r.Intn(len(c.Labels))
			}
			truth = false
		}
		label := c.optionText(l, style)
		q := strings.ReplaceAll(pick(r, c.YesNo), "{label}", label)
		var yes, no string
		if c.Explicit && r.Float64() < 0.4 {
			yes, no = "The text matches: "+label, "The text does not match: "+label
		}
		return noulExample(r, c.Name, state, q, truth, yes, no)
	}
	opts := make([]string, len(c.Labels))
	for i := range c.Labels {
		opts[i] = c.optionText(i, style)
	}
	return Example{
		Task: c.Name, Kind: KindChoice, Instructions: pick(r, c.Instructions), State: state,
		Options: opts, Gold: gold, Subsample: len(opts) > 8,
	}
}

// intentLabels builds labels from raw class names with a few description styles.
func intentLabels(names []string, what string) []Label {
	out := make([]Label, len(names))
	for i, n := range names {
		h := humanize(n)
		out[i] = Label{Name: n, Verb: []string{
			h,
			h,
			capitalize(h),
			what + " " + h,
			"Intent: " + h,
		}}
	}
	return out
}

// ScaleTask builds ordinal (Score) examples over an ordered scale.
type ScaleTask struct {
	Name         string
	Levels       [][]string // per level, alternative texts (same style index across levels)
	Instructions []string
}

// Make builds an ordinal example; the target is smoothed onto neighbouring levels.
func (s *ScaleTask) Make(r *rand.Rand, state string, level int) Example {
	style := r.Intn(len(s.Levels[0]))
	opts := make([]string, len(s.Levels))
	for i, l := range s.Levels {
		opts[i] = l[style%len(l)]
	}
	soft := make([]float32, len(opts))
	const eps = 0.12
	switch {
	case level == 0:
		soft[0], soft[1] = 1-eps, eps
	case level == len(opts)-1:
		soft[level], soft[level-1] = 1-eps, eps
	default:
		soft[level], soft[level-1], soft[level+1] = 1-eps, eps/2, eps/2
	}
	return Example{
		Task: s.Name, Kind: KindScore, Instructions: pick(r, s.Instructions), State: state,
		Options: opts, Gold: level, Soft: soft, Ordinal: true,
	}
}

// mcExample builds a multiple-choice example whose options are real answer texts.
func mcExample(r *rand.Rand, task, state string, instr []string, options []string, gold int) (Example, bool) {
	if gold < 0 || gold >= len(options) || len(options) < 2 {
		return Example{}, false
	}
	opts := make([]string, len(options))
	for i, o := range options {
		o = oneLine(o)
		if o == "" {
			return Example{}, false
		}
		opts[i] = clip(o, 300)
	}
	return Example{Task: task, Kind: KindChoice, Instructions: pick(r, instr), State: state, Options: opts, Gold: gold}, true
}
