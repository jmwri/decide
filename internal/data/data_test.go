package data

import (
	"math/rand"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestHelpers(t *testing.T) {
	if got := oneLine("  a \n\t b   c "); got != "a b c" {
		t.Fatal(got)
	}
	if got := clip("hello world this is long", 12); !strings.HasSuffix(got, "...") || len(got) > 20 {
		t.Fatal(got)
	}
	if got := clip("short", 100); got != "short" {
		t.Fatal(got)
	}
	// clipping must not split a multi-byte rune
	if got := clip(strings.Repeat("é", 50), 11); !strings.HasSuffix(got, "...") {
		t.Fatal(got)
	}
	if capitalize("émile") != "Émile" || capitalize("") != "" {
		t.Fatal("capitalize")
	}
	if humanize("card_arrival") != "card arrival" {
		t.Fatal(humanize("card_arrival"))
	}
}

func TestFieldsStyles(t *testing.T) {
	multi, single := 0, 0
	for i := 0; i < 200; i++ {
		s := fields(rand.New(rand.NewSource(int64(i))), "premise", "a b", "hypothesis", "c\nd")
		switch {
		case s == "premise: a b\nhypothesis: c d":
			multi++
		case s == "Premise: a b Hypothesis: c d":
			single++
		default:
			t.Fatalf("unexpected layout %q", s)
		}
	}
	if multi < 100 || single < 20 {
		t.Fatalf("style mix off: %d multi, %d single", multi, single)
	}
}

func TestClassTaskMake(t *testing.T) {
	ct := &ClassTask{
		Name:         "t",
		Labels:       []Label{{Name: "a", Verb: []string{"alpha", "the A"}}, {Name: "b", Verb: []string{"beta", "the B"}}, {Name: "c", Verb: []string{"gamma", "the C"}}},
		Instructions: []string{"Which?"},
		YesNo:        []string{"Is it {label}?"},
		NoulProb:     0.5,
	}
	r := rand.New(rand.NewSource(1))
	var choices, nouls, yes int
	for i := 0; i < 400; i++ {
		e := ct.Make(r, "state", 1)
		if err := e.Validate(); err != nil {
			t.Fatal(err)
		}
		switch e.Kind {
		case KindChoice:
			choices++
			if len(e.Options) != 3 || e.Gold != 1 {
				t.Fatalf("%+v", e)
			}
			// one style across all options of an example
			if (e.Options[0] == "alpha") != (e.Options[1] == "beta") {
				t.Fatalf("mixed verbalisation styles: %v", e.Options)
			}
		case KindNoul:
			nouls++
			if e.Gold == 0 {
				yes++
				if !strings.Contains(e.Instructions, "beta") && !strings.Contains(e.Instructions, "the B") {
					t.Fatalf("a true question must name the gold label: %q", e.Instructions)
				}
			} else if strings.Contains(e.Instructions, "beta") || strings.Contains(e.Instructions, "the B") {
				t.Fatalf("a false question must not name the gold label: %q", e.Instructions)
			}
		}
	}
	if choices < 120 || nouls < 120 || yes < 60 || yes > nouls-60 {
		t.Fatalf("mix: %d choice, %d noul (%d yes)", choices, nouls, yes)
	}
}

func TestScaleTaskSoftTargets(t *testing.T) {
	st := &ScaleTask{Name: "s", Levels: [][]string{{"l0"}, {"l1"}, {"l2"}}, Instructions: []string{"rate"}}
	r := rand.New(rand.NewSource(1))
	for level := 0; level < 3; level++ {
		e := st.Make(r, "x", level)
		if err := e.Validate(); err != nil {
			t.Fatal(err)
		}
		var sum float32
		for _, p := range e.Soft {
			sum += p
		}
		if sum < 0.999 || sum > 1.001 || e.Soft[level] < 0.85 || !e.Ordinal {
			t.Fatalf("level %d: %+v", level, e.Soft)
		}
	}
}

func TestNLIExamples(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	kinds := map[string]int{}
	for i := 0; i < 300; i++ {
		exs := nliExample(r, "mnli", "A man sleeps.", "A person rests.", i%3)
		if len(exs) != 1 {
			t.Fatal("expected one example")
		}
		e := exs[0]
		if err := e.Validate(); err != nil {
			t.Fatal(err)
		}
		kinds[e.Kind]++
		if !strings.Contains(strings.ToLower(e.State), "premise") || !strings.Contains(strings.ToLower(e.State), "hypothesis") {
			t.Fatalf("state should carry both fields: %q", e.State)
		}
	}
	if kinds[KindChoice] < 120 || kinds[KindNoul] < 60 {
		t.Fatalf("kinds %v", kinds)
	}
	if nliExample(r, "x", "", "h", 0) != nil || nliExample(r, "x", "p", "h", -1) != nil {
		t.Fatal("invalid rows must be dropped")
	}
}

func TestSubsampleFlag(t *testing.T) {
	opts := make([]Label, 20)
	for i := range opts {
		opts[i] = Label{Name: "x", Verb: []string{"opt " + string(rune('a'+i))}}
	}
	ct := &ClassTask{Name: "big", Labels: opts, Instructions: []string{"?"}}
	if e := ct.Make(rand.New(rand.NewSource(1)), "s", 3); !e.Subsample || len(e.Options) != 20 {
		t.Fatalf("large label spaces must be marked for subsampling: %+v", e)
	}
}

func TestExampleValidate(t *testing.T) {
	good := Example{Task: "t", Kind: KindChoice, Instructions: "i", State: "s", Options: []string{"a", "b"}, Gold: 1}
	if err := good.Validate(); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(e *Example){
		"one option":    func(e *Example) { e.Options = []string{"a"} },
		"gold range":    func(e *Example) { e.Gold = 2 },
		"empty option":  func(e *Example) { e.Options = []string{"a", ""} },
		"soft mismatch": func(e *Example) { e.Soft = []float32{1} },
	} {
		e := good
		mutate(&e)
		if e.Validate() == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestJSONLRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x.jsonl")
	in := []Example{
		{Task: "a", Kind: KindChoice, Instructions: "i <b>", State: "s & é", Options: []string{"x", "y"}, Gold: 0, Subsample: true},
		{Task: "b", Kind: KindScore, State: "s", Options: []string{"l0", "l1"}, Gold: 1, Soft: []float32{0.1, 0.9}, Ordinal: true},
	}
	if err := WriteJSONL(path, in); err != nil {
		t.Fatal(err)
	}
	out, err := ReadJSONL(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(in, out) {
		t.Fatalf("round trip:\n%+v\n%+v", in, out)
	}
}

func TestRowAccessors(t *testing.T) {
	r := Row{"s": "x", "i": int32(4), "f": float32(2.5), "b": true, "l": []any{"a", "b"}}
	if r.Str("s") != "x" || r.Int("i") != 4 || r.Float("f") != 2.5 || !r.Bool("b") || !reflect.DeepEqual(r.Strings("l"), []string{"a", "b"}) || r.Str("missing") != "" {
		t.Fatal("accessors")
	}
}

func TestParquetMetadataNames(t *testing.T) {
	var m Meta
	m.Names = map[string][]string{}
	mergeNames(&m, `{"info":{"features":{"label":{"names":["neg","pos"],"_type":"ClassLabel"},"text":{"dtype":"string"}}}}`)
	if !reflect.DeepEqual(m.Names["label"], []string{"neg", "pos"}) || m.Names["text"] != nil {
		t.Fatalf("%v", m.Names)
	}
	mergeNames(&m, "not json") // must not panic
}

func TestCivilConverterBalancesClasses(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	kinds := map[string]int{}
	yes, no := 0, 0
	for i := 0; i < 4000; i++ {
		row := Row{"text": "some comment", "toxicity": float32(0.02)}
		if i%10 == 0 {
			row = Row{"text": "awful comment", "toxicity": float32(0.9), "insult": float32(0.8)}
		}
		for _, e := range civilConverter(r, row) {
			if err := e.Validate(); err != nil {
				t.Fatal(err)
			}
			kinds[e.Kind]++
			if e.Kind == KindNoul {
				if e.Gold == 0 {
					yes++
				} else {
					no++
				}
			}
		}
	}
	if yes == 0 || no == 0 || float64(yes)/float64(yes+no) < 0.3 {
		t.Fatalf("toxic yes/no not balanced: %d yes, %d no", yes, no)
	}
	if kinds[KindScore] == 0 || kinds[KindChoice] == 0 {
		t.Fatalf("kinds: %v", kinds)
	}
}

func TestRegistrySanity(t *testing.T) {
	seen := map[string]bool{}
	var holdouts int
	for _, tk := range Tasks() {
		if tk.Name == "" || seen[tk.Name] {
			t.Fatalf("bad or duplicate task name %q", tk.Name)
		}
		seen[tk.Name] = true
		if tk.Prepare == nil || tk.Val.Repo == "" || tk.License == "" {
			t.Errorf("%s: incomplete registration", tk.Name)
		}
		if tk.Holdout {
			holdouts++
			if tk.Train.Repo != "" || tk.TrainCap != 0 {
				t.Errorf("%s: a holdout must not carry a train split", tk.Name)
			}
		} else if tk.Train.Repo == "" || tk.TrainCap == 0 {
			t.Errorf("%s: missing train split", tk.Name)
		}
	}
	if holdouts < 3 {
		t.Fatalf("only %d held-out tasks", holdouts)
	}
}
