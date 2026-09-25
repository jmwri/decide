package decide

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestQuestionsRoundTripPreservesOrder(t *testing.T) {
	src := `{"z":{"type":"choice","instructions":"pick","criteria":{"b":"B desc","a":null,"c":"C"}},
	         "a":{"type":"noul","instructions":"yes?","criteria":{"true":"T"}},
	         "m":{"type":"score","instructions":"rate","criteria":["low",{"what":"high","examples":["x","y"]}]}}`
	var qs Questions
	if err := json.Unmarshal([]byte(src), &qs); err != nil {
		t.Fatal(err)
	}
	ids := []string{}
	for _, q := range qs {
		ids = append(ids, q.ID)
	}
	if !reflect.DeepEqual(ids, []string{"z", "a", "m"}) {
		t.Fatalf("question order lost: %v", ids)
	}
	c := qs[0].Question.(Choice)
	if got := []string{c.Criteria[0].Key, c.Criteria[1].Key, c.Criteria[2].Key}; !reflect.DeepEqual(got, []string{"b", "a", "c"}) {
		t.Fatalf("option order lost: %v", got)
	}
	if c.Criteria[1].Description != "" {
		t.Fatal("null description should be empty")
	}
	n := qs[1].Question.(Noul)
	if n.Criteria == nil || n.Criteria.True != "T" || n.Criteria.False != "" {
		t.Fatalf("noul criteria: %+v", n.Criteria)
	}
	s := qs[2].Question.(Score)
	if s.Criteria[1].Text() != "high Examples: x, y" {
		t.Fatalf("level text: %q", s.Criteria[1].Text())
	}

	out, err := json.Marshal(qs)
	if err != nil {
		t.Fatal(err)
	}
	var again Questions
	if err := json.Unmarshal(out, &again); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(qs, again) {
		t.Fatalf("round trip changed questions:\n%s", out)
	}
}

func TestQuestionDefaultsAndErrors(t *testing.T) {
	q, err := ParseQuestion([]byte(`{"instructions":"x","criteria":{"a":"b"}}`))
	if err != nil || q.Type() != "choice" {
		t.Fatalf("missing type should mean choice: %v %v", q, err)
	}
	for name, body := range map[string]string{
		"unknown type":   `{"type":"bogus","instructions":"x"}`,
		"no instruction": `{"type":"noul"}`,
		"no criteria":    `{"type":"choice","instructions":"x"}`,
		"bad option":     `{"type":"choice","instructions":"x","criteria":{"a":3}}`,
		"score not list": `{"type":"score","instructions":"x","criteria":{"a":"b"}}`,
	} {
		if _, err := ParseQuestion([]byte(body)); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
}

func TestStructuredInstructionsAreStringified(t *testing.T) {
	q, err := ParseQuestion([]byte(`{"type":"noul","instructions":{"b":1,"a":[true,null,"é"]}}`))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := q.(Noul).Instructions, "{\"a\": [true, null, \"\\u00e9\"], \"b\": 1}"; got != want {
		t.Fatalf("got %s want %s", got, want)
	}
	q, _ = ParseQuestion([]byte(`{"type":"noul","instructions":[{"b":1,"a":2}]}`))
	if got, want := q.(Noul).Instructions, `[{"b": 1, "a": 2}]`; got != want {
		t.Fatalf("got %s want %s", got, want)
	}
}

func TestNoulLegacyCriteriaFold(t *testing.T) {
	q, err := ParseQuestion([]byte(`{"type":"noul","instructions":"x","pos_criteria":"P","neg_criteria":"N"}`))
	if err != nil {
		t.Fatal(err)
	}
	if c := q.(Noul).Criteria; c == nil || c.True != "P" || c.False != "N" {
		t.Fatalf("legacy keys not folded: %+v", c)
	}
	_, err = ParseQuestion([]byte(`{"type":"noul","instructions":"x","pos_criteria":"P","criteria":{"true":"T"}}`))
	if err == nil || !strings.Contains(err.Error(), "pos_criteria") {
		t.Fatalf("expected conflict error, got %v", err)
	}
}

func TestChoicesRejectsDuplicates(t *testing.T) {
	if _, err := Choices("a", "b", "a"); err == nil {
		t.Fatal("expected duplicate error")
	}
}

func TestParseRequest(t *testing.T) {
	req, err := ParseRequest([]byte(`{"state":{"b":1,"a":"x"},"questions":{"q":{"type":"noul","instructions":"i"}}}`))
	if err != nil {
		t.Fatal(err)
	}
	if req.Model != "decide-latest" {
		t.Fatalf("model default: %q", req.Model)
	}
	if got := formatState(req.State); got != "b: 1\na: x" {
		t.Fatalf("state must keep key order, got %q", got)
	}
	for name, body := range map[string]string{
		"no state":        `{"questions":{"q":{"type":"noul","instructions":"i"}}}`,
		"no questions":    `{"state":"s"}`,
		"empty questions": `{"state":"s","questions":{}}`,
		"not an object":   `[]`,
		"bad json":        `{`,
	} {
		if _, err := ParseRequest([]byte(body)); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
}

func TestResponseJSON(t *testing.T) {
	r := &Response{
		Model: ModelID,
		Answers: map[string]Answer{
			"b": &NoulAnswer{Noul: 0.25},
			"a": &ChoiceAnswer{Choice: "x", Probabilities: map[string]float64{"x": 0.9, "y": 0.1}, Confidence: 0.8},
			"c": &ScoreAnswer{Score: 1.5, Confidence: 0.5, Legend: map[string]string{"0": "lo", "1": "hi"}, Probabilities: map[string]float64{"0": 0.5, "1": 0.5}},
		},
		Order: []string{"b", "a", "c"},
		Usage: Usage{InputTokens: 3, OutputTokens: 3},
	}
	raw, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(raw), `{"model":"`+ModelID+`","answers":{"b":{"type":"noul","noul":0.25},"a":{"type":"choice"`) {
		t.Fatalf("unexpected encoding: %s", raw)
	}
	var back Response
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(&back, r) {
		t.Fatalf("round trip mismatch:\n got %+v\nwant %+v", back, *r)
	}
	if a, ok := back.Choice("a"); !ok || a.Choice != "x" {
		t.Fatal("Choice accessor")
	}
	// TypeSafe omits "type" on some answers; infer from shape.
	var inferred Response
	if err := json.Unmarshal([]byte(`{"model":"m","answers":{"q":{"noul":0.5}},"usage":{}}`), &inferred); err != nil {
		t.Fatal(err)
	}
	if _, ok := inferred.Noul("q"); !ok {
		t.Fatal("answer type not inferred")
	}
}

func TestFormatState(t *testing.T) {
	type ticket struct {
		ID   string  `json:"id"`
		Tier string  `json:"tier"`
		Amt  float64 `json:"amount"`
	}
	cases := []struct {
		in   any
		want string
	}{
		{"plain", "plain"},
		{Object{{"k", "v"}, {"n", json.Number("3")}, {"f", json.Number("2.50")}, {"t", true}, {"z", nil}}, "k: v\nn: 3\nf: 2.5\nt: True\nz: None"},
		{Object{{"list", []any{"a", json.Number("1"), false}}, {"o", Object{{"x", "it's"}}}}, "list: ['a', 1, False]\no: {'x': \"it's\"}"},
		{map[string]any{"b": 2, "a": "x"}, "a: x\nb: 2"},
		{ticket{"INC-1", "ent", 10.5}, "id: INC-1\ntier: ent\namount: 10.5"},
		{42, "42"},
		{[]string{"a", "b"}, "['a', 'b']"},
		{nil, "None"},
	}
	for _, c := range cases {
		if got := formatState(c.in); got != c.want {
			t.Errorf("formatState(%#v) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestPyFloat(t *testing.T) {
	cases := map[float64]string{
		1: "1.0", 0.5: "0.5", 100: "100.0", 1e16: "1e+16", 1e15: "1000000000000000.0", 0.0001: "0.0001",
		0.00001: "1e-05", 123456.789: "123456.789", -2.5: "-2.5", 1.5e-7: "1.5e-07", 0: "0.0",
	}
	for f, want := range cases {
		if got := pyFloat(f); got != want {
			t.Errorf("pyFloat(%v) = %q, want %q", f, got, want)
		}
	}
}

func TestMarginConfidence(t *testing.T) {
	if got := marginConfidence([]float64{1}); got != 1 {
		t.Fatal(got)
	}
	if got := marginConfidence([]float64{0.5, 0.5}); got != 0 {
		t.Fatal(got)
	}
	if got := marginConfidence([]float64{0.9, 0.05, 0.05}); got != 0.85 {
		t.Fatal(got)
	}
}

func TestLevelText(t *testing.T) {
	if got := (Level{What: "  hi  "}).Text(); got != "hi" {
		t.Fatal(got)
	}
}
