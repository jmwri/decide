package decide

import (
	"context"
	"errors"
	"testing"
)

// fakeEval answers with canned values keyed by question id.
type fakeEval struct {
	answers map[string]Answer
	calls   int
}

func (f *fakeEval) SystemOne(_ context.Context, _ any, qs Questions) (*Response, error) {
	f.calls++
	r := &Response{Model: ModelID, Answers: map[string]Answer{}}
	for _, q := range qs {
		a, ok := f.answers[q.ID]
		if !ok {
			return nil, errors.New("no canned answer for " + q.ID)
		}
		r.Answers[q.ID] = a
		r.Order = append(r.Order, q.ID)
	}
	return r, nil
}

func TestPresetsAreWellFormed(t *testing.T) {
	for name, qs := range map[string]Questions{
		"triage": TriagePreset(), "email": EmailPreset(nil), "moderation": ModerationPreset(), "security": SecurityPreset(),
	} {
		if len(qs) < 3 {
			t.Errorf("%s: too few questions", name)
		}
		for _, q := range qs {
			switch v := q.Question.(type) {
			case Noul:
				if v.Criteria == nil || v.Criteria.True == "" || v.Criteria.False == "" {
					t.Errorf("%s/%s: noul preset must carry explicit criteria", name, q.ID)
				}
			case Choice:
				if len(v.Criteria) < 2 {
					t.Errorf("%s/%s: choice needs options", name, q.ID)
				}
			case Score:
				if len(v.Criteria) < 2 {
					t.Errorf("%s/%s: score needs levels", name, q.ID)
				}
			}
		}
	}
	custom := EmailPreset(Options{{"a", "A"}, {"b", "B"}})
	if c := custom[0].Question.(Choice); len(c.Criteria) != 2 {
		t.Fatal("custom categories ignored")
	}
}

func TestConfidenceGate(t *testing.T) {
	ev := &fakeEval{answers: map[string]Answer{
		"sure":   &ChoiceAnswer{Choice: "a", Confidence: 0.95},
		"unsure": &ChoiceAnswer{Choice: "a", Confidence: 0.4},
		"yes":    &NoulAnswer{Noul: 0.97}, // distance from 0.5 -> 0.94
		"maybe":  &NoulAnswer{Noul: 0.55}, // -> 0.1
		"score":  &ScoreAnswer{Score: 1, Confidence: 0.85},
	}}
	qs := Questions{{"sure", NewChoice("q", nil)}, {"unsure", NewChoice("q", nil)}, {"yes", NewNoul("q", nil)}, {"maybe", NewNoul("q", nil)}, {"score", NewScore("q", nil)}}
	res, err := ConfidenceGate(context.Background(), ev, "s", qs, 0.8)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"sure", "yes", "score"} {
		if _, ok := res.Automatic[id]; !ok {
			t.Errorf("%s should be automatic", id)
		}
	}
	for _, id := range []string{"unsure", "maybe"} {
		if _, ok := res.Escalate[id]; !ok {
			t.Errorf("%s should escalate", id)
		}
	}
	if _, err := ConfidenceGate(context.Background(), ev, "s", qs, 1.5); err == nil {
		t.Fatal("threshold out of range must fail")
	}
}

func TestRoute(t *testing.T) {
	ev := &fakeEval{answers: map[string]Answer{"route_question": &ChoiceAnswer{Choice: "refund", Confidence: 0.6}}}
	var hit string
	routes := map[string]func(*ChoiceAnswer) error{
		"refund":   func(*ChoiceAnswer) error { hit = "refund"; return nil },
		"escalate": func(*ChoiceAnswer) error { hit = "escalate"; return nil },
	}
	q := NewChoice("q", Options{{"refund", "R"}, {"escalate", "E"}})
	ctx := context.Background()

	if _, err := Route(ctx, ev, "s", q, routes, RouteConfig{}); err != nil || hit != "refund" {
		t.Fatalf("route: hit=%q err=%v", hit, err)
	}
	hit = ""
	def := RouteConfig{MinConfidence: 0.9, Default: func(a *ChoiceAnswer) error { hit = "default"; return nil }}
	if _, err := Route(ctx, ev, "s", q, routes, def); err != nil || hit != "default" {
		t.Fatalf("low confidence should use default: hit=%q err=%v", hit, err)
	}
	boom := errors.New("boom")
	ans, err := Route(ctx, ev, "s", q, map[string]func(*ChoiceAnswer) error{"refund": func(*ChoiceAnswer) error { return boom }}, RouteConfig{})
	if !errors.Is(err, boom) || ans == nil {
		t.Fatalf("handler error must propagate with the answer: %v %v", ans, err)
	}
}

func TestCompositeScore(t *testing.T) {
	ev := &fakeEval{answers: map[string]Answer{
		"severity":  &ScoreAnswer{Score: 3, Legend: map[string]string{"0": "", "1": "", "2": "", "3": ""}}, // 3/3 = 1.0
		"is_threat": &NoulAnswer{Noul: 0.5},
		"kind":      &ChoiceAnswer{Choice: "x"}, // excluded
	}}
	qs := Questions{{"severity", NewScore("q", nil)}, {"is_threat", NewNoul("q", nil)}, {"kind", NewChoice("q", nil)}}
	res, err := CompositeScore(context.Background(), ev, "s", qs, map[string]float64{"severity": 2, "is_threat": 3})
	if err != nil {
		t.Fatal(err)
	}
	// (1.0*2 + 0.5*3) / 5 = 0.7
	if res.Score != 0.7 {
		t.Fatalf("score = %v", res.Score)
	}
	if _, ok := res.Breakdown["kind"]; ok {
		t.Fatal("choice answers must be excluded")
	}
	if res.Breakdown["severity"].Normalized != 1 || res.Breakdown["is_threat"].Weight != 3 {
		t.Fatalf("breakdown: %+v", res.Breakdown)
	}
}

func TestTwoStageChoice(t *testing.T) {
	ev := &fakeEval{answers: map[string]Answer{
		"category": &ChoiceAnswer{Choice: "database", Confidence: 0.9},
		"option":   &ChoiceAnswer{Choice: "postgres", Confidence: 0.5},
	}}
	tax := Taxonomy{
		{"cloud", Options{{"aws", "Amazon"}}},
		{"database", Options{{"postgres", "PostgreSQL"}, {"redis", "Redis"}}},
	}
	res, err := TwoStageChoice(context.Background(), ev, "pg lag", tax, TwoStageConfig{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Category != "database" || res.Choice != "postgres" || res.CombinedConfidence != 0.45 || ev.calls != 2 {
		t.Fatalf("%+v calls=%d", res, ev.calls)
	}
}

func TestClientHelpers(t *testing.T) {
	ev := &fakeEval{answers: map[string]Answer{
		"decision": &ChoiceAnswer{Choice: "a", Confidence: 1},
		"judgment": &NoulAnswer{Noul: 0.9},
		"rating":   &ScoreAnswer{Score: 1.5},
	}}
	c := NewClient(ev)
	ctx := context.Background()
	if a, err := c.Decide(ctx, "s", Options{{"a", ""}, {"b", ""}}, ""); err != nil || a.Choice != "a" {
		t.Fatal(a, err)
	}
	if p, err := c.Judge(ctx, "s", "q", nil); err != nil || p != 0.9 {
		t.Fatal(p, err)
	}
	if a, err := c.Rate(ctx, "s", Levels("l", "h"), ""); err != nil || a.Score != 1.5 {
		t.Fatal(a, err)
	}
}
