package decide

import (
	"context"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/jmwri/decide/internal/bundle"
	"github.com/jmwri/decide/internal/nn"
	"github.com/jmwri/decide/internal/tokenizer"
)

const tinyConfig = `{"hidden_size":32,"intermediate_size":48,"num_attention_heads":4,"num_hidden_layers":4,
"vocab_size":300,"norm_eps":1e-5,"local_attention":8,"global_attn_every_n_layers":3,
"global_rope_theta":160000.0,"local_rope_theta":10000.0}`

// tinyBundle writes a random-weight bundle (real architecture, tiny sizes,
// byte-level tokenizer) so the whole inference path can be tested hermetically.
func tinyBundle(t *testing.T, cal *bundle.Calibration) string {
	t.Helper()
	src, dir := t.TempDir(), t.TempDir()
	cfgPath, tokPath := filepath.Join(src, "config.json"), filepath.Join(src, "tokenizer.json")
	if err := os.WriteFile(cfgPath, []byte(tinyConfig), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(tokPath, tokenizer.MinimalJSON(), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := nn.LoadConfig(cfgPath)
	if err != nil {
		t.Fatal(err)
	}
	m := nn.New(cfg)
	m.InitRandom(rand.New(rand.NewSource(7)), 0.3)
	meta := bundle.Meta{Name: "decide", ModelID: "decide-test-0", IndependentOptions: true, MaxTokens: 8192, Calibration: cal}
	if err := bundle.Save(dir, m, cfgPath, tokPath, meta); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestLocalEndToEnd(t *testing.T) {
	cal := &bundle.Calibration{Temperature: map[string]float64{"choice": 2, "noul": 1.5, "score": 3}, Default: 1}
	l := NewLocal(tinyBundle(t, cal))
	ctx := context.Background()

	resp, err := l.SystemOne(ctx, Object{{Key: "ticket", Value: "Payment failed twice"}}, Questions{
		{ID: "c", Question: NewChoice("Which team?", Options{{Key: "billing", Description: "Invoices and payments"}, {Key: "tech", Description: "Bugs"}, {Key: "sales"}})},
		{ID: "n", Question: NewNoul("Is it urgent?", nil)},
		{ID: "n2", Question: NewNoul("Is it urgent?", &NoulCriteria{True: "Urgent", False: "Routine"})},
		{ID: "s", Question: NewScore("How severe?", Levels("low", "medium", "high"))},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Model != "decide-test-0" || len(resp.Order) != 4 {
		t.Fatalf("%+v", resp)
	}
	c, _ := resp.Choice("c")
	var sum float64
	for _, p := range c.Probabilities {
		sum += p
	}
	if math.Abs(sum-1) > 5e-4 || c.Confidence < 0 || c.Confidence > 1 || c.Probabilities[c.Choice] < 1.0/3-1e-3 {
		t.Fatalf("choice answer: %+v (sum %v)", c, sum)
	}
	if n, _ := resp.Noul("n"); n.Noul < 0 || n.Noul > 1 {
		t.Fatalf("noul: %+v", n)
	}
	s, _ := resp.Score("s")
	if s.Score < 0 || s.Score > 2 || len(s.Legend) != 3 {
		t.Fatalf("score: %+v", s)
	}

	// Determinism.
	again, err := l.SystemOne(ctx, Object{{Key: "ticket", Value: "Payment failed twice"}}, Questions{{ID: "c", Question: NewChoice("Which team?", Options{{Key: "billing", Description: "Invoices and payments"}, {Key: "tech", Description: "Bugs"}, {Key: "sales"}})}})
	if err != nil {
		t.Fatal(err)
	}
	if a, _ := again.Choice("c"); a.Choice != c.Choice || a.Probabilities["billing"] != c.Probabilities["billing"] {
		t.Fatalf("not deterministic: %+v vs %+v", a, c)
	}
}

// Permuting the options must permute the probabilities and nothing else.
func TestLocalOrderInvariant(t *testing.T) {
	l := NewLocal(tinyBundle(t, nil))
	opts := Options{{Key: "a", Description: "first option text"}, {Key: "b", Description: "second"}, {Key: "c", Description: "the third one"}, {Key: "d", Description: "4th"}}
	perm := Options{opts[2], opts[0], opts[3], opts[1]}
	ask := func(o Options) *ChoiceAnswer {
		r, err := l.SystemOne(context.Background(), "some state text", Questions{{ID: "q", Question: NewChoice("pick", o)}})
		if err != nil {
			t.Fatal(err)
		}
		a, _ := r.Choice("q")
		return a
	}
	x, y := ask(opts), ask(perm)
	if x.Choice != y.Choice {
		t.Fatalf("winner changed with option order: %s vs %s", x.Choice, y.Choice)
	}
	for k, p := range x.Probabilities {
		if math.Abs(p-y.Probabilities[k]) > 2e-4 {
			t.Fatalf("option %s: %v vs %v", k, p, y.Probabilities[k])
		}
	}
}

func TestLocalErrors(t *testing.T) {
	l := NewLocal(tinyBundle(t, nil))
	ctx := context.Background()
	if _, err := l.SystemOne(ctx, "text with [MASK] injected", Questions{{ID: "q", Question: NewChoice("i", Options{{Key: "a"}, {Key: "b"}})}}); err == nil {
		t.Fatal("reserved token in input must be rejected")
	}
	if _, err := l.SystemOne(ctx, "s", Questions{{ID: "q", Question: NewChoice("i", Options{{Key: "a"}, {Key: "a"}})}}); err == nil {
		t.Fatal("duplicate option keys must be rejected")
	}
	if _, err := NewLocal(t.TempDir()).SystemOne(ctx, "s", Questions{{ID: "q", Question: NewNoul("i", nil)}}); err == nil {
		t.Fatal("missing bundle must error")
	}
	// single-option questions skip the model
	r, err := l.SystemOne(ctx, "s", Questions{{ID: "q", Question: NewChoice("i", Options{{Key: "only"}})}})
	if err != nil {
		t.Fatal(err)
	}
	if a, _ := r.Choice("q"); a.Choice != "only" || a.Confidence != 1 || a.Probabilities["only"] != 1 {
		t.Fatalf("%+v", a)
	}
}
