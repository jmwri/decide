package decide

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
)

// GateResult is the outcome of ConfidenceGate.
type GateResult struct {
	Automatic map[string]Answer
	Escalate  map[string]Answer
	Response  *Response
}

// ConfidenceGate is the selective-automation pattern: answers whose
// calibrated confidence reaches threshold go to Automatic, the ambiguous tail
// to Escalate. Noul answers carry no confidence, so their distance from
// uncertainty, |p - 0.5| * 2, is used instead.
func ConfidenceGate(ctx context.Context, ev Evaluator, state any, questions Questions, threshold float64) (*GateResult, error) {
	if threshold < 0 || threshold > 1 {
		return nil, fmt.Errorf("decide: threshold must be in [0.0, 1.0], got %v", threshold)
	}
	resp, err := evaluator(ev).SystemOne(ctx, state, questions)
	if err != nil {
		return nil, err
	}
	res := &GateResult{Automatic: map[string]Answer{}, Escalate: map[string]Answer{}, Response: resp}
	for _, id := range resp.Order {
		var conf float64
		switch a := resp.Answers[id].(type) {
		case *NoulAnswer:
			conf = math.Abs(a.Noul-0.5) * 2
		case *ChoiceAnswer:
			conf = a.Confidence
		case *ScoreAnswer:
			conf = a.Confidence
		}
		if conf >= threshold {
			res.Automatic[id] = resp.Answers[id]
		} else {
			res.Escalate[id] = resp.Answers[id]
		}
	}
	return res, nil
}

// RouteConfig tunes Route.
type RouteConfig struct {
	// Default handles answers with no matching route or below MinConfidence.
	Default func(*ChoiceAnswer) error
	// MinConfidence below which the Default handler is used.
	MinConfidence float64
}

// Route runs a Choice decision and dispatches to the handler registered under
// the winning option's key. It returns the answer alongside any handler error.
// With no matching handler (or confidence under MinConfidence) it calls
// cfg.Default if set, and otherwise just returns the answer.
func Route(ctx context.Context, ev Evaluator, state any, question Choice, routes map[string]func(*ChoiceAnswer) error, cfg RouteConfig) (*ChoiceAnswer, error) {
	resp, err := evaluator(ev).SystemOne(ctx, state, Questions{{"route_question", question}})
	if err != nil {
		return nil, err
	}
	ans, ok := resp.Choice("route_question")
	if !ok {
		return nil, errors.New("decide: response has no choice answer")
	}
	handler := routes[ans.Choice]
	if handler == nil || ans.Confidence < cfg.MinConfidence {
		if cfg.Default != nil {
			return ans, cfg.Default(ans)
		}
		return ans, nil
	}
	return ans, handler(ans)
}

// CompositeItem is one term of a composite score.
type CompositeItem struct {
	Raw        float64
	Normalized float64
	Weight     float64
}

// CompositeResult is the outcome of CompositeScore.
type CompositeResult struct {
	Score     float64
	Breakdown map[string]CompositeItem
	Response  *Response
}

// CompositeScore combines Score and Noul answers into one weighted risk index
// in [0, 1]. Score answers are normalised by their top level index, Noul
// answers use P(true), and Choice answers are excluded. Questions missing from
// weights weigh 1.0.
func CompositeScore(ctx context.Context, ev Evaluator, state any, questions Questions, weights map[string]float64) (*CompositeResult, error) {
	resp, err := evaluator(ev).SystemOne(ctx, state, questions)
	if err != nil {
		return nil, err
	}
	var sum, total float64
	breakdown := map[string]CompositeItem{}
	for _, id := range resp.Order {
		var raw, norm float64
		switch a := resp.Answers[id].(type) {
		case *ScoreAnswer:
			raw, norm = a.Score, a.Score/float64(max(1, len(a.Legend)-1))
		case *NoulAnswer:
			raw, norm = a.Noul, a.Noul
		default:
			continue
		}
		w := 1.0
		if v, ok := weights[id]; ok {
			w = v
		}
		sum += norm * w
		total += w
		breakdown[id] = CompositeItem{Raw: raw, Normalized: round(norm, 4), Weight: w}
	}
	score := sum
	if total > 0 {
		score = sum / total
	}
	return &CompositeResult{Score: round(score, 4), Breakdown: breakdown, Response: resp}, nil
}

// Taxonomy is a two-level option hierarchy.
type Taxonomy []TaxonomyCategory

// TaxonomyCategory is one branch of a Taxonomy.
type TaxonomyCategory struct {
	Name    string
	Options Options
}

// TwoStageResult is the outcome of TwoStageChoice.
type TwoStageResult struct {
	Category           string
	CategoryConfidence float64
	Choice             string
	ChoiceConfidence   float64
	CombinedConfidence float64
}

// TwoStageConfig overrides the default instructions of TwoStageChoice. The
// option instructions may contain "{category}".
type TwoStageConfig struct {
	InstructionsCategory string
	InstructionsOption   string
}

// TwoStageChoice routes high-cardinality taxonomies (>20 options)
// hierarchically: first the category, then the option within it. This avoids
// single-pass option saturation.
func TwoStageChoice(ctx context.Context, ev Evaluator, state any, taxonomy Taxonomy, cfg TwoStageConfig) (*TwoStageResult, error) {
	if len(taxonomy) == 0 {
		return nil, errors.New("decide: empty taxonomy")
	}
	if cfg.InstructionsCategory == "" {
		cfg.InstructionsCategory = "Which broad category best matches the state?"
	}
	if cfg.InstructionsOption == "" {
		cfg.InstructionsOption = "Which specific sub-option applies within {category}?"
	}
	e := evaluator(ev)
	catOpts := make(Options, len(taxonomy))
	for i, c := range taxonomy {
		catOpts[i] = Option{c.Name, "Category for " + c.Name + " operations and topics"}
	}
	r1, err := e.SystemOne(ctx, state, Questions{{"category", NewChoice(cfg.InstructionsCategory, catOpts)}})
	if err != nil {
		return nil, err
	}
	cat, ok := r1.Choice("category")
	if !ok {
		return nil, errors.New("decide: response has no category answer")
	}
	var sub Options
	for _, c := range taxonomy {
		if c.Name == cat.Choice {
			sub = c.Options
		}
	}
	r2, err := e.SystemOne(ctx, state, Questions{{"option", NewChoice(strings.ReplaceAll(cfg.InstructionsOption, "{category}", cat.Choice), sub)}})
	if err != nil {
		return nil, err
	}
	opt, ok := r2.Choice("option")
	if !ok {
		return nil, errors.New("decide: response has no option answer")
	}
	return &TwoStageResult{
		Category:           cat.Choice,
		CategoryConfidence: cat.Confidence,
		Choice:             opt.Choice,
		ChoiceConfidence:   opt.Confidence,
		CombinedConfidence: round(cat.Confidence*opt.Confidence, 4),
	}, nil
}

func evaluator(ev Evaluator) Evaluator {
	if ev == nil {
		return Default()
	}
	return ev
}
