package decide

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// Questions

// Question is one of Noul, Choice or Score.
type Question interface {
	// Type is the wire discriminator: "noul", "choice" or "score".
	Type() string
	json.Marshaler
}

// Option is one candidate of a Choice. An empty Description means the key
// itself is used as the option text.
type Option struct {
	Key         string
	Description string
}

// Options is an ordered list of Choice candidates. It marshals as a JSON
// object, {"key": "description", ...}, preserving order.
type Options []Option

// Choices builds Options whose description is the key itself.
func Choices(keys ...string) (Options, error) {
	seen := map[string]bool{}
	opts := make(Options, 0, len(keys))
	for _, k := range keys {
		if seen[k] {
			return nil, fmt.Errorf("decide: duplicate choices found in options list: %q", keys)
		}
		seen[k] = true
		opts = append(opts, Option{Key: k})
	}
	return opts, nil
}

// MarshalJSON implements json.Marshaler.
func (o Options) MarshalJSON() ([]byte, error) {
	obj := make(Object, len(o))
	for i, opt := range o {
		var v any
		if opt.Description != "" {
			v = opt.Description
		}
		obj[i] = KV{opt.Key, v}
	}
	if len(obj) == 0 {
		return []byte("{}"), nil
	}
	return obj.MarshalJSON()
}

// UnmarshalJSON implements json.Unmarshaler.
func (o *Options) UnmarshalJSON(data []byte) error {
	v, err := decodeOrdered(data)
	if err != nil {
		return err
	}
	opts, err := optionsFromValue(v)
	if err != nil {
		return err
	}
	*o = opts
	return nil
}

func optionsFromValue(v any) (Options, error) {
	obj, ok := v.(Object)
	if !ok {
		return nil, errors.New("choice criteria must be an object of option: description")
	}
	opts := make(Options, 0, len(obj))
	for _, kv := range obj {
		switch d := kv.Value.(type) {
		case nil:
			opts = append(opts, Option{Key: kv.Key})
		case string:
			opts = append(opts, Option{Key: kv.Key, Description: d})
		default:
			return nil, fmt.Errorf("choice criteria[%q] must be a string or null", kv.Key)
		}
	}
	return opts, nil
}

// NoulCriteria optionally describes what "true" and "false" mean for a Noul.
type NoulCriteria struct {
	True  string `json:"true,omitempty"`
	False string `json:"false,omitempty"`
}

// Noul is a yes/no probability question.
type Noul struct {
	Instructions string
	Criteria     *NoulCriteria
}

// Choice picks one option from a fixed list, with a probability distribution.
type Choice struct {
	Instructions string
	Criteria     Options
}

// Level is one step of an ordered Score scale. Examples, when present, are
// appended to What as " Examples: a, b".
type Level struct {
	What     string
	Examples []string
}

// Levels builds plain-text scale levels.
func Levels(descriptions ...string) []Level {
	out := make([]Level, len(descriptions))
	for i, d := range descriptions {
		out[i] = Level{What: d}
	}
	return out
}

// MarshalJSON writes a plain string when there are no examples, else {"what","examples"}.
func (l Level) MarshalJSON() ([]byte, error) {
	if len(l.Examples) == 0 {
		return json.Marshal(l.What)
	}
	return json.Marshal(struct {
		What     string   `json:"what"`
		Examples []string `json:"examples"`
	}{l.What, l.Examples})
}

// Text is the level description as shown to the model.
func (l Level) Text() string {
	if len(l.Examples) == 0 {
		return strings.TrimSpace(l.What)
	}
	return strings.TrimSpace(l.What + " Examples: " + strings.Join(l.Examples, ", "))
}

// Score is a position on an ordered scale (2 to 10 levels).
type Score struct {
	Instructions string
	Criteria     []Level
}

// NewNoul builds a Noul. criteria may be nil.
func NewNoul(instructions string, criteria *NoulCriteria) Noul {
	return Noul{Instructions: instructions, Criteria: criteria}
}

// NewChoice builds a Choice.
func NewChoice(instructions string, criteria Options) Choice {
	return Choice{Instructions: instructions, Criteria: criteria}
}

// NewScore builds a Score.
func NewScore(instructions string, criteria []Level) Score {
	return Score{Instructions: instructions, Criteria: criteria}
}

func (Noul) Type() string   { return "noul" }
func (Choice) Type() string { return "choice" }
func (Score) Type() string  { return "score" }

// MarshalJSON implements json.Marshaler.
func (n Noul) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type         string        `json:"type"`
		Instructions string        `json:"instructions"`
		Criteria     *NoulCriteria `json:"criteria,omitempty"`
	}{"noul", n.Instructions, n.Criteria})
}

// MarshalJSON implements json.Marshaler.
func (c Choice) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type         string  `json:"type"`
		Instructions string  `json:"instructions"`
		Criteria     Options `json:"criteria"`
	}{"choice", c.Instructions, c.Criteria})
}

// MarshalJSON implements json.Marshaler.
func (s Score) MarshalJSON() ([]byte, error) {
	crit := s.Criteria
	if crit == nil {
		crit = []Level{}
	}
	return json.Marshal(struct {
		Type         string  `json:"type"`
		Instructions string  `json:"instructions"`
		Criteria     []Level `json:"criteria"`
	}{"score", s.Instructions, crit})
}

// asValue normalises pointer question types to values.
func asValue(q Question) Question {
	switch x := q.(type) {
	case *Noul:
		return *x
	case *Choice:
		return *x
	case *Score:
		return *x
	}
	return q
}

// instructionsText mirrors the Python SDK: TypeSafe's API lets instructions be
// a string, object or array; only text reaches the model, so structured
// instructions are serialised to JSON.
func instructionsText(v any) (string, error) {
	switch x := v.(type) {
	case string:
		return x, nil
	case Object:
		return pyJSONDumps(x, true), nil
	case []any:
		return pyJSONDumps(x, false), nil
	case nil:
		return "", errors.New("instructions is required")
	}
	return "", errors.New("instructions must be a string, object or array")
}

// ParseQuestion decodes a wire-format question. A missing "type" means
// "choice", as in the reference server.
func ParseQuestion(data []byte) (Question, error) {
	v, err := decodeOrdered(data)
	if err != nil {
		return nil, err
	}
	obj, ok := v.(Object)
	if !ok {
		return nil, errors.New("question must be a JSON object")
	}
	return questionFromObject(obj)
}

func questionFromObject(obj Object) (Question, error) {
	typ := "choice"
	if t, ok := obj.Get("type"); ok {
		s, isStr := t.(string)
		if !isStr {
			return nil, errors.New("question type must be a string")
		}
		typ = s
	}
	rawInst, _ := obj.Get("instructions")
	inst, err := instructionsText(rawInst)
	if err != nil {
		return nil, err
	}
	crit, hasCrit := obj.Get("criteria")
	switch typ {
	case "choice":
		if !hasCrit {
			return nil, errors.New("choice question requires criteria")
		}
		opts, err := optionsFromValue(crit)
		if err != nil {
			return nil, err
		}
		return Choice{Instructions: inst, Criteria: opts}, nil
	case "noul":
		return noulFromObject(obj, inst, crit)
	case "score":
		arr, ok := crit.([]any)
		if !ok {
			return nil, errors.New("score question requires a criteria array")
		}
		levels := make([]Level, 0, len(arr))
		for i, e := range arr {
			switch x := e.(type) {
			case string:
				levels = append(levels, Level{What: x})
			case Object:
				var lv Level
				if w, ok := x.Get("what"); ok {
					if lv.What, ok = w.(string); !ok && w != nil {
						return nil, fmt.Errorf("score criteria[%d].what must be a string", i)
					}
				}
				if ex, ok := x.Get("examples"); ok && ex != nil {
					list, ok := ex.([]any)
					if !ok {
						return nil, fmt.Errorf("score criteria[%d].examples must be an array", i)
					}
					for _, e := range list {
						lv.Examples = append(lv.Examples, pyStr(e))
					}
				}
				levels = append(levels, lv)
			default:
				return nil, fmt.Errorf("score criteria[%d] must be a string or object", i)
			}
		}
		return Score{Instructions: inst, Criteria: levels}, nil
	}
	return nil, fmt.Errorf("unknown question type %q", typ)
}

// Legacy spellings of Noul's criteria keys; the presets once passed these.
var legacyNoulCriteria = [][2]string{{"pos_criteria", "true"}, {"neg_criteria", "false"}}

func noulFromObject(obj Object, inst string, crit any) (Question, error) {
	c := map[string]string{}
	if crit != nil {
		co, ok := crit.(Object)
		if !ok {
			return nil, errors.New("noul criteria must be an object")
		}
		for _, kv := range co {
			s, ok := kv.Value.(string)
			if !ok {
				return nil, fmt.Errorf("noul criteria[%q] must be a string", kv.Key)
			}
			c[kv.Key] = s
		}
	}
	for _, lg := range legacyNoulCriteria {
		v, ok := obj.Get(lg[0])
		if !ok {
			continue
		}
		if _, dup := c[lg[1]]; dup {
			return nil, fmt.Errorf("Noul got both '%s' and criteria['%s']; use criteria only", lg[0], lg[1])
		}
		if s, _ := v.(string); s != "" {
			c[lg[1]] = s
		}
	}
	n := Noul{Instructions: inst}
	if len(c) > 0 {
		n.Criteria = &NoulCriteria{True: c["true"], False: c["false"]}
	}
	return n, nil
}

// NamedQuestion pairs a question with its id.
type NamedQuestion struct {
	ID       string
	Question Question
}

// Questions is an ordered set of questions; it marshals as a JSON object
// keyed by id and preserves order in both directions.
type Questions []NamedQuestion

// QuestionMap converts a map into Questions, sorted by id.
func QuestionMap(m map[string]Question) Questions {
	ids := make([]string, 0, len(m))
	for id := range m {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	qs := make(Questions, len(ids))
	for i, id := range ids {
		qs[i] = NamedQuestion{id, m[id]}
	}
	return qs
}

// Get returns the question with the given id.
func (qs Questions) Get(id string) (Question, bool) {
	for _, q := range qs {
		if q.ID == id {
			return q.Question, true
		}
	}
	return nil, false
}

// MarshalJSON implements json.Marshaler.
func (qs Questions) MarshalJSON() ([]byte, error) {
	obj := make(Object, len(qs))
	for i, q := range qs {
		obj[i] = KV{q.ID, q.Question}
	}
	if len(obj) == 0 {
		return []byte("{}"), nil
	}
	return obj.MarshalJSON()
}

// UnmarshalJSON implements json.Unmarshaler.
func (qs *Questions) UnmarshalJSON(data []byte) error {
	v, err := decodeOrdered(data)
	if err != nil {
		return err
	}
	obj, ok := v.(Object)
	if !ok {
		return errors.New("questions must be a JSON object")
	}
	out := make(Questions, 0, len(obj))
	for _, kv := range obj {
		qo, ok := kv.Value.(Object)
		if !ok {
			return fmt.Errorf("question %q must be a JSON object", kv.Key)
		}
		q, err := questionFromObject(qo)
		if err != nil {
			return fmt.Errorf("question %q: %w", kv.Key, err)
		}
		out = append(out, NamedQuestion{kv.Key, q})
	}
	*qs = out
	return nil
}

// ---------------------------------------------------------------------------
// Answers

// Answer is one of *NoulAnswer, *ChoiceAnswer or *ScoreAnswer.
type Answer interface {
	Type() string
}

// NoulAnswer is the answer to a Noul question.
type NoulAnswer struct {
	// Noul is the probability, in [0, 1], that the condition is true.
	Noul float64 `json:"noul"`
}

// ChoiceAnswer is the answer to a Choice question.
type ChoiceAnswer struct {
	// Choice is the key of the most probable option.
	Choice        string             `json:"choice"`
	Probabilities map[string]float64 `json:"probabilities"`
	// Confidence is (n*p_max - 1)/(n - 1), in [0, 1].
	Confidence float64 `json:"confidence"`
}

// ScoreAnswer is the answer to a Score question.
type ScoreAnswer struct {
	// Score is the probability-weighted expectation over level indexes.
	Score         float64            `json:"score"`
	Confidence    float64            `json:"confidence"`
	Legend        map[string]string  `json:"legend"`
	Probabilities map[string]float64 `json:"probabilities"`
}

func (*NoulAnswer) Type() string   { return "noul" }
func (*ChoiceAnswer) Type() string { return "choice" }
func (*ScoreAnswer) Type() string  { return "score" }

// MarshalJSON implements json.Marshaler.
func (a *NoulAnswer) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type string  `json:"type"`
		Noul float64 `json:"noul"`
	}{"noul", a.Noul})
}

// MarshalJSON implements json.Marshaler.
func (a *ChoiceAnswer) MarshalJSON() ([]byte, error) {
	type plain ChoiceAnswer
	return json.Marshal(struct {
		Type string `json:"type"`
		*plain
	}{"choice", (*plain)(a)})
}

// MarshalJSON implements json.Marshaler.
func (a *ScoreAnswer) MarshalJSON() ([]byte, error) {
	type plain ScoreAnswer
	return json.Marshal(struct {
		Type string `json:"type"`
		*plain
	}{"score", (*plain)(a)})
}

// Usage is token accounting metadata.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Response is the result of a System One evaluation.
type Response struct {
	Model string
	// Answers by question id.
	Answers map[string]Answer
	// Order lists the answer ids in question order.
	Order []string
	Usage Usage
}

// Choice returns the answer to a Choice question.
func (r *Response) Choice(id string) (*ChoiceAnswer, bool) {
	a, ok := r.Answers[id].(*ChoiceAnswer)
	return a, ok
}

// Noul returns the answer to a Noul question.
func (r *Response) Noul(id string) (*NoulAnswer, bool) {
	a, ok := r.Answers[id].(*NoulAnswer)
	return a, ok
}

// Score returns the answer to a Score question.
func (r *Response) Score(id string) (*ScoreAnswer, bool) {
	a, ok := r.Answers[id].(*ScoreAnswer)
	return a, ok
}

// MarshalJSON implements json.Marshaler.
func (r *Response) MarshalJSON() ([]byte, error) {
	order := r.Order
	if len(order) != len(r.Answers) {
		order = make([]string, 0, len(r.Answers))
		for id := range r.Answers {
			order = append(order, id)
		}
		sort.Strings(order)
	}
	var b bytes.Buffer
	b.WriteString(`{"model":`)
	m, _ := json.Marshal(r.Model)
	b.Write(m)
	b.WriteString(`,"answers":{`)
	for i, id := range order {
		if i > 0 {
			b.WriteByte(',')
		}
		k, _ := json.Marshal(id)
		b.Write(k)
		b.WriteByte(':')
		a, err := json.Marshal(r.Answers[id])
		if err != nil {
			return nil, err
		}
		b.Write(a)
	}
	b.WriteString(`},"usage":`)
	u, _ := json.Marshal(r.Usage)
	b.Write(u)
	b.WriteByte('}')
	return b.Bytes(), nil
}

// UnmarshalJSON implements json.Unmarshaler.
func (r *Response) UnmarshalJSON(data []byte) error {
	var env struct {
		Model   string          `json:"model"`
		Answers json.RawMessage `json:"answers"`
		Usage   Usage           `json:"usage"`
	}
	if err := json.Unmarshal(data, &env); err != nil {
		return err
	}
	r.Model, r.Usage = env.Model, env.Usage
	r.Answers, r.Order = map[string]Answer{}, nil
	v, err := decodeOrdered(env.Answers)
	if err != nil {
		return err
	}
	obj, ok := v.(Object)
	if !ok {
		return errors.New("decide: response answers must be an object")
	}
	dec := map[string]json.RawMessage{}
	if err := json.Unmarshal(env.Answers, &dec); err != nil {
		return err
	}
	for _, kv := range obj {
		a, err := parseAnswer(dec[kv.Key])
		if err != nil {
			return fmt.Errorf("decide: answer %q: %w", kv.Key, err)
		}
		r.Answers[kv.Key] = a
		r.Order = append(r.Order, kv.Key)
	}
	return nil
}

func parseAnswer(raw json.RawMessage) (Answer, error) {
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(raw, &probe); err != nil {
		return nil, err
	}
	typ := ""
	if t, ok := probe["type"]; ok {
		_ = json.Unmarshal(t, &typ)
	}
	if typ == "" {
		switch {
		case probe["legend"] != nil:
			typ = "score"
		case probe["choice"] != nil:
			typ = "choice"
		case probe["noul"] != nil:
			typ = "noul"
		}
	}
	switch typ {
	case "noul":
		var a NoulAnswer
		return &a, json.Unmarshal(raw, &a)
	case "choice":
		var a ChoiceAnswer
		return &a, json.Unmarshal(raw, &a)
	case "score":
		var a ScoreAnswer
		return &a, json.Unmarshal(raw, &a)
	}
	return nil, fmt.Errorf("unknown answer type %q", typ)
}
