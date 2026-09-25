package decide

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// Evaluator answers System One questions. *Local and *Remote implement it.
type Evaluator interface {
	SystemOne(ctx context.Context, state any, questions Questions) (*Response, error)
}

// Client offers the high-level decision helpers on top of any Evaluator.
type Client struct {
	Evaluator
}

// NewClient wraps an Evaluator.
func NewClient(ev Evaluator) *Client { return &Client{ev} }

var (
	defaultMu sync.Mutex
	defaultEv Evaluator
)

// Default returns the process-wide Client. Unless replaced with SetDefault it
// talks to $DECIDE_BASE_URL when set (authenticating with $DECIDE_API_KEY or
// $TYPESAFE_API_KEY), and otherwise runs the model in-process.
func Default() *Client {
	defaultMu.Lock()
	defer defaultMu.Unlock()
	if defaultEv == nil {
		if u := os.Getenv("DECIDE_BASE_URL"); u != "" {
			key := os.Getenv("DECIDE_API_KEY")
			if key == "" {
				key = os.Getenv("TYPESAFE_API_KEY")
			}
			defaultEv = NewRemote(u, key)
		} else {
			defaultEv = NewLocal("")
		}
	}
	return &Client{defaultEv}
}

// SetDefault replaces the process-wide Evaluator.
func SetDefault(ev Evaluator) {
	defaultMu.Lock()
	defer defaultMu.Unlock()
	defaultEv = ev
}

// SystemOne evaluates several questions against one state, using the default client.
func SystemOne(ctx context.Context, state any, questions Questions) (*Response, error) {
	return Default().SystemOne(ctx, state, questions)
}

// Decide makes a discrete choice among options, using the default client.
func Decide(ctx context.Context, state any, choices Options, instructions string) (*ChoiceAnswer, error) {
	return Default().Decide(ctx, state, choices, instructions)
}

// Judge returns P(condition holds), using the default client.
func Judge(ctx context.Context, state any, instructions string, criteria *NoulCriteria) (float64, error) {
	return Default().Judge(ctx, state, instructions, criteria)
}

// Rate scores state on an ordered scale, using the default client.
func Rate(ctx context.Context, state any, levels []Level, instructions string) (*ScoreAnswer, error) {
	return Default().Rate(ctx, state, levels, instructions)
}

// Decide makes a fast discrete decision among options. Empty instructions
// default to "Which option best describes the state?".
func (c *Client) Decide(ctx context.Context, state any, choices Options, instructions string) (*ChoiceAnswer, error) {
	if instructions == "" {
		instructions = "Which option best describes the state?"
	}
	resp, err := c.SystemOne(ctx, state, Questions{{"decision", NewChoice(instructions, choices)}})
	if err != nil {
		return nil, err
	}
	a, ok := resp.Choice("decision")
	if !ok {
		return nil, errors.New("decide: response has no choice answer")
	}
	return a, nil
}

// Judge evaluates a yes/no question and returns its probability in [0, 1].
func (c *Client) Judge(ctx context.Context, state any, instructions string, criteria *NoulCriteria) (float64, error) {
	resp, err := c.SystemOne(ctx, state, Questions{{"judgment", NewNoul(instructions, criteria)}})
	if err != nil {
		return 0, err
	}
	a, ok := resp.Noul("judgment")
	if !ok {
		return 0, errors.New("decide: response has no noul answer")
	}
	return a.Noul, nil
}

// Rate evaluates state on an ordered multi-level scale. Empty instructions
// default to "Rate where the state falls on this scale:".
func (c *Client) Rate(ctx context.Context, state any, levels []Level, instructions string) (*ScoreAnswer, error) {
	if instructions == "" {
		instructions = "Rate where the state falls on this scale:"
	}
	resp, err := c.SystemOne(ctx, state, Questions{{"rating", NewScore(instructions, levels)}})
	if err != nil {
		return nil, err
	}
	a, ok := resp.Score("rating")
	if !ok {
		return nil, errors.New("decide: response has no score answer")
	}
	return a, nil
}

// Remote is an Evaluator that calls a /v1/systemone HTTP endpoint: a `decide
// serve` instance, or TypeSafe's hosted API.
type Remote struct {
	BaseURL string
	APIKey  string
	// Model is sent as the request's "model"; default "decide-latest".
	Model string
	HTTP  *http.Client
}

// NewRemote builds a Remote for baseURL (default http://localhost:8000).
func NewRemote(baseURL, apiKey string) *Remote {
	if baseURL == "" {
		baseURL = "http://localhost:8000"
	}
	return &Remote{BaseURL: baseURL, APIKey: apiKey, Model: "decide-latest", HTTP: &http.Client{Timeout: 30 * time.Second}}
}

// HTTPError is returned for non-2xx responses.
type HTTPError struct {
	Status int
	Body   string
}

func (e *HTTPError) Error() string { return fmt.Sprintf("decide: HTTP %d: %s", e.Status, e.Body) }

// SystemOne implements Evaluator.
func (r *Remote) SystemOne(ctx context.Context, state any, questions Questions) (*Response, error) {
	model := r.Model
	if model == "" {
		model = "decide-latest"
	}
	body, err := json.Marshal(struct {
		Model     string    `json:"model"`
		State     any       `json:"state"`
		Questions Questions `json:"questions"`
	}{model, normalize(state), questions})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(r.BaseURL, "/")+"/v1/systemone", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if r.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+r.APIKey)
	}
	hc := r.HTTP
	if hc == nil {
		hc = http.DefaultClient
	}
	resp, err := hc.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode/100 != 2 {
		return nil, &HTTPError{resp.StatusCode, strings.TrimSpace(string(data))}
	}
	var out Response
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("decide: decoding response: %w", err)
	}
	return &out, nil
}
