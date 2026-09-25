package decide

import (
	"errors"
	"fmt"
)

// SDKVersion is the version of this SDK.
const SDKVersion = "1.0.0"

// Request is the /v1/systemone request body.
type Request struct {
	Model     string
	State     any
	Questions Questions
}

// ParseRequest decodes and validates a wire-format request. "model" defaults
// to "decide-latest"; "state" is required; "questions" must be a non-empty object.
func ParseRequest(data []byte) (*Request, error) {
	v, err := decodeOrdered(data)
	if err != nil {
		return nil, fmt.Errorf("invalid JSON: %w", err)
	}
	obj, ok := v.(Object)
	if !ok {
		return nil, errors.New("request body must be a JSON object")
	}
	req := &Request{Model: "decide-latest"}
	if m, ok := obj.Get("model"); ok {
		s, isStr := m.(string)
		if !isStr {
			return nil, errors.New("model must be a string")
		}
		req.Model = s
	}
	state, ok := obj.Get("state")
	if !ok {
		return nil, errors.New("state is required")
	}
	req.State = state
	qv, ok := obj.Get("questions")
	if !ok {
		return nil, errors.New("questions is required")
	}
	qo, ok := qv.(Object)
	if !ok {
		return nil, errors.New("questions must be a JSON object")
	}
	if len(qo) == 0 {
		// OpenAPI declares minProperties: 1; an empty object is malformed, not a no-op.
		return nil, errors.New("questions must contain at least one entry")
	}
	for _, kv := range qo {
		q, ok := kv.Value.(Object)
		if !ok {
			return nil, fmt.Errorf("question %q must be a JSON object", kv.Key)
		}
		question, err := questionFromObject(q)
		if err != nil {
			return nil, fmt.Errorf("question %q: %w", kv.Key, err)
		}
		req.Questions = append(req.Questions, NamedQuestion{kv.Key, question})
	}
	return req, nil
}
