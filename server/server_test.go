package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jmwri/decide"
)

type stubEval struct {
	gotState any
	gotQs    decide.Questions
}

func (s *stubEval) SystemOne(_ context.Context, state any, qs decide.Questions) (*decide.Response, error) {
	s.gotState, s.gotQs = state, qs
	r := &decide.Response{Model: decide.ModelID, Answers: map[string]decide.Answer{}}
	for _, q := range qs {
		r.Answers[q.ID] = &decide.NoulAnswer{Noul: 0.75}
		r.Order = append(r.Order, q.ID)
	}
	return r, nil
}

const validBody = `{"model":"decide-0.1.0","state":{"error":"Disk at 98%"},"questions":{"z":{"type":"noul","instructions":"intervene?"},"a":{"type":"noul","instructions":"again?"}}}`

func do(h http.Handler, method, path, body string, hdr map[string]string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestHealthAndModels(t *testing.T) {
	h := New(&stubEval{}, Config{})
	for _, path := range []string{"/", "/health"} {
		rec := do(h, "GET", path, "", nil)
		var out map[string]string
		if rec.Code != 200 || json.Unmarshal(rec.Body.Bytes(), &out) != nil || out["status"] != "ok" || out["engine"] != "decide-"+decide.Version {
			t.Fatalf("%s: %d %s", path, rec.Code, rec.Body)
		}
	}
	rec := do(h, "GET", "/v1/models", "", nil)
	var models struct {
		Models []struct{ Name string } `json:"models"`
		Data   []struct{ ID string }   `json:"data"`
		Object string                  `json:"object"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &models); err != nil || len(models.Models) != 2 || len(models.Data) != 2 || models.Object != "list" {
		t.Fatalf("models: %v %s", err, rec.Body)
	}
}

func TestSystemOne(t *testing.T) {
	ev := &stubEval{}
	rec := do(New(ev, Config{}), "POST", "/v1/systemone", validBody, nil)
	if rec.Code != 200 {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	// Question order (z before a) survives the request and the response.
	if ev.gotQs[0].ID != "z" || ev.gotQs[1].ID != "a" {
		t.Fatalf("question order lost: %+v", ev.gotQs)
	}
	if !strings.Contains(rec.Body.String(), `"answers":{"z":{"type":"noul","noul":0.75},"a":`) {
		t.Fatalf("response order: %s", rec.Body)
	}
	// And the wire format decodes back through the SDK's own client types.
	var resp decide.Response
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if a, ok := resp.Noul("z"); !ok || a.Noul != 0.75 {
		t.Fatalf("%+v", resp)
	}
}

func TestValidationErrors(t *testing.T) {
	h := New(&stubEval{}, Config{})
	for name, body := range map[string]string{
		"empty questions": `{"state":"s","questions":{}}`,
		"missing state":   `{"questions":{"q":{"type":"noul","instructions":"i"}}}`,
		"bad question":    `{"state":"s","questions":{"q":{"type":"nope","instructions":"i"}}}`,
		"not json":        `not json`,
	} {
		rec := do(h, "POST", "/v1/systemone", body, nil)
		var out map[string]string
		if rec.Code != 422 || json.Unmarshal(rec.Body.Bytes(), &out) != nil || out["detail"] == "" {
			t.Errorf("%s: %d %s", name, rec.Code, rec.Body)
		}
	}
}

func TestBearerAuth(t *testing.T) {
	h := New(&stubEval{}, Config{APIKey: "sekret"})
	if rec := do(h, "POST", "/v1/systemone", validBody, nil); rec.Code != 401 {
		t.Fatalf("missing token: %d", rec.Code)
	}
	if rec := do(h, "POST", "/v1/systemone", validBody, map[string]string{"Authorization": "Bearer nope"}); rec.Code != 401 {
		t.Fatalf("wrong token: %d", rec.Code)
	}
	if rec := do(h, "POST", "/v1/systemone", validBody, map[string]string{"Authorization": "Bearer sekret"}); rec.Code != 200 {
		t.Fatalf("right token: %d %s", rec.Code, rec.Body)
	}
	// Health stays open.
	if rec := do(h, "GET", "/health", "", nil); rec.Code != 200 {
		t.Fatalf("health: %d", rec.Code)
	}
}

func TestAPIKeyFromEnvironment(t *testing.T) {
	t.Setenv("DECIDE_API_KEY", "envkey")
	h := New(&stubEval{}, Config{})
	if rec := do(h, "POST", "/v1/systemone", validBody, nil); rec.Code != 401 {
		t.Fatalf("env key not enforced: %d", rec.Code)
	}
	if rec := do(h, "POST", "/v1/systemone", validBody, map[string]string{"Authorization": "Bearer envkey"}); rec.Code != 200 {
		t.Fatalf("env key rejected: %d", rec.Code)
	}
}

func TestCORS(t *testing.T) {
	rec := do(New(&stubEval{}, Config{}), "OPTIONS", "/v1/systemone", "", map[string]string{
		"Origin": "https://app.example", "Access-Control-Request-Method": "POST",
	})
	if rec.Code != 200 || rec.Header().Get("Access-Control-Allow-Origin") != "*" || rec.Header().Get("Access-Control-Allow-Credentials") != "" {
		t.Fatalf("wildcard preflight: %d %v", rec.Code, rec.Header())
	}
	h := New(&stubEval{}, Config{CORSOrigins: []string{"https://app.example"}})
	rec = do(h, "GET", "/health", "", map[string]string{"Origin": "https://app.example"})
	if rec.Header().Get("Access-Control-Allow-Origin") != "https://app.example" || rec.Header().Get("Access-Control-Allow-Credentials") != "true" {
		t.Fatalf("explicit origin: %v", rec.Header())
	}
	rec = do(h, "GET", "/health", "", map[string]string{"Origin": "https://evil.example"})
	if rec.Header().Get("Access-Control-Allow-Origin") != "" {
		t.Fatalf("unlisted origin allowed: %v", rec.Header())
	}
}

// The SDK's Remote evaluator and this handler must speak the same protocol.
func TestRemoteAgainstServer(t *testing.T) {
	ev := &stubEval{}
	ts := httptest.NewServer(New(ev, Config{APIKey: "k"}))
	defer ts.Close()
	r := decide.NewRemote(ts.URL, "k")
	state := decide.Object{{Key: "b", Value: "1"}, {Key: "a", Value: "2"}}
	resp, err := r.SystemOne(context.Background(), state, decide.Questions{
		{ID: "q", Question: decide.NewChoice("pick", decide.Options{{Key: "x", Description: "X"}, {Key: "y"}})},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := resp.Noul("q"); !ok {
		t.Fatalf("%+v", resp)
	}
	c := ev.gotQs[0].Question.(decide.Choice)
	if c.Instructions != "pick" || len(c.Criteria) != 2 || c.Criteria[0].Key != "x" || c.Criteria[1].Description != "" {
		t.Fatalf("server saw %+v", c)
	}
	if obj, ok := ev.gotState.(decide.Object); !ok || obj[0].Key != "b" {
		t.Fatalf("state order lost: %#v", ev.gotState)
	}

	bad := decide.NewRemote(ts.URL, "wrong")
	_, err = bad.SystemOne(context.Background(), "s", decide.Questions{{ID: "q", Question: decide.NewNoul("i", nil)}})
	var he *decide.HTTPError
	if e, ok := err.(*decide.HTTPError); !ok || e.Status != 401 {
		t.Fatalf("expected 401 HTTPError, got %v (%v)", err, he)
	}
}
