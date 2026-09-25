// Package server exposes a decide.Evaluator over the TypeSafe-compatible
// /v1/systemone HTTP protocol.
package server

import (
	"crypto/subtle"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/jmwri/decide"
)

// Config configures the handler.
type Config struct {
	// APIKey, when non-empty, requires "Authorization: Bearer <key>". If empty,
	// $DECIDE_API_KEY is consulted on every request.
	APIKey string
	// CORSOrigins is the allowed-origin list; empty means "*". Credentials are
	// only enabled for an explicit list, never for the wildcard.
	CORSOrigins []string
	// MaxBodyBytes bounds request bodies (default 8 MiB).
	MaxBodyBytes int64
}

// New returns an http.Handler serving ev.
func New(ev decide.Evaluator, cfg Config) http.Handler {
	if cfg.MaxBodyBytes == 0 {
		cfg.MaxBodyBytes = 8 << 20
	}
	if len(cfg.CORSOrigins) == 0 {
		cfg.CORSOrigins = []string{"*"}
	}
	s := &srv{ev: ev, cfg: cfg}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", s.health)
	mux.HandleFunc("GET /health", s.health)
	mux.HandleFunc("GET /v1/models", s.models)
	mux.HandleFunc("POST /v1/systemone", s.systemOne)
	return s.cors(mux)
}

type srv struct {
	ev  decide.Evaluator
	cfg Config
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}

func detail(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"detail": msg})
}

func (s *srv) cors(next http.Handler) http.Handler {
	wildcard := len(s.cfg.CORSOrigins) == 1 && s.cfg.CORSOrigins[0] == "*"
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if origin := r.Header.Get("Origin"); origin != "" {
			allowed := wildcard
			for _, o := range s.cfg.CORSOrigins {
				if o == origin {
					allowed = true
				}
			}
			if allowed {
				if wildcard {
					w.Header().Set("Access-Control-Allow-Origin", "*")
				} else {
					w.Header().Set("Access-Control-Allow-Origin", origin)
					w.Header().Set("Access-Control-Allow-Credentials", "true")
					w.Header().Add("Vary", "Origin")
				}
			}
			if r.Method == http.MethodOptions && r.Header.Get("Access-Control-Request-Method") != "" {
				w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
				if h := r.Header.Get("Access-Control-Request-Headers"); h != "" {
					w.Header().Set("Access-Control-Allow-Headers", h)
				} else {
					w.Header().Set("Access-Control-Allow-Headers", "*")
				}
				w.WriteHeader(http.StatusOK)
				return
			}
		}
		next.ServeHTTP(w, r)
	})
}

func (s *srv) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{
		"status":  "ok",
		"service": "decide-server",
		"version": decide.SDKVersion,
		"engine":  "decide-" + decide.Version,
		"homage":  "System One decision model",
	})
}

func (s *srv) models(w http.ResponseWriter, r *http.Request) {
	type entry struct {
		Name        string `json:"name"`
		Description string `json:"description"`
		ReleaseDate string `json:"release_date"`
	}
	models := []entry{
		{"decide-latest", "Current Decide System One decision model", "2026-09-25"},
		{decide.ModelID, "Decide " + decide.Version + " (order-invariant option scoring)", "2026-09-25"},
	}
	type dataEntry struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		OwnedBy string `json:"owned_by"`
	}
	data := make([]dataEntry, len(models))
	for i, m := range models {
		data[i] = dataEntry{m.Name, "model", "decide"}
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": models, "object": "list", "data": data})
}

func (s *srv) authorized(r *http.Request) (ok bool, msg string) {
	key := s.cfg.APIKey
	if key == "" {
		key = os.Getenv("DECIDE_API_KEY")
	}
	if key == "" {
		return true, ""
	}
	h := r.Header.Get("Authorization")
	if !strings.HasPrefix(h, "Bearer ") {
		return false, "Missing or invalid Bearer token"
	}
	token := strings.TrimSpace(strings.TrimPrefix(h, "Bearer "))
	// Constant-time compare: a plain == leaks how many leading bytes of a
	// guess were right through response timing.
	if subtle.ConstantTimeCompare([]byte(token), []byte(key)) != 1 {
		return false, "Unauthorized: invalid API key"
	}
	return true, ""
}

func (s *srv) systemOne(w http.ResponseWriter, r *http.Request) {
	if ok, msg := s.authorized(r); !ok {
		detail(w, http.StatusUnauthorized, msg)
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, s.cfg.MaxBodyBytes))
	if err != nil {
		detail(w, http.StatusRequestEntityTooLarge, err.Error())
		return
	}
	req, err := decide.ParseRequest(body)
	if err != nil {
		detail(w, http.StatusUnprocessableEntity, err.Error())
		return
	}
	resp, err := s.ev.SystemOne(r.Context(), req.State, req.Questions)
	if err != nil {
		detail(w, http.StatusUnprocessableEntity, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, resp)
}
