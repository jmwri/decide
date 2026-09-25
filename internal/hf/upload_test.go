package hf

import (
	"bufio"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// fakeHub implements just enough of the hub protocol to check the client
// speaks it correctly: repo creation, LFS batch/PUT/verify, and commit.
type fakeHub struct {
	mu        sync.Mutex
	created   map[string]any
	stored    map[string][]byte // LFS objects by oid
	verified  []string
	commit    []map[string]any
	gotAuth   []string
	srv       *httptest.Server
	existing  map[string]bool // oids the hub already has
	putAuthed bool
}

func newFakeHub(t *testing.T) *fakeHub {
	h := &fakeHub{stored: map[string][]byte{}, existing: map[string]bool{}}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/repos/create", func(w http.ResponseWriter, r *http.Request) {
		h.gotAuth = append(h.gotAuth, r.Header.Get("Authorization"))
		json.NewDecoder(r.Body).Decode(&h.created)
		w.Write([]byte(`{"url":"x"}`))
	})
	mux.HandleFunc("POST /{user}/{repo}/info/lfs/objects/batch", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Operation string
			Objects   []struct {
				Oid  string
				Size int64
			}
		}
		json.NewDecoder(r.Body).Decode(&req)
		o := req.Objects[0]
		obj := map[string]any{"oid": o.Oid, "size": o.Size}
		if !h.existing[o.Oid] {
			obj["actions"] = map[string]any{
				"upload": map[string]any{"href": h.srv.URL + "/s3/" + o.Oid, "header": map[string]string{"X-Sig": "abc"}},
				"verify": map[string]any{"href": h.srv.URL + "/verify", "header": map[string]string{"Authorization": "Bearer verify-token"}},
			}
		}
		w.Header().Set("Content-Type", "application/vnd.git-lfs+json")
		json.NewEncoder(w).Encode(map[string]any{"objects": []any{obj}})
	})
	mux.HandleFunc("PUT /s3/{oid}", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			h.putAuthed = true
		}
		if r.Header.Get("X-Sig") != "abc" {
			http.Error(w, "bad signature", 403)
			return
		}
		b, _ := io.ReadAll(r.Body)
		h.mu.Lock()
		h.stored[r.PathValue("oid")] = b
		h.mu.Unlock()
	})
	mux.HandleFunc("POST /verify", func(w http.ResponseWriter, r *http.Request) {
		var v struct{ Oid string }
		json.NewDecoder(r.Body).Decode(&v)
		h.verified = append(h.verified, v.Oid)
	})
	mux.HandleFunc("POST /api/models/{user}/{repo}/commit/main", func(w http.ResponseWriter, r *http.Request) {
		sc := bufio.NewScanner(r.Body)
		sc.Buffer(make([]byte, 1<<20), 1<<26)
		for sc.Scan() {
			var m map[string]any
			json.Unmarshal(sc.Bytes(), &m)
			h.commit = append(h.commit, m)
		}
		w.Write([]byte(`{"success":true}`))
	})
	h.srv = httptest.NewServer(mux)
	t.Cleanup(h.srv.Close)
	return h
}

func TestUploadDir(t *testing.T) {
	dir := t.TempDir()
	weights := []byte(strings.Repeat("W", 4096))
	os.WriteFile(filepath.Join(dir, "model.safetensors"), weights, 0o644)
	os.WriteFile(filepath.Join(dir, "decide.json"), []byte(`{"name":"decide"}`), 0o644)
	os.WriteFile(filepath.Join(dir, "README.md"), []byte("# card <b>&"), 0o644)
	os.WriteFile(filepath.Join(dir, "model.safetensors.tmp"), []byte("junk"), 0o644)

	hub := newFakeHub(t)
	c := &Client{Endpoint: hub.srv.URL, Token: "hf_secret"}
	if err := c.CreateRepo(t.Context(), "someone/decide", false); err != nil {
		t.Fatal(err)
	}
	if hub.created["name"] != "decide" || hub.created["organization"] != "someone" || hub.created["type"] != "model" || hub.gotAuth[0] != "Bearer hf_secret" {
		t.Fatalf("create repo: %v %v", hub.created, hub.gotAuth)
	}
	if err := c.UploadDir(t.Context(), "someone/decide", dir, "Add model"); err != nil {
		t.Fatal(err)
	}

	sum := sha256.Sum256(weights)
	oid := hex.EncodeToString(sum[:])
	if string(hub.stored[oid]) != string(weights) {
		t.Fatal("weights were not uploaded to the LFS URL under their sha256")
	}
	if hub.putAuthed {
		t.Fatal("the hub token must not be sent to the pre-signed storage URL")
	}
	if len(hub.verified) != 1 || hub.verified[0] != oid {
		t.Fatalf("verify: %v", hub.verified)
	}
	// commit: header, then files sorted by path; weights as lfsFile, small files inline base64.
	if len(hub.commit) != 4 || hub.commit[0]["key"] != "header" {
		t.Fatalf("commit ops: %v", hub.commit)
	}
	ops := map[string]map[string]any{}
	for _, m := range hub.commit[1:] {
		v := m["value"].(map[string]any)
		ops[v["path"].(string)] = m
	}
	if ops["model.safetensors"]["key"] != "lfsFile" || ops["model.safetensors"]["value"].(map[string]any)["oid"] != oid {
		t.Fatalf("weights op: %v", ops["model.safetensors"])
	}
	readme := ops["README.md"]["value"].(map[string]any)
	if b, _ := base64.StdEncoding.DecodeString(readme["content"].(string)); string(b) != "# card <b>&" || ops["README.md"]["key"] != "file" {
		t.Fatalf("readme op: %v", ops["README.md"])
	}
	if _, junk := ops["model.safetensors.tmp"]; junk {
		t.Fatal("temporary files must not be uploaded")
	}
}

func TestUploadSkipsObjectsTheHubHas(t *testing.T) {
	dir := t.TempDir()
	weights := []byte("already there")
	os.WriteFile(filepath.Join(dir, "model.safetensors"), weights, 0o644)
	hub := newFakeHub(t)
	sum := sha256.Sum256(weights)
	hub.existing[hex.EncodeToString(sum[:])] = true
	c := &Client{Endpoint: hub.srv.URL, Token: "t"}
	if err := c.UploadDir(t.Context(), "u/r", dir, "m"); err != nil {
		t.Fatal(err)
	}
	if len(hub.stored) != 0 || len(hub.commit) != 2 {
		t.Fatalf("stored %d objects, %d commit ops", len(hub.stored), len(hub.commit))
	}
}

func TestErrorsSurface(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"Invalid credentials"}`, http.StatusUnauthorized)
	}))
	defer srv.Close()
	c := &Client{Endpoint: srv.URL, Token: "bad"}
	err := c.CreateRepo(t.Context(), "u/r", true)
	if err == nil || !strings.Contains(err.Error(), "401") || !strings.Contains(err.Error(), "Invalid credentials") {
		t.Fatalf("%v", err)
	}
	if err := c.UploadDir(t.Context(), "u/r", t.TempDir(), "m"); err == nil {
		t.Fatal("empty directory must not upload")
	}
}
