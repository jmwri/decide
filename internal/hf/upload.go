// Package hf uploads a model directory to the Hugging Face hub using its
// HTTP API (repo creation, the git-LFS batch protocol for large files, and the
// commit endpoint), with no Python or git dependency.
package hf

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Client talks to a hub endpoint.
type Client struct {
	Endpoint string // default https://huggingface.co
	Token    string
	HTTP     *http.Client
	Log      func(format string, args ...any)
}

func (c *Client) endpoint() string {
	if c.Endpoint != "" {
		return strings.TrimRight(c.Endpoint, "/")
	}
	return "https://huggingface.co"
}

func (c *Client) client() *http.Client {
	if c.HTTP != nil {
		return c.HTTP
	}
	return http.DefaultClient
}

func (c *Client) logf(format string, args ...any) {
	if c.Log != nil {
		c.Log(format, args...)
	}
}

func (c *Client) do(req *http.Request, wantStatus ...int) ([]byte, int, error) {
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
	resp, err := c.client().Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	for _, s := range wantStatus {
		if resp.StatusCode == s {
			return body, resp.StatusCode, nil
		}
	}
	return body, resp.StatusCode, fmt.Errorf("%s %s: %s: %s", req.Method, req.URL.Path, resp.Status, strings.TrimSpace(string(body)))
}

// CreateRepo creates a model repository (an existing one is fine).
func (c *Client) CreateRepo(ctx context.Context, repo string, private bool) error {
	name := repo
	body := map[string]any{"type": "model", "private": private}
	if org, n, ok := strings.Cut(repo, "/"); ok {
		body["name"], body["organization"] = n, org
		name = n
	} else {
		body["name"] = name
	}
	b, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint()+"/api/repos/create", bytes.NewReader(b))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	_, status, err := c.do(req, http.StatusOK, http.StatusConflict)
	if err != nil {
		return err
	}
	if status == http.StatusConflict {
		c.logf("repo %s already exists", repo)
	}
	return nil
}

// lfsThreshold is the size above which a file goes through LFS. The hub also
// requires LFS for weights regardless of size.
const lfsThreshold = 10 << 20

func isLFS(name string, size int64) bool {
	return size >= lfsThreshold || strings.HasSuffix(name, ".safetensors") || strings.HasSuffix(name, ".bin")
}

type fileEntry struct {
	path string // path inside the repo
	src  string
	size int64
	lfs  bool
	oid  string
}

// UploadDir commits every file of dir (recursively) to the repo's main branch.
func (c *Client) UploadDir(ctx context.Context, repo, dir, message string) error {
	var files []fileEntry
	err := filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		rel, _ := filepath.Rel(dir, p)
		rel = filepath.ToSlash(rel)
		if strings.HasPrefix(rel, ".") || strings.HasSuffix(rel, ".tmp") || strings.HasSuffix(rel, ".part") {
			return nil
		}
		st, err := d.Info()
		if err != nil {
			return err
		}
		files = append(files, fileEntry{path: rel, src: p, size: st.Size(), lfs: isLFS(rel, st.Size())})
		return nil
	})
	if err != nil {
		return err
	}
	if len(files) == 0 {
		return fmt.Errorf("hf: nothing to upload in %s", dir)
	}
	sort.Slice(files, func(i, j int) bool { return files[i].path < files[j].path })

	for i := range files {
		f := &files[i]
		if !f.lfs {
			continue
		}
		oid, err := sha256File(f.src)
		if err != nil {
			return err
		}
		f.oid = oid
		if err := c.uploadLFS(ctx, repo, f); err != nil {
			return fmt.Errorf("uploading %s: %w", f.path, err)
		}
	}
	return c.commit(ctx, repo, files, message)
}

func sha256File(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

type lfsAction struct {
	Href   string            `json:"href"`
	Header map[string]string `json:"header"`
}

func (c *Client) uploadLFS(ctx context.Context, repo string, f *fileEntry) error {
	batch, _ := json.Marshal(map[string]any{
		"operation": "upload", "transfers": []string{"basic"}, "hash_algo": "sha256",
		"objects": []map[string]any{{"oid": f.oid, "size": f.size}},
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("%s/%s.git/info/lfs/objects/batch", c.endpoint(), repo), bytes.NewReader(batch))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/vnd.git-lfs+json")
	req.Header.Set("Accept", "application/vnd.git-lfs+json")
	body, _, err := c.do(req, http.StatusOK)
	if err != nil {
		return err
	}
	var resp struct {
		Objects []struct {
			Error   *struct{ Message string } `json:"error"`
			Actions map[string]lfsAction      `json:"actions"`
		} `json:"objects"`
	}
	if err := json.Unmarshal(body, &resp); err != nil || len(resp.Objects) != 1 {
		return fmt.Errorf("unexpected LFS batch response: %s", strings.TrimSpace(string(body)))
	}
	obj := resp.Objects[0]
	if obj.Error != nil {
		return fmt.Errorf("LFS: %s", obj.Error.Message)
	}
	up, ok := obj.Actions["upload"]
	if !ok {
		c.logf("%s already on the hub", f.path) // deduplicated by content hash
		return nil
	}
	if up.Header["chunk_size"] != "" {
		return fmt.Errorf("the hub asked for a multipart upload (file too large for this client)")
	}
	c.logf("uploading %s (%d MB)", f.path, f.size>>20)
	fh, err := os.Open(f.src)
	if err != nil {
		return err
	}
	defer fh.Close()
	put, err := http.NewRequestWithContext(ctx, http.MethodPut, up.Href, fh)
	if err != nil {
		return err
	}
	put.ContentLength = f.size
	for k, v := range up.Header {
		put.Header.Set(k, v)
	}
	// The pre-signed URL carries its own credentials; do not forward the hub token.
	resp2, err := c.client().Do(put)
	if err != nil {
		return err
	}
	rb, _ := io.ReadAll(resp2.Body)
	resp2.Body.Close()
	if resp2.StatusCode/100 != 2 {
		return fmt.Errorf("PUT %s: %s: %s", f.path, resp2.Status, strings.TrimSpace(string(rb)))
	}
	if v, ok := obj.Actions["verify"]; ok {
		vb, _ := json.Marshal(map[string]any{"oid": f.oid, "size": f.size})
		vr, err := http.NewRequestWithContext(ctx, http.MethodPost, v.Href, bytes.NewReader(vb))
		if err != nil {
			return err
		}
		vr.Header.Set("Content-Type", "application/vnd.git-lfs+json")
		vr.Header.Set("Accept", "application/vnd.git-lfs+json")
		for k, val := range v.Header {
			vr.Header.Set(k, val)
		}
		if _, _, err := c.do(vr, http.StatusOK); err != nil {
			return err
		}
	}
	return nil
}

func (c *Client) commit(ctx context.Context, repo string, files []fileEntry, message string) error {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(map[string]any{"key": "header", "value": map[string]any{"summary": message, "description": ""}})
	for _, f := range files {
		if f.lfs {
			_ = enc.Encode(map[string]any{"key": "lfsFile", "value": map[string]any{"path": f.path, "algo": "sha256", "oid": f.oid, "size": f.size}})
			continue
		}
		b, err := os.ReadFile(f.src)
		if err != nil {
			return err
		}
		_ = enc.Encode(map[string]any{"key": "file", "value": map[string]any{"path": f.path, "encoding": "base64", "content": base64.StdEncoding.EncodeToString(b)}})
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("%s/api/models/%s/commit/main", c.endpoint(), repo), &buf)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/x-ndjson")
	if _, _, err := c.do(req, http.StatusOK); err != nil {
		return err
	}
	c.logf("committed %d files to %s", len(files), repo)
	return nil
}
