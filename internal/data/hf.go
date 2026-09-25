package data

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/parquet-go/parquet-go"
)

// Row is one dataset row, decoded from parquet into plain Go values
// (strings, numbers, bools, []any, map[string]any).
type Row map[string]any

// Str returns a string field ("" if absent or not a string).
func (r Row) Str(k string) string {
	s, _ := r[k].(string)
	return s
}

// Int returns an integer field.
func (r Row) Int(k string) int {
	switch v := r[k].(type) {
	case int:
		return v
	case int8:
		return int(v)
	case int16:
		return int(v)
	case int32:
		return int(v)
	case int64:
		return int(v)
	case uint8:
		return int(v)
	case uint32:
		return int(v)
	case uint64:
		return int(v)
	case float32:
		return int(v)
	case float64:
		return int(v)
	case bool:
		if v {
			return 1
		}
	}
	return 0
}

// Float returns a float field.
func (r Row) Float(k string) float64 {
	switch v := r[k].(type) {
	case float32:
		return float64(v)
	case float64:
		return v
	}
	return float64(r.Int(k))
}

// Bool returns a bool field.
func (r Row) Bool(k string) bool {
	switch v := r[k].(type) {
	case bool:
		return v
	}
	return r.Int(k) != 0
}

// Strings returns a []string field.
func (r Row) Strings(k string) []string {
	l, _ := r[k].([]any)
	out := make([]string, 0, len(l))
	for _, e := range l {
		s, _ := e.(string)
		out = append(out, s)
	}
	return out
}

// Source names a hub dataset split.
type Source struct {
	Repo   string
	Config string // "" = "default"
	Split  string
}

func (s Source) key() string {
	cfg := s.Config
	if cfg == "" {
		cfg = "default"
	}
	return strings.NewReplacer("/", "__").Replace(s.Repo) + "/" + cfg + "/" + s.Split
}

// Fetcher downloads and reads parquet splits, caching them on disk.
type Fetcher struct {
	CacheDir string
	Endpoint string // default https://huggingface.co
	Token    string // optional HF token
	HTTP     *http.Client
	Log      func(format string, args ...any)
}

func (f *Fetcher) logf(format string, args ...any) {
	if f.Log != nil {
		f.Log(format, args...)
	}
}

func (f *Fetcher) endpoint() string {
	if f.Endpoint != "" {
		return f.Endpoint
	}
	return "https://huggingface.co"
}

func (f *Fetcher) client() *http.Client {
	if f.HTTP != nil {
		return f.HTTP
	}
	return http.DefaultClient
}

func (f *Fetcher) get(ctx context.Context, u string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	if f.Token != "" {
		req.Header.Set("Authorization", "Bearer "+f.Token)
	}
	resp, err := f.client().Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("GET %s: %s", u, resp.Status)
	}
	return resp, nil
}

// Files downloads (once) the parquet shards of a split and returns their paths.
// At most maxFiles shards are fetched (0 = all).
func (f *Fetcher) Files(ctx context.Context, s Source, maxFiles int) ([]string, error) {
	dir := filepath.Join(f.CacheDir, s.key())
	if entries, err := filepath.Glob(filepath.Join(dir, "*.parquet")); err == nil && len(entries) > 0 && (maxFiles == 0 || len(entries) >= maxFiles) {
		return entries, nil
	}
	cfg := s.Config
	if cfg == "" {
		cfg = "default"
	}
	listURL := fmt.Sprintf("%s/api/datasets/%s/parquet/%s/%s", f.endpoint(), s.Repo, url.PathEscape(cfg), url.PathEscape(s.Split))
	resp, err := f.get(ctx, listURL)
	if err != nil {
		return nil, fmt.Errorf("listing parquet shards for %s: %w", s.key(), err)
	}
	var urls []string
	err = json.NewDecoder(resp.Body).Decode(&urls)
	resp.Body.Close()
	if err != nil || len(urls) == 0 {
		return nil, fmt.Errorf("no parquet shards for %s/%s/%s (the hub has not converted this dataset)", s.Repo, cfg, s.Split)
	}
	if maxFiles > 0 && len(urls) > maxFiles {
		urls = urls[:maxFiles]
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	var paths []string
	for i, u := range urls {
		dst := filepath.Join(dir, fmt.Sprintf("%04d.parquet", i))
		if _, err := os.Stat(dst); err != nil {
			f.logf("downloading %s", u)
			if err := f.download(ctx, u, dst); err != nil {
				return nil, err
			}
		}
		paths = append(paths, dst)
	}
	return paths, nil
}

func (f *Fetcher) download(ctx context.Context, u, dst string) error {
	resp, err := f.get(ctx, u)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	tmp := dst + ".part"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, resp.Body); err != nil {
		out.Close()
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, dst)
}

// Meta is what a parquet file's embedded HuggingFace metadata says about columns.
type Meta struct {
	// Names maps a ClassLabel column to its label names.
	Names map[string][]string
}

// Read streams a split's rows. When limit > 0 it keeps a uniform random
// sample of at most limit rows (reservoir sampling) drawn from the first
// scanCap rows (0 = no scan cap), which bounds work on huge datasets.
func (f *Fetcher) Read(ctx context.Context, s Source, limit, scanCap, maxFiles int, rng *rand.Rand) ([]Row, Meta, error) {
	paths, err := f.Files(ctx, s, maxFiles)
	if err != nil {
		return nil, Meta{}, err
	}
	meta := Meta{Names: map[string][]string{}}
	var rows []Row
	seen := 0
	for _, p := range paths {
		fh, err := os.Open(p)
		if err != nil {
			return nil, meta, err
		}
		st, _ := fh.Stat()
		pf, err := parquet.OpenFile(fh, st.Size())
		if err != nil {
			fh.Close()
			return nil, meta, fmt.Errorf("%s: %w", p, err)
		}
		if hf, ok := pf.Lookup("huggingface"); ok {
			mergeNames(&meta, hf)
		}
		r := parquet.NewReader(pf)
		for {
			row := map[string]any{}
			if err := r.Read(&row); err != nil {
				if err == io.EOF {
					break
				}
				r.Close()
				fh.Close()
				return nil, meta, fmt.Errorf("%s: %w", p, err)
			}
			seen++
			switch {
			case limit <= 0 || len(rows) < limit:
				rows = append(rows, Row(row))
			default:
				if j := rng.Intn(seen); j < limit {
					rows[j] = Row(row)
				}
			}
			if scanCap > 0 && seen >= scanCap {
				break
			}
		}
		r.Close()
		fh.Close()
		if scanCap > 0 && seen >= scanCap {
			break
		}
	}
	return rows, meta, nil
}

func mergeNames(m *Meta, hf string) {
	var v struct {
		Info struct {
			Features map[string]struct {
				Names []string `json:"names"`
			} `json:"features"`
		} `json:"info"`
	}
	if json.Unmarshal([]byte(hf), &v) != nil {
		return
	}
	for col, f := range v.Info.Features {
		if len(f.Names) > 0 {
			m.Names[col] = f.Names
		}
	}
}
