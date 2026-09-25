package decide

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

// DefaultModelRepo is the Hugging Face repository the model is pulled from
// when none is given (see DownloadModel). Override with $DECIDE_MODEL_REPO.
const DefaultModelRepo = "jmwri/decide"

var modelFiles = []string{"decide.json", "config.json", "tokenizer.json", "model.safetensors"}

// CacheDir is where downloaded weights live: $DECIDE_CACHE_DIR, else the user cache dir.
func CacheDir() string {
	if d := os.Getenv("DECIDE_CACHE_DIR"); d != "" {
		return d
	}
	if d, err := os.UserCacheDir(); err == nil {
		return filepath.Join(d, "decide")
	}
	return filepath.Join(os.TempDir(), "decide")
}

func cachedModelDir() string { return filepath.Join(CacheDir(), "models", "decide") }

// DownloadModel fetches a model bundle from a Hugging Face model repository
// (default: $DECIDE_MODEL_REPO, else DefaultModelRepo) into dir (default: the
// user cache) unless the files are already present, and returns the
// directory. Interrupted downloads resume.
func DownloadModel(ctx context.Context, repo, dir string, progress func(format string, args ...any)) (string, error) {
	if repo == "" {
		repo = os.Getenv("DECIDE_MODEL_REPO")
	}
	if repo == "" {
		repo = DefaultModelRepo
	}
	if repo == "" {
		return "", fmt.Errorf("decide: no model found. Point DECIDE_MODEL_DIR at a model bundle directory, " +
			"or set DECIDE_MODEL_REPO (a Hugging Face repo) and run `decide pull`")
	}
	if dir == "" {
		dir = cachedModelDir()
	}
	if progress == nil {
		progress = func(string, ...any) {}
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	base := os.Getenv("HF_ENDPOINT")
	if base == "" {
		base = "https://huggingface.co"
	}
	for _, name := range modelFiles {
		dst := filepath.Join(dir, name)
		if _, err := os.Stat(dst); err == nil {
			continue
		}
		url := fmt.Sprintf("%s/%s/resolve/main/%s", base, repo, name)
		progress("downloading %s", url)
		if err := download(ctx, url, dst, progress); err != nil {
			return "", fmt.Errorf("decide: downloading %s: %w", name, err)
		}
	}
	return dir, nil
}

func download(ctx context.Context, url, dst string, progress func(string, ...any)) error {
	part := dst + ".part"
	var have int64
	if st, err := os.Stat(part); err == nil {
		have = st.Size()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	if tok := os.Getenv("HF_TOKEN"); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	if have > 0 {
		req.Header.Set("Range", "bytes="+strconv.FormatInt(have, 10)+"-")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	flags := os.O_CREATE | os.O_WRONLY
	switch resp.StatusCode {
	case http.StatusOK:
		flags |= os.O_TRUNC
		have = 0
	case http.StatusPartialContent:
		flags |= os.O_APPEND
	case http.StatusRequestedRangeNotSatisfiable:
		return os.Rename(part, dst)
	default:
		return fmt.Errorf("HTTP %s", resp.Status)
	}
	f, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return err
	}
	total := have + resp.ContentLength
	buf := make([]byte, 1<<20)
	last := time.Now()
	done := have
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := f.Write(buf[:n]); werr != nil {
				f.Close()
				return werr
			}
			done += int64(n)
			if resp.ContentLength > 0 && time.Since(last) > 5*time.Second {
				progress("  %d / %d MB", done>>20, total>>20)
				last = time.Now()
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			f.Close()
			return rerr
		}
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(part, dst)
}
