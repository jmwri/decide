// Command decide is the CLI for the Decide System One decision model.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"slices"
	"strings"
	"syscall"
	"time"

	"github.com/jmwri/decide"
	"github.com/jmwri/decide/server"
)

const usage = `Decide - System One decision model.

Usage:
  decide serve   [--host H] [--port N] [--model M] [--model-dir DIR]   start the HTTP server
  decide choose  TEXT -c a,b,c [-i INSTRUCTIONS]                        classify among choices
  decide judge   TEXT -i QUESTION [--pos TEXT] [--neg TEXT]             yes/no probability
  decide rate    TEXT -l low,mid,high [-i INSTRUCTIONS]                 rate on an ordered scale
  decide eval    REQUEST.json                                           evaluate a request file
  decide pull    [--repo HF_REPO] [--dir DIR]                           download the model bundle
  decide version

Common flags: --model-dir DIR (weights directory), --device cpu.
Environment: DECIDE_MODEL_DIR, DECIDE_CACHE_DIR, DECIDE_BASE_URL, DECIDE_API_KEY, DECIDE_CORS_ORIGINS.
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	cmd, args := os.Args[1], os.Args[2:]
	var err error
	switch cmd {
	case "serve":
		err = runServe(ctx, args)
	case "choose":
		err = runChoose(ctx, args)
	case "judge":
		err = runJudge(ctx, args)
	case "rate":
		err = runRate(ctx, args)
	case "eval":
		err = runEval(ctx, args)
	case "pull":
		err = runPull(ctx, args)
	case "version", "--version", "-version":
		fmt.Printf("decide %s (model %s, %s)\n", decide.SDKVersion, decide.ModelID, runtime.Version())
	case "help", "-h", "--help", "-help":
		fmt.Print(usage)
	default:
		fmt.Fprintf(os.Stderr, "decide: unknown command %q\n\n%s", cmd, usage)
		os.Exit(2)
	}
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return
		}
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
}

// parseInterspersed parses flags that may follow positional arguments, as click does.
func parseInterspersed(fs *flag.FlagSet, args []string) ([]string, error) {
	var pos []string
	for {
		if err := fs.Parse(args); err != nil {
			return nil, err
		}
		args = fs.Args()
		if len(args) == 0 {
			return pos, nil
		}
		pos = append(pos, args[0])
		args = args[1:]
	}
}

type common struct {
	checkpoint string
	device     string
}

func (c *common) register(fs *flag.FlagSet) {
	fs.StringVar(&c.checkpoint, "model-dir", "", "model bundle directory (decide.json, model.safetensors, config.json, tokenizer.json)")
	fs.StringVar(&c.device, "device", "auto", "compute device: 'auto' or 'cpu' (this build is CPU-only)")
}

// install selects the evaluator: a remote one if DECIDE_BASE_URL is set, else a local model.
func (c *common) install() error {
	switch strings.ToLower(c.device) {
	case "", "auto", "cpu":
	default:
		return fmt.Errorf("device %q is not supported: the Go build runs on the CPU", c.device)
	}
	if os.Getenv("DECIDE_BASE_URL") != "" && c.checkpoint == "" {
		return nil
	}
	l := decide.NewLocal(c.checkpoint)
	l.Progress = func(m string) { fmt.Fprintln(os.Stderr, "[decide]", m) }
	decide.SetDefault(l)
	return nil
}

func printJSON(v any) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}

func splitList(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func newFlagSet(name string) *flag.FlagSet {
	fs := flag.NewFlagSet("decide "+name, flag.ContinueOnError)
	return fs
}

func runServe(ctx context.Context, args []string) error {
	fs := newFlagSet("serve")
	var c common
	c.register(fs)
	host := fs.String("host", "0.0.0.0", "host interface to bind on")
	port := fs.Int("port", 8000, "port to listen on")
	model := fs.String("model", "decide-"+decide.Version, "model version to load")
	if _, err := parseInterspersed(fs, args); err != nil {
		return err
	}
	if !slices.Contains(decide.ModelAliases, strings.ToLower(strings.TrimSpace(*model))) {
		return fmt.Errorf("unknown model %q; Decide %s is the only model, accepted aliases: %s",
			*model, decide.Version, strings.Join(decide.ModelAliases, ", "))
	}
	if err := c.install(); err != nil {
		return err
	}
	if l, ok := decide.Default().Evaluator.(*decide.Local); ok {
		if err := l.Load(ctx); err != nil {
			return err
		}
	}
	origins := splitList(os.Getenv("DECIDE_CORS_ORIGINS"))
	addr := fmt.Sprintf("%s:%d", *host, *port)
	srv := &http.Server{
		Addr:              addr,
		Handler:           server.New(decide.Default(), server.Config{CORSOrigins: origins}),
		ReadHeaderTimeout: 10 * time.Second,
	}
	fmt.Fprintf(os.Stderr, "Starting Decide server [%s on CPU x%d] on http://%s\n", *model, runtime.GOMAXPROCS(0), addr)
	errc := make(chan error, 1)
	go func() { errc <- srv.ListenAndServe() }()
	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
		shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		return srv.Shutdown(shutdown)
	}
}

func runChoose(ctx context.Context, args []string) error {
	fs := newFlagSet("choose")
	var c common
	c.register(fs)
	choices := fs.String("c", "", "comma-separated choices (e.g. 'billing,bug_report,feature_request')")
	fs.StringVar(choices, "choices", "", "alias for -c")
	instr := fs.String("i", "Which option best describes the input?", "instructions for classification")
	fs.StringVar(instr, "instructions", "Which option best describes the input?", "alias for -i")
	pos, err := parseInterspersed(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 {
		return errors.New("choose takes exactly one TEXT argument")
	}
	opts, err := decide.Choices(splitList(*choices)...)
	if err != nil {
		return err
	}
	if len(opts) == 0 {
		return errors.New("at least one choice must be provided (-c)")
	}
	if err := c.install(); err != nil {
		return err
	}
	ans, err := decide.Decide(ctx, pos[0], opts, *instr)
	if err != nil {
		return err
	}
	return printJSON(map[string]any{"choice": ans.Choice, "confidence": ans.Confidence, "probabilities": ans.Probabilities})
}

func runJudge(ctx context.Context, args []string) error {
	fs := newFlagSet("judge")
	var c common
	c.register(fs)
	instr := fs.String("i", "", "boolean judgment question (e.g. 'Is the server down?')")
	fs.StringVar(instr, "instructions", "", "alias for -i")
	posC := fs.String("pos", "", "explicit criteria description for the True condition")
	negC := fs.String("neg", "", "explicit criteria description for the False condition")
	pos, err := parseInterspersed(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 {
		return errors.New("judge takes exactly one TEXT argument")
	}
	if *instr == "" {
		return errors.New("-i/--instructions is required")
	}
	if err := c.install(); err != nil {
		return err
	}
	var crit *decide.NoulCriteria
	if *posC != "" || *negC != "" {
		crit = &decide.NoulCriteria{True: *posC, False: *negC}
	}
	p, err := decide.Judge(ctx, pos[0], *instr, crit)
	if err != nil {
		return err
	}
	return printJSON(map[string]any{"type": "noul", "instructions": *instr, "noul": p})
}

func runRate(ctx context.Context, args []string) error {
	fs := newFlagSet("rate")
	var c common
	c.register(fs)
	levels := fs.String("l", "", "comma-separated descriptions of ordered levels from 0 to N-1")
	fs.StringVar(levels, "levels", "", "alias for -l")
	instr := fs.String("i", "Rate where the state falls on this scale:", "instructions for rating")
	fs.StringVar(instr, "instructions", "Rate where the state falls on this scale:", "alias for -i")
	pos, err := parseInterspersed(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 {
		return errors.New("rate takes exactly one TEXT argument")
	}
	list := splitList(*levels)
	if len(list) < 2 {
		return errors.New("at least two levels must be provided (-l)")
	}
	if err := c.install(); err != nil {
		return err
	}
	ans, err := decide.Rate(ctx, pos[0], decide.Levels(list...), *instr)
	if err != nil {
		return err
	}
	return printJSON(map[string]any{"type": "score", "score": ans.Score, "confidence": ans.Confidence, "legend": ans.Legend, "probabilities": ans.Probabilities})
}

func runEval(ctx context.Context, args []string) error {
	fs := newFlagSet("eval")
	var c common
	c.register(fs)
	pos, err := parseInterspersed(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 {
		return errors.New("eval takes exactly one REQUEST_FILE argument")
	}
	f, err := os.Open(pos[0])
	if err != nil {
		return fmt.Errorf("cannot read %s: %w", pos[0], err)
	}
	data, err := io.ReadAll(f)
	f.Close()
	if err != nil {
		return err
	}
	req, err := decide.ParseRequest(data)
	if err != nil {
		return fmt.Errorf("%s: %w", pos[0], err)
	}
	if err := c.install(); err != nil {
		return err
	}
	resp, err := decide.SystemOne(ctx, req.State, req.Questions)
	if err != nil {
		return err
	}
	return printJSON(resp)
}

func runPull(ctx context.Context, args []string) error {
	fs := newFlagSet("pull")
	dir := fs.String("dir", "", "destination directory (default: the user cache)")
	repo := fs.String("repo", "", "Hugging Face model repo (default: $DECIDE_MODEL_REPO)")
	if _, err := parseInterspersed(fs, args); err != nil {
		return err
	}
	out, err := decide.DownloadModel(ctx, *repo, *dir, func(f string, a ...any) { fmt.Fprintf(os.Stderr, "[decide] "+f+"\n", a...) })
	if err != nil {
		return err
	}
	fmt.Println(out)
	return nil
}
