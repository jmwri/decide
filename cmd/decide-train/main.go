// Command decide-train builds the training corpus for, trains, evaluates and
// exports the Decide model. Everything runs in pure Go on the CPU.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"

	"github.com/jmwri/decide/internal/bundle"
	"github.com/jmwri/decide/internal/data"
	"github.com/jmwri/decide/internal/hf"
	"github.com/jmwri/decide/internal/nn"
	"github.com/jmwri/decide/internal/tokenizer"
	"github.com/jmwri/decide/internal/train"
)

const usage = `decide-train: build, train, evaluate and publish the Decide model (pure Go; CPU or NVIDIA GPU).

  decide-train data    --dir DIR [--only a,b] [--scale X]   download public datasets, build train/val/ood JSONL
  decide-train train   --data DIR --base DIR --out DIR      fine-tune ModernBERT-base (--gpu auto|on|off, --resume,
                       [--epochs N] [--init MODEL] ...      --init to continue from a trained model)
  decide-train eval    --model FILE --base DIR --data DIR   accuracy / calibration report on --split val|ood
  decide-train export  --model FILE --base DIR --data DIR --out DIR --id decide-0.2.0
                                                            calibrate and write a model bundle for inference
  decide-train card    --bundle DIR --repo USER/NAME        write README.md (model card) into a bundle
  decide-train publish --bundle DIR --repo USER/NAME --yes  upload a bundle to the Hugging Face hub ($HF_TOKEN)

See docs/training.md.
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	var err error
	switch os.Args[1] {
	case "data":
		err = runData(ctx, os.Args[2:])
	case "train":
		err = runTrain(ctx, os.Args[2:])
	case "eval":
		err = runEval(ctx, os.Args[2:])
	case "card":
		err = runCard(os.Args[2:])
	case "publish":
		err = runPublish(ctx, os.Args[2:])
	case "export":
		err = runExport(ctx, os.Args[2:])
	case "help", "-h", "--help":
		fmt.Print(usage)
	default:
		fmt.Fprintf(os.Stderr, "decide-train: unknown command %q\n\n%s", os.Args[1], usage)
		os.Exit(2)
	}
	if err != nil && err != flag.ErrHelp {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
}

func logf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "%s "+format+"\n", append([]any{timestamp()}, args...)...)
}

func runData(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("data", flag.ContinueOnError)
	dir := fs.String("dir", "corpus", "output directory")
	only := fs.String("only", "", "comma-separated task names (default: all)")
	seed := fs.Int64("seed", 1, "random seed")
	scale := fs.Float64("scale", 1, "multiplier for per-task train caps")
	strict := fs.Bool("strict", false, "fail when an optional task cannot be downloaded")
	if err := fs.Parse(args); err != nil {
		return err
	}
	cfg := data.BuildConfig{Dir: *dir, Seed: *seed, CapScale: *scale, Strict: *strict, Log: logf}
	if *only != "" {
		cfg.Only = strings.Split(*only, ",")
	}
	m, err := data.Build(ctx, cfg)
	if err != nil {
		return err
	}
	fmt.Printf("train %d, val %d, ood %d examples in %s\n", m.Train, m.Val, m.OOD, *dir)
	return nil
}

func runTrain(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("train", flag.ContinueOnError)
	var c train.Config
	fs.StringVar(&c.DataDir, "data", "corpus", "corpus directory")
	fs.StringVar(&c.BaseDir, "base", "", "ModernBERT-base directory (config.json, model.safetensors, tokenizer.json)")
	fs.StringVar(&c.OutDir, "out", "run", "run directory")
	fs.Float64Var(&c.Epochs, "epochs", 1, "passes over the training set")
	fs.IntVar(&c.MaxExamples, "max-examples", 0, "use only this many training examples per epoch (0 = all)")
	fs.IntVar(&c.BatchExamples, "batch", 32, "examples per optimizer step")
	fs.IntVar(&c.TokenBudget, "token-budget", 0, "tokens per micro-batch (default 1200 on the CPU, 4000 on a GPU)")
	fs.StringVar(&c.GPU, "gpu", "auto", "use an NVIDIA GPU: auto, on or off")
	fs.IntVar(&c.MaxLen, "max-len", 384, "longest packed sequence")
	fs.IntVar(&c.KMax, "kmax", 10, "most options shown per example")
	fs.Float64Var(&c.LR, "lr", 4e-5, "encoder learning rate")
	fs.Float64Var(&c.HeadLR, "head-lr", 3e-4, "scorer head learning rate")
	fs.Float64Var(&c.WeightDecay, "wd", 0.01, "weight decay")
	fs.StringVar(&c.InitModel, "init", "", "start from this trained model.safetensors instead of the base model")
	fs.IntVar(&c.TrainFrom, "train-from", 0, "freeze encoder layers below this index")
	noEmb := fs.Bool("freeze-embeddings", false, "do not train the token embeddings")
	fs.IntVar(&c.EvalEvery, "eval-every", 250, "steps between validation runs")
	fs.IntVar(&c.EvalPerTask, "eval-per-task", 30, "validation examples per task during training")
	fs.IntVar(&c.SaveEvery, "save-every", 250, "steps between checkpoints")
	fs.BoolVar(&c.Resume, "resume", false, "continue from the latest checkpoint in --out")
	fs.Int64Var(&c.Seed, "seed", 1, "random seed")
	exclude := fs.String("exclude", "", "comma-separated tasks to leave out of training")
	threads := fs.Int("threads", 0, "CPU threads (default: all)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if c.BaseDir == "" {
		return fmt.Errorf("--base is required")
	}
	if *threads > 0 {
		runtime.GOMAXPROCS(*threads)
	}
	c.TrainEmb = !*noEmb
	c.Independent = true
	if *exclude != "" {
		c.Exclude = strings.Split(*exclude, ",")
	}
	c.Log = logf
	return train.Run(ctx, c)
}

func runEval(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("eval", flag.ContinueOnError)
	modelPath := fs.String("model", "", "model.safetensors written by training")
	baseDir := fs.String("base", "", "ModernBERT-base directory (for config.json and tokenizer.json)")
	dataDir := fs.String("data", "corpus", "corpus directory")
	split := fs.String("split", "val", "val or ood")
	perTask := fs.Int("per-task", 0, "examples per task (0 = all)")
	temp := fs.Float64("temp", 1, "softmax temperature")
	kmax := fs.Int("kmax", 10, "most options shown per example")
	gpu := fs.String("gpu", "auto", "use an NVIDIA GPU: auto, on or off")
	if err := fs.Parse(args); err != nil {
		return err
	}
	cfg, err := nn.LoadConfig(filepath.Join(*baseDir, "config.json"))
	if err != nil {
		return err
	}
	m := nn.New(cfg)
	if err := m.LoadWeights(*modelPath); err != nil {
		return err
	}
	tok, err := tokenizer.Load(filepath.Join(*baseDir, "tokenizer.json"))
	if err != nil {
		return err
	}
	exs, err := data.ReadJSONL(filepath.Join(*dataDir, *split+".jsonl"))
	if err != nil {
		return err
	}
	items := train.PrepareEval(tok, exs, *perTask, *kmax, 384, true)
	logf("evaluating %d examples", len(items))
	fwd, closeFn, where, err := train.NewInferer(m, *gpu, logf)
	if err != nil {
		return err
	}
	defer closeFn()
	logf("scoring on %s", where)
	logits, err := train.PredictWith(fwd, items, 4000)
	if err != nil {
		return err
	}
	fmt.Print(train.Score(items, logits, *temp).String())
	return nil
}

func runExport(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("export", flag.ContinueOnError)
	var c train.ExportConfig
	fs.StringVar(&c.ModelPath, "model", "", "model.safetensors written by training")
	fs.StringVar(&c.BaseDir, "base", "", "ModernBERT-base directory")
	fs.StringVar(&c.DataDir, "data", "corpus", "corpus directory")
	fs.StringVar(&c.OutDir, "out", "", "bundle directory to write")
	fs.StringVar(&c.ModelID, "id", "decide-0.1.0", "model id stamped on responses")
	fs.IntVar(&c.KMax, "kmax", 10, "most options shown per example")
	fs.StringVar(&c.GPU, "gpu", "auto", "use an NVIDIA GPU: auto, on or off")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if c.ModelPath == "" || c.BaseDir == "" || c.OutDir == "" {
		return fmt.Errorf("--model, --base and --out are required")
	}
	c.Log = logf
	return train.Export(c)
}

func writeCard(dir, repo string) error {
	b, err := bundle.Load(dir)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "README.md"), []byte(train.ModelCard(b.Meta, repo)), 0o644)
}

func runCard(args []string) error {
	fs := flag.NewFlagSet("card", flag.ContinueOnError)
	dir := fs.String("bundle", "", "bundle directory")
	repo := fs.String("repo", "USER/decide", "hub repo the card will live in")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *dir == "" {
		return fmt.Errorf("--bundle is required")
	}
	if err := writeCard(*dir, *repo); err != nil {
		return err
	}
	fmt.Println(filepath.Join(*dir, "README.md"))
	return nil
}

func runPublish(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("publish", flag.ContinueOnError)
	dir := fs.String("bundle", "", "bundle directory")
	repo := fs.String("repo", "", "hub repo, USER/NAME")
	private := fs.Bool("private", false, "create the repo as private")
	yes := fs.Bool("yes", false, "confirm that you want to upload (publishing is not reversible)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *dir == "" || *repo == "" {
		return fmt.Errorf("--bundle and --repo are required")
	}
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		return fmt.Errorf("set HF_TOKEN to a Hugging Face access token with write permission")
	}
	if !*yes {
		return fmt.Errorf("this uploads %s to https://huggingface.co/%s; re-run with --yes to confirm", *dir, *repo)
	}
	if err := writeCard(*dir, *repo); err != nil {
		return err
	}
	c := &hf.Client{Token: token, Endpoint: os.Getenv("HF_ENDPOINT"), Log: logf}
	if err := c.CreateRepo(ctx, *repo, *private); err != nil {
		return err
	}
	if err := c.UploadDir(ctx, *repo, *dir, "Upload model"); err != nil {
		return err
	}
	fmt.Printf("https://huggingface.co/%s\n", *repo)
	return nil
}
