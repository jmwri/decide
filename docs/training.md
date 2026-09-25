# Training a Decide model

Everything below is one Go binary, `decide-train`, and works on Windows, Linux and macOS. Training needs no
Python, PyTorch or CUDA Toolkit. It runs on the CPU, or much faster on an NVIDIA GPU using only the GPU driver.

```
go install github.com/jmwri/decide/cmd/decide-train@latest     # or: go build ./cmd/decide-train
```

## 1. Get the base model

Decide fine-tunes [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) (Apache-2.0). Download three
files into one directory (called `base` below):

```
config.json   tokenizer.json   model.safetensors
```

from https://huggingface.co/answerdotai/ModernBERT-base/tree/main. Only the tokenizer, the architecture config and the
weights are used.

## 2. Build the corpus

```
decide-train data --dir corpus
```

Downloads the public datasets listed in `internal/data/tasks.go` from the Hugging Face hub (as parquet, cached under
`corpus/raw`) and writes `train.jsonl`, `val.jsonl`, `ood.jsonl` and `manifest.json`.

- **train / val**: one example per source row, from the dataset's own train and validation splits.
- **ood**: tasks marked `Holdout` are never trained on; they measure zero-shot behaviour.
- Training rows whose text also occurs in val or ood are dropped.
- `manifest.json` records every task's row counts and its license as declared on the hub. **Read it before you
  redistribute weights.**
- A task that cannot be downloaded is skipped with a warning (`--strict` makes it an error); `--only a,b` builds a
  subset; `--scale 0.5` halves every task's cap.

Each example is *state + instructions + options + gold*. Classification tasks appear both as multiple choice and as
yes/no verification questions; ordinal tasks carry soft targets on neighbouring levels; label spaces larger than 10 are
sub-sampled at training time to a random subset that always contains the answer.

To add a dataset, register a `Task` in `tasks.go` with a converter from a parquet row to `Example`s; the registry test
checks the shape, and `data.Validate` checks every generated example.

## 3. Train

```
decide-train train --data corpus --base base --out run1 --epochs 2
```

| flag | default | meaning |
| :-- | :-- | :-- |
| `--epochs`, `--max-examples` | 1, all | how long to train / cap the examples per epoch |
| `--batch` | 32 | examples per optimizer step |
| `--lr`, `--head-lr`, `--wd` | 4e-5, 3e-4, 0.01 | encoder LR, scorer-head LR, weight decay (linear warm-up 5%, decay to 5%) |
| `--gpu` | auto | `auto` uses an NVIDIA GPU when present, `on` requires one, `off` forces the CPU |
| `--token-budget` | 1200 CPU / 4000 GPU | tokens per forward/backward micro-batch |
| `--init FILE` | | start from a previously trained `model.safetensors` instead of the base model |
| `--train-from N`, `--freeze-embeddings` | 0 | freeze the lowest N layers / the token embeddings (cheaper, usually worse) |
| `--eval-every`, `--save-every` | 250 | steps between validation runs / checkpoints |
| `--resume` | | continue from the newest checkpoint in `--out` |
| `--exclude a,b` | | leave tasks out of training |

The loss is cross-entropy over the options (soft targets for ordinal tasks) with AdamW and gradient clipping at 1.0.
`--out` receives `train.log`, `eval.jsonl` (a validation report per evaluation), rolling checkpoints
(`ckpt-*/`: weights, Adam moments and progress, so a resumed run continues exactly) and finally `model.safetensors`.
Ctrl-C saves a checkpoint before exiting.

**Speed.** On a Ryzen 7 5800X3D the CPU path trains at about 450 tokens/s; on an RTX 3090 Ti the GPU path reaches
6,000-9,000 tokens/s, so a pass over a 300k-example corpus takes about an hour instead of half a day.

### How the GPU path works

`internal/cuda` loads `nvcuda.dll` / `libcuda.so` directly (via `purego`, no cgo). `internal/cuda/kernels/kernels.cu`
holds the kernels (a tiled fp32 matrix multiply, LayerNorm, masked attention softmax, RoPE, GEGLU, AdamW and their
backward passes). `cmd/ptxgen` compiles them to `internal/cuda/kernels.ptx` with NVRTC, which is embedded in the binary
and JIT-compiled by the driver for the installed GPU. You only need NVRTC (the `nvidia-cuda-nvrtc` wheel, or a CUDA
Toolkit) if you change the kernels:

```
NVRTC_DLL=/path/to/nvrtc64_120_0.dll go run ./cmd/ptxgen
```

`internal/gpunn` runs the encoder (weights, gradients and Adam state stay on the card); the final norm, scorer head
and loss run on the CPU because they touch only a few rows per example. Requirements: an NVIDIA GPU of compute
capability 8.0 or newer and enough memory for the batch (about 10 GB for 5,000-token micro-batches).

## 4. Evaluate

```
decide-train eval --model run1/model.safetensors --base base --data corpus --split val    # or --split ood
```

Prints accuracy, negative log-likelihood and expected calibration error (ECE) per task, per kind and overall, plus a
macro average over tasks. Large option pools are sub-sampled deterministically to at most 10 options.

## 5. Calibrate and export a bundle

```
decide-train export --model run1/model.safetensors --base base --data corpus --out bundle --id decide-0.2.0
```

Fits one softmax temperature per question kind on the validation split (temperature changes how confident a
probability claims to be, never which option wins), scores the held-out tasks, measures order invariance, and writes a
model bundle: `model.safetensors`, `config.json`, `tokenizer.json` and `decide.json` (id, calibration, provenance,
evaluation reports). `decide` and `decide serve` load a bundle from `--model-dir` or `$DECIDE_MODEL_DIR`.

## 6. Publish (optional)

```
decide-train card    --bundle bundle --repo <user>/decide           # writes bundle/README.md (model card)
HF_TOKEN=hf_... decide-train publish --bundle bundle --repo <user>/decide --yes
```

`publish` creates the repository (add `--private` for a private one) and uploads the files with the hub's HTTP API. It
refuses to run without `--yes` because publishing is not reversible.

## How the model reads a question

```
<instructions> <state> [SEP] [MASK] option 0 [MASK] option 1 ...
```

One encoder pass; a small head scores the hidden state at each `[MASK]`. With *independent options* (always on) every
option attends only to the shared prefix and to itself, and its position ids restart right after the prefix, so an option's
score depends on *(premise, that option)* alone. The bundle's `independent_options` flag records this and `decide-train
export` measures the effect (the maximum logit change when options are permuted is around 1e-6).

## How it is verified

- `internal/nn` (the CPU forward and backward pass) is compared with PyTorch autograd: `internal/nn/testdata/gen_ref.py`
  builds a tiny random ModernBERT with `transformers`, runs a batch in both attention modes, and stores logits and every
  parameter's gradient; `go test ./internal/nn` checks against them. On real ModernBERT-base weights it is compared with
  `transformers` hidden states (see `base_test.go`, `DECIDE_BASE_DIR` / `DECIDE_BASE_REF`).
- `internal/gpunn` is compared with `internal/nn` tensor by tensor (forward, all gradients, an AdamW step) on the tiny model
  and on the real base weights.
- `internal/train` has a toy end-to-end test (train, checkpoint, resume, export, reload) that runs on the CPU and, when a
  GPU is present, on the GPU.

Tests that need weights or a GPU skip themselves when those are absent.
