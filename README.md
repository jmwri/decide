# Decide

A fast, local **decision model** in pure Go. Give it some state, an instruction and a set of options; it scores
every option in **one forward pass** and returns calibrated probabilities. No text generation, no Python, no cgo.

```go
ans, _ := decide.Decide(ctx, "Replication lag on cluster us-west-2 exceeded 45 seconds.",
    decide.Options{
        {Key: "infrastructure", Description: "Database, hardware, network, or server failures"},
        {Key: "billing", Description: "Invoices, payments, refunds, subscription queries"},
        {Key: "feature_request", Description: "Requests for new platform capabilities"},
    }, "Classify the root cause domain of this incident.")
fmt.Println(ans.Choice, ans.Confidence, ans.Probabilities)
```

Everything in this repository is written in Go and trained in Go: the model runtime (`internal/nn`, a
ModernBERT encoder with a hand-written backward pass on an AVX2 matrix-multiply kernel), the tokenizer, the data
pipeline, the trainer, the evaluator, the calibrator and the Hugging Face uploader. Decide's own model is
fine-tuned from [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) on public datasets.

## The three primitives

| Primitive | Returns |
| :-- | :-- |
| **Choice** | calibrated probability distribution over K options, plus a confidence `(K·p_max − 1)/(K − 1)` |
| **Noul** | `P(condition is true)` in [0, 1] |
| **Score** | expected level over an ordered scale of 2–10 levels, in `[0, K−1]` |

Any number of questions can be asked of the same state in one call. Option order never matters: each option
attends only to the shared premise and to itself, and its position ids restart after the premise, so permuting the
options permutes the scores and changes nothing else (guaranteed by the attention mask, and tested).

## Getting the model

Decide loads a *model bundle*: a directory holding `decide.json`, `model.safetensors`, `config.json` and
`tokenizer.json`. Either point at one, or pull one from the Hugging Face hub:

```
export DECIDE_MODEL_DIR=/path/to/bundle          # use a local bundle
decide pull                                      # or download jmwri/decide into the user cache
```

Resolution order: `NewLocal(dir)`, `$DECIDE_MODEL_DIR`, `./models/decide`, the download cache
(`$DECIDE_CACHE_DIR`), then a download from `$DECIDE_MODEL_REPO` (default `jmwri/decide`). To build your own bundle see [Training](#training).

## Library usage

```go
import "github.com/jmwri/decide"

ctx := context.Background()

// Choice
ans, err := decide.Decide(ctx, state, decide.Options{{Key: "a", Description: "..."}, {Key: "b"}}, "Which one?")

// Noul (yes/no probability), optionally with explicit criteria for true and false
p, err := decide.Judge(ctx, "Connection pool exhausted; handshakes timing out.",
    "Is this issue actively blocking customer operations?", nil)

// Score (ordered scale)
rating, err := decide.Rate(ctx, "Memory at 98% with frequent OOM kills.",
    decide.Levels("Nominal", "Degraded performance", "Imminent service termination"),
    "Assess system degradation level.")
```

### Several questions about one state

```go
state := decide.Object{ // ordered: keys are rendered to the model in this order
    {Key: "ticket_id", Value: "INC-4091"},
    {Key: "customer_tier", Value: "enterprise"},
    {Key: "message", Value: "Payment gateway reports timeout on charge authorizations. Urgent."},
}
resp, err := decide.SystemOne(ctx, state, decide.Questions{
    {ID: "intent", Question: decide.NewChoice("What is the nature of this ticket?", decide.Options{
        {Key: "payment_failure", Description: "Failures processing charges, gateway timeouts, card declines"},
        {Key: "access_issue", Description: "Login, SSO, authentication, or permission errors"},
    })},
    {ID: "is_urgent", Question: decide.NewNoul("Does the request require immediate SLA intervention?", nil)},
    {ID: "severity", Question: decide.NewScore("Rate the incident severity.", decide.Levels("Low", "Medium", "High", "Critical"))},
})
intent, _ := resp.Choice("intent")
urgent, _ := resp.Noul("is_urgent")
severity, _ := resp.Score("severity")
```

`state` may be a string, a `decide.Object` (ordered), a `map[string]any` (sorted key order), or anything that
marshals to JSON (struct fields keep declaration order).

### Presets and patterns

```go
resp, _ := decide.SystemOne(ctx, ticket, decide.TriagePreset()) // also EmailPreset, ModerationPreset, SecurityPreset

gate, _ := decide.ConfidenceGate(ctx, nil, payload, decide.TriagePreset(), 0.85) // gate.Automatic / gate.Escalate
decide.Route(ctx, nil, event, choice, map[string]func(*decide.ChoiceAnswer) error{...}, decide.RouteConfig{})
risk, _ := decide.CompositeScore(ctx, nil, telemetry, questions, map[string]float64{"is_threat": 3})
res, _ := decide.TwoStageChoice(ctx, nil, "Postgres replica lag", taxonomy, decide.TwoStageConfig{}) // >10 options
```

Passing `nil` as the evaluator uses `decide.Default()`.

### Where the model runs

Anything implementing `decide.Evaluator` works with the helpers and patterns:

| Evaluator | What it does |
| :-- | :-- |
| `decide.NewLocal(dir)` | in-process pure-Go inference from a model bundle |
| `decide.NewRemote(url, key)` | calls any `/v1/systemone` endpoint, e.g. `decide serve` |

`decide.Default()` picks `Remote` when `$DECIDE_BASE_URL` is set (key from `$DECIDE_API_KEY`) and `Local` otherwise.

## Server

```
decide serve --host 0.0.0.0 --port 8000 --model-dir /path/to/bundle
```

`GET /`, `GET /health`, `GET /v1/models`, `POST /v1/systemone`. Set `DECIDE_API_KEY` to require a Bearer token
(compared in constant time) and `DECIDE_CORS_ORIGINS` for an explicit origin allow-list.

```
curl -X POST http://localhost:8000/v1/systemone -H "Content-Type: application/json" -d '{
  "state": { "error": "Disk volume /var/log at 98% capacity." },
  "questions": { "requires_intervention": {
      "type": "noul", "instructions": "Does this disk space condition require operational intervention?" } }
}'
```

Embed it with `http.ListenAndServe(addr, server.New(decide.Default(), server.Config{}))`.

## CLI

```
decide serve   [--host H] [--port N] [--model-dir DIR]
decide choose  TEXT -c a,b,c [-i INSTRUCTIONS]
decide judge   TEXT -i QUESTION [--pos TEXT] [--neg TEXT]
decide rate    TEXT -l low,mid,high [-i INSTRUCTIONS]
decide eval    REQUEST.json
decide pull    [--repo HF_REPO] [--dir DIR]
```

## Training

The whole pipeline is `decide-train`, a single Go binary. On the reference machine (Ryzen 7 5800X3D, 8 cores) the
matrix-multiply kernel reaches 400–500 GFLOPS and a full fine-tune of ModernBERT-base (all 149M parameters) runs at
about 500 tokens/s.

```
# 1. Build the corpus from public Hugging Face datasets (downloads parquet, no Python)
decide-train data   --dir corpus

# 2. Fine-tune ModernBERT-base (download config.json, tokenizer.json, model.safetensors from
#    answerdotai/ModernBERT-base into ./base). Checkpoints every 250 steps; add --resume to continue.
decide-train train  --data corpus --base base --out run1 --max-examples 120000

# 3. Evaluate, calibrate on held-out data, and write a model bundle
decide-train export --model run1/model.safetensors --base base --data corpus --out bundle --id decide-0.1.0

# 4. Optional: model card + upload to the Hugging Face hub
decide-train card    --bundle bundle --repo <user>/decide
HF_TOKEN=... decide-train publish --bundle bundle --repo <user>/decide --yes
```

**Data.** `internal/data` turns 24 public datasets (NLI, intent and topic classification, sentiment, emotion,
toxicity, multiple-choice QA, yes/no QA and ordinal ratings) into *state + instruction + options* examples. Large
label spaces are sub-sampled to a random subset of at most 10 options that always contains the answer;
classification tasks are also posed as yes/no verification questions; options are verbalised in several styles
(bare label, descriptive sentence, ...). Four tasks (MMLU, SMS spam, dair-ai/emotion, STS-B) are held out
entirely and only used to measure zero-shot behaviour. Training examples whose text also appears in the validation
or held-out sets are dropped. The corpus manifest records every dataset's declared license.

**Model.** ModernBERT-base plus a small scorer head (LayerNorm → Linear → GELU → LayerNorm → Linear) applied to the
hidden state at every `[MASK]`. Trained with cross-entropy over the options (soft targets on neighbouring levels
for ordinal tasks), AdamW, linear warm-up and decay.

**Calibration.** A softmax temperature per question kind is fitted on held-out validation data, which fixes
over-confidence without changing any answer.

**Verification.** The forward and backward passes are checked in tests against PyTorch autograd (every parameter's
gradient) and against HuggingFace `transformers` on the real ModernBERT-base weights, in both attention modes.
`go test ./...` also runs a toy end-to-end job that trains, resumes, exports and reloads a tiny model.

## Performance and limits

- CPU only. The hot loops use an AVX2/FMA micro-kernel on amd64 and a portable fallback elsewhere (arm64 works
  but is several times slower).
- Inference calls are serialised (each already uses every core); the server stays responsive.
- Inputs are capped at 8192 tokens; keep option sets under about ten (use `TwoStageChoice` for larger taxonomies).
- English only. Strongest on the kinds of task in the training corpus; give options descriptive text rather than
  bare labels.

## Acknowledgements

The option-marker formulation (score every option in one pass from `[MASK]` markers) and the order-invariant
option attention follow the ideas of [wfzyx/von](https://github.com/wfzyx/von); this repository is an independent
implementation with its own model, data and training code, and shares no weights with it. Built on
[ModernBERT](https://arxiv.org/abs/2412.13663) (Answer.AI and LightOn, Apache-2.0).

## License

Apache-2.0. See [LICENSE.md](LICENSE.md). Model weights inherit the terms of the training datasets listed in the
model card.
