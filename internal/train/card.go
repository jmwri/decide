package train

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/jmwri/decide/internal/bundle"
)

type cardReports struct {
	Val map[string]json.RawMessage `json:"val"`
	OOD map[string]json.RawMessage `json:"ood"`
	// Order invariance
	OrderDelta float64 `json:"order_invariance_max_logit_delta"`
}

type cardTraining struct {
	Reports cardReports `json:"reports"`
	Corpus  struct {
		Train int `json:"train"`
		Val   int `json:"val"`
		OOD   int `json:"ood"`
		Tasks []struct {
			Name    string `json:"name"`
			License string `json:"license"`
			Train   int    `json:"train"`
			Val     int    `json:"val"`
			OOD     int    `json:"ood"`
			Skipped string `json:"skipped"`
		} `json:"tasks"`
	} `json:"corpus"`
}

func decodeMetrics(raw map[string]json.RawMessage, key string) map[string]*Metrics {
	out := map[string]*Metrics{}
	if b, ok := raw[key]; ok {
		_ = json.Unmarshal(b, &out)
	}
	return out
}

// ModelCard renders a Hugging Face model card (README.md) for a bundle from
// the metadata recorded at export time.
func ModelCard(meta bundle.Meta, repo string) string {
	var tr cardTraining
	if b, err := json.Marshal(meta.Training); err == nil {
		_ = json.Unmarshal(b, &tr)
	}
	var w strings.Builder
	p := func(format string, args ...any) { fmt.Fprintf(&w, format+"\n", args...) }

	p("---")
	p("license: apache-2.0")
	p("language: en")
	p("base_model: answerdotai/ModernBERT-base")
	p("library_name: decide")
	p("pipeline_tag: zero-shot-classification")
	p("tags:")
	p("  - modernbert")
	p("  - decision-model")
	p("  - zero-shot-classification")
	p("  - calibrated")
	p("---")
	p("")
	p("# %s", meta.ModelID)
	p("")
	p("A small, fast, **non-autoregressive decision model**. Given some state, an instruction and a set of candidate")
	p("options, it scores every option in a single forward pass and returns calibrated probabilities. It answers")
	p("three kinds of question with the same weights:")
	p("")
	p("- **choice**: a distribution over K mutually exclusive options")
	p("- **noul**: the probability that a condition holds (a yes/no question)")
	p("- **score**: an expected level on an ordered scale")
	p("")
	p("It is fine-tuned from [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) (149M parameters).")
	p("The packed input is `<instructions> <state> [SEP] [MASK] option0 [MASK] option1 ...`; a small MLP head scores the")
	p("encoder state at each `[MASK]`. Each option attends only to the shared prefix and to itself and its position ids restart")
	p("after the prefix, so an option's score depends on *(premise, that option)* alone: **permuting the options permutes the")
	p("scores and changes nothing else** (measured max logit change under permutation: %.1e).", tr.Reports.OrderDelta)
	p("")
	p("The model, tokenizer, training loop, evaluation and this card were all produced with the pure-Go toolchain in")
	p("[jmwri/decide](https://github.com/jmwri/decide); no Python or PyTorch was used to train it.")
	p("")
	p("## Usage")
	p("")
	p("```go")
	p("import \"github.com/jmwri/decide\"")
	p("")
	p("decide.SetDefault(decide.NewLocal(\"/path/to/this/model\")) // or: decide pull --repo %s", repo)
	p("ans, _ := decide.Decide(ctx, \"Replication lag exceeded 45 seconds.\", decide.Options{")
	p("    {Key: \"infrastructure\", Description: \"Database, hardware, network or server failures\"},")
	p("    {Key: \"billing\", Description: \"Invoices, payments, refunds\"},")
	p("}, \"Classify the root cause domain.\")")
	p("```")
	p("")

	if len(tr.Reports.Val) > 0 {
		p("## Evaluation")
		p("")
		p("Accuracy is top-option accuracy. ECE is the expected calibration error (15 bins) of the top-option confidence,")
		p("after the per-kind temperature scaling shipped in `decide.json`.")
		p("")
		table := func(title string, byTask map[string]*Metrics, kinds map[string]*Metrics, overall *Metrics) {
			p("### %s", title)
			p("")
			p("| task | n | accuracy | NLL | ECE |")
			p("| :-- | --: | --: | --: | --: |")
			if overall != nil {
				p("| **all** | %d | **%.1f%%** | %.3f | %.3f |", overall.N, 100*overall.Acc, overall.NLL, overall.ECE)
			}
			for _, k := range sortedKeys(kinds) {
				m := kinds[k]
				p("| kind: %s | %d | %.1f%% | %.3f | %.3f |", k, m.N, 100*m.Acc, m.NLL, m.ECE)
			}
			names := make([]string, 0, len(byTask))
			for n := range byTask {
				names = append(names, n)
			}
			sort.Strings(names)
			for _, n := range names {
				m := byTask[n]
				extra := ""
				if m.MAE > 0 {
					extra = fmt.Sprintf(" (MAE %.2f levels)", m.MAE)
				}
				p("| %s | %d | %.1f%%%s | %.3f | %.3f |", n, m.N, 100*m.Acc, extra, m.NLL, m.ECE)
			}
			p("")
		}
		var valOverall, oodOverall *Metrics
		if b, ok := tr.Reports.Val["overall"]; ok {
			valOverall = &Metrics{}
			_ = json.Unmarshal(b, valOverall)
		}
		if b, ok := tr.Reports.OOD["overall"]; ok {
			oodOverall = &Metrics{}
			_ = json.Unmarshal(b, oodOverall)
		}
		table("Held-out validation splits of the training tasks", decodeMetrics(tr.Reports.Val, "by_task"), decodeMetrics(tr.Reports.Val, "by_kind"), valOverall)
		if len(tr.Reports.OOD) > 0 {
			p("Tasks in this second table were **never trained on** (zero-shot to the model):")
			p("")
			table("Held-out tasks", decodeMetrics(tr.Reports.OOD, "by_task"), decodeMetrics(tr.Reports.OOD, "by_kind"), oodOverall)
		}
	}

	if len(tr.Corpus.Tasks) > 0 {
		p("## Training data")
		p("")
		p("Fine-tuned on %d examples built from the public datasets below (options are re-sampled and verbalised in several", tr.Corpus.Train)
		p("styles; classification tasks are also posed as yes/no verification questions). Licenses are as declared on the")
		p("Hugging Face hub; **review them before redistributing or commercialising the weights**, since several datasets")
		p("carry share-alike or unspecified terms.")
		p("")
		p("| dataset | train examples | license (as declared) |")
		p("| :-- | --: | :-- |")
		for _, t := range tr.Corpus.Tasks {
			switch {
			case t.Skipped != "":
				continue
			case t.OOD > 0:
				p("| %s (held out, evaluation only) | 0 | %s |", t.Name, t.License)
			default:
				p("| %s | %d | %s |", t.Name, t.Train, t.License)
			}
		}
		p("")
	}

	p("## Limitations")
	p("")
	p("- English only. Inputs are limited to 8192 tokens; option sets are best kept under ~10 (use a two-stage taxonomy beyond that).")
	p("- Trained on public classification, inference and multiple-choice datasets, so it is strongest on those styles of")
	p("  question. Explicit, descriptive option texts work better than bare labels. Domain-specific jargon or rubrics with")
	p("  no descriptive criteria fall back to general language priors.")
	p("- Probabilities are calibrated on held-out data from the training tasks; on very different distributions they will")
	p("  be over-confident. Use a confidence gate and route uncertain cases to review.")
	p("- Not for decisions that affect people's rights, safety or livelihood without human oversight.")
	p("")
	p("## License")
	p("")
	p("Weights: Apache-2.0 (ModernBERT-base is Apache-2.0), subject to the training-data terms above.")
	return w.String()
}
