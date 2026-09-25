// Package decide is a fast, local decision model in pure Go.
//
// Given some state, an instruction and a set of candidate options, it scores
// every option in a single forward pass of a ModernBERT encoder and returns
// calibrated probabilities, with no text generation. There are three question
// primitives:
//
//   - Choice: a calibrated distribution over K mutually exclusive options.
//   - Noul: P(condition holds), a yes/no probability.
//   - Score: an expected level over an ordered scale.
//
// Each option attends only to the shared premise and to itself, so the answer
// never depends on the order options are listed in.
//
// The model runs in-process on the CPU with no cgo, Python or native runtime:
// the tokenizer, the ModernBERT forward pass and (in cmd/decide-train) the
// training loop are all implemented in this module. A model is a bundle
// directory (decide.json, model.safetensors, config.json, tokenizer.json), read
// from a path, $DECIDE_MODEL_DIR, or downloaded from Hugging Face (see
// DownloadModel).
//
//	ans, err := decide.Decide(ctx, "Replication lag exceeded 45 seconds.",
//		decide.Options{
//			{Key: "infrastructure", Description: "Database, hardware, network, or server failures"},
//			{Key: "billing", Description: "Invoices, payments, refunds"},
//		}, "Classify the root cause domain.")
//
// The same API works against a remote /v1/systemone endpoint (see NewRemote),
// and package server exposes any Evaluator over that protocol.
package decide
