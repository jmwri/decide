package data

import (
	"math"
	"math/rand"
	"strings"
)

// Converter turns one dataset row into zero or more examples.
type Converter func(r *rand.Rand, row Row) []Example

// Task describes one public dataset and how to turn it into examples.
type Task struct {
	Name    string
	License string // as declared on the hub; review before redistributing weights
	Train   Source // empty Repo for holdout tasks
	Val     Source
	// TrainCap / ValCap bound the rows taken from each split.
	TrainCap, ValCap int
	// ScanCap / MaxFiles bound how much of a huge split is read.
	ScanCap, MaxFiles int
	// Holdout tasks never enter training: their Val split is the out-of-domain test set.
	Holdout bool
	// Optional tasks are skipped with a warning when the download fails.
	Optional bool
	// DeriveNames names a string column whose distinct values (sorted) become
	// the label set, exposed to Prepare as meta.Names[derivedNames].
	DeriveNames string
	Prepare     func(meta Meta) Converter
}

func one(e Example, ok bool) []Example {
	if !ok {
		return nil
	}
	return []Example{e}
}

func static(c Converter) func(Meta) Converter { return func(Meta) Converter { return c } }

// ---------------------------------------------------------------------------
// natural language inference

var nliLabels = []Label{
	{Name: "entailment", Verb: []string{"entailment", "The hypothesis follows from the premise.", "The premise implies the hypothesis.", "true"}},
	{Name: "neutral", Verb: []string{"neutral", "The hypothesis may or may not be true given the premise.", "The premise neither confirms nor rules out the hypothesis.", "unknown"}},
	{Name: "contradiction", Verb: []string{"contradiction", "The hypothesis contradicts the premise.", "The hypothesis is false given the premise.", "false"}},
}

var nliChoice = ClassTask{
	Name:   "nli",
	Labels: nliLabels,
	Instructions: []string{
		"What is the relationship between the premise and the hypothesis?",
		"Does the premise support the hypothesis?",
		"Given the premise, what can we say about the hypothesis?",
		"Classify the logical relation between the two statements.",
		"Determine whether the hypothesis is entailed by, neutral to, or contradicted by the premise.",
	},
}

func nliExample(r *rand.Rand, task, premise, hypothesis string, label int) []Example {
	premise, hypothesis = clip(oneLine(premise), 900), clip(oneLine(hypothesis), 400)
	if premise == "" || hypothesis == "" || label < 0 || label > 2 {
		return nil
	}
	state := fields(r, "premise", premise, "hypothesis", hypothesis)
	switch x := r.Float64(); {
	case x < 0.6:
		e := nliChoice.Make(r, state, label)
		e.Task = task
		return []Example{e}
	case x < 0.85:
		q := pick(r, []string{
			"Does the premise entail the hypothesis?",
			"Is the hypothesis true given the premise?",
			"Can the hypothesis be concluded from the premise?",
			"Does the hypothesis follow from the premise?",
		})
		return []Example{noulExample(r, task, state, q, label == 0, "", "")}
	default:
		q := pick(r, []string{
			"Does the hypothesis contradict the premise?",
			"Is the hypothesis false given the premise?",
			"Are the premise and the hypothesis in conflict?",
		})
		return []Example{noulExample(r, task, state, q, label == 2, "", "")}
	}
}

// ---------------------------------------------------------------------------
// sentiment and emotion

func sentimentTask(name string) *ClassTask {
	return &ClassTask{
		Name: name,
		Labels: []Label{
			{Name: "negative", Verb: []string{"negative", "The text expresses a negative opinion.", "Unfavorable", "The author disliked it."}},
			{Name: "positive", Verb: []string{"positive", "The text expresses a positive opinion.", "Favorable", "The author liked it."}},
		},
		Instructions: []string{
			"What is the sentiment of the text?",
			"Is the opinion expressed positive or negative?",
			"Classify the sentiment.",
			"How does the author feel about the subject?",
		},
		YesNo:    []string{"Is the sentiment of the text {label}?", "Does the author express a {label} opinion?"},
		NoulProb: 0.3,
	}
}

var goEmotionNames = []string{"admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"}

func emotionTask(name string, names []string) *ClassTask {
	labels := make([]Label, len(names))
	for i, n := range names {
		labels[i] = Label{Name: n, Verb: []string{n, n, "The writer feels " + n + ".", capitalize(n), "Emotion: " + n}}
	}
	return &ClassTask{
		Name: name, Labels: labels,
		Instructions: []string{
			"Which emotion does the text express?",
			"What is the writer feeling?",
			"Classify the emotion of the message.",
			"Identify the dominant emotion.",
		},
		YesNo: []string{"Does the text express {label}?", "Is the writer feeling {label}?"},
	}
}

// ---------------------------------------------------------------------------
// registry

// Tasks returns every task the corpus builder knows.
func Tasks() []Task {
	pairNoul := func(name, k1, k2, f1, f2 string, question []string, truthLabel int) func(Meta) Converter {
		return static(func(r *rand.Rand, row Row) []Example {
			a, b := clip(oneLine(row.Str(f1)), 700), clip(oneLine(row.Str(f2)), 700)
			if a == "" || b == "" {
				return nil
			}
			return []Example{noulExample(r, name, fields(r, k1, a, k2, b), pick(r, question), row.Int("label") == truthLabel, "", "")}
		})
	}
	tasks := []Task{
		// ---- NLI
		{Name: "mnli", License: "cc-by-3.0/cc-by-sa-3.0/mit/other", Train: Source{Repo: "nyu-mll/multi_nli", Split: "train"}, Val: Source{Repo: "nyu-mll/multi_nli", Split: "validation_matched"},
			TrainCap: 30000, ValCap: 300, ScanCap: 120000, MaxFiles: 1,
			Prepare: static(func(r *rand.Rand, row Row) []Example {
				return nliExample(r, "mnli", row.Str("premise"), row.Str("hypothesis"), row.Int("label"))
			})},
		{Name: "snli", License: "cc-by-sa-4.0", Train: Source{Repo: "stanfordnlp/snli", Config: "plain_text", Split: "train"}, Val: Source{Repo: "stanfordnlp/snli", Config: "plain_text", Split: "validation"},
			TrainCap: 16000, ValCap: 300, ScanCap: 150000, MaxFiles: 1,
			Prepare: static(func(r *rand.Rand, row Row) []Example {
				return nliExample(r, "snli", row.Str("premise"), row.Str("hypothesis"), row.Int("label"))
			})},
		{Name: "wanli", License: "cc-by-4.0", Train: Source{Repo: "alisawuffles/WANLI", Split: "train"}, Val: Source{Repo: "alisawuffles/WANLI", Split: "test"},
			TrainCap: 16000, ValCap: 300,
			Prepare: static(func(r *rand.Rand, row Row) []Example {
				label := map[string]int{"entailment": 0, "neutral": 1, "contradiction": 2}
				l, ok := label[row.Str("gold")]
				if !ok {
					return nil
				}
				return nliExample(r, "wanli", row.Str("premise"), row.Str("hypothesis"), l)
			})},

		// ---- yes/no over text pairs and passages
		{Name: "boolq", License: "cc-by-sa-3.0", Train: Source{Repo: "google/boolq", Split: "train"}, Val: Source{Repo: "google/boolq", Split: "validation"},
			TrainCap: 9000, ValCap: 300,
			Prepare: static(func(r *rand.Rand, row Row) []Example {
				q, p := oneLine(row.Str("question")), clip(oneLine(row.Str("passage")), 1100)
				if q == "" || p == "" {
					return nil
				}
				q = capitalize(strings.TrimRight(q, "?. ")) + "?"
				if r.Float64() < 0.6 {
					return []Example{noulExample(r, "boolq", p, q, row.Bool("answer"), "", "")}
				}
				return []Example{noulExample(r, "boolq", fields(r, "question", q, "passage", p),
					pick(r, []string{"Is the answer to the question yes, based on the passage?", "Does the passage support a yes answer to the question?"}), row.Bool("answer"), "", "")}
			})},
		{Name: "qnli", License: "other (GLUE)", Train: Source{Repo: "nyu-mll/glue", Config: "qnli", Split: "train"}, Val: Source{Repo: "nyu-mll/glue", Config: "qnli", Split: "validation"},
			TrainCap: 8000, ValCap: 200, ScanCap: 60000, MaxFiles: 1,
			Prepare: pairNoul("qnli", "question", "sentence", "question", "sentence",
				[]string{"Does the sentence contain the answer to the question?", "Is the question answered by the sentence?", "Can the question be answered from the sentence?"}, 0)},
		{Name: "rte", License: "other (GLUE)", Train: Source{Repo: "nyu-mll/glue", Config: "rte", Split: "train"}, Val: Source{Repo: "nyu-mll/glue", Config: "rte", Split: "validation"},
			TrainCap: 2500, ValCap: 200,
			Prepare: pairNoul("rte", "premise", "hypothesis", "sentence1", "sentence2",
				[]string{"Does the first text imply the second?", "Is the hypothesis entailed by the premise?", "Does the premise entail the hypothesis?"}, 0)},
		{Name: "mrpc", License: "other (GLUE)", Train: Source{Repo: "nyu-mll/glue", Config: "mrpc", Split: "train"}, Val: Source{Repo: "nyu-mll/glue", Config: "mrpc", Split: "validation"},
			TrainCap: 3700, ValCap: 200,
			Prepare: pairNoul("mrpc", "sentence 1", "sentence 2", "sentence1", "sentence2",
				[]string{"Do the two sentences have the same meaning?", "Are these sentences paraphrases of each other?", "Do both sentences say the same thing?"}, 1)},
		{Name: "qqp", License: "other (GLUE)", Train: Source{Repo: "nyu-mll/glue", Config: "qqp", Split: "train"}, Val: Source{Repo: "nyu-mll/glue", Config: "qqp", Split: "validation"},
			TrainCap: 8000, ValCap: 200, ScanCap: 80000, MaxFiles: 1,
			Prepare: pairNoul("qqp", "question 1", "question 2", "question1", "question2",
				[]string{"Are these two questions asking the same thing?", "Do the two questions have the same intent?", "Is the second question a duplicate of the first?"}, 1)},
		{Name: "cola", License: "other (GLUE)", Train: Source{Repo: "nyu-mll/glue", Config: "cola", Split: "train"}, Val: Source{Repo: "nyu-mll/glue", Config: "cola", Split: "validation"},
			TrainCap: 6000, ValCap: 200,
			Prepare: static(func(r *rand.Rand, row Row) []Example {
				s := oneLine(row.Str("sentence"))
				if s == "" {
					return nil
				}
				return []Example{noulExample(r, "cola", s, pick(r, []string{"Is this sentence grammatically acceptable?", "Is the sentence well-formed English?", "Does the sentence follow the rules of English grammar?"}), row.Int("label") == 1, "", "")}
			})},

		// ---- intent, topic, sentiment, emotion
		{Name: "banking77", License: "cc-by-4.0", Train: Source{Repo: "mteb/banking77", Split: "train"}, Val: Source{Repo: "mteb/banking77", Split: "test"},
			TrainCap: 10000, ValCap: 300, Optional: true, DeriveNames: "label_text",
			Prepare: namedIntent("banking77", "Customer is asking about", "label_text")},
		{Name: "clinc150", License: "cc-by-3.0", Train: Source{Repo: "clinc/clinc_oos", Config: "plus", Split: "train"}, Val: Source{Repo: "clinc/clinc_oos", Config: "plus", Split: "validation"},
			TrainCap: 15000, ValCap: 300,
			Prepare: func(meta Meta) Converter {
				names := labelNames(meta, "intent")
				ct := intentTask("clinc150", "User wants help with", names)
				return func(r *rand.Rand, row Row) []Example {
					text := oneLine(row.Str("text"))
					i := row.Int("intent")
					if text == "" || i < 0 || i >= len(names) {
						return nil
					}
					return []Example{ct.Make(r, text, i)}
				}
			}},
		{Name: "bitext_support", License: "cdla-sharing-1.0", Train: Source{Repo: "bitext/Bitext-customer-support-llm-chatbot-training-dataset", Split: "train"}, Val: Source{Repo: "bitext/Bitext-customer-support-llm-chatbot-training-dataset", Split: "train"},
			TrainCap: 9000, ValCap: 300, Optional: true, DeriveNames: "intent",
			Prepare: namedIntent("bitext_support", "Customer request about", "intent")},
		{Name: "dbpedia14", License: "cc-by-sa-3.0", Train: Source{Repo: "fancyzhx/dbpedia_14", Config: "dbpedia_14", Split: "train"}, Val: Source{Repo: "fancyzhx/dbpedia_14", Config: "dbpedia_14", Split: "test"},
			TrainCap: 8000, ValCap: 300, ScanCap: 100000, MaxFiles: 1,
			Prepare: func(meta Meta) Converter {
				names := labelNames(meta, "label")
				ct := topicTask("dbpedia14", names, []string{"What kind of entity does the text describe?", "Which category best describes the subject?", "Classify the entity type."})
				return func(r *rand.Rand, row Row) []Example {
					i := row.Int("label")
					body := clip(oneLine(row.Str("content")), 700)
					if i < 0 || i >= len(names) || body == "" {
						return nil
					}
					return []Example{ct.Make(r, fields(r, "title", oneLine(row.Str("title")), "text", body), i)}
				}
			}},
		{Name: "sst2", License: "other (GLUE)", Train: Source{Repo: "nyu-mll/glue", Config: "sst2", Split: "train"}, Val: Source{Repo: "nyu-mll/glue", Config: "sst2", Split: "validation"},
			TrainCap: 8000, ValCap: 300, ScanCap: 70000, MaxFiles: 1,
			Prepare: func(Meta) Converter {
				ct := sentimentTask("sst2")
				return func(r *rand.Rand, row Row) []Example {
					s := oneLine(row.Str("sentence"))
					if s == "" {
						return nil
					}
					return []Example{ct.Make(r, s, row.Int("label"))}
				}
			}},
		{Name: "amazon_polarity", License: "apache-2.0", Train: Source{Repo: "fancyzhx/amazon_polarity", Config: "amazon_polarity", Split: "train"}, Val: Source{Repo: "fancyzhx/amazon_polarity", Config: "amazon_polarity", Split: "test"},
			TrainCap: 8000, ValCap: 300, ScanCap: 120000, MaxFiles: 1,
			Prepare: func(Meta) Converter {
				ct := sentimentTask("amazon_polarity")
				return func(r *rand.Rand, row Row) []Example {
					body := clip(oneLine(row.Str("content")), 900)
					if body == "" {
						return nil
					}
					state := body
					if t := oneLine(row.Str("title")); t != "" && r.Float64() < 0.5 {
						state = fields(r, "title", t, "review", body)
					}
					return []Example{ct.Make(r, state, row.Int("label"))}
				}
			}},
		{Name: "go_emotions", License: "apache-2.0", Train: Source{Repo: "google-research-datasets/go_emotions", Config: "simplified", Split: "train"}, Val: Source{Repo: "google-research-datasets/go_emotions", Config: "simplified", Split: "validation"},
			TrainCap: 9000, ValCap: 300,
			Prepare: func(Meta) Converter {
				ct := emotionTask("go_emotions", goEmotionNames)
				return func(r *rand.Rand, row Row) []Example {
					labels, _ := row["labels"].([]any)
					text := oneLine(row.Str("text"))
					if len(labels) != 1 || text == "" {
						return nil
					}
					i := Row{"v": labels[0]}.Int("v")
					if i < 0 || i >= len(goEmotionNames) {
						return nil
					}
					return []Example{ct.Make(r, text, i)}
				}
			}},

		// ---- moderation (civil comments, CC0)
		{Name: "civil_comments", License: "cc0-1.0", Train: Source{Repo: "google/civil_comments", Split: "train"}, Val: Source{Repo: "google/civil_comments", Split: "validation"},
			TrainCap: 40000, ValCap: 600, ScanCap: 400000, MaxFiles: 1,
			Prepare: static(civilConverter)},

		// ---- multiple choice with real answer texts
		{Name: "arc", License: "cc-by-sa-4.0", Train: Source{Repo: "allenai/ai2_arc", Config: "ARC-Challenge", Split: "train"}, Val: Source{Repo: "allenai/ai2_arc", Config: "ARC-Challenge", Split: "validation"},
			TrainCap: 3000, ValCap: 200,
			Prepare: static(func(r *rand.Rand, row Row) []Example {
				ch, _ := row["choices"].(map[string]any)
				return arcLike(r, "arc", row, ch)
			})},
		{Name: "arc_easy", License: "cc-by-sa-4.0", Train: Source{Repo: "allenai/ai2_arc", Config: "ARC-Easy", Split: "train"}, Val: Source{Repo: "allenai/ai2_arc", Config: "ARC-Easy", Split: "validation"},
			TrainCap: 4000, ValCap: 200,
			Prepare: static(func(r *rand.Rand, row Row) []Example {
				ch, _ := row["choices"].(map[string]any)
				return arcLike(r, "arc_easy", row, ch)
			})},
		{Name: "commonsense_qa", License: "mit", Train: Source{Repo: "tau/commonsense_qa", Split: "train"}, Val: Source{Repo: "tau/commonsense_qa", Split: "validation"},
			TrainCap: 9000, ValCap: 300,
			Prepare: static(func(r *rand.Rand, row Row) []Example {
				ch, _ := row["choices"].(map[string]any)
				return arcLike(r, "commonsense_qa", row, ch)
			})},
		{Name: "hellaswag", License: "mit", Train: Source{Repo: "Rowan/hellaswag", Split: "train"}, Val: Source{Repo: "Rowan/hellaswag", Split: "validation"},
			TrainCap: 22000, ValCap: 300, ScanCap: 40000, MaxFiles: 1,
			Prepare: static(func(r *rand.Rand, row Row) []Example {
				ctx := clip(oneLine(row.Str("ctx")), 600)
				gold := row.Int("label")
				if s := row.Str("label"); s != "" {
					gold = int(s[0] - '0')
				}
				e, ok := mcExample(r, "hellaswag", ctx, []string{"Which ending best continues the text?", "What most plausibly happens next?", "Choose the most sensible continuation."}, row.Strings("endings"), gold)
				return one(e, ok && ctx != "")
			})},
		{Name: "winogrande", License: "cc-by-4.0", Train: Source{Repo: "allenai/winogrande", Config: "winogrande_xl", Split: "train"}, Val: Source{Repo: "allenai/winogrande", Config: "winogrande_xl", Split: "validation"},
			TrainCap: 22000, ValCap: 300,
			Prepare: static(func(r *rand.Rand, row Row) []Example {
				gold := row.Int("answer") - 1
				if s := row.Str("answer"); s == "1" || s == "2" {
					gold = int(s[0]-'0') - 1
				}
				e, ok := mcExample(r, "winogrande", oneLine(row.Str("sentence")),
					[]string{"Which option correctly fills the blank (_)?", "What should replace the underscore?", "Choose the word or name that fits the blank."},
					[]string{row.Str("option1"), row.Str("option2")}, gold)
				return one(e, ok)
			})},

		// ---- ordinal
		{Name: "sst5", License: "unspecified (SetFit/sst5)", Train: Source{Repo: "SetFit/sst5", Split: "train"}, Val: Source{Repo: "SetFit/sst5", Split: "validation"},
			TrainCap: 8500, ValCap: 300,
			Prepare: func(Meta) Converter {
				st := &ScaleTask{Name: "sst5",
					Levels: [][]string{
						{"very negative", "Strongly negative sentiment.", "1 - terrible", "Hates it"},
						{"negative", "Somewhat negative sentiment.", "2 - poor", "Dislikes it"},
						{"neutral", "Neutral or mixed sentiment.", "3 - average", "Indifferent"},
						{"positive", "Somewhat positive sentiment.", "4 - good", "Likes it"},
						{"very positive", "Strongly positive sentiment.", "5 - excellent", "Loves it"},
					},
					Instructions: []string{"Rate the sentiment of the review.", "How positive is the text?", "Place the sentiment on the scale.", "What rating does this review deserve?"}}
				return func(r *rand.Rand, row Row) []Example {
					s := oneLine(row.Str("text"))
					l := row.Int("label")
					if s == "" || l < 0 || l > 4 {
						return nil
					}
					return []Example{st.Make(r, s, l)}
				}
			}},

		// ---- holdouts: evaluation only, never trained on
		{Name: "mmlu", License: "mit", Holdout: true, Val: Source{Repo: "cais/mmlu", Config: "all", Split: "test"}, ValCap: 1500, ScanCap: 20000, MaxFiles: 1,
			Prepare: static(func(r *rand.Rand, row Row) []Example {
				q := clip(oneLine(row.Str("question")), 800)
				e, ok := mcExample(r, "mmlu", q, []string{"Choose the correct answer.", "Which option is correct?", "Select the best answer."}, row.Strings("choices"), row.Int("answer"))
				return one(e, ok && q != "")
			})},
		{Name: "emotion", License: "other (dair-ai)", Holdout: true, Val: Source{Repo: "dair-ai/emotion", Config: "split", Split: "test"}, ValCap: 1000,
			Prepare: func(meta Meta) Converter {
				names := labelNames(meta, "label")
				if len(names) == 0 {
					names = []string{"sadness", "joy", "love", "anger", "fear", "surprise"}
				}
				ct := emotionTask("emotion", names)
				ct.YesNo = nil // choice only, so the holdout stays a pure classification check
				return func(r *rand.Rand, row Row) []Example {
					i := row.Int("label")
					text := oneLine(row.Str("text"))
					if i < 0 || i >= len(names) || text == "" {
						return nil
					}
					return []Example{ct.Make(r, text, i)}
				}
			}},

		// ---- added in the second corpus: intents, topics, spam, toxicity, science QA, similarity
		{Name: "massive_intent", License: "apache-2.0", Train: Source{Repo: "mteb/amazon_massive_intent", Config: "en", Split: "train"}, Val: Source{Repo: "mteb/amazon_massive_intent", Config: "en", Split: "validation"},
			TrainCap: 11500, ValCap: 300, DeriveNames: "label_text", Optional: true,
			Prepare: namedIntent("massive_intent", "User request about", "label_text")},
		{Name: "massive_scenario", License: "apache-2.0", Train: Source{Repo: "mteb/amazon_massive_scenario", Config: "en", Split: "train"}, Val: Source{Repo: "mteb/amazon_massive_scenario", Config: "en", Split: "validation"},
			TrainCap: 11500, ValCap: 300, DeriveNames: "label_text", Optional: true,
			Prepare: namedIntent("massive_scenario", "Request in the area of", "label_text")},
		{Name: "newsgroups", License: "unspecified (SetFit/20_newsgroups)", Train: Source{Repo: "SetFit/20_newsgroups", Split: "train"}, Val: Source{Repo: "SetFit/20_newsgroups", Split: "test"},
			TrainCap: 11000, ValCap: 300, DeriveNames: "label_text", Optional: true,
			Prepare: func(meta Meta) Converter {
				names := meta.Names[derivedNames]
				ct := topicTask("newsgroups", names, []string{"Which discussion group does this post belong to?", "What is this post about?", "Categorise the message by topic."})
				idx := map[string]int{}
				for i, n := range names {
					idx[n] = i
				}
				return func(r *rand.Rand, row Row) []Example {
					i, ok := idx[row.Str("label_text")]
					text := clip(oneLine(row.Str("text")), 700)
					if !ok || text == "" {
						return nil
					}
					return []Example{ct.Make(r, text, i)}
				}
			}},
		{Name: "qasc", License: "cc-by-4.0", Train: Source{Repo: "allenai/qasc", Split: "train"}, Val: Source{Repo: "allenai/qasc", Split: "validation"},
			TrainCap: 8000, ValCap: 250,
			Prepare: static(func(r *rand.Rand, row Row) []Example {
				ch, _ := row["choices"].(map[string]any)
				return arcLike(r, "qasc", row, ch)
			})},
		{Name: "openbookqa", License: "unknown", Train: Source{Repo: "allenai/openbookqa", Config: "main", Split: "train"}, Val: Source{Repo: "allenai/openbookqa", Config: "main", Split: "validation"},
			TrainCap: 5000, ValCap: 250,
			Prepare: static(func(r *rand.Rand, row Row) []Example {
				ch, _ := row["choices"].(map[string]any)
				row["question"] = row["question_stem"]
				return arcLike(r, "openbookqa", row, ch)
			})},
		{Name: "toxic_conversations", License: "cc-by-4.0", Train: Source{Repo: "mteb/toxic_conversations_50k", Split: "train"}, Val: Source{Repo: "mteb/toxic_conversations_50k", Split: "test"},
			TrainCap: 30000, ValCap: 600, ScanCap: 60000, Optional: true,
			Prepare: static(func(r *rand.Rand, row Row) []Example {
				text := clip(oneLine(row.Str("text")), 900)
				toxic := row.Int("label") == 1
				// about 8% of the source is toxic: keep every toxic row but only some clean ones
				if text == "" || (!toxic && r.Float64() > 0.14) {
					return nil
				}
				q := pick(r, []string{"Is this comment toxic?", "Does this comment contain abuse, harassment or hate?", "Should this comment be flagged for moderation?", "Is the tone of this comment hostile or offensive?"})
				return []Example{noulExample(r, "toxic_conversations", text, q, toxic, "", "")}
			})},
		{Name: "sms_spam", License: "unknown (UCI)", Train: Source{Repo: "ucirvine/sms_spam", Config: "plain_text", Split: "train"}, Val: Source{Repo: "ucirvine/sms_spam", Config: "plain_text", Split: "train"}, TrainCap: 4500, ValCap: 300,
			Prepare: static(func(r *rand.Rand, row Row) []Example {
				s := oneLine(row.Str("sms"))
				if s == "" {
					return nil
				}
				return []Example{noulExample(r, "sms_spam", s, pick(r, []string{"Is this message spam?", "Is this an unsolicited promotional text message?"}), row.Int("label") == 1, "", "")}
			})},
		{Name: "stsb", License: "other (GLUE)", Train: Source{Repo: "nyu-mll/glue", Config: "stsb", Split: "train"}, Val: Source{Repo: "nyu-mll/glue", Config: "stsb", Split: "validation"}, TrainCap: 5800, ValCap: 300,
			Prepare: func(Meta) Converter {
				st := &ScaleTask{Name: "stsb",
					Levels: [][]string{
						{"0 - unrelated", "The sentences are completely unrelated.", "Completely different meaning"},
						{"1 - slightly related", "The sentences share only a little in common.", "Barely similar"},
						{"2 - somewhat related", "The sentences are on the same topic but differ in meaning.", "Loosely similar"},
						{"3 - roughly equivalent", "The sentences mean roughly the same thing.", "Fairly similar"},
						{"4 - mostly equivalent", "The sentences are mostly equivalent, with minor differences.", "Very similar"},
						{"5 - equivalent", "The sentences mean exactly the same thing.", "Identical in meaning"},
					},
					Instructions: []string{"How similar in meaning are the two sentences?", "Rate the semantic similarity of the sentences.", "Score how closely the sentences match."}}
				return func(r *rand.Rand, row Row) []Example {
					a, b := oneLine(row.Str("sentence1")), oneLine(row.Str("sentence2"))
					if a == "" || b == "" {
						return nil
					}
					level := int(math.Round(row.Float("label")))
					return []Example{st.Make(r, fields(r, "sentence 1", a, "sentence 2", b), min(max(level, 0), 5))}
				}
			}},

		// ---- more holdouts: a different spam corpus and a different ordinal domain
		{Name: "enron_spam", License: "unspecified (SetFit/enron_spam)", Holdout: true, Val: Source{Repo: "SetFit/enron_spam", Split: "test"}, ValCap: 1000, Optional: true,
			Prepare: static(func(r *rand.Rand, row Row) []Example {
				text := clip(oneLine(row.Str("text")), 800)
				if text == "" {
					return nil
				}
				return []Example{noulExample(r, "enron_spam", text, pick(r, []string{"Is this email spam?", "Is this an unsolicited or fraudulent email?"}), row.Int("label") == 1, "", "")}
			})},
		{Name: "amazon_stars", License: "unspecified (mteb/amazon_reviews_multi)", Holdout: true, Val: Source{Repo: "mteb/amazon_reviews_multi", Config: "en", Split: "test"}, ValCap: 1000, Optional: true,
			Prepare: func(Meta) Converter {
				st := &ScaleTask{Name: "amazon_stars",
					Levels: [][]string{
						{"1 star", "Terrible: the reviewer is very unhappy with the product.", "Very negative"},
						{"2 stars", "Poor: the reviewer is disappointed.", "Negative"},
						{"3 stars", "Average: mixed feelings about the product.", "Mixed"},
						{"4 stars", "Good: the reviewer is satisfied.", "Positive"},
						{"5 stars", "Excellent: the reviewer loves the product.", "Very positive"},
					},
					Instructions: []string{"How many stars did the reviewer give?", "Rate the product review on a five-star scale.", "What rating does this review imply?"}}
				return func(r *rand.Rand, row Row) []Example {
					text := clip(oneLine(row.Str("text")), 800)
					l := row.Int("label")
					if text == "" || l < 0 || l > 4 {
						return nil
					}
					return []Example{st.Make(r, text, l)}
				}
			}},
	}
	return tasks
}

const derivedNames = "_derived"

func labelNames(meta Meta, col string) []string { return meta.Names[col] }

func intentTask(name, what string, names []string) *ClassTask {
	ct := &ClassTask{
		Name: name,
		Instructions: []string{
			"What is the customer asking about?",
			"Which intent does this message express?",
			"Route this message to the correct intent.",
			"Classify the intent of the message.",
			"What does the user want?",
		},
		YesNo:    []string{"Is the customer asking about {label}?", "Does this message express the intent: {label}?", "Is this request related to {label}?"},
		Explicit: true,
	}
	if len(names) > 0 {
		ct.Labels = intentLabels(names, what)
	}
	return ct
}

func topicTask(name string, names []string, instr []string) *ClassTask {
	labels := make([]Label, len(names))
	for i, n := range names {
		h := humanize(n)
		labels[i] = Label{Name: n, Verb: []string{h, h, "The text is about " + strings.ToLower(h) + ".", capitalize(h)}}
	}
	return &ClassTask{Name: name, Labels: labels, Instructions: instr,
		YesNo: []string{"Is the text about {label}?", "Does this belong to the category: {label}?"}, NoulProb: 0.15}
}

// namedIntent converts rows whose label is a string column; the label set is
// derived from the data before conversion (Task.DeriveNames).
func namedIntent(name, what, col string) func(Meta) Converter {
	return func(meta Meta) Converter {
		names := meta.Names[derivedNames]
		ct := intentTask(name, what, names)
		idx := make(map[string]int, len(names))
		for i, n := range names {
			idx[n] = i
		}
		return func(r *rand.Rand, row Row) []Example {
			text := oneLine(row.Str("text"))
			if text == "" {
				text = oneLine(row.Str("instruction"))
			}
			i, ok := idx[row.Str(col)]
			if !ok || text == "" {
				return nil
			}
			return []Example{ct.Make(r, text, i)}
		}
	}
}

func arcLike(r *rand.Rand, task string, row Row, ch map[string]any) []Example {
	if ch == nil {
		return nil
	}
	texts := (Row{"v": ch["text"]}).Strings("v")
	labels := (Row{"v": ch["label"]}).Strings("v")
	gold := -1
	for i, l := range labels {
		if l == row.Str("answerKey") {
			gold = i
		}
	}
	q := clip(oneLine(row.Str("question")), 800)
	e, ok := mcExample(r, task, q, []string{"Choose the correct answer.", "Which option is correct?", "Select the best answer.", "Answer the question."}, texts, gold)
	return one(e, ok && q != "")
}

// civilConverter derives yes/no, category and severity questions from civil_comments.
func civilConverter(r *rand.Rand, row Row) []Example {
	text := clip(oneLine(row.Str("text")), 900)
	if text == "" {
		return nil
	}
	tox := row.Float("toxicity")
	switch x := r.Float64(); {
	case x < 0.4: // toxic yes/no; skip the ambiguous middle
		// the corpus is ~90% clean: keep all toxic comments but only some clean ones
		if tox >= 0.5 || (tox <= 0.1 && r.Float64() < 0.12) {
			return []Example{noulExample(r, "civil_comments", text, pick(r, []string{"Is this comment toxic?", "Is this comment abusive or offensive?", "Should this comment be flagged for harassment?", "Would most readers find this comment rude or hostile?"}), tox >= 0.5, "", "")}
		}
	case x < 0.7: // category
		cats := []struct {
			col  string
			verb []string
		}{
			{"insult", []string{"insult", "Personal insult or attack.", "insulting"}},
			{"obscene", []string{"obscene", "Obscene or vulgar language.", "vulgar"}},
			{"threat", []string{"threat", "Threat of harm or violence.", "threatening"}},
			{"identity_attack", []string{"identity attack", "Hostile toward a group or identity.", "hateful"}},
			{"sexual_explicit", []string{"sexually explicit", "Sexually explicit content.", "explicit"}},
		}
		best, bestV := -1, 0.5
		for i, c := range cats {
			if v := row.Float(c.col); v >= bestV {
				best, bestV = i, v
			}
		}
		clean := []string{"clean", "Civil and acceptable content.", "acceptable"}
		if best < 0 && (tox > 0.1 || r.Float64() < 0.85) {
			return nil
		}
		style := r.Intn(3)
		opts := []string{clean[style]}
		gold := 0
		for i, c := range cats {
			opts = append(opts, c.verb[style])
			if i == best {
				gold = i + 1
			}
		}
		return []Example{{Task: "civil_comments", Kind: KindChoice, Instructions: pick(r, []string{"Which kind of policy violation, if any, does the comment contain?", "Classify the comment for moderation.", "What is wrong with this comment, if anything?"}),
			State: text, Options: opts, Gold: gold}}
	default: // severity scale
		level := 0
		if tox < 0.1 && r.Float64() < 0.85 {
			return nil
		}
		switch {
		case tox >= 0.7:
			level = 3
		case tox >= 0.4:
			level = 2
		case tox >= 0.1:
			level = 1
		}
		st := &ScaleTask{Name: "civil_comments", Levels: [][]string{
			{"not toxic", "Civil, no problem.", "0 - safe"},
			{"mildly rude", "Rude or dismissive but tolerable.", "1 - low"},
			{"toxic", "Clearly toxic or insulting.", "2 - medium"},
			{"severely toxic", "Abusive, hateful or threatening.", "3 - high"},
		}, Instructions: []string{"Rate how toxic the comment is.", "How severe is the content risk?", "Assign a toxicity level."}}
		return []Example{st.Make(r, text, level)}
	}
	return nil
}
