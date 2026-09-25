package data

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"
)

// BuildConfig controls corpus construction.
type BuildConfig struct {
	// Dir receives raw/ (download cache), train.jsonl, val.jsonl, ood.jsonl and manifest.json.
	Dir  string
	Seed int64
	// Only restricts the build to the named tasks (empty = all).
	Only []string
	// Strict makes a failing optional task an error instead of a warning.
	Strict bool
	// CapScale multiplies every task's train cap (default 1).
	CapScale float64
	Log      func(format string, args ...any)
}

// TaskStat summarises one task in the manifest.
type TaskStat struct {
	Name    string `json:"name"`
	License string `json:"license"`
	Train   int    `json:"train"`
	Val     int    `json:"val"`
	OOD     int    `json:"ood"`
	Skipped string `json:"skipped,omitempty"`
}

// Manifest describes a built corpus.
type Manifest struct {
	Created string     `json:"created"`
	Seed    int64      `json:"seed"`
	Train   int        `json:"train"`
	Val     int        `json:"val"`
	OOD     int        `json:"ood"`
	Tasks   []TaskStat `json:"tasks"`
}

func (c *BuildConfig) logf(format string, args ...any) {
	if c.Log != nil {
		c.Log(format, args...)
	}
}

func seedFor(base int64, name string) int64 {
	h := fnv.New64a()
	h.Write([]byte(name))
	return base ^ int64(h.Sum64())
}

func exampleKey(e *Example) string {
	return e.Task + "\x00" + e.Instructions + "\x00" + e.State + "\x00" + strings.Join(e.Options, "\x01")
}

func stateKey(s string) string { return strings.ToLower(oneLine(s)) }

// Build downloads the public datasets and writes the training corpus.
func Build(ctx context.Context, cfg BuildConfig) (*Manifest, error) {
	if cfg.CapScale == 0 {
		cfg.CapScale = 1
	}
	if err := os.MkdirAll(cfg.Dir, 0o755); err != nil {
		return nil, err
	}
	f := &Fetcher{CacheDir: filepath.Join(cfg.Dir, "raw"), Token: os.Getenv("HF_TOKEN"), Log: cfg.Log}
	if e := os.Getenv("HF_ENDPOINT"); e != "" {
		f.Endpoint = e
	}

	var train, val, ood []Example
	m := &Manifest{Created: time.Now().UTC().Format(time.RFC3339), Seed: cfg.Seed}
	for _, t := range Tasks() {
		if len(cfg.Only) > 0 && !slices.Contains(cfg.Only, t.Name) {
			continue
		}
		stat := TaskStat{Name: t.Name, License: t.License}
		tr, va, err := buildTask(ctx, f, &cfg, t)
		if err != nil {
			if t.Optional && !cfg.Strict {
				cfg.logf("skipping %s: %v", t.Name, err)
				stat.Skipped = err.Error()
				m.Tasks = append(m.Tasks, stat)
				continue
			}
			return nil, fmt.Errorf("task %s: %w", t.Name, err)
		}
		if t.Holdout {
			ood = append(ood, va...)
			stat.OOD = len(va)
		} else {
			train = append(train, tr...)
			val = append(val, va...)
			stat.Train, stat.Val = len(tr), len(va)
		}
		cfg.logf("%-16s train %6d  val %5d  ood %5d", t.Name, stat.Train, stat.Val, stat.OOD)
		m.Tasks = append(m.Tasks, stat)
	}

	// Anything in train that also appears in val/ood (same state text) is leakage.
	held := map[string]bool{}
	for i := range val {
		held[stateKey(val[i].State)] = true
	}
	for i := range ood {
		held[stateKey(ood[i].State)] = true
	}
	kept := train[:0]
	dropped := 0
	for _, e := range train {
		if held[stateKey(e.State)] {
			dropped++
			continue
		}
		kept = append(kept, e)
	}
	train = kept
	if dropped > 0 {
		cfg.logf("dropped %d train examples whose state also appears in val/ood", dropped)
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	rng.Shuffle(len(train), func(i, j int) { train[i], train[j] = train[j], train[i] })
	rng.Shuffle(len(val), func(i, j int) { val[i], val[j] = val[j], val[i] })
	m.Train, m.Val, m.OOD = len(train), len(val), len(ood)
	sort.Slice(m.Tasks, func(i, j int) bool { return m.Tasks[i].Name < m.Tasks[j].Name })

	for name, exs := range map[string][]Example{"train.jsonl": train, "val.jsonl": val, "ood.jsonl": ood} {
		if err := WriteJSONL(filepath.Join(cfg.Dir, name), exs); err != nil {
			return nil, err
		}
	}
	mb, _ := json.MarshalIndent(m, "", "  ")
	if err := os.WriteFile(filepath.Join(cfg.Dir, "manifest.json"), mb, 0o644); err != nil {
		return nil, err
	}
	return m, nil
}

func buildTask(ctx context.Context, f *Fetcher, cfg *BuildConfig, t Task) (train, val []Example, err error) {
	rng := rand.New(rand.NewSource(seedFor(cfg.Seed, t.Name)))
	trainCap := int(float64(t.TrainCap) * cfg.CapScale)

	var trainRows, valRows []Row
	var meta Meta
	sameSource := !t.Holdout && t.Train == t.Val
	if !t.Holdout {
		limit := trainCap
		if sameSource {
			limit = trainCap + t.ValCap
		}
		trainRows, meta, err = f.Read(ctx, t.Train, limit, t.ScanCap, t.MaxFiles, rng)
		if err != nil {
			return nil, nil, err
		}
	}
	if sameSource {
		rng.Shuffle(len(trainRows), func(i, j int) { trainRows[i], trainRows[j] = trainRows[j], trainRows[i] })
		n := min(t.ValCap, len(trainRows)/5)
		valRows, trainRows = trainRows[:n], trainRows[n:]
	} else {
		var vmeta Meta
		valRows, vmeta, err = f.Read(ctx, t.Val, t.ValCap, t.ScanCap, t.MaxFiles, rng)
		if err != nil {
			return nil, nil, err
		}
		if meta.Names == nil {
			meta = vmeta
		}
		for k, v := range vmeta.Names {
			if _, ok := meta.Names[k]; !ok {
				meta.Names[k] = v
			}
		}
	}
	if t.DeriveNames != "" {
		set := map[string]bool{}
		for _, rows := range [][]Row{trainRows, valRows} {
			for _, r := range rows {
				if s := r.Str(t.DeriveNames); s != "" {
					set[s] = true
				}
			}
		}
		names := make([]string, 0, len(set))
		for n := range set {
			names = append(names, n)
		}
		sort.Strings(names)
		meta.Names[derivedNames] = names
	}
	conv := t.Prepare(meta)
	convert := func(rows []Row) ([]Example, error) {
		seen := map[string]bool{}
		var out []Example
		for _, row := range rows {
			for _, e := range conv(rng, row) {
				if err := e.Validate(); err != nil {
					return nil, err
				}
				if k := exampleKey(&e); !seen[k] {
					seen[k] = true
					out = append(out, e)
				}
			}
		}
		return out, nil
	}
	if train, err = convert(trainRows); err != nil {
		return nil, nil, err
	}
	if val, err = convert(valRows); err != nil {
		return nil, nil, err
	}
	if !t.Holdout && len(train) == 0 {
		return nil, nil, fmt.Errorf("no examples produced")
	}
	return train, val, nil
}
