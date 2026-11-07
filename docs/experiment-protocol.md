# Experiment Protocol

## Split discipline

1. **Train** (official UIT-VSFC train file): used for fitting models and for
   learning the TF-IDF vocabulary.
2. **Validation** (official dev file): used exclusively for hyperparameter
   selection and error analysis. Never used for fitting.
3. **Test** (official test file): touched only by `vfi evaluate`, after model
   selection for the milestone is finalized. Never used for tuning, and never
   merged with train/validation.

If the official validation file were unavailable, `vfi train` can carve a
deterministic stratified validation split from the train file
(`use_official_splits: false`, seeded). The test file is never generated.

## Metrics

- Primary selection metric: **macro-F1** over the canonical labels
  (`negative`, `neutral`, `positive`), computed with `labels=` forced so all
  classes appear even if absent from predictions.
- `zero_division=0` is passed explicitly: a class with no true and no
  predicted samples scores 0.0 (documented, not NaN).
- Also recorded: accuracy, macro precision/recall, per-class
  precision/recall/F1/support, 3x3 confusion matrix (rows = true label,
  columns = predicted label, canonical order).

## Latency protocol

Batch size 1; 10 untimed warm-up calls; 100 timed single-text calls sampled
uniformly with a seeded RNG; wall clock via `time.perf_counter`; reported as
mean/median/p95 in milliseconds. The protocol parameters are stored in the
evaluation JSON alongside the numbers.

## Hyperparameter search space (justified)

| Candidate | Analyzer | n-grams | C    | Rationale |
| --------- | -------- | ------- | ---- | --------- |
| 1         | word     | 1-2     | 1.0  | Word bigrams capture negation ("không tốt"); moderate regularization |
| 2         | word     | 1-2     | 4.0  | Weaker regularization for sparse word features |
| 3         | char_wb  | 2-5     | 1.0  | Robust to Vietnamese spelling/teencode variants |
| 4         | char_wb  | 2-5     | 4.0  | Same representation, weaker regularization |

`min_df=2`, `max_df=0.95`, `sublinear_tf=true`, `class_weight="balanced"`
are fixed corpus-scale priors, not tuned. Selection is by validation
macro-F1; ties break to the first candidate. This is a small, documented
search — not exhaustive tuning, and definitely not "fine-tuning a language
model".

## Reproducibility record (per experiment)

Every trained artifact's `meta.json` and every report JSON records:

- dataset source/URL, local fingerprints per split (sha256)
- random seed, Python version, scikit-learn version
- full model configuration + short config hash
- training/evaluation command (below), trial results, timings

Commands:

```powershell
# 1. Validate raw data, write reports/artifacts/data_report.json
vfi prepare-data --config configs/baseline.yaml

# 2. Train both baselines; select on validation; save artifacts + report
vfi train --config configs/baseline.yaml

# 3. Evaluate a saved artifact on the official test split
vfi evaluate --model-id tfidf-logreg --config configs/baseline.yaml
vfi evaluate --model-id majority --config configs/baseline.yaml
```

## Honesty rules

- No metric is reported unless produced by a real run on real, legitimately
  obtained data. Synthetic-fixture unit tests are not model evaluations.
- The test split is never used for any selection decision.
- Zero-division handling and any degenerate behavior (e.g., the majority
  baseline's one-hot probabilities) are documented, never hidden.
