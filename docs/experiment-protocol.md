# Experiment Protocol

## Split discipline

1. **Train** is used for fitting models and learning the TF-IDF vocabulary.
2. **Validation** is used for hyperparameter selection and error analysis.
3. **Test** is a separate holdout accessed only by `vfi evaluate`. Before
   training, `--preflight-only` checks integrity without scoring a model or
   reporting the canonical test label distribution. For NEU-ESC it does
   record Toxic policy counts. Score test only after finalizing model
   selection. Keep test out of fitting, tuning, and error-driven iteration.

With `data.use_official_splits: false`, the commands consistently derive a
stratified validation subset from the physical train file using
`validation_size` and `seed`. Identical normalized texts stay together.
Fingerprints and sizes describe the effective subsets. With no separate test
file, you can train and select on validation, but preflight and final test
scoring fail until a holdout is available.

`prepare-data` summarizes the effective train and validation data. `train`
independently rejects cross-split matches between those subsets before fitting.
`evaluate --preflight-only` checks all three split pairs. Reports contain counts
of distinct overlapping texts, counts per pair, and whole-split fingerprints.
Exact matching uses Unicode and whitespace normalization. Cross-split matches
and contradictory labels for the same normalized text fail with exit code 1.
Review the conflicts under the dataset's annotation protocol. The optional
`prepare-data --near-duplicates` command compares up to 500 train/validation
records using character TF-IDF similarity. Its bounded candidate count is an
audit signal and does not remove rows or block training. Review sources and
collection groups separately when they could correlate across splits.

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

`training_report.json` and model sidecars fingerprint the ordered normalized
text and label pairs used for fitting and selection. Sidecars also record the
configuration, seed, Python and scikit-learn versions, training start time, and
the artifact's SHA-256 checksum. Adapter provenance records dataset version,
mapping, policy, and source and effective split fingerprints. CSV and
UIT-VSFC bind source bytes. JSONL binds parsed records and annotation
provenance. Evidence and
policy explanations are stored as hashes in model and report metadata.
Training reads only sources for fit and selection, including when test files
are unavailable.

To score a benchmark, `evaluate` requires a passing `evaluation_preflight.json`
created before fitting. It checks the current configuration and split fingerprints
against the receipt and the model's training provenance. A change to the test
snapshot, fit/selection data, or configuration blocks scoring. Rechecking an
unchanged snapshot preserves the original receipt time. For a changed snapshot,
run preflight and train again, and report the change as a separate benchmark.
Evaluation JSON identifies the exact test fingerprint and the training and
artifact provenance. Keep the preflight report alongside the run's artifacts.
Preflight also binds source provenance such as a change to a reviewed JSONL
record's annotation reference even when its text and label stay the same.
Synthetic fixture evaluations are explicitly labeled and are software checks.

Load only locally trained artifacts from directories you control. Deserializing
Joblib can execute arbitrary code. A sidecar and checksum can detect malformed
metadata or accidental changes. Anyone who supplies a model file can also forge
its sidecar. Retrain legacy artifacts that lack a checksum or the required
evaluation provenance. Audited PR #1 models with a valid checksum remain
loadable for local prediction and must be retrained for the integrated benchmark.
Use separate model and report directories for concurrent or independent runs.

Commands:

```powershell
# 1. Validate train/validation, then preflight all splits before fitting
vfi prepare-data --config configs/baseline.yaml
vfi evaluate --preflight-only --config configs/baseline.yaml

# 2. Train both baselines; select on validation; save artifacts + report
vfi train --config configs/baseline.yaml

# 3. Evaluate a saved artifact on the declared test split
vfi evaluate --model-id tfidf-logreg --config configs/baseline.yaml
vfi evaluate --model-id majority --config configs/baseline.yaml
```

## Honesty rules

- No metric is reported unless produced by a real run on real, legitimately
  obtained data. Synthetic-fixture unit tests are not model evaluations.
- The test split is never used for any selection decision.
- Repeatedly inspecting test results and changing the model in response amounts
  to implicit test tuning. Finalize selection before scoring and record any
  subsequent iterations separately.
- Zero-division handling and any degenerate behavior (e.g., the majority
  baseline's one-hot probabilities) are documented, never hidden.
- The baseline training command requires all three canonical classes in the
  effective train set. For prediction, Logistic Regression provides one column
  per fitted class. A trusted binary artifact lists omitted classes in
  `untrained_labels`. The majority baseline reports a constant one-hot vector
  over the canonical labels.
