# Phase 2 Evaluation Protocol

Status: **design only**. This protocol extends the released
[Phase 1 experiment protocol](../experiment-protocol.md). Every arm uses a
single selected, authorized dataset version, the same verified label
mapping, and the same effective train, validation, and test holdout
fingerprints. The primary metric is macro-F1 in canonical label order
(negative, neutral, positive), using the same metric code.

## 1. Result categories (mandatory labeling)

Every number in any Phase 2 table, plot, or claim is labeled with exactly one
of these categories. Mixing categories in one table is allowed only when each
row is labeled.

| Category | What it is | Produced by | May support selection? | May be quoted as "accuracy"? |
| -------- | ---------- | ----------- | ---------------------- | ---------------------------- |
| **A. Validation performance** | Metrics on the frozen effective validation split during/after selection | `vfi train` / fine-tuning loop | Yes — it is the selection signal | No — say "validation" |
| **B. Final test performance** | Metrics on the same frozen test holdout, produced once per final configuration (per seed) | `vfi evaluate` | Never | Yes — this is the headline |
| **C. Synthetic unit tests** | Hand-computed fixture metrics proving code correctness | pytest suite | Never | Never — "tests passing" is not model quality (Phase 1 honesty rule) |
| **D. Latency benchmarks** | Wall-clock timings under the fixed protocol (§4) | latency harness | Never | No — environment-relative, hardware-relative; not accuracy |
| **E. Author-reported context** | Numbers from the UIT-VSFC paper (e.g. MaxEnt ≈ 88% F1) or third-party papers | Citation | Never | No — different protocol, quoted as context only |

The UIT-VSFC authors' approximately 88% Maximum Entropy F1 is historical
category E context. The optional UIT-VSFC adapter still requires permission
and verified mapping. The published result cannot be a comparator for a
run on the selected independent dataset.

### Prerequisites and holdout access

Before any real fit, verify dataset rights and label semantics and run the
released all-split preflight. Retain the passing receipt covering dataset
identity, configuration, adapter provenance and split fingerprints. Only the
preflight and final evaluation paths read the raw test source. Fit and
selection have access to train and validation only, including segmentation
and preprocessing diagnostics.

Before starting the PhoBERT pilot, fit the two Phase 1 baselines on the
selected data. Retain actual validation reports and hashes of the frozen
artifacts. Once all final model choices are locked, score each on the same
test holdout once per predeclared seed. Evaluation requires that its current
data match the preflight and trained model provenance. Run a new comparison
on a changed dataset version, mapping or holdout.

## 2. Metrics recorded (exact fields)

Every evaluation — validation or test, any model — calls
`compute_metrics(y_true, y_pred)` and records the full returned dict:

```json
{
  "labels": ["negative", "neutral", "positive"],
  "accuracy": 0.0,
  "macro_precision": 0.0,
  "macro_recall": 0.0,
  "macro_f1": 0.0,
  "per_class": {
    "negative": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
    "neutral":  {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
    "positive": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
  },
  "confusion_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
  "n_samples": 0
}
```

Conventions (inherited, restated):

- Primary metric: **macro-F1** with `labels=` forced to the canonical three,
  so every class appears even if absent from predictions.
- Confusion matrix: 3×3, rows = true label, columns = predicted label,
  canonical order.
- `support` = true class count.
- Probabilities: models that support them report class probabilities
  (PhoBERT softmax over 3 logits). `majority` reports its degenerate one-hot
  values as-is, honestly, per Phase 1.
- For fine-tuned runs, additionally record per-epoch validation metrics and
  the selected epoch/step of the chosen checkpoint.

## 3. Zero-division and degenerate cases

- `zero_division=0` is passed explicitly everywhere: a class with no true
  **and** no predicted samples scores 0.0 — documented, not NaN, not an
  exception. A class with true support but zero predicted samples also
  scores 0.0 precision and recall under this policy.
- Degenerate outputs are reported, never excluded:
  - `majority`: constant predictions, one-hot probabilities.
  - A fine-tuned run that collapses to one class: full metrics + confusion
    matrix are recorded for that run/seed; the run is flagged in the report;
    it is neither dropped nor silently averaged away (it counts in the seed
    mean — that is what the mean is for).
- If a class is absent from the validation predictions of a search
  candidate, the trial's macro-F1 is still computed by the same rule; the
  report shows the per-class row so the 0.0 is attributable.

## 4. Latency protocol

Reused verbatim from Phase 1 (`measure_latency`), extended only with device
metadata:

- Batch size 1 (single text); 10 untimed warm-up calls; 100 timed calls on
  samples drawn uniformly with a seeded RNG (seed 42); wall clock via
  `time.perf_counter`; reported as mean / median / p95 in milliseconds.
- The protocol parameters are stored in the JSON next to the numbers.
- Phase 2 additions: every latency record also stores device name, precision
  (fp16/fp32), and whether the text path included segmentation (transformer
  path) — segmentation time is a pipeline cost and must be reported with the
  model's latency, not hidden.
- Measured configurations: all classical models on **local CPU**; PhoBERT on
  **both** local CPU and local GPU. Category D, always.
- Caveat recorded with the numbers: laptop-GPU timings vary with thermals
  and power state; they are environment-relative measurements, not
  deployment SLAs.

## 5. Repeated seeds and variance reporting

- Classical models: single deterministic run (seed 42); no variance to
  report.
- M1 uses a single pre-declared seed (42), with single-run claim language.
- M2 fine-tuned models use 3 seeds minimum ({42, 7, 2026}), 5 if the budget allows
  (`experiment-blueprint.md` §6). For each seed: validation macro-F1 of the
  selected checkpoint, test macro-F1 of that checkpoint. Reported as:
  per-seed table + mean ± std across seeds, both on validation and test.
- The mean is the headline; it is fixed in advance; no seed is dropped or
  re-weighted after seeing test output (`experiment-blueprint.md` §6).
- With 3 seeds, the std estimate itself is unstable — report it, but do not
  build claims on its magnitude; use claim phrasing from §6.

## 6. Confidence limitations and claim phrasing

What this protocol **cannot** deliver, and must never pretend to:

- **No confidence intervals on single runs.** A one-seed test macro-F1 comes
  with no CI. Bootstrap-over-samples CIs are allowed as a *descriptive*
  display of sampling noise but must be labeled as such and never called
  "significance".
- **No significance tests across few runs.** With 3-5 seeds, a t-test across
  runs is statistically fragile; it is not part of this protocol.
- **Optional descriptive paired test.** McNemar's test on paired test-set
  predictions (same test samples through both models) may be reported once,
  post-hoc, to characterize disagreement between two models' error patterns.
  It is descriptive support for a direction, never the basis of a
  "statistically significant" headline, and it uses the test predictions
  that already exist — it triggers no new test evaluation.

| Claim situation | Allowed phrasing | Forbidden phrasing |
| --------------- | ---------------- | ------------------ |
| Single-seed fine-tune vs classical baseline | "In this run, fine-tuned PhoBERT reached validation macro-F1 X vs Y for TF-IDF+LogReg; seed variance is unmeasured." | "PhoBERT significantly outperforms the baseline." |
| 3-5 seeds, consistent direction, gap > seed std | "Across 3 seeds, fine-tuning averaged X ± Y vs Z for the baseline; the direction was consistent in all runs." | "Proven better (p < 0.05)." (no such test in this protocol) |
| 3-5 seeds, overlapping ranges | "Fine-tuning and the baseline are comparable within measured seed variance (X ± Y vs Z ± W)." | Any "better/worse" language for the overlapping quantity. |
| Difference smaller than measured seed std | Explicitly "comparable within variance" | "Higher" based on point estimates alone. |
| Comparing to author-reported ≈88% F1 | "For context, the corpus authors report ≈88% F1 with a Maximum Entropy baseline under a different protocol." | Any table row implying protocol-equal comparison. |
| Any claim beyond this dataset | "On the selected dataset and version, on its held-out test split, under this protocol." | Generalization to "Vietnamese sentiment" or "student feedback in general". |

Rule of thumb: every comparative sentence names (1) the split, (2) the
number of runs/seeds, and (3) the variance when measured.

## 7. Error-analysis procedure

Scope: **validation split only** during Phase 2. The test split gets no
error analysis while any selection or retraining remains possible; a test
error-analysis section may be written only in the final Phase 2 report,
after all test evaluations are done and labeled as post-hoc.

Procedure:

1. **Dump.** For the final (selected) model configuration, write a per-sample
   validation dump locally (gitignored): line index, true label, predicted
   label, predicted probability of the predicted class, and (for the
   fine-tuned model) the probability vector. No dumps for search candidates
   — analysis follows selection, not the reverse.
2. **Cell counts.** Aggregate misclassifications by confusion cell
   (true → predicted). Report the 3×3 matrix of error counts and per-cell
   share of all errors.
3. **Recurring categories, evidence-gated.** A "recurring error pattern"
   (e.g. "short declarative praise misread as neutral") may be stated only
   with the cell count, the share of the cell it covers, and at least three
   locally reviewed examples. Keep indices and examples local. Publish
   only aggregate counts and non-identifying categories.
4. **Cross-model comparison.** Where classical and deep models err
   differently, compare per-cell rates on the same validation samples
   (paired by line index) — this is the honest way to say "the transformer
   fixes X but breaks Y".
5. **Privacy.** Per-sample dumps and raw feedback stay local in ignored
   paths. Committed findings use aggregate counts and non-identifying
   categories. Phase 1 now emits privacy-safe aggregate data reports and
   includes tracked-file and ignore protections. Keep the same restrictions
   for Phase 2-derived files, including checkpoints and segmentation caches.

## 8. Reproducibility metadata (per experiment)

Training and evaluation records extend the Phase 1 evidence. Keep the
preflight receipt separately from training metadata. Attach its test
fingerprint and test-derived metadata to final evaluation records.

| Field | Content |
| ----- | ------- |
| `dataset` | ID/version, adapter identity and version, rights evidence reference, verified mapping and policy, source and effective fingerprints for the splits used, admissible aggregate counts |
| `preflight` | Passing receipt identity and SHA-256, effective and source fingerprints, and matching recorded test fingerprint in final evaluation |
| `derived_data` | segmentation tool + version, segmented source fingerprints for the splits accessed, `max_seq_length`, truncation/padding policy, p99 train token length. Holdout-derived diagnostics go only in evaluation. |
| `code` | git SHA of the code commit used, `vfi` package version |
| `config` | full configuration dict + `config_hash` (same scheme as Phase 1) |
| `libraries` | Python version; torch, transformers, tokenizers, CUDA/driver versions (deep arm); scikit-learn (classical arm); py_vncorenlp or underthesea version |
| `hardware` | device name (e.g. RTX 3060 Laptop 6 GB), precision used, batch size, grad-accum factor |
| `seeds` | all of: data seed, torch/numpy/python random seeds, dataloader seed, latency RNG seed |
| `search` | full trial list (candidate params, per-candidate best validation macro-F1, fit seconds) when a search ran |
| `checkpoint` | selected epoch/step, its validation macro-F1, sha256 of the checkpoint file |
| `artifacts` | sha256 of every saved artifact file referenced by the report |
| `commands` | exact command lines for every recorded number |
| `timings` | per-run and total wall time (also feeds the `[ESTIMATE]` refinements in the matrix) |

Rule: a number in a report whose provenance fields are incomplete is not a
reportable number — it is an anecdote and gets labeled as such or omitted.

## 9. Reporting rules

- **Committed** (`reports/`, markdown): aggregates, category-labeled tables
  (§1), per-class tables, confusion matrices, seed means ± std, error-
  analysis findings per §7, and non-sensitive provenance summaries.
- **Local only** (gitignored): raw dataset, segmented splits, per-sample
  dumps, model checkpoints, raw training logs.
- **No numbers without a category label and provenance.** A table mixing
  validation and test rows without §1 labels is a protocol violation.
- **Test evaluations are counted.** The report states how many times the
  test split was evaluated during Phase 2 and for which configurations;
  the expected count is one per final configuration and seed (three total
  for M1 if all models qualify). An integrity-only preflight is not a
  scoring run. Document any departure from the ledger.
