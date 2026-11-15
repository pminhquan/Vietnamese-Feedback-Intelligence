# Phase 2 Pilot Experiment Plan — Milestone M1

Status: **design only**. No model has been run for this pilot. Real runs
need a selected authorized dataset, verified labels, a passing preflight,
two real classical baseline artifacts with validation evidence, and
approval for the deep-learning stack and compute. Runtime and VRAM figures
are [ESTIMATE] planning scenarios. Strategy and cost details live in
[experiment-blueprint.md](experiment-blueprint.md) and
[experiment-matrix.md](experiment-matrix.md). For the historical hardware
and preprocessing reasoning, see [compute-feasibility.md](compute-feasibility.md)
and [tokenization-design.md](tokenization-design.md).

Evidence labels: `[VERIFIED FACT]`, `[MEASURED]`, `[ESTIMATE]`,
`[ASSUMPTION]`, `[UNVERIFIED]`, `[BLOCKED]`.

## 0. Headline decisions (the ten that define the pilot)

1. **Exactly three models in M1**: `majority` (Phase 1 code), `tfidf-logreg`
   (Phase 1 code), `phobert-base-ft` (one fine-tuned
   `vinai/phobert-base`). Nothing else is trained in M1.
2. **No hyperparameter search.** One pre-declared PhoBERT configuration
   (§5), fixed before the first run. Search is the pre-designed M2
   escalation, not a pilot activity.
3. **Single seed (42) for the pilot.** Multi-seed {42, 7, 2026} is the
   pre-declared M2 escalation path (§4), never a pilot fallback triggered
   by disliking a result.
4. **Approve and freeze one dataset before fitting.** Verify rights, mapping,
   and a separate holdout. Pass the integrity preflight, then retain real
   majority and TF-IDF+LogReg validation results with both artifacts
   frozen before any PhoBERT fit (§2).
5. **max_length 256** (PhoBERT's model maximum), right-side truncation,
   measured length distribution and truncation rate recorded (§3.4).
6. **Effective batch 32 via micro-batch 8 × gradient accumulation 4**
   `[ESTIMATE]` — sized for the `[MEASURED]` 6 GB VRAM but not guaranteed
   before measurement; the OOM ladder (§7.2) is pre-declared, not improvised.
7. **Validation macro-F1 is the only selection metric.** Preflight reads the
   test source for integrity. Final scoring runs once per frozen model, with
   all fits locked before any score is released (§6, §8).
8. **Same metric code as Phase 1** (`vfi.evaluation.metrics.compute_metrics`)
   and the same documented latency protocol (`measure_latency`), extended
   with device/precision metadata and segmentation time reported separately
   (§8, §12).
9. **Logs and metadata carry the reproducibility chain**: per-epoch JSONL
   with no raw text; `meta.json` extends the Phase 1 provenance pattern with
   torch/transformers/segmenter versions, git SHA, per-stage seeds,
   tokenizer/segmenter config hashes, and artifact sha256 (§9, §10).
10. **XLM-R is not mandatory in M1** — it is the fallback only if PhoBERT
    itself is blocked (checkpoint/license failure, or segmentation
    determinism cannot be shown); see §1.2 for the reasoning.

## 1. Models under comparison in M1

### 1.1 The three models

| ID | Model | Code origin | Input text | Trained in M1? |
| -- | ----- | ----------- | ---------- | -------------- |
| `majority` | Majority-class classifier, Phase 1 (`MajorityBaseline`, canonical-label tie-break) | Phase 1, reused unchanged | train labels | Yes, deterministic count |
| `tfidf-logreg` | TF-IDF + Logistic Regression, Phase 1 4-candidate search on validation, select by validation macro-F1, ties to first candidate | Phase 1, reused unchanged (`run_small_search`) | normalized raw text, vectorizer-internal tokenization | Yes, minutes `[ESTIMATE]` |
| `phobert-base-ft` | `vinai/phobert-base` (~135M params `[VERIFIED FACT]` card/config), full fine-tune, linear head on first token (`<s>`), dropout 0.1 `[ESTIMATE]` | New Phase 2 training loop (this plan) | NFC-normalized → word-segmented → BPE, max 256 | Yes, one fixed-recipe run, seed 42 |

Run both classical models using the released Phase 1 protocol on the same
frozen dataset, mapping and splits planned for PhoBERT. Retain their actual
validation metrics, configuration, preflight provenance, and artifact
hashes before starting the deep arm.

### 1.2 Why XLM-R is not mandatory for M1 (and what it is instead)

- **The M1 question is directional**: "does a fine-tuned Vietnamese
  pretrained model beat the strong classical baseline under this protocol?"
  One deep model answers it. A second pretrained model answers a *different*
  question (does the choice of pretraining recipe change the ranking?) —
  that is the M2+ escalation, and `experiment-blueprint.md` §2 already
  marks it OPTIONAL.
- **Budget honesty**: on the `[MEASURED]` 6 GB laptop GPU, runs are serial
  (§7); every additional model spends the same scarce GPU-hours. Spending
  them on M2 seeds of the selected model buys more valid inference than a
  second architecture buys novelty.
- **Segmentation asymmetry**: XLM-R takes raw-syllable input (no segmenter)
  `[ASSUMPTION per model-landscape.md]`; adding it to M1 would confound the
  pilot's tokenization contract with a second preprocessing chain the pilot
  does not need.
- **Declared fallback role**: XLM-R-base replaces `phobert-base-ft` in M1
  **only if PhoBERT is blocked** — (a) the checkpoint cannot be legitimately
  obtained/verified (license tag re-check fails), or (b) segmentation
  determinism cannot be demonstrated (§13b). If that substitution happens,
  it is a documented protocol deviation with owner approval, and every
  claim states that the deep model is XLM-R-base, not PhoBERT.

## 2. Data contract

### 2.1 Dataset layout and access

Select one authorized local adapter using the
[Phase 1 data layouts](../../data/README.md). The default independent
workflow exports human-approved JSONL to physically separate `train.jsonl`,
`validation.jsonl`, and `test.jsonl`. Authorized CSV adapters use separate CSV
files for the same splits. Record the label mapping and any exclusions for
the chosen adapter. NEU-ESC requires provider access approval and an explicit
policy for Toxic. The five-level CSV mapping and domain limits must travel
with its results.

UIT-VSFC is optional and remains blocked without its usage agreement and a
verified raw label mapping. It is outside the current plan. Its historical
roughly 16k-row scale is used only for the compute estimates in §7.

### 2.2 Fingerprint expectations

1. Before fitting, use the selected dataset configuration to run `vfi prepare-data`
   and `vfi evaluate --preflight-only`. Retain the passing
   `evaluation_preflight.json`: identity/version, adapter and mapping
   provenance, source digests, and effective record fingerprints for all
   three splits.
2. Run `vfi train` with that configuration for the two classical fits.
   Training metadata contains effective train and validation fingerprints
   and configuration and artifact hashes. Save both real classical
   validation reports and their frozen artifact evidence before fine-tuning.
3. Segment and fingerprint train and validation for fit and selection.
   Produce and fingerprint the holdout's segmented sequence only in final
   evaluation. Final scoring must match the preflight and training record.
   Any change to the data, mapping, or holdout requires a fresh comparison.

### 2.3 Behavior on schema violation

| Violation | Behavior |
| --------- | -------- |
| Missing or malformed train/validation split, empty feedback, invalid labels or annotation | Fail before fit. The adapter reports safe split and location details where available. |
| No independent test holdout | Allow validation-only work. Preflight and benchmark scoring remain blocked. |
| Cross-split exact duplicates or contradictory labels | Fail preflight. Resolve locally and record the policy without exposing feedback. |
| Possible near-duplicate matches | Review locally. The bounded audit is advisory and does not remove records. |
| Mismatched identity, mapping, or fingerprint during final evaluation | Fail before scoring and establish a new preflight and fit record if the dataset version changes. |

### 2.4 Verify label semantics

Audit the selected dataset's documented labels and local human annotations,
then freeze their mapping into the canonical classes. Resolve unsupported
or ambiguous labels before training classical or deep models. Summary
scores cannot verify semantics; a label permutation can preserve aggregate
scores while changing the class meanings. The optional UIT-VSFC adapter's
historical 0/1/2 conflict and alternatives are documented in
[label-semantics-evidence.md](label-semantics-evidence.md); neither meaning
may be assumed if that adapter is ever authorized.

## 3. Tokenization contract

The three models intentionally consume different text pipelines; the
differences are documented, not hidden (`experiment-blueprint.md` §4). All
chains start at the same place: Phase 1's `normalize_text` (NFC + whitespace
collapse), reused unmodified.

### 3.1 Per-model preprocessing chain (exact)

| Step | `majority` | `tfidf-logreg` | `phobert-base-ft` |
| ---- | ---------- | -------------- | ----------------- |
| 1. Load | Selected Phase 1 adapter | Same adapter and records | Same effective fit and selection records |
| 2. Normalize | `normalize_text` (in loader) | `normalize_text` (in loader) | `normalize_text` (in loader) |
| 3. Word segmentation | none | none | `py_vncorenlp` RDRsegmenter, `annotators=["wseg"]`, `word_segment(text)` → underscore-joined string; then a defensive whitespace re-collapse `[VERIFIED FACT for the requirement; chain per tokenization-design.md §7]` |
| 4. Tokenization | none needed | `TfidfVectorizer` internals (`lowercase=True`, word 1-2 / char_wb 2-5) | `AutoTokenizer.from_pretrained("vinai/phobert-base")` (fastBPE, 64,001 types `[VERIFIED FACT]`) |
| 5. Length policy | n/a | unbounded (sparse) | `max_length=256`, truncation side **right**, dynamic padding to the longest sequence in batch |
| 6. Labels | canonical strings | canonical strings | canonical ids (`negative=0, neutral=1, positive=2`) |

Rationale preserved from Phase 1: no lowercasing/diacritic stripping in the
shared normalizer (signal-bearing in Vietnamese); TF-IDF lowercases inside
the vectorizer (config-recorded); PhoBERT's tokenizer is case-preserving.

### 3.2 Segmenter call and determinism expectations

- Tool: `py_vncorenlp` (RDRsegmenter, word-segmentation-only mode) — the
  segmenter PhoBERT's card recommends, i.e. the fidelity-first choice;
  VnCoreNLP tool + pre-trained models are **GPLv3** per
  `tokenization-design.md` §4 and `experiment-matrix.md` §2
  `[VERIFIED FACT]` — acceptable for local research use, re-examine before
  any Phase 3 serving. Requires Java 1.8+; JVM startup is amortized by batch
  segmentation.
  `underthesea` (`word_tokenize(..., format="text")`) is the pinned,
  documented fallback — a different segmenter, never mixed with RDRsegmenter
  within one experiment.
- Treat the segmenter as a **pure `str -> str` function**. Cache train
  and validation before fitting, and cache test inside final evaluation.
  Rebuild derived files from the frozen sources, keep them local, and
  record their fingerprints (§2.2).
- **Check determinism before fitting.** Segment train and validation twice
  in separate processes and compare the fingerprints. Use the same pinned
  transformation on the holdout during final evaluation. A golden-file
  test additionally pins N fixed sentences to an
  exact expected underscore-joined output for the recorded tool+model
  version, failing loudly on version drift. If determinism cannot be shown,
  the PhoBERT arm halts (§13b).
- The segmented text stays **local-only** (gitignored), same as raw data.

### 3.3 Identical treatment across splits

One segmentation run per split, same tool, same version, same code path —
with identical options. The training loop consumes only train/validation
caches. The final evaluation path alone segments the holdout. For inference
latency (§12), run the same chain per text and include segmentation in the
timing to measure end-to-end cost.

### 3.4 Truncation policy (max_length 256)

- `256` is PhoBERT's model maximum `[VERIFIED FACT — card/paper]`; sequences
  longer than 256 subword tokens are truncated from the **right**
  (single-sequence inputs; sentiment cues near the end are lost by design
  and documented).
- Before the first fine-tune, measure length on the **segmented train split**
  (p50/p95/p99/max) and record train and validation truncation rates.
  Record the holdout rate only in final evaluation.
- If the measured tail is meaningful (p99 > 256 or truncation rate > ~1% on
  any split), record the limitation. Freeze the policy before holdout scoring.

### 3.5 Segmentation error logging (no raw text ever)

- Every segmentation failure is logged with: split name, line index,
  exception class name, and a category code
  (`EMPTY_OUTPUT` | `JVM_ERROR` | `TIMEOUT` | `ENCODING`). Never the input
  text, never the partial segmented output, never a text payload inside an
  exception message (wrap the segmenter call so text never reaches a logger
  or traceback file).
- A per-split failure summary (count per category) goes into the run report;
  failures above a threshold (any `JVM_ERROR`, or > 0.1% of lines
  `[ESTIMATE]`) halt the pipeline before training.
- Unit tests assert the sanitizer: a deliberately failing segmenter fixture
  must produce a log line containing no substring of the input text.

## 4. Reproducible seed policy

Single-seed pilot, seed **42** everywhere (Phase 1's default); no exceptions
in M1.

| Where | Seed | Mechanism |
| ----- | ---- | --------- |
| Data preparation (integrity gate, fingerprinting, latency sampling) | 42 | Phase 1 config `seed`; `np.random.default_rng(42)` inside `measure_latency` `[VERIFIED FACT from source]` |
| Classical fitting (`majority`, `tfidf-logreg`) | 42 (`random_state`) | Deterministic given fixed data; one confirmation repeat with seed 43 is allowed (minutes `[ESTIMATE]`) and recorded if run |
| PhoBERT run (M1 pilot) | 42 | `torch.manual_seed`, `torch.cuda.manual_seed_all`, `numpy.random.seed`, `python.random.seed`, and the DataLoader shuffle `generator` all set to 42 before model init; model init, dropout masks, and shuffling all derive from it |
| Escalation (M2, pre-declared) | {42, 7, 2026} | Three independent runs, identical hyperparameters, differing only in the seed above; mean ± std reported; headline = mean, fixed in advance; no seed dropped after seeing test output |

Rules:

- The pilot's single-seed claim phrasing is mandatory: "in this run", no
  variance language (`evaluation-protocol.md` §6 table, row 1).
- CUDA nondeterminism is not bitwise-eliminated (no
  `torch.use_deterministic_algorithms(True)` in M1 — it can error or
  silently change performance on some ops); the residual nondeterminism is
  documented in metadata, not hidden (`experiment-blueprint.md` §6).
- Escalation seeds {42, 7, 2026} supersede the older {42, 43, 44(, 45, 46)}
  examples; `experiment-blueprint.md` §6 and `evaluation-protocol.md` §5 are
  updated accordingly (reconciled in the 2026-11-07 QA pass).

## 5. Pre-declared PhoBERT configuration (no search)

Exactly one configuration, fixed before the first run. All quantitative
values below are `[ESTIMATE]` unless marked otherwise — none is measured on
this hardware yet; the first run replaces them.

| Field | Value | Evidence |
| ----- | ----- | -------- |
| Checkpoint | `vinai/phobert-base` (~135M params, hidden 768, 12 layers, vocab 64,001, `max_position_embeddings=258`) | `[VERIFIED FACT]` HF card + config.json; license tag `license:mit` re-verified at download and recorded |
| Head | Linear(768 → 3) on the first token (`<s>`), dropout 0.1 | `[ESTIMATE]` standard recipe, no tuning |
| Optimizer | AdamW, LR 3e-5 | `[ESTIMATE]` common BERT-base fine-tune center; deliberately not searched in M1 |
| LR schedule | Linear warmup over the first 10% of total optimizer steps, then linear decay to 0 | `[ESTIMATE]` |
| Weight decay | 0.01 (excluded from bias and LayerNorm terms) | `[ESTIMATE]` |
| Gradient clipping | max_norm 1.0 | `[ESTIMATE]` |
| Batch geometry | micro-batch 8 × gradient accumulation 4 = **effective batch 32** | `[ESTIMATE]` sized for 6 GB at seq 256 (`compute-feasibility.md` §3.2: batch 8 comfortable, 16 tight, 32 OOM); not guaranteed before measurement |
| Max sequence length | 256, right truncation (§3.4) | `[VERIFIED FACT]` model max |
| Precision | AMP fp16 on GPU; fp32 if CPU fallback | fixed per device |
| Epochs | ≤ 5 with early stopping (§6) | `[ESTIMATE]` |
| Steps | optimizer updates/epoch ≈ ceil(11,426 / 32) ≈ 357 `[ESTIMATE]`; micro fwd/bwd steps/epoch ≈ ceil(11,426 / 8) ≈ 1,429 `[ESTIMATE]` | arithmetic |
| Optimizer state grouping | bias/LayerNorm no-decay; otherwise PyTorch AdamW defaults | `[ESTIMATE]` |

**No hyperparameter search in M1** — not "a small search": zero candidates
beyond this one. The search design (LR {2e-5, 3e-5, 5e-5}) stays pre-written
in `experiment-blueprint.md` §7 as the M2 escalation. If M1's validation
results motivate a different LR, that motivates M2 — it does not license an
M1 retry loop on validation.

## 6. Checkpoint selection and early stopping

- **Evaluation cadence**: validation macro-F1 (same metric code,
  `zero_division=0`, canonical labels) computed at the end of every epoch —
  ≤ 5 validation passes per run, affordable at ≈1.6k validation sentences
  `[ESTIMATE]`.
- **Selection metric**: validation macro-F1 only. Accuracy, per-class F1,
  confusion matrix are recorded alongside but never select.
- **Save-best-on-validation**: after each epoch's validation, if macro-F1
  strictly improves on the best so far, overwrite the `best` checkpoint
  (ties break to the **earliest** epoch — a later equal score does not
  overwrite). The `last` checkpoint is kept for crash forensics only.
  Storage: best + last; sha256 of each saved file recorded (§10).
- **Early stopping**: stop after 2 consecutive end-of-epoch evaluations
  without strict improvement (patience 2). With ≤ 5 epochs this bounds every
  run at 5 epochs; patience does not extend the epoch cap.
- **Freeze, then test once**: after the run, the selected checkpoint is
  frozen (config + selected epoch + val macro-F1 + sha256 recorded); the
  test split is evaluated **exactly once** for this configuration, through
  the evaluation path only (§8). No test-informed decision of any kind
  before that single evaluation, and no second test evaluation for the same
  configuration afterwards.
- **M1 test ledger** (expected total: 3 entries): `majority` (1),
  `tfidf-logreg` selected candidate (1), `phobert-base-ft` seed-42 best
  checkpoint (1). The count and the configurations are stated in the report
  (`evaluation-protocol.md` §9).
- **Artifact rule** (pre-fixed): the seed-42 best-validation checkpoint of
  `phobert-base-ft` is the M1 deep artifact carried forward. If M2 runs, its
  artifact is likewise the seed-42 run of the (possibly re-selected) config
  — the rule is stated before any run to prevent post-hoc artifact swapping.

## 7. Training budget and OOM recovery

### 7.1 Wall-clock `[ESTIMATE]` per stage, with uncertainty

These estimates use the measured laptop hardware (RTX 3060 Laptop 6 GB,
i5-10500H, 32 GB RAM) and the historical UIT-VSFC-sized planning scenario.
Recalculate using the selected dataset. The GPU-time uncertainty factor
is approximately 2× due to thermals and background load.

| Stage | Wall clock | Basis |
| ----- | ---------- | ----- |
| Environment bootstrap (venv, CUDA torch wheel, transformers, py_vncorenlp + JVM, model downloads: PhoBERT ≈1.2 GB `[VERIFIED FACT]`, VnCoreNLP models ≈115 MB `[VERIFIED FACT]`) | ≈ 1-2 h one-time, download-bound | `[ESTIMATE]` |
| Integrity gate + label audit (human steps) | ≈ 2-4 h elapsed, mostly human latency | `[ESTIMATE]` |
| Segmentation (≈16k sentences, batched JVM) | ≈ 5-15 min | `[ESTIMATE]` |
| `majority` | seconds | trivial |
| `tfidf-logreg` (Phase 1 4-candidate search on ≈11.4k docs) | ≈ 10-20 min | `[ESTIMATE]` per `compute-feasibility.md` §3.1 |
| One PhoBERT fine-tune run (≤5 epochs, AMP, effective batch 32, seq 256) | ≈ 5-9 min/epoch → **≈ 20-45 min/run** including ≤5 validation passes | `[ESTIMATE]`: fwd+bwd ≈ 6 × 135M × 8 × 256 ≈ 1.7e12 FLOPs per micro-step at 5-8 TFLOPS sustained (20-30% MFU `[ASSUMPTION]`) ⇒ ≈0.2-0.35 s/micro-step; per compute-feasibility.md §3.2 ≈40-70 samples/s ⇒ ≈3-7 min/epoch — the two methods bracket the range |
| Latency harness (PhoBERT CPU + GPU + segmentation timing) | ≈ 15-30 min CPU + minutes GPU | `[ESTIMATE]` (CPU ≈0.3-1 s/call × 110 calls) |
| Error analysis (validation, human) | ≈ 1-2 h | `[ESTIMATE]` |
| **M1 total compute** | ≈ 2-4 h GPU-adjacent wall clock, spread over 1-2 calendar days | `[ESTIMATE]` |

CPU-only fallback: ≈10-25× slower ⇒ ≈ 3.5-19 h for the single M1 fine-tune
`[ESTIMATE]` — tolerable once as an emergency path, not a planning
assumption (`compute-feasibility.md` §3.2).

### 7.2 OOM recovery procedure (pre-declared ladder)

No batch size is guaranteed before measurement. If CUDA OOM occurs:

1. Halve the micro-batch and double gradient accumulation —
   8×4 → **4×8** → (if needed) **2×16** → (if needed) **1×32**. Effective
   batch stays 32, so the optimizer-step arithmetic (LR schedule, warmup
   count) is unchanged; only wall clock grows `[ESTIMATE]` (more micro-steps
   per optimizer step, less GPU utilization per step).
2. Each deviation is recorded in the run's metadata (`hardware` block:
   micro_batch, grad_accum) and in the report.
3. fp16 non-finite loss / NaN validation is a **separate** failure: switch
   that run to fp32, record the deviation, re-run once
   (`experiment-blueprint.md` §11 row). Not an OOM.
4. If OOM persists at micro-batch 1 × seq 256 × fp16 — `[ESTIMATE]`
   extremely unlikely for a 135M model on 6 GB — that is a hardware-failure
   verdict, not a tuning problem: §13c applies.

## 8. Validation protocol vs evaluation protocol

| | Validation split | Test split |
| - | ---------------- | ---------- |
| When computed | Freely and repeatedly: per-epoch during training, checkpoint selection, error analysis, diagnostics, the §13d floor check | Exactly once per final configuration, after selection froze |
| Where computed | In the training loop | Only by the dedicated evaluation path, after checking the holdout against preflight and model provenance |
| Metrics | `compute_metrics` full schema; macro-F1 drives decisions | Same `compute_metrics` full schema |
| Artifacts | Per-epoch JSONL (§9), per-sample dump for error analysis (local, gitignored) | One ledger-recorded evaluation JSON per configuration, committed (aggregates only) |
| May it change the model? | Yes, before final selection freezes | Keep selection closed. Test-informed changes require a separate comparison and fresh holdout. |

Metrics schema (identical to Phase 1, `[VERIFIED FACT from
vfi.evaluation.metrics]`): accuracy, macro precision/recall/F1, per-class
precision/recall/F1/support over the canonical labels, 3×3 confusion matrix
(rows = true, columns = predicted, canonical order), `n_samples`;
`zero_division=0` explicit everywhere.

Latency (both splits' models, after training completes):

- Protocol: `measure_latency` reused verbatim — batch size 1, 10 untimed
  warm-up calls, 100 timed single-text calls sampled uniformly with seed 42,
  `time.perf_counter`, mean/median/p95 ms, protocol parameters stored
  alongside `[VERIFIED FACT from source]`.
- Configurations: classical models on **local CPU**; `phobert-base-ft` on
  **both** local CPU and local GPU; precision (fp16/fp32) recorded.
- Segmentation time is reported **separately** from model forward time
  (§12) — it is a pipeline cost and must be visible, not averaged into the
  model number.

## 9. Training logs

One JSONL file per fine-tune run (`logs/<run_id>/epochs.jsonl`, local-only,
gitignored), one JSON object per line, written at each end-of-epoch
validation:

```json
{"run_id": "phobert-base-ft_m1_seed42", "seed": 42, "epoch": 1,
 "optimizer_step": 357, "micro_steps": 1428,
 "train_loss_epoch_mean": 0.0,
 "val_macro_f1": 0.0, "lr_last": 3e-05,
 "wall_epoch_s": 0.0, "wall_cumulative_s": 0.0,
 "vram_peak_mib": 0,
 "micro_batch": 8, "grad_accum": 4,
 "max_seq_length": 256, "precision": "fp16",
 "truncated_train_fraction": 0.0,
 "early_stop_pending": false}
```

- `vram_peak_mib` from `torch.cuda.max_memory_allocated()` when CUDA is
  available; `null` otherwise (recorded as "unavailable", not 0).
- `train_loss_epoch_mean` is the mean over micro-batches of that epoch.
- Full validation metrics (not just macro-F1) go to a per-epoch sidecar JSON
  (`logs/<run_id>/val_epoch_<n>.json`), local-only.
- Classical models keep Phase 1's report JSONs (trial list, fit seconds) —
  no new log format invented for them.

**Log hygiene rules** (required and testable):

1. Keep raw feedback and decoded token sequences out of logs and error messages
   (§3.5 sanitizer).
2. Keep identifying metadata and any identifiers found in feedback out of logs.
   Keep per-sample indices local.
3. Line indices and split names are the only per-sample keys allowed.
4. Committed reports carry aggregates and category labels
   (`evaluation-protocol.md` §9); raw JSONL stays local.

## 10. Final artifact metadata (`meta.json` extension)

Phase 1's provenance pattern (`build_training_metadata`: created_at, config,
config_hash, dataset_fingerprints, sklearn_version, python_version, seed
`[VERIFIED FACT from source]`) extends to the example M1 deep-arm
`meta.json` below. Keep holdout data and derived diagnostics in the
separate preflight and final evaluation evidence.

```json
{
  "created_at": "<UTC ISO>",
  "model_id": "phobert-base-ft",
  "config": {"lr": 3e-05, "weight_decay": 0.01, "warmup_ratio": 0.1,
              "max_seq_length": 256, "micro_batch": 8, "grad_accum": 4,
              "epochs_cap": 5, "patience": 2, "precision": "fp16",
              "head": "linear_cls_dropout0.1"},
  "config_hash": "<same scheme as Phase 1>",
  "code": {"git_sha": "<exact commit of the run>", "vfi_version": "<x.y.z>"},
  "dataset": {"id": "<selected dataset>", "version": "<frozen version>",
               "adapter": "<name and version>", "rights_reference": "<local evidence reference>",
               "source_fingerprints": {"train": "<...>", "validation": "<...>"},
               "record_fingerprints": {"train": "<...>", "validation": "<...>"},
               "label_mapping_reference": "<verified mapping and policy evidence>"},
  "derived_data": {"segmenter": "py_vncorenlp", "segmenter_version": "<x.y.z>",
                    "segmenter_model_fingerprint": "<sha256 of model dir>",
                    "tokenizer_id": "vinai/phobert-base",
                    "tokenizer_config_hash": "<sha256 of tokenizer config>",
                    "segmented_split_fingerprints": {"train": "<...>", "validation": "<...>"},
                    "token_length_p99_train": 0, "truncation_rate": {"train": 0.0, "validation": 0.0}},
  "libraries": {"python": "<3.x>", "torch": "<x.y.z+cuXYZ>", "transformers": "<x.y.z>",
                 "tokenizers": "<x.y.z>", "cuda_driver": "<...>", "java": "<...>",
                 "sklearn": "<classical arm only>"},
  "hardware": {"device": "NVIDIA GeForce RTX 3060 Laptop GPU",
                "vram_mib": 6144, "precision": "fp16",
                "micro_batch": 8, "grad_accum": 4},
  "seeds": {"data": 42, "run": 42, "dataloader": 42, "latency_rng": 42,
             "confirmed_deterministic_repeat": null},
  "checkpoint": {"selected_epoch": 0, "selected_val_macro_f1": 0.0,
                  "checkpoint_sha256": "<...>"},
  "artifacts": {"model.safetensors": "<sha256>"},
  "commands": {"train": "<exact command line>"},
  "timings": {"train_wall_s": 0.0, "per_epoch_wall_s": [0.0]}
}
```

Use trusted local artifacts. Validate config and artifact hashes and check
software versions. Final evaluation evidence adds the matching preflight
receipt, test fingerprint, and scoring command and timing
(`evaluation-protocol.md` §8).

## 11. Error analysis

Scope: **validation split only**, on the final selected configuration (after
selection — never on search/retry candidates, of which M1 has none).

1. **Dump** (local, gitignored): per-sample line — index, true label,
   predicted label, probability of the predicted class, full probability
   vector for the deep model.
2. **Cell counts**: misclassifications aggregated by confusion cell
   (true → predicted); the 3×3 error-count matrix and per-cell share of all
   errors are committed.
3. **Recurring categories, evidence-gated**: a pattern may be stated only
   with the cell count, the share of the cell it covers, and at least three
   examples reviewed locally.
4. **Privacy**: publish aggregate counts and non-identifying categories.
   Keep per-sample dumps, indices, and feedback local in ignored directories.
   Compare the same validation records using paired indices locally
   (`evaluation-protocol.md` §7).
5. **Floor interaction**: if §13d triggered (below-baseline validation),
   error analysis is part of the investigation — it does not authorize any
   test run.

## 12. Inference profiling (end-to-end latency)

Phase 1's protocol measures `predict_fn(texts)` as one opaque call. M1 keeps
that number (comparability) and adds visibility into where the time goes:

- **E2E number** (headline, protocol-identical to Phase 1): `predict_fn`
  wraps normalize → segment → tokenize → forward → argmax.
  `measure_latency(predict_fn, samples, warmup_calls=10, timed_calls=100,
  seed=42)`, batch size 1. Samples: validation-split texts (after the gates)
  — before data arrives, synthetic fixtures exercise the harness itself only
  (category C, never quoted).
- **Decomposition** (recorded in the same JSON, clearly labeled):
  - `segmentation_ms` — per-text segmentation cost, measured two ways:
    batched (JVM amortized; the production-realistic number) and single-call
    (the worst case); both labeled.
  - `tokenize_forward_ms` — forward-only timing on pre-segmented, cached
    inputs (excludes segmentation).
- Devices: PhoBERT on local CPU and local GPU; classical on local CPU.
  Precision and device recorded per measurement (category D always).
- The E2E number is the honest deployment cost; the decomposition exists so
  segmentation is never hidden inside it. The comparison to Phase 1's
  classical latency is stated as "classical (CPU, no segmentation) vs
  PhoBERT E2E (CPU/GPU, segmentation included) — different device, different
  pipeline, not a like-for-like race".

## 13. Failure handling

| Case | Trigger | Action |
| ---- | ------- | ------ |
| (a) **Label audit fails** | Unknown or contradictory label meanings (§2.4) | Stop real fitting. Verify the selected adapter's mapping against local evidence and rerun preflight before either baseline or PhoBERT fits. |
| (b) **Segmentation determinism cannot be shown** | Double-segmentation fingerprints differ, or golden-file test fails on the pinned version | Halt the PhoBERT arm. Debug tool/JVM version drift; if a pinned working version is found, re-run the determinism proof and continue. If unresolvable: `phobert-base-ft` = `BLOCKED(segmentation)`; documented options (owner decides): pinned `underthesea` path (same determinism proof required) or the XLM-R fallback (§1.2) with the protocol deviation recorded. |
| (c) **VRAM insufficient even at minimum micro-batch** | OOM at micro-batch 1 × seq 256 × fp16 (§7.2 step 4) | Hardware-failure verdict: full fine-tune = `BLOCKED(compute)` with the OOM logs as evidence. Fallbacks (owner decides): frozen-encoder variant (encode once + logistic head, CPU-feasible, `experiment-matrix.md` E6) — noted as a protocol deviation from "one fine-tuned PhoBERT"; or cloud GPU (selected dataset's hosting permission and owner approval first). Claims scoped accordingly; M1 does not silently become a different experiment. |
| (d) **Validation macro-F1 below the majority baseline** | Best checkpoint's validation macro-F1 < majority's validation macro-F1 (same split, same metric code) | **No test run. No claims.** Investigate in order: (1) pipeline bug — label mapping, segmentation cache corruption, truncation anomaly (check `truncated_train_fraction`, confusion matrix shape); (2) training pathology — collapsed class (per-class recall ≈ 0), loss divergence (fp32 retry per §7.2.3 is the only allowed single re-run, recorded as a deviation); (3) if investigation finds no bug, the result stands as an honest single-run finding: reported as-is with single-seed phrasing, flagged to the owner, and M2 (seeds + optionally search) becomes the escalation path — *never* a seed retry loop run until the number beats the baseline. |
| (e) **Dependency version drift from the plan** | Installed torch/transformers/segmenter differ from the pinned stack (G4) at any stage | Before training: re-run the golden-file tests (segmentation, tokenization, seed determinism) against the actual versions; record actual versions in metadata regardless. Drift discovered mid-M1: stop, re-pin, re-run the affected stages from their fingerprints; the report states which versions produced which numbers. A major-version mismatch in torch/transformers invalidates prior checkpoints (Phase 1's sklearn rule, generalized). |

## 14. Implementable now vs gated work

The readiness branch includes released Phase 1 code. The deep-arm
implementation and runs still require approval. Synthetic fixtures can
exercise scaffolding without supporting any performance claim.

| Step | Needs dataset? | Needs torch/transformers? | Status today |
| ---- | -------------- | -------------------------- | ------------ |
| Data-contract module: schema-violation behaviors, fingerprint expectations (synthetic fixture files) | No (fixtures) | No | **Codeable + executable now** — Phase 1 loader/fingerprint code is importable and sklearn-only |
| Seed utility (set torch/numpy/random + dataloader generator) + determinism unit test | No | Yes (torch) | Codeable now (guarded by `pytest.importorskip("torch")`); executable after **G4** |
| Training-log JSONL writer + log-hygiene tests (text-never-leaks sanitizer) | No | No (writer is dependency-free) | **Codeable + executable now** |
| `meta.json` extension schema + builder + provenance-completeness test | No | No | **Codeable + executable now** |
| Classical-arm runner wiring (reuse `run_small_search`, `save_model`, `load_model` on synthetic records) | No (fixtures) | No | **Codeable + executable now** (Phase 1 code) |
| Tokenization chain implementation + truncation boundary tests (255/256/257) + golden-file segmenter tests | No (fixtures) | Yes (transformers) + JVM/py_vncorenlp | Codeable now; executable after **G4** (stack) — golden files themselves need the pinned segmenter |
| PhoBERT training loop (AMP, accumulation ladder, early stopping, JSONL hooks) + OOM-ladder logic, tested on synthetic tensors | No (synthetic) | Yes | Codeable now; executable on synthetic data after **G3 + G4** (first checkpoint download is gated by G3) |
| VRAM probe (forward/backward at seq 256, micro-batch ladder) — replaces §7.1's estimates with `[MEASURED]` | No | Yes | After **G3 + G4**; minutes `[ESTIMATE]`; no dataset needed |
| E1 integrity preflight on selected data | **Yes** | No | Requires authorization, verified mapping and a separate holdout |
| E2/E3 classical models on real data | **Yes** | No | Requires passing preflight. Freeze both real validation reports and fitted artifacts. |
| M1 PhoBERT fine-tune (seed 42) | **Yes** | Yes | Requires E2/E3 evidence on the same dataset, all readiness gates, and owner approval |
| Latency/error analysis on real validation samples | **Yes** | Yes (deep arm) | Requires selected dataset and frozen fitted artifacts |

Reading: the entire scaffolding layer (data contract, logging, metadata,
seed utility, classical wiring, and the code of the training loop) is
codeable immediately against synthetic fixtures — that is the honest
"progress without the dataset" story. Every number, including the
`[ESTIMATE]` replacements from the VRAM probe, waits on the gates.

## 15. Related plans

- [Experiment blueprint](experiment-blueprint.md) defines the shared dataset
  and holdout contract and the M2 escalation.
- [Experiment matrix](experiment-matrix.md) separates actual prerequisites
  from compute estimates and optional work.

- [Evaluation protocol](evaluation-protocol.md) defines comparable results,
  reporting boundaries, and the M2 seed set.
