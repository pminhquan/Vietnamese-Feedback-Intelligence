# Phase 2 Pilot Experiment Plan — Milestone M1

Status: **design only**. Nothing here has been run; no dataset is available
(G1 open), the label gate is unresolved (G2 open), the torch stack is not
installed (G3/G4 open). Every runtime or VRAM figure is `[ESTIMATE]` with
visible reasoning; no metric value exists anywhere in this document. This
plan is the implementation-level contract for the smallest reproducible
deep-learning pilot; the strategy-level view stays in
[experiment-blueprint.md](experiment-blueprint.md) and the cost map in
[experiment-matrix.md](experiment-matrix.md).

Written 2026-11-07 (hardening pass) against Phase 1 commit
`95f7687735c785367fa0a6f9b1efbf0e284aac7f`. Companion evidence:
[compute-feasibility.md](compute-feasibility.md) (hardware `[MEASURED]`),
[tokenization-design.md](tokenization-design.md) (preprocessing contract),
[label-semantics-evidence.md](label-semantics-evidence.md) (label conflict —
read-only for this plan).

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
4. **Label-audit gate blocks everything.** No model — classical or deep —
   consumes real data until the author answers + the blind audit settle the
   raw 0/1/2 mapping (§2.4). Evidence currently favors the authors' own
   layer (0=negative, 1=neutral, 2=positive) as a *planning prior only*
   (H1 in [label-semantics-evidence.md](label-semantics-evidence.md) §7).
5. **max_length 256** (PhoBERT's model maximum), right-side truncation,
   measured length distribution and truncation rate recorded (§3.4).
6. **Effective batch 32 via micro-batch 8 × gradient accumulation 4**
   `[ESTIMATE]` — sized for the `[MEASURED]` 6 GB VRAM but not guaranteed
   before measurement; the OOM ladder (§7.2) is pre-declared, not improvised.
7. **Selection on validation macro-F1 only; test touched exactly once** per
   final configuration, after freezing, ledger-recorded (§6, §8).
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
| `majority` | Majority-class classifier, Phase 1 (`MajorityBaseline`, canonical-label tie-break) | Phase 1, reused unchanged | normalized raw text (labels only — text unused) | No (deterministic count) |
| `tfidf-logreg` | TF-IDF + Logistic Regression, Phase 1 4-candidate search on validation, select by validation macro-F1, ties to first candidate | Phase 1, reused unchanged (`run_small_search`) | normalized raw text, vectorizer-internal tokenization | Yes, minutes `[ESTIMATE]` |
| `phobert-base-ft` | `vinai/phobert-base` (~135M params `[VERIFIED FACT]` card/config), full fine-tune, linear head on first token (`<s>`), dropout 0.1 `[ESTIMATE]` | New Phase 2 training loop (this plan) | NFC-normalized → word-segmented → BPE, max 256 | Yes, one fixed-recipe run, seed 42 |

The classical models run exactly as Phase 1 specified them — same search
space, same selection rule, same config — so the comparison's classical side
is a Phase 1-protocol rerun on real data, not a re-design.

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

### 2.1 Input files and layout

Exactly the Phase 1 layout, unchanged (`data/README.md`,
`vfi.data.loading.load_split`):

```text
data/raw/uit_vsfc/
  train/{sentences.txt, sentiments.txt}
  validation/{sentences.txt, sentiments.txt}
  test/{sentences.txt, sentiments.txt}
```

- One sentence per line in `sentences.txt`; one raw integer (0/1/2) per line
  in `sentiments.txt`; line i of one corresponds to line i of the other.
- Encoding: read as `utf-8-sig` (BOM-tolerant) — Phase 1 behavior
  `[VERIFIED FACT from source]`.
- Files are never committed; nothing under `data/raw/` is tracked.
- Expected scale for planning only: ≈16k sentences, commonly cited split
  train 11,426 / validation 1,583 / test 3,166 `[ESTIMATE — secondary
  sources disagree; the delivered files are authoritative]`.

### 2.2 Fingerprint expectations

Recorded once at the integrity gate and copied verbatim into every artifact
metadata and report:

1. **File fingerprints**: sha256 of each of the six raw files.
2. **Record fingerprints**: the order-sensitive record fingerprint from
   `vfi.data.validation.fingerprint` per split (already produced by
   `vfi prepare-data`) — captures the normalized text + label sequence, so
   any reordering or relabeling of the delivered files changes it.
3. **Derived-data fingerprints** (added by M1): after segmentation, a
   fingerprint of the segmented line sequence per split (sha256 of the
   `"\n".join(segmented_lines)`), recorded in `derived_data` metadata so
   experiments cite a rebuildable, verifiable artifact, not a hand-edited
   file.

Any change to a raw file invalidates all downstream fingerprints and reopens
the integrity gate — the same freeze rule as `experiment-blueprint.md` §3.2.

### 2.3 Behavior on schema violation

Phase 1's loader is already fail-fast; M1 adds no tolerance:

| Violation | Detected by | Behavior |
| --------- | ----------- | -------- |
| Missing file | `load_split` → `DataLoadingError` | Loud failure naming the path; no partial artifacts |
| Line-count mismatch | `load_split` → `DataLoadingError` | Loud failure naming the split and the two counts |
| Empty sentence line | `load_split` → `DataLoadingError` (line number included) | Loud failure; no silent skip |
| Non-integer or out-of-range label (not 0/1/2) | `load_split` → `DataLoadingError` (line number included) | Loud failure |
| Unknown canonical label after mapping | impossible by construction (mapping is total over {0,1,2}) | n/a |
| Label values distribution nonsensical (e.g. one class ≈ 0 everywhere) | integrity-gate distribution check | **Investigation gate**, not auto-pass — may indicate the wrong mapping (see §2.4) |

Failure output discipline: error messages carry split names and line
numbers only — never sentence text. The error report (committed) contains
counts and line indices; the offending lines themselves stay local.

### 2.4 The label-audit gate (blocking pre-step, part of G2)

**Nothing in M1 runs on real data until this gate passes — including the
classical models.** A cyclically-permuted label mapping leaves accuracy and
macro-F1 untouched (consistent permutations preserve both) while inverting
every per-class claim
([label-semantics-evidence.md](label-semantics-evidence.md) §6), so no
metric-based sanity check can substitute for the audit.

Procedure (unchanged from `experiment-blueprint.md` §3.3; hypotheses from
[label-semantics-evidence.md](label-semantics-evidence.md) §7):

1. **Author documentation first.** The signed-corpus package is checked for
   label-encoding documentation; the pre-written question block
   ([label-semantics-evidence.md](label-semantics-evidence.md) §8) was
   included in the G1 email. Whatever the author states wins over all
   secondary sources.
2. **Blind audit second.** 15 lines per raw value (0/1/2) per split —
   135 lines, 45 per class. Two readers judge polarity from text alone,
   blind to the integer and to each other's judgments.
3. **Pre-registered hypotheses.** H1 (expected): raw 0=negative, 1=neutral,
   2=positive — the authors' own card/script/served-data layer. H2:
   0=positive, 1=negative, 2=neutral — the SEACrowd layer, which Phase 1's
   hard-coded `RAW_SENTIMENT_TO_LABEL` currently assumes. The audit decides
   between them; a conflict between author answer and audit triggers
   investigation, not silent acceptance of either.
4. **Pass criteria** (per raw value): ≥ 90% of sampled lines judged with the
   mapped polarity, and both readers individually ≥ 85%; distribution
   corroboration shows a plausible profile (positive-majority expected for
   student feedback `[ASSUMPTION]`).
5. **Fail criteria**: any class < 80%, or a systematic inversion pattern
   (e.g. raw `0` lines consistently negative under H2's reading), or
   author-vs-audit conflict unresolved.
6. **On pass**: the mapping used by every M1 loader is the audited one; the
   audit report (sample design, per-reader per-class agreement, decision) is
   committed; fingerprints and the mapping hash enter all metadata.
7. **On fail**: stop (§13a). If H1 is confirmed, Phase 1's loader mapping
   must be corrected first — a Phase 1-branch code change, owned by the
   Phase 1 maintainer, **not** done inside this readiness clone.

## 3. Tokenization contract

The three models intentionally consume different text pipelines; the
differences are documented, not hidden (`experiment-blueprint.md` §4). All
chains start at the same place: Phase 1's `normalize_text` (NFC + whitespace
collapse), reused unmodified.

### 3.1 Per-model preprocessing chain (exact)

| Step | `majority` | `tfidf-logreg` | `phobert-base-ft` |
| ---- | ---------- | -------------- | ----------------- |
| 1. Load | `load_split` (labels only matter) | `load_split` | `load_split` |
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
- The segmenter is treated as a **pure `str -> str` function**, applied
  identically to train, validation, and test, once, before any training.
  Outputs are cached as derived files (rebuilt only from frozen raw files,
  never hand-edited) with per-split fingerprints (§2.2).
- **Determinism is demonstrated, not assumed** `[ASSUMPTION that RDRsegmenter
  is deterministic; verified empirically]`: each split is segmented twice in
  separate process invocations; the two per-split fingerprints must be
  identical. Additionally, a golden-file test pins N fixed sentences →
  exact expected underscore-joined output for the recorded tool+model
  version, failing loudly on version drift. If determinism cannot be shown,
  the PhoBERT arm halts (§13b).
- The segmented text stays **local-only** (gitignored), same as raw data.

### 3.3 Identical treatment across splits

One segmentation run per split, same tool, same version, same code path —
no split-specific options, no per-split "tuned" segmentation. The training
loop consumes the cached segmented files; validation and test inference
consume theirs. At inference time (latency protocol, §12), the same chain
runs per text — the predict function must include segmentation so the
measured latency is honest end-to-end cost.

### 3.4 Truncation policy (max_length 256)

- `256` is PhoBERT's model maximum `[VERIFIED FACT — card/paper]`; sequences
  longer than 256 subword tokens are truncated from the **right**
  (single-sequence inputs; sentiment cues near the end are lost by design
  and documented).
- Before the first fine-tune, the token-length distribution of the
  **segmented train split** is measured (p50/p95/p99/max) and the
  truncation rate per split is computed and recorded in metadata. UIT-VSFC
  sentences are expected to be short, so a p99 ≤ 256 is expected
  `[ESTIMATE, unverified]`.
- If the measured tail is meaningful (p99 > 256 or truncation rate > ~1% on
  any split), it is recorded as a **known limitation** in the report — not
  silently absorbed, and not a reason to raise max_length beyond the model
  maximum in M1.

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

All figures assume the `[MEASURED]` hardware (RTX 3060 Laptop 6 GB,
i5-10500H, 32 GB RAM) and cite their arithmetic. Uncertainty factor ≈ 2× on
GPU figures (thermals, background load) — nothing below is measured.

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
| Where computed | In the training loop | Only by the dedicated evaluation path (`vfi evaluate` equivalent), never inside the training loop |
| Metrics | `compute_metrics` full schema; macro-F1 drives decisions | Same `compute_metrics` full schema |
| Artifacts | Per-epoch JSONL (§9), per-sample dump for error analysis (local, gitignored) | One ledger-recorded evaluation JSON per configuration, committed (aggregates only) |
| May it change the model? | Yes — that is its purpose | No — a number that triggers any change converts the run into a **deviation**, documented, and the affected configuration's test count increments |

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

**Log hygiene rules** (hard requirements, unit-tested):

1. No raw student text anywhere in logs — not inputs, not decoded token
   sequences, not error messages that embed text (§3.5 sanitizer).
2. No person-identifying metadata (student names do not appear in the
   corpus, but any incidental identifier found in text must never be copied
   into a log — the rule is: indices and aggregates only).
3. Line indices and split names are the only per-sample keys allowed.
4. Committed reports carry aggregates and category labels
   (`evaluation-protocol.md` §9); raw JSONL stays local.

## 10. Final artifact metadata (`meta.json` extension)

Phase 1's provenance pattern (`build_training_metadata`: created_at, config,
config_hash, dataset_fingerprints, sklearn_version, python_version, seed
`[VERIFIED FACT from source]`) is extended, not replaced. M1 deep-arm
`meta.json` skeleton:

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
  "dataset": {"source": "UIT-VSFC via signed agreement",
               "agreement_signed": true,
               "file_sha256": {"train/sentences.txt": "<...>", "...": "<...>"},
               "record_fingerprints": {"train": "<...>", "validation": "<...>", "test": "<...>"},
               "label_mapping": {"0": "negative", "1": "neutral", "2": "positive"},
               "label_mapping_basis": "author + blind audit (G2 report)"},
  "derived_data": {"segmenter": "py_vncorenlp", "segmenter_version": "<x.y.z>",
                    "segmenter_model_fingerprint": "<sha256 of model dir>",
                    "tokenizer_id": "vinai/phobert-base",
                    "tokenizer_config_hash": "<sha256 of tokenizer config>",
                    "segmented_split_fingerprints": {"train": "<...>", "...": "<...>"},
                    "token_length_p99_train": 0, "truncation_rate": {"train": 0.0, "validation": 0.0, "test": 0.0}},
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
  "artifacts": {"model.safetensors": "<sha256>", "meta.json": "<sha256>"},
  "commands": {"train": "<exact command line>", "evaluate": "<exact command line>"},
  "timings": {"train_wall_s": 0.0, "per_epoch_wall_s": [0.0], "evaluate_wall_s": 0.0}
}
```

Rules carried over from Phase 1: the trusted-artifact load policy (only
load artifacts this project produced, config-hash present, version-match
warning) applies unchanged; a report number without complete provenance is
an anecdote, not a result (`evaluation-protocol.md` §8).

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
   with (i) the cell count, (ii) the share of the cell it covers, and
   (iii) ≥ 3 examples cited by **line index** with at most short truncated
   fragments — or paraphrases — explicitly marked "truncated" /
   "paraphrased". Patterns without counts are not published.
4. **Privacy**: raw student text must stay out of every committed path —
   conditional on the G3/R-08 hardening landing first (the gitignore gaps
   for torch-era `*.bin`/`*.pt` files are verified open, and Phase 1's
   `data_report.json` already embeds duplicate raw texts, R-05). Per-sample
   dumps and full texts stay in local gitignored directories.
   Classical-vs-deep error comparison uses paired line indices on the same
   validation samples (`evaluation-protocol.md` §7).
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
| (a) **Label audit fails** | §2.4 fail criteria, or author-vs-audit conflict | Full stop — no model (classical included) touches real data. The mapping correction is a Phase 1-branch change (owner's action, since it edits Phase 1 code); re-audit after correction; gate reopens. Nothing in M1 proceeds on a "probably right" mapping. |
| (b) **Segmentation determinism cannot be shown** | Double-segmentation fingerprints differ, or golden-file test fails on the pinned version | Halt the PhoBERT arm. Debug tool/JVM version drift; if a pinned working version is found, re-run the determinism proof and continue. If unresolvable: `phobert-base-ft` = `BLOCKED(segmentation)`; documented options (owner decides): pinned `underthesea` path (same determinism proof required) or the XLM-R fallback (§1.2) with the protocol deviation recorded. |
| (c) **VRAM insufficient even at minimum micro-batch** | OOM at micro-batch 1 × seq 256 × fp16 (§7.2 step 4) | Hardware-failure verdict: full fine-tune = `BLOCKED(compute)` with the OOM logs as evidence. Fallbacks (owner decides): frozen-encoder variant (encode once + logistic head, CPU-feasible, `experiment-matrix.md` E6) — noted as a protocol deviation from "one fine-tuned PhoBERT"; or cloud GPU (corpus-agreement hosting gate first). Claims scoped accordingly; M1 does not silently become a different experiment. |
| (d) **Validation macro-F1 below the majority baseline** | Best checkpoint's validation macro-F1 < majority's validation macro-F1 (same split, same metric code) | **No test run. No claims.** Investigate in order: (1) pipeline bug — label mapping, segmentation cache corruption, truncation anomaly (check `truncated_train_fraction`, confusion matrix shape); (2) training pathology — collapsed class (per-class recall ≈ 0), loss divergence (fp32 retry per §7.2.3 is the only allowed single re-run, recorded as a deviation); (3) if investigation finds no bug, the result stands as an honest single-run finding: reported as-is with single-seed phrasing, flagged to the owner, and M2 (seeds + optionally search) becomes the escalation path — *never* a seed retry loop run until the number beats the baseline. |
| (e) **Dependency version drift from the plan** | Installed torch/transformers/segmenter differ from the pinned stack (G4) at any stage | Before training: re-run the golden-file tests (segmentation, tokenization, seed determinism) against the actual versions; record actual versions in metadata regardless. Drift discovered mid-M1: stop, re-pin, re-run the affected stages from their fingerprints; the report states which versions produced which numbers. A major-version mismatch in torch/transformers invalidates prior checkpoints (Phase 1's sklearn rule, generalized). |

## 14. Implementable now vs gated work

Status of each pilot step **today**, with this clone holding docs only
(no venv for Phase 2 deps, no dataset, gates G1-G4 open). "Codeable" = can
be written and unit-tested with synthetic fixtures; "executable" = can
actually run.

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
| E1 integrity gate + label audit on delivered files | **Yes** | No | `[BLOCKED]` until **G1 + G2** |
| E2/E3 classical models on real data | **Yes** | No | `[BLOCKED]` until **G1 + G2** (audit gate blocks classical too, §2.4) |
| M1 PhoBERT fine-tune (seed 42) | **Yes** | Yes | `[BLOCKED]` until **G1 + G2 + G3 + G4** |
| Latency/error-analysis on real validation samples | **Yes** | Yes (deep arm) | `[BLOCKED]` until **G1 + G2** (+ training done) |

Reading: the entire scaffolding layer (data contract, logging, metadata,
seed utility, classical wiring, and the code of the training loop) is
codeable immediately against synthetic fixtures — that is the honest
"progress without the dataset" story. Every number, including the
`[ESTIMATE]` replacements from the VRAM probe, waits on the gates.

## 15. Consistency notes (what this plan changes elsewhere)

- `experiment-blueprint.md`: §2 (M1 marking, XLM-R fallback note), §3.3
  (evidence cross-reference, H1/H2 prior), §4 (max length 128 → 256), §6
  (seed set {42, 7, 2026}), §7 (search marked M2), §10 (batch geometry at
  seq 256), §11 (OOM ladder wording).
- `experiment-matrix.md`: §2 arithmetic re-based to micro-batch 8 × seq 256
  × accum 4; new row E4p (the M1 fixed-recipe run); E4/E5 marked M2;
  milestone column; completion criteria re-cut by milestone; escalation
  flow updated.
- Reconciled (2026-11-07 QA pass): `evaluation-protocol.md` §5 now uses the
  escalation set {42, 7, 2026}.
