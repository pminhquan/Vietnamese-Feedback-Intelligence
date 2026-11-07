# Phase 2 Experiment Blueprint

Status: **design only**. Phase 2 is not authorized to start; the UIT-VSFC
license is still blocked (see `docs/dataset-uit-vsfc.md` on the Phase 1
branch). No dataset file, no fine-tuning run, and no metric value exists
behind this document. Every runtime figure is labeled `[ESTIMATE]` with its
reasoning. Written 2026-11-07 against Phase 1 commit `95f7687` and main
`caddb4a`.

## 1. Research question

Does fine-tuning a pretrained Vietnamese language model (PhoBERT-base)
improve 3-class sentiment classification of student feedback — measured as
validation macro-F1 and final test macro-F1 on the official UIT-VSFC splits —
compared to the Phase 1 classical baselines, under a protocol where both
model families share the same data, splits, label space, metric code, and
selection rules?

Secondary questions, answered only if compute allows:

- Does a second pretrained model (different pretraining recipe or scale)
  change the ranking?
- How much of the fine-tuned gain survives a frozen-encoder + linear-head
  variant (a cheaper deployment shape)?

## 2. Models under comparison

| ID | Model | Family | Status in Phase 2 |
| -- | ----- | ------ | ----------------- |
| `majority` | Majority-class classifier (Phase 1, canonical-label tie-break) | Classical | Mandatory control floor |
| `tfidf-logreg` | TF-IDF + Logistic Regression, Phase 1 4-candidate search on validation (word 1-2 / char_wb 2-5 × C 1.0/4.0) | Classical | Mandatory strong baseline; re-run on real data under the Phase 1 protocol |
| `phobert-base-ft` | PhoBERT-base (~135M params) full fine-tune, classification head on `[CLS]` | Deep, pretrained | Primary Phase 2 model |
| `phobert-base-frozen` | PhoBERT-base frozen encoder + logistic regression head on `[CLS]` embeddings | Deep, frozen | Fallback only: used if full fine-tuning is compute-infeasible, or as a reported secondary variant |
| `second-pretrained` | One additional pretrained baseline — candidate order: XLM-R-base (multilingual, raw-syllable input), then PhoBERT-large | Deep, pretrained | **OPTIONAL**, only if the Phase 2 budget is not consumed by the mandatory cells; decision rule in `experiment-matrix.md` |

Notes:

- `phobert-base-frozen` is a **different model**, never conflated with
  `phobert-base-ft`. Its headline claim, if any, is about deployment cost,
  not quality parity.
- PhoBERT checkpoint facts (verified 2026-11-07 from the Hugging Face card
  `vinai/phobert-base` and `config.json`): ~135M parameters, hidden 768,
  12 layers, vocab 64,001 BPE tokens, `max_position_embeddings=258`, card
  license tag `license:mit`. Re-verify the license tag at Phase 2 kickoff and
  record it in the reproducibility metadata.

## 3. Dataset acquisition and integrity checks

### 3.1 Acquisition

Unchanged from Phase 1: the owner emails `kietnv@uit.edu.vn`, signs the
corpus user agreement, receives the official `train` / `validation` / `test`
file pairs, and places them under `data/raw/uit_vsfc/<split>/{sentences,sentiments}.txt`
exactly as `data/README.md` specifies. Files are never committed. The signed
agreement's terms (especially any prohibition on uploading the corpus to
third-party services) are recorded in `reports/` **before** any cloud-GPU
scenario is considered — see `experiment-matrix.md`.

Expected corpus scale for planning only: ≈16k sentences total; the commonly
cited official split is train 11,426 / validation 1,583 / test 3,166
`[ESTIMATE]` — secondary sources disagree slightly on train/test counts, so
the authoritative numbers are whatever the delivered files contain, recorded
at the integrity gate.

### 3.2 Integrity gate (must pass before any model sees the data)

Run in order; each step is recorded in a committed report (aggregate
statistics only; raw text never committed):

1. **Layout and line alignment.** Each split directory contains exactly
   `sentences.txt` and `sentiments.txt`; line counts match per split. The
   existing loader (`vfi.data.loading.load_split`) fails loudly on
   misalignment — this failure is the check, and its absence is confirmed by
   a clean `vfi prepare-data`.
2. **File and record fingerprints.** sha256 of each raw file, plus the
   order-sensitive record fingerprint from `vfi.data.validation.fingerprint`
   per split. These identifiers are copied into every experiment's
   reproducibility metadata and into the matrix when Phase 2 runs.
3. **Label-semantics verification** — see 3.3. Hard gate: no training until
   it passes.
4. **Schema and distribution.** `vfi.data.validation.validate_records`
   (non-empty normalized text, known labels), per-split class distribution
   over the canonical labels, and duplicates report
   (`find_duplicates` — reported, not dropped, per Phase 1 policy).
5. **Cross-split leakage.** `find_leakage` exact-match check across all
   three splits; any hit is reported and resolved (documented decision)
   before training. Near-duplicate leakage remains out of scope (Phase 1
   limitation), documented as such.
6. **Freeze.** Once 1-5 pass, the dataset version is frozen as
   "UIT-VSFC v1.0, fingerprints `<...>`" and every subsequent artifact
   references it. Any change to the raw files invalidates all fingerprints
   and reopens the gate.

### 3.3 Label-semantics verification of the raw 0/1/2 mapping

Phase 1's loader hard-codes `RAW_SENTIMENT_TO_LABEL = {0: "positive",
1: "negative", 2: "neutral"}`, sourced from the SEACrowd loader's
`ClassLabel` declaration — a secondary-source inference, explicitly flagged
for verification in `docs/dataset-uit-vsfc.md`. Because every metric in both
phases depends on this mapping, it is verified **before** trusting it:

1. **Authoritative documentation.** The corpus package delivered after
   signing is checked for label-encoding documentation (README or agreement
   material). Whatever it states is recorded verbatim-or-paraphrased in the
   report.
2. **Manual audit.** Stratify by raw label value: sample 15 lines per raw
   value (0/1/2) per split — 135 lines total, or 45 per class. Two readers
   (the owner and one independent reader) judge each line's polarity from
   the text alone, blind to the integer. Compare their judgments to the
   declared mapping.
3. **Acceptance rule.** For each raw value: ≥ 90% of sampled lines judged
   with the mapped polarity, and both readers individually ≥ 85%. If any
   class falls below 80% or shows a systematic inversion (e.g., raw `0`
   lines consistently negative), the mapping is **not trusted**: loading is
   stopped, `RAW_SENTIMENT_TO_LABEL` is corrected against the author's
   documentation, and the audit is repeated before anything trains.
4. **Distribution corroboration.** The per-split class proportions under the
   declared mapping are sanity-checked against any distribution the author
   publishes (the 2018 KSE paper reports sentiment stats; inter-annotator
   agreement > 91%). A positive-majority profile is expected for student
   feedback; a nonsensical profile triggers investigation, not silent
   acceptance.
5. **Audit trail.** The committed report records: sample design, both
   readers' per-class agreement rates, the decision, and at most short
   truncated fragments as evidence. No full student sentences.

Because Phase 1 has never run on real data, this gate is a shared
prerequisite for both phases; passing it unblocks Phase 1's real-data runs
too.

## 4. Preprocessing pipelines (documented, not hidden)

The two families intentionally use different text pipelines. The differences
are stated here and in every Phase 2 report; hiding them is a fairness
violation, not a convenience.

| Step | Classical (`majority`, `tfidf-logreg`) | Transformer (`phobert-base-ft`, frozen, optional second model) |
| ---- | -------------------------------------- | -------------------------------------------------------------- |
| Raw input | Official split files, one sentence per line | Same files, same lines |
| Normalization | `normalize_text`: Unicode NFC + whitespace collapse | Same `normalize_text` first (shared, minimal) |
| Tokenization | TF-IDF analyzer (whitespace words or char_wb), `lowercase=True` inside the vectorizer | Word segmentation, then PhoBERT BPE WordPiece tokenizer |
| Word segmentation | Not applied (whitespace tokens) | **Mandatory**: PhoBERT was pre-trained on word-segmented Vietnamese (underscore-joined syllables, e.g. `sinh_viên`); unsegmented input measurably degrades it |
| Case handling | Lowercased by the vectorizer | Preserved (tokenizer is case-preserving) |
| Sequence length | Unbounded (sparse features) | `max_seq_length=128`, truncation from the right; derived from the train-split token-length distribution (p99 must fit; verify empirically after segmentation and record the percentile) |
| Fitted on our data | TF-IDF vocabulary + IDF, train split only | Nothing (segmenter and tokenizer are pre-trained externally; their models were not trained on UIT-VSFC) |

Segmentation tooling: **VnCoreNLP RDRSegmenter** (`py_vncorenlp`), Apache-2.0
including the pre-trained models — verified 2026-11-07. `underthesea` is the
backup; its license changed from GPL-3.0 to Apache-2.0 only in recent
releases, so if used, the pinned version's license is recorded. The
segmentation step's runtime on ≈16k short sentences is minutes on CPU
`[ESTIMATE]` and is recorded once in the report.

Segmentation is applied identically to every split, once, and the segmented
split files are treated as derived data: rebuilt only from frozen raw files,
never hand-edited, and fingerprinted (hash of the segmented line sequence per
split) so experiments can cite them.

## 5. Split discipline

Identical to Phase 1, restated because Phase 2 adds new ways to violate it:

1. **Official train** split: fitting only — TF-IDF vocabulary, logistic
   regression weights, the fine-tuning updates (all epochs), the frozen head.
2. **Official validation** split: hyperparameter selection, early stopping,
   checkpoint choice, error analysis. Never used for fitting; never merged
   with train.
3. **Official test** split: evaluated only after model selection is finalized
   and frozen. Never used for tuning, early stopping, checkpoint choice, or
   error-analysis-driven fixes. Each final configuration (per seed) is
   evaluated once; the resulting test predictions may support descriptive
   post-hoc statistics (e.g. McNemar's test, `evaluation-protocol.md` §6)
   but may not trigger retraining or reselection.

The `use_official_splits: false` fallback from Phase 1 stays available for
validation-file unavailability, but Phase 2 fine-tuning requires the official
splits — the fallback split must not become the default just because it is
convenient.

## 6. Seed policy

Sources of randomness differ by family, so the policy does too:

| Where | Seeds | Repetition | Variance reported |
| ----- | ----- | ---------- | ----------------- |
| Data preparation (fallback split, latency sampling) | 42 | Single | Not stochastic |
| Classical fitting (`majority`, `tfidf-logreg`) | 42 (`random_state`) | Single run | ≈0: deterministic given fixed data and settings; optionally one repeat with seed 43 to confirm determinism (minutes `[ESTIMATE]`) |
| Fine-tuning search runs (§7) | 42 only | Single run per candidate | Not used for claims; search only |
| Fine-tuning final configuration | Plan A: {42, 43, 44} (3 seeds); Plan B: {42, 43, 44, 45, 46} (5 seeds) if Plan A's total wall time fits the budget (`experiment-matrix.md`) | 3-5 independent runs, identical hyperparameters | mean ± std of validation macro-F1 and test macro-F1 across seeds |
| Transformer internal seeds | Per run: `torch`, `numpy`, `python.random` all set to the run's seed; dataloader shuffling seeded | Same as above | CUDA nondeterminism is not bitwise-eliminated; this is documented, not hidden |

Rules:

- The **headline test number for a multi-seed model is the mean across all
  seeds, decided in advance**. No seed result may be dropped, replaced, or
  re-weighted after seeing test output. Per-seed numbers are reported
  alongside.
- If compute allows only one fine-tuning seed, the report must say so and
  §6 of `evaluation-protocol.md` restricts the claim phrasing accordingly
  ("in this run", no variance language).
- Seed 42 remains the project default for everything deterministic, matching
  Phase 1.

## 7. Fine-tuning hyperparameter search budget

### 7.1 The search

| Dimension | Values | Tuned? |
| --------- | ------ | ------ |
| Learning rate | {2e-5, 3e-5, 5e-5} | Tuned (3 candidates, one seed each) |
| Epochs | ≤ 5 with early stopping | Tuned implicitly by early stopping, not by the grid |
| Batch size | 16 effective (fallback 8 × grad-accum 2 on OOM, recorded per run) | Fixed |
| Max sequence length | 128 (from the data, §4) | Fixed |
| Weight decay | 0.01 | Fixed |
| Warmup | Linear schedule, 10% warmup steps | Fixed |
| Precision | fp16 on GPU (fp32 on CPU fallback) | Fixed per device |
| Gradient clipping | 1.0 | Fixed |
| Optimizer | AdamW defaults otherwise | Fixed |
| Classification head | Linear on `[CLS]`, dropout 0.1 | Fixed |
| Class weighting | None in the primary recipe | Conditional candidate: if validation shows the neutral class collapsing (recall ≈ 0 at any searched LR), a class-weighted variant joins the search **before** any test run — a planned fourth candidate, not a post-hoc fix |

Total: **3 search runs** (+ at most 1 conditional), then **3-5 final-seed
runs** of the winning configuration (§6).

### 7.2 Early stopping and evaluation cadence

Validation macro-F1 (same metric code, `zero_division=0`) is computed at the
end of every epoch. Training stops after 2 consecutive evaluations without
improvement; the best-validation checkpoint is kept. With ≤ 5 epochs this
bounds every run at 5 validation passes — affordable at ≈1.6k validation
sentences.

### 7.3 Why the grid stays small

- **Cost.** Each run is a full fine-tune: ≈715 steps/epoch × ≤ 5 epochs on
  train ≈ 11.4k sentences `[ESTIMATE]`. On the RTX 3060 Laptop 6 GB that is
  ≈15-40 min per run `[ESTIMATE]` (fp16, seq 128, batch 16; ~0.3-0.6
  s/step). The 3-candidate search ≈ 1-2 h; a 3×3×3 grid (LR × batch ×
  epochs) would be ≈ 27 runs ≈ 9-18 h of serial GPU time on the same 6 GB
  card — runs cannot be parallelized on it.
- **Information.** Published fine-tuning studies (Dodge et al. 2020 "Show
  Your Work"; Mosbach et al., ICLR 2021) show that at small-dataset scale,
  seed-to-seed variance routinely rivals or exceeds the deltas between
  adjacent reasonable hyperparameters. Spending the budget on 3-5 seeds of
  the selected configuration buys more valid inference than a wider grid.
- **Precedent.** Phase 1 set the pattern: a small, documented, justified
  search — "a demonstration of disciplined search, not exhaustive tuning".

## 8. Checkpoint selection criteria

All selection uses validation macro-F1 with the same metric code as Phase 1;
`test` is never an input to any selection decision.

1. **Within a run**: the end-of-epoch checkpoint with the highest validation
   macro-F1; ties break to the **earliest** epoch (less overfitting).
   Storage: keep best + last only; record sha256 of each saved checkpoint.
2. **Across search candidates**: the candidate whose best checkpoint has the
   highest validation macro-F1; ties break to the **first** candidate in the
   fixed order [2e-5, 3e-5, 5e-5] — mirroring Phase 1's "ties break to the
   first candidate".
3. **Final artifact**: the seed-42 run of the selected configuration, at its
   best-validation checkpoint, becomes the artifact carried into Phase 3.
   The other seeds exist for variance reporting only. This rule is fixed in
   advance to prevent post-hoc artifact swapping.
4. **Artifact provenance**: each checkpoint's `meta.json` gains Phase 2
   fields on top of the Phase 1 provenance (config hash, dataset
   fingerprints, versions, seed): segmentation tool + version, tokenizer id,
   `max_seq_length`, precision, final LR schedule parameters, selected epoch
   and its validation macro-F1, and sha256 of the checkpoint file. The
   trusted-artifact load policy carries over unchanged.

## 9. Fairness rules for the classical-vs-deep comparison

1. **Same splits**: official train/validation/test for every model. No
   model gets a different train set, resampling, or augmentation.
2. **Same label space**: canonical `negative/neutral/positive` end-to-end;
   the raw 0/1/2 mapping is verified once (§3.3) and shared by all loaders.
3. **Same metric implementation**: every number in the comparison tables is
   produced by `vfi.evaluation.metrics.compute_metrics` — one code path, one
   `zero_division=0` policy, one confusion-matrix convention.
4. **Same selection procedure**: validation macro-F1; ties to first
   candidate; test evaluated once per final configuration.
5. **Pipeline differences documented, not hidden**: §4's table is reproduced
   in the Phase 2 report. The classical arm runs as Phase 1 specified it
   (whitespace-token TF-IDF, lowercased by the vectorizer). A secondary
   **optional** variant `tfidf-logreg-seg` — the Phase 1-selected candidate
   re-run once on segmented text, same fixed configuration, no re-search —
   answers the predictable question "would the baseline also gain from
   segmentation?". It is reported as a variant; it cannot replace the
   Phase 1-protocol baseline, and it is never selected on.
6. **No re-selection on the deep side**: the fine-tune recipe is frozen
   before its first test run; if a variant (e.g. class-weighted) joins the
   search, it competes on validation only and before any test evaluation,
   like every other candidate.
7. **Compute asymmetry acknowledged**: the comparison's question is "does a
   pretrained representation help this task?", not "which model wins a
   fixed-FLOP contest?". Latency and cost are reported separately under the
   latency protocol; they do not adjust the accuracy comparison.
8. **Same honesty rules**: no metric without a real run on legitimately
   obtained data; synthetic-fixture tests are never quoted as results;
   degenerate behavior (majority one-hot probabilities; a fine-tuned run
   collapsing to one class) is reported, never excluded.

## 10. Compute plan and VRAM feasibility

Grounded on the owner's machine (verified 2026-11-07): Intel i5-10500H
(6C/12T), 32 GB RAM, NVIDIA RTX 3060 Laptop 6 GB VRAM, Windows. Full
per-cell status and cost estimates live in `experiment-matrix.md`; the
load-bearing facts:

- **PhoBERT-base full fine-tune fits 6 GB** `[ESTIMATE]`: mixed precision
  ≈ 0.27 GB fp16 weights + 0.54 GB fp32 master + 1.08 GB AdamW states +
  0.27 GB gradients ≈ 2.2 GB, plus ≈ 1-1.5 GB activations at batch 16 ×
  seq 128 — roughly 3.5-4.5 GB total. Batch 8 × grad-accum 2 is the
  documented fallback if OOM appears.
- **PhoBERT-large does not fit 6 GB** for training (`[ESTIMATE]` ≈ 6-7 GB
  optimizer state alone); it is cloud-GPU-only and optional.
- **CPU-only fine-tuning** is ≈10-25× slower than the local GPU
  `[ESTIMATE]` → ≈4-10 h per run → the full protocol (search + seeds) is
  compute-infeasible on CPU; the frozen-encoder variant (encode once, fit a
  logistic head) is the CPU-feasible fallback.
- **Cloud GPU** changes hours, not science: the protocol is identical; the
  corpus agreement's hosting terms gate whether the dataset may be uploaded
  at all.

## 11. Risks and fallbacks

| Risk | Trigger | Fallback |
| ---- | ------- | -------- |
| Dataset license stays blocked | No signed agreement | Nothing runs; the matrix keeps everything `BLOCKED(dataset license)`; Phase 2 remains design-only (current state) |
| Label mapping fails the audit | §3.3 acceptance rule fails | Stop; correct mapping against author documentation; re-audit; no training until it passes |
| VRAM OOM at batch 16 | CUDA OOM on RTX 3060 6 GB | batch 8 × grad-accum 2 (recorded deviation); fp16 already default |
| fp16 loss instability | Non-finite loss / NaN validation | Switch the affected run to fp32, record the deviation, re-run that candidate |
| Neutral class collapse in validation | Neutral recall ≈ 0 across searched LRs | Planned class-weighted candidate joins the search on validation (§7.1), before any test run |
| Fine-tuning infeasible on local GPU | Sustained hardware failure / VRAM below expectation | Frozen-encoder variant as the reported deep model; full fine-tune stays `BLOCKED(compute)`; claims scoped accordingly |
| Cloud hosting prohibited by corpus agreement | Agreement terms forbid third-party upload | Cloud cells become `BLOCKED(license)`; local GPU remains the only compute path |

## 12. Phase 1 protocol items that Phase 2 changes or extends

- **Reproducibility metadata** gains fields (git SHA, torch/transformers/
  CUDA versions, segmentation tool, tokenizer, precision, per-seed records,
  artifact hashes) — schema in `evaluation-protocol.md` §8.
- **Latency protocol** is reused verbatim but extended with device/precision
  metadata, and PhoBERT latency is measured on both local-CPU and local-GPU
  so the classical (CPU) numbers have a like-for-like reference.
- **Error analysis** moves from "allowed on validation" to a mandatory,
  evidence-gated procedure (`evaluation-protocol.md` §7).
- **Repeated seeds and claim phrasing** are new obligations Phase 1 could
  not have: classical single-run reporting was fine when the model was
  deterministic; fine-tuning is not.
- Everything else — split discipline, macro-F1 primary metric,
  `zero_division=0`, canonical labels, seed 42, honesty rules, no-raw-text
  rule — carries over **unchanged**.
