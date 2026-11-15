# Phase 2 Experiment Blueprint

Status: **design only**. The released Phase 1 adapters support independent
datasets. Phase 2 runs await a selected, authorized dataset, verified label
semantics, a passing integrity preflight, actual classical baseline evidence,
and the approved compute stack. Runtime figures marked `[ESTIMATE]` are
planning scenarios. **M1 is the three-model pilot** described in
[pilot-experiment-plan.md](pilot-experiment-plan.md): majority, TF-IDF+LogReg,
and one pre-declared PhoBERT fine-tune (seed 42). Search and extra seeds belong
to M2.

## 1. Research question

Does fine-tuning a pretrained Vietnamese language model (PhoBERT-base)
improve 3-class Vietnamese sentiment classification on the selected dataset,
as measured by validation macro-F1 and final holdout macro-F1? Compare it
with the majority and TF-IDF+LogReg baselines under identical records,
split assignments, canonical labels, and metric code.

Secondary questions, answered only if compute allows:

- Does a second pretrained model (different pretraining recipe or scale)
  change the ranking?
- How much of the fine-tuned gain survives a frozen-encoder + linear-head
  variant (a cheaper deployment shape)?

## 2. Models under comparison

| ID | Model | Family | Status in Phase 2 |
| -- | ----- | ------ | ----------------- |
| `majority` | Majority-class classifier (Phase 1, canonical-label tie-break) | Classical | **M1 (pilot)** — mandatory control floor |
| `tfidf-logreg` | TF-IDF + Logistic Regression, Phase 1 4-candidate search on validation (word 1-2 / char_wb 2-5 × C 1.0/4.0) | Classical | **M1 (pilot)** — mandatory strong baseline; re-run on real data under the Phase 1 protocol |
| `phobert-base-ft` | PhoBERT-base (~135M params) full fine-tune, classification head on `[CLS]`, one fixed pre-declared recipe in M1 (no search) | Deep, pretrained | **M1 (pilot)** — primary Phase 2 model |
| `phobert-base-frozen` | PhoBERT-base frozen encoder + logistic regression head on `[CLS]` embeddings | Deep, frozen | Fallback only: used if full fine-tuning is compute-infeasible, or as a reported secondary variant (M2+) |
| `second-pretrained` | One additional pretrained baseline — candidate order: XLM-R-base (multilingual, raw-syllable input), then PhoBERT-large | Deep, pretrained | **OPTIONAL** (M2+), only if the Phase 2 budget is not consumed by the mandatory cells; decision rule in `experiment-matrix.md` |

Notes:

- `phobert-base-frozen` is a **different model**, never conflated with
  `phobert-base-ft`. Its headline claim, if any, is about deployment cost,
  not quality parity.
- **XLM-R is not mandatory for M1.** The pilot's question is directional
  ("does one fine-tuned Vietnamese pretrained model beat the strong
  classical baseline under this protocol?") and one deep model answers it;
  on the 6 GB GPU runs are serial, so the budget goes to M2 seeds of the
  selected model, not a second architecture. XLM-R-base's declared role is
  **fallback only**: it replaces `phobert-base-ft` in M1 solely if PhoBERT
  is blocked (checkpoint/license verification fails, or segmentation
  determinism cannot be shown — `pilot-experiment-plan.md` §13(b)), as a
  documented deviation with owner approval.
- PhoBERT checkpoint facts (verified 2026-11-07 from the Hugging Face card
  `vinai/phobert-base` and `config.json`): ~135M parameters, hidden 768,
  12 layers, vocab 64,001 BPE tokens, `max_position_embeddings=258`, card
  license tag `license:mit`. Re-verify the license tag at Phase 2 kickoff and
  record it in the reproducibility metadata.

## 3. Dataset selection and integrity checks

### 3.1 Select one eligible dataset

Use a locally held dataset with documented permission for the planned use.
The default Phase 1 path uses human-approved custom JSONL. An authorized
local five-level CSV or NEU-ESC CSV is also eligible after verifying label
semantics and recording any collapse or exclusion policy. NEU-ESC access
remains subject to its provider's gate; Toxic stays distinct and requires
an explicit policy for a three-class comparison. See the
[adapter layouts](../../data/README.md) and
[Phase 1 experiment protocol](../experiment-protocol.md).

UIT-VSFC remains an optional, agreement-gated adapter and is excluded from
the current plan. Its published scale (roughly 16k rows, with 11,426 train,
1,583 validation and 3,166 test per the authors' card as read on 2026-11-07)
is historical context used in compute estimates. Use actual effective
counts and measurements for the selected dataset.

### 3.2 Integrity gate before fitting

1. Freeze the dataset identity, version, adapter, verified label mapping,
   applicable exclusions, and physically separate holdout. Check that
   collection groups do not leak across splits. Use group-aware splitting
   where the source has repeated or related examples.
2. Run `vfi prepare-data` on the effective train and validation sources. Run
   `vfi evaluate --preflight-only` before fitting to check all three splits
   and record source provenance and effective fingerprints. Retain the
   passing `evaluation_preflight.json`. Use the receipt to verify final
   evaluation provenance and keep its test data out of fit and selection.
3. Resolve exact cross-split matches and contradictory labels before fitting.
   Treat bounded near-duplicate signals as a review queue; related texts
   or known collection groups still need separate local review. Keep raw
   feedback and row-level evidence local.
4. Fit and select both real Phase 1 baselines on the same approved snapshot.
   Retain their real validation reports, artifact hashes, and training
   provenance before starting PhoBERT. Keep test scoring closed until all
   final configurations are frozen. A dataset, mapping, or holdout change
   invalidates the comparison and requires a fresh preflight and baselines.

### 3.3 Verify the selected dataset's label semantics

Document the source label definitions and mapping to negative, neutral and
positive. Check their meaning against source documentation and a local
human review where needed. For owner-created records, retain annotation
provenance and approval. A permutation of labels can preserve aggregate
scores while changing class meanings, so scores cannot verify the mapping.
The disputed UIT-VSFC raw 0/1/2 order is documented in
[label-semantics-evidence.md](label-semantics-evidence.md). Follow the
explicit mapping gate if that optional adapter ever becomes authorized.

## 4. Preprocessing pipelines (documented, not hidden)

The two families intentionally use different text pipelines. The differences
are stated here and in every Phase 2 report; hiding them is a fairness
violation, not a convenience.

| Step | Classical (`majority`, `tfidf-logreg`) | Transformer (`phobert-base-ft`, frozen, optional second model) |
| ---- | -------------------------------------- | -------------------------------------------------------------- |
| Raw input | Selected adapter's frozen records | Same texts, labels, and split order |
| Normalization | `normalize_text`: Unicode NFC + whitespace collapse | Same `normalize_text` first (shared, minimal) |
| Tokenization | TF-IDF analyzer (whitespace words or char_wb), `lowercase=True` inside the vectorizer | Word segmentation, then PhoBERT's BPE tokenizer |
| Word segmentation | Not applied (whitespace tokens) | **Mandatory**: PhoBERT was pre-trained on word-segmented Vietnamese (underscore-joined syllables, e.g. `sinh_viên`); unsegmented input measurably degrades it |
| Case handling | Lowercased by the vectorizer | Preserved (tokenizer is case-preserving) |
| Sequence length | Unbounded (sparse features) | `max_seq_length=256` (PhoBERT's model maximum, `tokenization-design.md` §3), truncation from the right; measure train token lengths (p99) and train/validation truncation before selection; record holdout truncation only in final evaluation — a meaningful tail is a documented limitation, not silent truncation |
| Fitted on our data | TF-IDF vocabulary + IDF, train split only | No preprocessing fit on this project's data. Segmenter and tokenizer are external, fixed components. |

Segmentation tooling: **VnCoreNLP RDRSegmenter** (`py_vncorenlp`) — the
VnCoreNLP tool + pre-trained models are **GPLv3** (`LICENSE.md`, verified
2026-11-07; `tokenization-design.md` §4): acceptable for local research use,
re-examine before any Phase 3 serving. `underthesea` is the license-safe
backup — **Apache-2.0** on the main repo (older releases were GPL-3.0, so
the pinned version's license is recorded). The historical ≈16k-row size
suggests minutes of CPU work `[ESTIMATE]`. Measure the selected dataset.

Apply the same pinned segmenter to each split when that stage is allowed
to read it. Fit and selection use only train and validation caches. Build
the holdout cache in the dedicated final evaluation path once selection is
frozen. Derived outputs are rebuilt from the frozen source, kept local,
and fingerprinted.

## 5. Split discipline

1. **Effective train** is used for all fitting, including the TF-IDF vocabulary,
   classifier weights, fine-tuning updates, and any frozen-encoder head.
2. **Effective validation** is used for hyperparameter selection, early
   stopping, checkpoint choice, and error analysis.
3. **Test holdout** is accessed by the evaluation path only. Its preflight
   verifies integrity and fingerprints before fitting. Matching provenance
   is required at final evaluation. Score each frozen configuration once per
   seed; existing paired test predictions may support descriptive post-hoc
   statistics, with no retraining or reselection in response.

A Phase 1 grouped validation fallback is allowed when there is a separate
test holdout. Freeze and reuse its effective train and validation
assignments for all arms, including Phase 2. Without a usable holdout,
report validation evidence only and stop before the benchmark pilot.

## 6. Seed policy

Sources of randomness differ by family, so the policy does too:

| Where | Seeds | Repetition | Variance reported |
| ----- | ----- | ---------- | ----------------- |
| Data preparation (fallback split, latency sampling) | 42 | Single | Not stochastic |
| Classical fitting (`majority`, `tfidf-logreg`) | 42 (`random_state`) | Single run | ≈0: deterministic given fixed data and settings; optionally one repeat with seed 43 to confirm determinism (minutes `[ESTIMATE]`) |
| Fine-tuning pilot run (M1, §7) | 42 only | Single fixed-recipe run | Not used for variance claims; single-seed phrasing ("in this run") |
| Fine-tuning escalation (M2, pre-declared) | {42, 7, 2026} (3 seeds) | 3 independent runs, identical hyperparameters, differing only in seed | mean ± std of validation macro-F1 and test macro-F1 across seeds; headline = mean, fixed in advance |
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

## 7. Fine-tuning hyperparameter search budget (M2 escalation — not the pilot)

**M1 uses none of this section**: the pilot runs the single pre-declared
configuration of `pilot-experiment-plan.md` §5 (LR 3e-5, effective batch 32
via micro-batch 8 × accumulation 4, max length 256, ≤ 5 epochs, seed 42) —
zero search. This section describes the M2 escalation. Run it only after
owner approval and protect model selection from any M1 test scores already
released. If those scores informed a decision, any new benchmark requires a
fresh holdout and matching baselines.

### 7.1 The search

| Dimension | Values | Tuned? |
| --------- | ------ | ------ |
| Learning rate | {2e-5, 3e-5, 5e-5} | Tuned (3 candidates, one seed each) |
| Epochs | ≤ 5 with early stopping | Tuned implicitly by early stopping, not by the grid |
| Batch size | 32 effective (micro-batch 8 × grad-accum 4; OOM ladder halves the micro-batch and doubles accumulation — `pilot-experiment-plan.md` §7.2), recorded per run | Fixed |
| Max sequence length | 256 (PhoBERT model maximum, §4) | Fixed |
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
bounds every run at 5 validation passes. The historical ≈1.6k-row validation
set served only as a planning example.

### 7.3 Why the grid stays small

- **Cost.** Each run is a full fine-tune: ≈1,429 micro-steps and ≈357
  optimizer updates per epoch × ≤ 5 epochs on train ≈ 11.4k sentences
  `[ESTIMATE]`. On the RTX 3060 Laptop 6 GB that is ≈20-45 min per run
  `[ESTIMATE]` (fp16, seq 256, micro-batch 8 × grad-accum 4; ~0.2-0.35
  s/micro-step — `pilot-experiment-plan.md` §7.1 arithmetic). The
  3-candidate search ≈ 1-2.5 h; a 3×3×3 grid (LR × batch × epochs) would be
  ≈ 27 runs ≈ 9-20 h of serial GPU time on the same 6 GB card — runs cannot
  be parallelized on it.
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
2. **Across search candidates** (M2 only — M1 has a single pre-declared
   candidate): the candidate whose best checkpoint has the highest
   validation macro-F1; ties break to the **first** candidate in the fixed
   order [2e-5, 3e-5, 5e-5] — mirroring Phase 1's "ties break to the first
   candidate".
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

1. **Same snapshot**: use identical effective fit and selection records,
   holdout fingerprint, and adapter provenance. Changes to examples,
   split policy, resampling, or augmentation require a new comparison.
2. **Same label space**: use canonical `negative/neutral/positive`.
   Verify the selected adapter's mapping once (§3.3) and bind it to all runs.
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
  0.27 GB gradients ≈ 2.2 GB, plus ≈ 1-2 GB activations at micro-batch 8 ×
  seq 256 (`compute-feasibility.md` §3.2) — roughly 3.2-4.2 GB total, with
  headroom on 6 GB. The pre-declared OOM ladder (halve micro-batch, double
  accumulation, effective batch unchanged) is the fallback if measurement
  disagrees — `pilot-experiment-plan.md` §7.2; no batch size is guaranteed
  before measurement.
- **PhoBERT-large does not fit 6 GB** for training (`[ESTIMATE]` ≈ 6-7 GB
  optimizer state alone); it is cloud-GPU-only and optional.
- **CPU-only fine-tuning** is ≈10-25× slower than the local GPU
  `[ESTIMATE]` → ≈4-10 h per run → the full protocol (search + seeds) is
  compute-infeasible on CPU; the frozen-encoder variant (encode once, fit a
  logistic head) is the CPU-feasible fallback.
- **Cloud GPU** changes hours, not science: the protocol is identical; the
  selected dataset's terms and explicit owner approval determine whether
  any third-party upload is allowed.

## 11. Risks and fallbacks

| Risk | Trigger | Fallback |
| ---- | ------- | -------- |
| Data or holdout unavailable | No authorized, integrity-checked benchmark snapshot | Continue readiness while the selected dataset and holdout are prepared. |
| Label mapping fails the audit | Source label meanings remain uncertain (§3.3) | Resolve them locally and rerun preflight before fitting either model family. |
| Segmentation determinism cannot be shown | Double-segmentation fingerprints differ or golden-file test fails | Halt the PhoBERT arm; debug/pin the tool version; if unresolvable, `phobert-base-ft` = `BLOCKED(segmentation)` and the XLM-R fallback (§2) becomes available with owner approval (`pilot-experiment-plan.md` §13(b)) |
| VRAM OOM at micro-batch 8 | CUDA OOM on RTX 3060 6 GB | Pre-declared ladder: 8×4 → 4×8 → 2×16 → 1×32 (effective batch stays 32, recorded deviation); OOM even at micro-batch 1 = hardware-failure verdict, fallback per `pilot-experiment-plan.md` §13(c) |
| fp16 loss instability | Non-finite loss / NaN validation | Switch the affected run to fp32, record the deviation, re-run that candidate |
| Neutral class collapse in validation | Neutral recall ≈ 0 across searched LRs | Planned class-weighted candidate joins the search on validation (§7.1), before any test run |
| Fine-tuning infeasible on local GPU | Sustained hardware failure / VRAM below expectation | Frozen-encoder variant as the reported deep model; full fine-tune stays `BLOCKED(compute)`; claims scoped accordingly |
| Cloud hosting prohibited by dataset terms | Dataset terms forbid third-party upload | Cloud cells become `BLOCKED(license)`; local GPU remains the only compute path |

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
