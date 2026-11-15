# 00 — Executive Decision: Phase 2 Readiness

Research snapshot: 2026-11-07. Release reconciliation: 2026-11-15.
Phase 2 remains a readiness proposal. Real experiments require approved data,
verified baseline evidence and explicit owner approval.

Verified starting checkpoints (live `git ls-remote` + `gh pr view` at program
start, 2026-11-07):

- `origin/main` = `caddb4a59c2744817c6877e8675af1eb8e69c946` (bootstrap only)
- Phase 1 branch `phase1/classical-ml-baselines` / PR #1 head =
  `95f7687735c785367fa0a6f9b1efbf0e284aac7f` (7 commits, Draft/Open, CI green)
- Phase 1 was used as a READ-ONLY reference (`git show <sha>:<path>`); no
  Phase 1 file was modified anywhere by this program.

Phase 1 is now released at `db9b921791832337ac0536ba5b1ae03f664ba2d0`.
Its adapters accept authorized local data and owner-collected JSONL after human
annotation and approval. See [the released protocol](../experiment-protocol.md)
and [CI on main](https://github.com/pminhquan/Vietnamese-Feedback-Intelligence/actions/runs/43682522477).

---

## Decision: CONDITIONAL GO

Proceed with planning and the first authorized classical baseline. Fine-tuning
requires completed evidence for the chosen dataset, both Phase 1 baselines, and
the remaining operational gates. No source has been selected for a real run.

| # | Gate (all mandatory before fine-tuning) | Status | Owner action |
| - | --------------------------------------- | ------ | ------------ |
| G1 | Verify and record the selected source's permitted access and use, or the collection and consent basis for owner-collected data | **OPEN** | Follow [dataset acquisition](dataset-acquisition-plan.md) |
| G2 | Verify the source-specific mapping and annotations, freeze physical splits, and pass preflight | **OPEN** | Use the Phase 1 protocol on the selected source before either baseline is fitted for comparison |
| G3 | Confirm checkpoint license, pinned revision, safe serialization, checksum and repository hygiene | **PARTIAL** | Git/CI protections released in Phase 1. Checkpoint evidence is pending. |
| G4 | Pinned dependency stack for Phase 2 (torch/transformers/segmenter) in a Phase 2 branch | **OPEN** | Approve after G1–G3 |
| G5 | Owner completes the preprocessing and tokenization learning tasks and approves the pilot scope | **OPEN** | [Learning plan](owner-learning-roadmap.md) weeks 1–2 and recorded approval |

## What can begin NOW (GO)

- Model study and tokenizer experiments on synthetic text (no weights needed
  to study docs; first real download only after G3).
- Segmentation toolchain decision and local smoke tests on sample sentences
  using the documented pilot candidate and fallback. See
  [tokenization design](tokenization-design.md).
- Hardware preparation: install CUDA-enabled torch in a dedicated venv and
  verify `torch.cuda.is_available()` (GPU measured: RTX 3060 Laptop 6 GB —
  sufficient for the planned scope, see `compute-feasibility.md`).
- All owner learning tasks that do not need the corpus.
- Synthetic-only smoke tests (E0 in `experiment-matrix.md`).

## What is BLOCKED (NO-GO)

- Downloading UIT-VSFC from any mirror while the license is unverified — the
  author org's own HF card states "Licensing Information: Unknown" and the
  official channel requires a signed agreement.
- Fine-tuning until the evidence and approval gates pass. No real metric is
  available until an authorized run actually produces it.

## Record release readiness evidence

Generate a fresh template from the branch, then fill its fields from the
actual dataset decision and saved Phase 1 artifacts. Keep evidence and
reports under an ignored local directory.

```powershell
New-Item -ItemType Directory -Force reports/artifacts | Out-Null
python tools/phase2_readiness/readiness_gate.py --schema |
    Set-Content -Encoding utf8 reports/artifacts/phase2-evidence.json
python tools/phase2_readiness/readiness_gate.py --input reports/artifacts/phase2-evidence.json --json
```

The template starts incomplete. Evidence must identify the dataset and
adapter, the source's access terms or collection permission, and the reviewed
label interpretation. Record the effective split fingerprints and the
`created_at` timestamp and SHA-256 of the passing `evaluation_preflight.json`.
Record a separate near-duplicate audit covering all three splits. Save the
frozen preflight digest with the baseline evidence and record
`training_started_at` from `training_report.json`. Confirm that the dataset
identity and label definition match across the baseline and preflight
reports. Validation results from both baselines are sufficient while the
test remains sealed. The gate checks declared evidence for completeness
and consistency. The owner reviews the cited material and approves the
PhoBERT pilot.

## Answers to the twelve readiness questions

1. **Which dataset can we use?** Prefer independent feedback with documented
   collection rights, human annotation and approval, or an external dataset
   with verified terms and a fitting label schema.
2. **What is the UIT-VSFC status?** Optional adapter only. The owner chose
   to pursue independent data. Any future use of UIT-VSFC still requires its
   signed agreement and verified mapping of the actual files.
3. **Suitable alternatives?** NEU-ESC is a candidate after Hugging Face
   acceptance and a review of the source terms. Toxic requires an explicit
   policy and never becomes negative. See [dataset research](dataset-landscape.md)
   and [access evidence](dataset-license-matrix.md).
4. **Which pretrained models?** First candidate `vinai/phobert-base` (135M,
   MIT on card, sentence-classification compatible); fallback
   `xlm-roberta-base` (279M, MIT, no segmentation needed). Deferred:
   phobert-base-v2 (AGPL-3.0), phobert-large (size), ViT5 (encoder-decoder
   friction), FPT.AI ViBERT/vELECTRA (no model card → excluded).
   `model-landscape.md`, `model-selection-decision.md`.
5. **Preprocessing/tokenization required?** PhoBERT requires **pre-segmented**
   Vietnamese (underscore-joined syllables) BEFORE its BPE tokenizer, max 256
   BPE tokens — the model card states this verbatim. Contract, tool licenses
   and the default chain: `tokenization-design.md`.
6. **Compute/memory/time/storage?** Measured laptop: RTX 3060 Laptop 6 GB,
   i5-10500H 6C/12T, 32 GB RAM, ~521 GiB free on D:. Estimates (labeled):
   PhoBERT-base fine-tune ~20–40 min/run with AMP batch 8–16; full protocol
   ~3–6 GPU-hours; disk need < 5 GiB. `compute-feasibility.md`.
7. **Can the laptop run it?** Yes for the entire planned Phase 2 scope
   (estimated); CPU-only fallback is viable but slow (~4–10 h for 3 epochs).
8. **Alternatives if no GPU?** (a) CPU-only patience runs, (b) cloud
   notebooks — but they force corpus upload (privacy/license risk; the user
   agreement may forbid third-party hosting), (c) paid GPU VM only on
   explicit triggers. `compute-feasibility.md` §environment comparison.
9. **Fair classical-vs-deep comparison?** Same dataset version, verified
   mapping, split fingerprints and holdout as the real Phase 1 baselines.
   Use the same metric code, document preprocessing, and consider an optional
   segmented-text TF-IDF variant so the baseline gets the same segmentation
   benefit. Fairness rules: `experiment-blueprint.md` §8.
10. **Leakage/test-tuning prevention?** Explicit integrity preflight before
    fitting, exact and near-duplicate audits, validation-only selection, and
    one final scoring per locked configuration. The Phase 1 near-duplicate
    sample alone does not cover all split pairs. See [evaluation](evaluation-protocol.md).
11. **Reproducibility?** Per-run metadata schema (git SHA, seeds per stage,
    torch/transformers/CUDA/segmenter versions, config hash, artifact
    hashes); Phase 1's sklearn-only record is insufficient and must be
    extended. `evaluation-protocol.md` §metadata.
12. **Owner's independent understanding?** 14-topic Vietnamese roadmap +
    20 interview questions + evidence checklist: `owner-learning-roadmap.md`,
    `interview-readiness.md`, `portfolio-evidence-plan.md`.

## Verified conflicts and their resolutions

1. **Raw label order (CRITICAL).** `SEACrowd/uit_vsfc` loader declares
   `ClassLabel(names=["positive","negative","neutral"])` → 0=positive, while
   the author org's own card `uitnlp/vietnamese_students_feedback` states
   "`sentiment`: values 0 (negative), 1 (neutral) and 2 (positive)". Both
   sides recorded 2026-11-07. The mapping of original files remains unresolved.
   The released optional adapter requires rights, an explicit mapping and
   evidence. Independent adapters use their own verified label semantics.

   Hardening pass 2026-11-07 (`label-semantics-evidence.md`): both mirrors
   consume the identical nine Google Drive files with unmodified integer
   passthrough, so their contradictory declarations mean one mirror
   systematically mislabels every row; SEACrowd's label reorder (PR #246)
   landed with zero verification discussion, while the authors' own org
   carries three mutually consistent artifacts (card prose, migrated loader
   `ClassLabel`, served parquet) favoring 0=negative. That evidential
   balance is recorded as a planning prior only; the audit hypotheses are
   pre-registered (§7 of that document) for any future authorized use.
2. **Segmenter licenses.** Agent outputs disagreed. Primary-source check by
   the orchestrator: VnCoreNLP `LICENSE.md` = **GPLv3**;
   `underthesea` `LICENSE` (main) = **Apache-2.0**. The pilot proposes
   VnCoreNLP for fidelity to PhoBERT pretraining and records underthesea as
   a separately pinned fallback. Recheck distribution implications before
   any later serving work.
3. **Split sizes.** Now [VERIFIED FACT] from the author org card:
   train 11,426 / validation 1,583 / test 3,166 (total 16,175).
4. **"PhoBERT paper reports UIT-VSFC F1"** — FALSE per the paper itself
   (tasks: POS/NER/NLI/parsing); community F1 figures remain
   [UNVERIFIED]. Do not cite them.

## Smallest scientifically valuable experiment (when G1–G5 close)

First fit majority and TF-IDF + Logistic Regression on authorized, preflighted
data and freeze their artifacts and validation records. Then fine-tune
`vinai/phobert-base` once — seed 42, pre-declared recipe (LR 3e-5,
effective batch 32 via micro-batch 8 × grad-accum 4, OOM ladder
8×4→4×8→2×16→1×32 pre-declared — no batch size is guaranteed before
measurement; ≤5 epochs, early stopping on validation macro-F1) — then
select using validation, then evaluate all three locked configurations once
each on the same test holdout, with identical metric code and
full provenance. Implementation contract: `pilot-experiment-plan.md`. Claims
phrased as single-run ("in this run"). Full matrix with
search and multi-seed variance only if the owner wants stronger claims
(`experiment-matrix.md` E0–E9).

## Objective Phase 2 completion criteria

1. G1–G5 closed with evidence.
2. E0 synthetic smoke test green (no reported metrics).
3. All three mandatory M1 models evaluated per protocol (majority,
   TF-IDF+LogReg, one PhoBERT fine-tune; XLM-R is the declared fallback
   only); test evaluated exactly once per final configuration.
4. Reproducibility record complete for every run.
5. Error analysis with indexed examples (no raw student text).
6. Comparison report scopes claims to the matched benchmark and reports
   per-seed variation only if multiple seeds were run.
7. Owner learning checklist items independently signed off by the owner.

## Top risks (see `risk-register.md` for the full register)

| ID | Risk | Severity |
| -- | ---- | -------- |
| R-01 | Untrusted checkpoint loading (pickle `pytorch_model.bin`; use safetensors + pinned revision + checksum) | High |
| R-06 | Near-duplicate cross-split leakage invisible to exact-match checks | High |
| R-08 | Accidental commit of weights/raw data despite released path guards | High |
| R-07 | Label mapping ambiguity, including the historical UIT-VSFC conflict | High |
| R-04 | License enforcement pressure once data is local | High |

## Evidence table (decision → primary source)

| Decision | Primary source (fetched 2026-11-07) |
| -------- | ----------------------------------- |
| Optional UIT-VSFC access requirement | https://github.com/kietnv/VietnameseDatasets (agreement clause) |
| License "Unknown" on both mirrors | https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback (card, "Licensing Information: Unknown"); https://huggingface.co/datasets/SEACrowd/uit_vsfc (`Licenses.UNKNOWN`) |
| Label-order conflict | Same uitnlp card "Data Fields" section vs https://huggingface.co/datasets/SEACrowd/uit_vsfc/raw/main/uit_vsfc.py (ClassLabel order) |
| Split sizes | uitnlp card "Data Splits" table (11,426/1,583/3,166) |
| PhoBERT choice + segmentation contract | https://huggingface.co/vinai/phobert-base (card: "INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!", MIT) + https://arxiv.org/abs/2003.00744 |
| VnCoreNLP GPLv3 / underthesea Apache-2.0 | https://github.com/vncorenlp/VnCoreNLP/blob/master/LICENSE.md ; https://github.com/undertheseanlp/underthesea (LICENSE) |
| Hardware sufficiency | `tools/phase2_readiness/hardware_probe.py` measured output, 2026-11-07 (sanitized) |
| NEU-ESC as a gated candidate | [Research and access caveats](dataset-landscape.md) |

## Program transparency

Six workstreams were dispatched to independent subagents (A dataset/legal,
B models/tokenization, C hardware/compute + probe tool, D experiment design,
E security/dependency review, F education/portfolio). The host environment
limited concurrent subagents: rounds ran with 2→1 concurrent workers
(`user concurrency limit exceeded` on two early dispatch attempts); every
workstream completed exactly once with its scoped files. The orchestrator
independently re-verified the four highest-impact claims (label conflict,
VnCoreNLP/underthesea licenses, split sizes, hardware) against primary
sources and resolved the conflicts above. No dataset, weight, or secret was
downloaded; no training was run; no Phase 1 file was modified; PR #1 was not
touched.

## Historical hardening pass (2026-11-07)

An independent hardening round reviewed and extended this package:

- **Probe privacy holes found AND fixed.** A red-team agent reproduced five
  real holes in `hardware_probe.py` end-to-end (default-run temp-file
  persistence, CWD-relative repo refusal, `\\?\`/trailing-space/UNC path
  bypasses, environment-spoofed sanitization, argparse echo of user paths);
  all five are fixed with regression tests, and the strict-xfail pins were
  converted to passing tests. Guarantees and remaining limits are documented
  in `hardware-probe-guide.md`.
- **Offline readiness gate added.** `tools/phase2_readiness/readiness_gate.py`
  checks an owner-prepared evidence JSON against the eleven readiness
  categories (dataset provenance, label verification, splits, leakage
  preflight, Phase 1 integrity, checkpoint/license, tokenization contract,
  dependency plan, compute, artifact/privacy controls, owner approval) and
  fails closed (exit 1 unless every category passes); it is an operational
  checklist — it does not validate legal rights or model quality.
  `readiness_gate.py --schema` prints an intentionally INCOMPLETE onboarding
  template (TODO placeholders and not-done flags; fed back through `--input`
  it exits 1 with overall "blocked") — a schema example only, never readiness
  evidence. Owner flow: fill the package with real evidence, run the gate,
  and require PASS before the PhoBERT pilot.
- **Pilot plan finalized.** `pilot-experiment-plan.md` is the
  implementation-level M1 contract: exactly three models (majority,
  TF-IDF+LogReg, one PhoBERT), no hyperparameter search, seed 42, label
  audit gate blocking every real-data run, pre-declared OOM ladder,
  single-touch test evaluation with a ledger.
- **Phase 1 handoff prepared.** `handoff-phase1-findings.md` is the
  copy-ready handoff for the Phase 1 maintainer (label conflict + confirmed
  issues + new red-team findings; Phase 1 reviewed read-only only).
- Full suite after hardening: 105 tests passing, ruff clean, zero files
  changed outside the authorized paths.
