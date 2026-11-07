# 00 — Executive Decision: Phase 2 Readiness

Program date: 2026-11-07. Program type: research & readiness only — **no
fine-tuning, no dataset downloads, no weight downloads** were performed.

Verified starting checkpoints (live `git ls-remote` + `gh pr view` at program
start, 2026-11-07):

- `origin/main` = `caddb4a59c2744817c6877e8675af1eb8e69c946` (bootstrap only)
- Phase 1 branch `phase1/classical-ml-baselines` / PR #1 head =
  `95f7687735c785367fa0a6f9b1efbf0e284aac7f` (7 commits, Draft/Open, CI green)
- Phase 1 was used as a READ-ONLY reference (`git show <sha>:<path>`); no
  Phase 1 file was modified anywhere by this program.

---

## Decision: CONDITIONAL GO

Phase 2 may proceed **conditionally**. Everything that does not require the
corpus is GO now. Any training or real-data experiment is NO-GO until every
gate below is closed by the owner.

| # | Gate (all mandatory before fine-tuning) | Status | Owner action |
| - | --------------------------------------- | ------ | ------------ |
| G1 | UIT-VSFC usage authorization: email `kietnv@uit.edu.vn`, sign the corpus user agreement, receive corpus | **OPEN** | Email + sign (template in `dataset-acquisition-plan.md`) |
| G2 | Label semantics settled: the two public sources DISAGREE on the raw integer order (see Conflicts below); must be settled by the author's answer + a blind audit of delivered label files vs sentence sentiment | **OPEN** | Ask author in the G1 email; run the audit from `experiment-blueprint.md` on receipt |
| G3 | Artifact/ignore hardening for torch-era files (`*.bin`, `*.pt`, `*.ckpt`, `.env` tracked-file checks) before the FIRST HF checkpoint download | **OPEN** | Approve the hardening patch (owned by Phase 1 maintainer; see `risk-register.md` R-08) |
| G4 | Pinned dependency stack for Phase 2 (torch/transformers/segmenter) in a Phase 2 branch | **OPEN** | Approve after G1–G3 |
| G5 | Owner completes the preprocessing/tokenization learning tasks | **OPEN** | `owner-learning-roadmap.md` weeks 1–2 |

## What can begin NOW (GO)

- Model study and tokenizer experiments on synthetic text (no weights needed
  to study docs; first real download only after G3).
- Segmentation toolchain decision and local smoke tests on sample sentences
  (underthesea is Apache-2.0 — the license-safe default; VnCoreNLP is GPLv3 —
  verify yourself in `tokenization-design.md`).
- Hardware preparation: install CUDA-enabled torch in a dedicated venv and
  verify `torch.cuda.is_available()` (GPU measured: RTX 3060 Laptop 6 GB —
  sufficient for the planned scope, see `compute-feasibility.md`).
- All owner learning tasks that do not need the corpus.
- Synthetic-only smoke tests (E0 in `experiment-matrix.md`).

## What is BLOCKED (NO-GO)

- Downloading UIT-VSFC from any mirror while the license is unverified — the
  author org's own HF card states "Licensing Information: Unknown" and the
  official channel requires a signed agreement.
- Any fine-tuning, any real-data metric, any leaderboard-style claim.

## Answers to the twelve readiness questions

1. **Which dataset can we use?** None is usable today without permission.
   Primary target remains **UIT-VSFC** (best domain/label/schema fit; author
   agreement required). Best USABLE-NOW alternative is **NEU-ESC**
   (Apache-2.0, education domain, 33k items) with documented caveats: 4-class
   schema, 69% neutral skew, no published annotation protocol. Details:
   `dataset-landscape.md`, `dataset-license-matrix.md`.
2. **Can we obtain UIT-VSFC under author conditions?** Yes, via email to
   `kietnv@uit.edu.vn` and signing the corpus user agreement; the condition
   was re-verified unchanged on 2026-11-07 (`dataset-acquisition-plan.md`).
3. **Suitable alternatives?** Ranked table in `dataset-license-matrix.md`;
   NEU-ESC (usable now), UIT-ViSFD and SA-VLSP2016 (permission required),
   NTC-SCV and two scraped/unknown-provenance mirrors REJECTED.
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
9. **Fair classical-vs-deep comparison?** Same splits, same canonical labels,
   same metric code, documented preprocessing differences, plus an optional
   segmented-text TF-IDF variant so the baseline gets the same segmentation
   benefit. Fairness rules: `experiment-blueprint.md` §8.
10. **Leakage/test-tuning prevention?** Integrity gate + blind label audit +
    exact-match AND near-duplicate checks (Phase 1 only does exact-match —
    confirmed risk R-06), validation-only selection, single-touch test
    evaluation with a written ledger. `evaluation-protocol.md`, `risk-register.md`.
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
   sides verified 2026-11-07. **Resolution:** no trust in either mapping for
   real training; settle via the author (question pre-written in the G1
   email) AND a blind audit of the delivered files (G2). Phase 1's
   documented mapping follows SEACrowd and is now flagged
   [UNVERIFIED AGAINST AUTHOR SOURCE] — reported to the Phase 1 maintainer,
   not edited here.
2. **Segmenter licenses.** Agent outputs disagreed. Primary-source check by
   the orchestrator: VnCoreNLP `LICENSE.md` = **GPLv3**;
   `underthesea` `LICENSE` (main) = **Apache-2.0**. Resolution: underthesea
   is the license-safe default; VnCoreNLP remains acceptable for local
   research use with the GPL implication documented (relevant later for
   Phase 3 serving).
3. **Split sizes.** Now [VERIFIED FACT] from the author org card:
   train 11,426 / validation 1,583 / test 3,166 (total 16,175).
4. **"PhoBERT paper reports UIT-VSFC F1"** — FALSE per the paper itself
   (tasks: POS/NER/NLI/parsing); community F1 figures remain
   [UNVERIFIED]. Do not cite them.

## Smallest scientifically valuable experiment (when G1–G5 close)

Fine-tune `vinai/phobert-base` ONCE — seed 42, pre-declared recipe (LR 3e-5,
batch 16 with accumulation, ≤5 epochs, early stopping on validation
macro-F1) — then evaluate it and the Phase 1 TF-IDF+LogReg winner on
validation, then once each on test, with identical metric code and full
provenance. Claims phrased as single-run ("in this run"). Full matrix with
search and multi-seed variance only if the owner wants stronger claims
(`experiment-matrix.md` E0–E9).

## Objective Phase 2 completion criteria

1. G1–G5 closed with evidence.
2. E0 synthetic smoke test green (no reported metrics).
3. All four models evaluated per protocol; test evaluated exactly once each.
4. Reproducibility record complete for every run.
5. Error analysis with indexed examples (no raw student text).
6. Comparison report states per-seed variance and restricted claims.
7. Owner learning checklist items independently signed off by the owner.

## Top risks (see `risk-register.md` for the full register)

| ID | Risk | Severity |
| -- | ---- | -------- |
| R-01 | Untrusted checkpoint loading (pickle `pytorch_model.bin`; use safetensors + pinned revision + checksum) | High |
| R-06 | Near-duplicate cross-split leakage invisible to exact-match checks | High |
| R-08 | Accidental commit of weights/raw data (gitignore/verify gaps for torch-era files) | High |
| R-07 | Label mapping ambiguity (now a VERIFIED two-source conflict) | High |
| R-04 | License enforcement pressure once data is local | High |

## Evidence table (decision → primary source)

| Decision | Primary source (fetched 2026-11-07) |
| -------- | ----------------------------------- |
| G1 requirement | https://github.com/kietnv/VietnameseDatasets (agreement clause, verbatim) |
| License "Unknown" on both mirrors | https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback (card, "Licensing Information: Unknown"); https://huggingface.co/datasets/SEACrowd/uit_vsfc (`Licenses.UNKNOWN`) |
| Label-order conflict | Same uitnlp card "Data Fields" section vs https://huggingface.co/datasets/SEACrowd/uit_vsfc/raw/main/uit_vsfc.py (ClassLabel order) |
| Split sizes | uitnlp card "Data Splits" table (11,426/1,583/3,166) |
| PhoBERT choice + segmentation contract | https://huggingface.co/vinai/phobert-base (card: "INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!", MIT) + https://arxiv.org/abs/2003.00744 |
| VnCoreNLP GPLv3 / underthesea Apache-2.0 | https://github.com/vncorenlp/VnCoreNLP/blob/master/LICENSE.md ; https://github.com/undertheseanlp/underthesea (LICENSE) |
| Hardware sufficiency | `tools/phase2_readiness/hardware_probe.py` measured output, 2026-11-07 (sanitized) |
| NEU-ESC as usable-now alternative | https://huggingface.co/datasets/ (see `dataset-landscape.md` §NEU-ESC for exact gated link and license tag) |

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
