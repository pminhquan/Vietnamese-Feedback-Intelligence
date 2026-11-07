# Phase 2 Model Selection Decision — Pretrained Model for First Fine-Tuning

Status: research recommendation (Phase 2 readiness only). **This document authorizes nothing**:
Phase 2 fine-tuning remains unapproved and no weights were downloaded while writing it. It fixes a
defensible ordering so that when Phase 2 starts, the first experiment is not a debate.

Companion notes: [model-landscape.md](model-landscape.md) (verified per-model evidence) and
[tokenization-design.md](tokenization-design.md) (preprocessing/segmentation contract).

Evidence labels: **[VERIFIED FACT]**, **[ESTIMATE]**, **[ASSUMPTION]**, **[UNVERIFIED]**.

## 1. Decision criteria (in priority order)

1. **License clarity.** The exact license must be readable from the model card. An empty card with
   no license tag is an automatic exclusion for anything downstream of research — especially with
   Phase 3 serving ambitions.
2. **Documented task fit.** Architecture and preprocessing contract must be verifiable from the
   card and/or the paper, with enough documentation that a segmentation bug can be diagnosed
   against a primary source rather than folklore.
3. **Hardware fit.** Phase 1 CI is CPU-only [VERIFIED FACT, project spec]; Phase 2 fine-tuning
   hardware is not yet fixed. Base-size (~135–300M) models are the working envelope; 370M+ is
   deferred.
4. **Preprocessing complexity.** An external segmentation dependency is acceptable (and manageable,
   see tokenization-design.md) but must be paid for knowingly; a raw-text model has intrinsic ops
   value.
5. **Published evidence.** Direct Vietnamese sentiment-classification numbers on UIT-VSFC carry
   weight only when verified from a primary source.

## 2. Recommendation

### First fine-tuning candidate: `vinai/phobert-base`

Justification, each item anchored in [model-landscape.md](model-landscape.md):

- **License: MIT** (`license: mit` on the card). The most permissive license on the Vietnamese
  shortlist and unproblematic for any future phase. [VERIFIED FACT]
- **Vietnamese-specialized with a fully documented contract.** 135M-parameter RoBERTa-style
  encoder trained on 20GB of Vietnamese Wikipedia + news; max input 256 subword tokens; explicit
  word-segmentation-before-BPE requirement ("INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!").
  [VERIFIED FACT] The preprocessing chain it imposes is fully specified in
  [tokenization-design.md](tokenization-design.md), so the main operational risk of the model is
  controlled.
- **Paper evidence for the design choice.** The PhoBERT paper reports state-of-the-art Vietnamese
  results over multilingual models on POS, dependency parsing, NER, and NLI, outperforming XLM-R
  "using far fewer parameters" (135M vs 250M as counted in the paper). [VERIFIED FACT] Sentiment
  classification on UIT-VSFC is **not** reported in the paper [VERIFIED FACT], so no claimed
  benchmark number enters this decision. Secondary literature reports roughly 89–94% F1 for
  PhoBERT-based systems on UIT-VSFC. [UNVERIFIED — treated as a reason for optimism, not as
  evidence]
- **Size fits the known envelope.** 135M parameters is the smallest verified model on the
  shortlist and the most feasible for CPU inference and modest-GPU fine-tuning. [ESTIMATE table
  below]
- **Natural comparison arm.** The XLM-R fallback is the same multilingual baseline the PhoBERT
  paper itself beat, so the Phase 2 experiment design inherits a published comparison axis.
  [VERIFIED FACT for the paper's comparison]

### Fallback (and planned comparison arm): `xlm-roberta-base`

- MIT license, 278,885,778 parameters (safetensors metadata), 2.5TB CC-100 across 100 languages,
  card explicitly targets "sequence classification, token classification or question answering" on
  raw text — no segmentation dependency, no Java runtime. [VERIFIED FACT]
- Use it as the first candidate if: no GPU becomes available and CPU-only fine-tuning time is the
  binding constraint [ESTIMATE: both base models are CPU-fine-tunable; XLM-R's main advantage is
  operational, not accuracy]; or the Java dependency for VnCoreNLP cannot be satisfied in the
  environment; or as the **planned multilingual comparison arm regardless** — it isolates the value
  of Vietnamese-specialized pretraining and of the segmentation pipeline.
- Known trade-off: the PhoBERT paper's own evidence says a Vietnamese-specialized model with fewer
  parameters beats it on Vietnamese tasks. [VERIFIED FACT]

### Secondary fallback ("lighter"): none currently qualifies

No smaller Vietnamese encoder with a verifiable license and card was found in this research round
(distilled PhoBERT variants exist in the community but none was verified with a proper card here
[UNVERIFIED]). If a lighter candidate is needed later, re-run this diligence rather than adopting
an undocumented checkpoint.

## 3. Deferred and excluded, with reasons

| Checkpoint | Verdict | Reason |
|------------|---------|--------|
| `vinai/phobert-base-v2` (135M) | **Deferred** | Best-documented card in the family [VERIFIED FACT], but **AGPL-3.0** [VERIFIED FACT] — acceptable for research, the wrong default for a portfolio that may serve a product in Phase 3. Optional later arm if its extra 120GB of OSCAR-2301 data is hypothesized to help. |
| `vinai/phobert-large` (370M) | **Deferred** | ~3x the fine-tuning cost of base for short-sentence classification [ESTIMATE]; revisit only if base models saturate. |
| `VietAI/vit5-base` (~310M) | **Excluded from round 1** | MIT and well-documented [VERIFIED FACT], but encoder-decoder (classification needs the `T5ForSequenceClassification` path [ASSUMPTION]), 2.3x the params of phobert-base, generation-oriented card, no documented sentiment-classification evidence. Keep as a stretch arm for a later round. |
| `FPTAI/vibert-base-cased` | **Excluded** | No model card, no license tag on HF [VERIFIED FACT]. License diligence impossible. |
| `FPTAI/velectra-base-discriminator-cased` | **Excluded** | No model card, no license tag on HF [VERIFIED FACT], despite community UIT-VSFC fine-tunes existing [VERIFIED FACT for the derivative's existence]. Re-evaluate if FPT.AI publishes terms. |
| BamiBERT | **Excluded** | Recommended on the PhoBERT card (129GB corpus, 2,048 context, raw input) [VERIFIED FACT], but no checkpoint is findable on the HF Hub by that name [VERIFIED FACT — empty search]. |

## 4. Memory/hardware implications (all [ESTIMATE])

AdamW fine-tuning in fp32 costs roughly 16 bytes/parameter (weights + gradients + optimizer state)
before activations.

| Model | Params | Fixed optimizer+weights | Rough total VRAM, batch 16, len 256 | CPU RAM fine-tune | CPU inference |
|-------|--------|------------------------|--------------------------------------|-------------------|---------------|
| `vinai/phobert-base` | 135M | ~2.2 GB | ~3–6 GB | 8–16 GB practical | yes, seconds-scale [ESTIMATE] |
| `xlm-roberta-base` | 279M | ~4.5 GB | ~5–9 GB | 16 GB practical | yes, slower [ESTIMATE] |
| `VietAI/vit5-base` | ~310M | ~5.0 GB | ~6–10 GB | 16 GB practical | yes, slower [ESTIMATE] |
| `vinai/phobert-large` | 370M | ~6.0 GB | ~8–14 GB | 32 GB practical | painful [ESTIMATE] |

Mixed precision (bf16/fp16) and gradient checkpointing reduce these further; 8-bit optimizers more
so. All numbers are planning estimates, not measurements — the binding constraint must be settled
before Phase 2 kicks off, because it decides between the primary candidate and the fallback.

## 5. Decision record

- **Decision (as of 2026-11-07):** first fine-tuning candidate = `vinai/phobert-base` (MIT,
  135M, segmentation required); fallback + comparison arm = `xlm-roberta-base` (MIT, 279M,
  raw input).
- **Preprocessing contract** for the primary candidate is fixed in
  [tokenization-design.md](tokenization-design.md) and must be implemented (including its tests)
  as part of the first Phase 2 milestone.
- **Not authorized by this document:** downloading weights, running fine-tuning, or committing to
  Phase 2 timelines. This is a readiness recommendation only.

## 6. Re-evaluation triggers

1. FPT.AI publishes license/model-card documentation for ViBERT/vELECTRA → re-evaluate vELECTRA as
   a raw-input (or at least alternative-preprocessing) Vietnamese encoder arm. [ESTIMATE that it
   would be competitive]
2. UIT-VSFC real token-length distribution shows a meaningful fraction above 256 PhoBERT tokens →
   revisit truncation strategy and the 512-context / raw-input models (XLM-R; BamiBERT if it ever
   becomes available).
3. No GPU materializes for Phase 2 → prefer CPU-feasible paths; XLM-R vs phobert-base choice
   becomes an ops decision, with fine-tuning-time measurements on the actual hardware
   ([ESTIMATE] today).
4. A licensed, documented, lighter Vietnamese encoder (e.g., a properly carded distilled PhoBERT)
   appears → repeat this diligence; do not adopt undocumented checkpoints.
5. Phase 3 scope confirms a network-served product → AGPL models (`phobert-base-v2`) move from
   "deferred" to "excluded for production" regardless of accuracy.

## 7. Shortlist table (canonical verdicts)

| Checkpoint | Params | License | Segmentation required | Verdict |
|------------|--------|---------|----------------------|---------|
| `vinai/phobert-base` | 135M | MIT | Yes (RDRsegmenter/py_vncorenlp) | **First candidate** |
| `xlm-roberta-base` | 278.9M | MIT | No | **Fallback + comparison arm** |
| `vinai/phobert-base-v2` | 135M | AGPL-3.0 | Yes | Deferred (license) |
| `vinai/phobert-large` | 370M | MIT [ASSUMPTION — via family cards] | Yes | Deferred (size) |
| `VietAI/vit5-base` | ~310M | MIT | No | Later stretch arm (seq2seq) |
| `FPTAI/vibert-base-cased` | unknown | none shown | unknown | Excluded (no card/license) |
| `FPTAI/velectra-base-discriminator-cased` | unknown | none shown | unknown | Excluded (no card/license) |

## 8. Key uncertainties

- Fine-tuning hardware (GPU availability and VRAM) is unresolved and drives the primary/fallback
  choice more than model quality does. [OPEN]
- No verified published PhoBERT-on-UIT-VSFC F1 number exists in this research; secondary sources
  suggest ~89–94% macro-F1 but none was fetched from a primary source. [UNVERIFIED]
- The FPT.AI checkpoints' licenses and segmentation requirements remain unknown. [UNVERIFIED]
- Exact `sentencepiece` dependency for XLM-R and the per-checkpoint max-length configs for ViT5
  are implementation details to confirm at Phase 2 environment setup. [ASSUMPTION]
