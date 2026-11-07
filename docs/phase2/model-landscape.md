# Phase 2 Model Landscape — Pretrained Models for Vietnamese Sentiment Classification

Status: research note (Phase 2 readiness only). Phase 2 fine-tuning is **not** authorized yet and
no model weights were downloaded while producing this document. All facts below were checked
against primary sources on 2026-11-07 unless labeled otherwise.

Evidence labels used throughout: **[VERIFIED FACT]** (fetched from the cited primary source),
**[ESTIMATE]**, **[ASSUMPTION]**, **[UNVERIFIED]**.

Related notes: [tokenization-design.md](tokenization-design.md) covers the preprocessing/segmentation
contract in depth; [model-selection-decision.md](model-selection-decision.md) records the shortlist
verdict and the recommended first candidate.

## 1. Scope and method

- Target task: 3-class sentence-level sentiment classification (`negative` / `neutral` / `positive`)
  over the UIT-VSFC format, as fixed by the Phase 1 pipeline.
- Method: for every candidate we fetched the actual Hugging Face model card or HF Hub API record,
  and where the card was thin, the underlying paper (arXiv) or the author's GitHub repository.
  Facts that could not be anchored to a primary source are explicitly labeled, not guessed.
- The shortlist must cover: the PhoBERT family (including `phobert-base-v2`, which does exist),
  at least one credible Vietnamese-specific alternative, and one multilingual model for comparison.

## 2. Shortlist summary

| # | Checkpoint | Architecture | Params | License (as shown on card) | Word segmentation required | Card quality | Verdict |
|---|------------|--------------|--------|----------------------------|---------------------------|--------------|---------|
| 1 | [`vinai/phobert-base`](https://huggingface.co/vinai/phobert-base) | RoBERTa-style encoder | 135M [VERIFIED] | `license: mit` [VERIFIED] | **Yes** [VERIFIED] | Thin card; details recoverable from paper + v2 card | **Primary candidate** |
| 2 | [`vinai/phobert-base-v2`](https://huggingface.co/vinai/phobert-base-v2) | RoBERTa-style encoder | 135M [VERIFIED] | AGPL-3.0 [VERIFIED] | **Yes** [VERIFIED] | Good card | Upgrade path, license-gated |
| 3 | [`vinai/phobert-large`](https://huggingface.co/vinai/phobert-large) | RoBERTa-style encoder (large) | 370M [VERIFIED] | `license: mit` [VERIFIED]* | **Yes** [VERIFIED] | Same family card | Excluded for now (size) |
| 4 | [`VietAI/vit5-base`](https://huggingface.co/VietAI/vit5-base) | T5 encoder-decoder | ~310M [VERIFIED] | `license: mit` [VERIFIED] | No (not mentioned anywhere) [VERIFIED] | Decent card + paper | Strong alternative, seq2seq friction |
| 5 | [`FPTAI/vibert-base-cased`](https://huggingface.co/FPTAI/vibert-base-cased) | BERT | unknown [UNVERIFIED] | **No license tag, no card** [VERIFIED] | Unknown | Empty card | Excluded (license diligence fails) |
| 6 | [`FPTAI/velectra-base-discriminator-cased`](https://huggingface.co/FPTAI/velectra-base-discriminator-cased) | ELECTRA discriminator | unknown [UNVERIFIED] | **No license tag, no card** [VERIFIED] | Unknown | Empty card | Excluded (license diligence fails) |
| 7 | [`xlm-roberta-base`](https://huggingface.co/xlm-roberta-base) (canonical: `FacebookAI/xlm-roberta-base`) | XLM-RoBERTa encoder | 278,885,778 (safetensors metadata) [VERIFIED] | `license: mit` [VERIFIED] | No [VERIFIED — card targets raw text, "sequence classification, token classification or question answering"] | Good card | **Fallback / comparison baseline** |

\* The MIT tag for `phobert-large` is asserted by the `vinai/phobert-base-v2` card's model table and
the family's MIT licensing; the large model's own card was not fetched directly [ASSUMPTION].

## 3. Detailed entries

### 3.1 `vinai/phobert-base` — primary candidate

- Checkpoint: `vinai/phobert-base` — https://huggingface.co/vinai/phobert-base [VERIFIED FACT]
- Card content observed: MIT license tag; RoBERTa-based pretraining; "Please CITE our paper when
  PhoBERT is used to help produce published results"; pointer to the EMNLP-2020 Findings paper
  (arXiv:2003.00744); a recommendation to also look at BamiBERT (see 3.7). [VERIFIED FACT]
- Architecture: RoBERTa-style encoder (`RobertaForMaskedLM` config; `model_type: roberta` per HF
  API). [VERIFIED FACT]
- Parameters: **135M** (PhoBERT-base) — stated both in the paper and in the
  `phobert-base-v2` card's model table. [VERIFIED FACT]
- Pre-training data: 20GB of Vietnamese text — "a concatenation of two corpora": Vietnamese
  Wikipedia (~1GB) + ~19GB deduplicated news. [VERIFIED FACT, paper + v2 card]
- License: the card shows only the metadata tag `license: mit`. No extended license section is
  present on the current card. [VERIFIED FACT] MIT permits research and commercial use subject to
  attribution; for anything beyond the standard MIT terms, read `LICENSE` in the repo files
  before Phase 2 training starts. [ASSUMPTION]
- Max input length: **256 subword tokens** — "We set a maximum length at 256 subword tokens"
  (paper) and "Max length 256" in the v2 card table. [VERIFIED FACT]
- Tokenizer: BPE — "we then apply fastBPE to segment these sentences with subword units, using a
  vocabulary of 64K subword types" (paper). [VERIFIED FACT] The `transformers` slow tokenizer is
  merged into mainline transformers; a fast tokenizer is available from a separate branch per the
  v2 card. [VERIFIED FACT]
- Word segmentation requirement: **mandatory**. The PhoBERT paper: "Recall that PhoBERT performs
  Vietnamese word segmentation to segment syllable-level sentences into word tokens before
  applying BPE"; the v2 card states verbatim: "INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!" and
  recommends using the same segmenter as pre-training (RDRSegmenter from VnCoreNLP, installable
  via `pip install py_vncorenlp`). Full contract in [tokenization-design.md](tokenization-design.md).
- Sequence-classification compatibility: `AutoModelForSequenceClassification` resolves to the
  RoBERTa sequence-classification head for a `roberta`-type checkpoint — standard `transformers`
  behavior. [ASSUMPTION, not explicitly demonstrated on the card]
- Hub/format notes: the repo ships `pytorch_model.bin`, `tf_model.h5`, `flax_model.msgpack` and
  has **no safetensors file** (HF API). [VERIFIED FACT] Loading under recent `transformers` works
  from the bin weights, but `safe_serialization`-only tooling would need a conversion. [ASSUMPTION]
- Known limitations: external segmentation dependency (Java, latency, error propagation — see
  tokenization-design.md); 256-token ceiling; current card is thin (most operational facts must be
  cited from the paper or the v2 card).

### 3.2 `vinai/phobert-base-v2` — same family, more data, restrictive license

- Checkpoint: `vinai/phobert-base-v2` — https://huggingface.co/vinai/phobert-base-v2 [VERIFIED FACT]
- Architecture / size / max length: identical to PhoBERT-base (135M, RoBERTa-style, max 256).
  [VERIFIED FACT, card table]
- Pre-training data: "20GB of Wikipedia and News texts + 120GB of texts from OSCAR-2301".
  [VERIFIED FACT]
- License: **AGPL-3.0** — the card states "GNU Affero General Public License v3, Copyright (c)
  2023 VinAI Research". [VERIFIED FACT] AGPL is a strong copyleft license; fine-tuning and
  publishing a research artifact is unproblematic, but any future network-served product built on
  it inherits AGPL obligations. This matters for Phase 3 and is the main reason it is not the
  first candidate despite being the best-documented card in the family.
- Segmentation: same requirement as PhoBERT-base — verbatim "INPUT TEXT MUST BE ALREADY
  WORD-SEGMENTED!", RDRSegmenter/py_vncorenlp recommended. [VERIFIED FACT]
- Known limitations: AGPL-3.0; OSCAR-2301 is web-crawled data (quality/noise considerations).
  [VERIFIED FACT for provenance; noise impact — ESTIMATE]

### 3.3 `vinai/phobert-large` — recorded for completeness

- 370M parameters, max length 256, same 20GB corpus, same segmentation requirement (paper + v2
  card table). [VERIFIED FACT] Roughly 3x the training compute of base for marginal gains on short
  feedback sentences [ESTIMATE] — excluded from the first round on size, kept as a later option.

### 3.4 `VietAI/vit5-base` — Vietnamese alternative (encoder-decoder)

- Checkpoint: `VietAI/vit5-base` — https://huggingface.co/VietAI/vit5-base [VERIFIED FACT]
- Architecture: T5-style encoder-decoder for Vietnamese; card usage shows
  `AutoModelForSeq2SeqLM`. [VERIFIED FACT]
- Parameters: the ViT5 paper (arXiv:2205.06457, NAACL 2022 SRW) states: "we adapt the base (310M
  parameters) and large (866M parameters) models for ViT5" — i.e., ViT5-base is the T5 v1.1 base
  scale (~310M), **not** the 220M T5-base scale. [VERIFIED FACT]
- Pre-training data: Vietnamese subset of CC100 — "138GB of raw text", filtered to 69GB of short
  paragraphs plus 71GB of long paragraphs for the 1024-length variant. [VERIFIED FACT]
- License: `license: mit` on the card; the GitHub repo (https://github.com/vietai/ViT5) is MIT.
  [VERIFIED FACT]
- Tokenizer: SentencePiece, "We fixed the size of vocabulary to 36K sub-words"; the GitHub repo
  links a `spiece.model` vocabulary. [VERIFIED FACT]
- Word segmentation: **not required and not mentioned anywhere** — the paper's preprocessing is
  limited to "normalizing punctuation and capitalization, splitting numbers". [VERIFIED FACT]
- Max length: the paper trains variants at "256 and 1024-length"; the card does not state a
  serving max length. [VERIFIED FACT for the paper claim; per-checkpoint config — UNVERIFIED]
- Sequence-classification compatibility: classification would use the encoder with a
  classification head (`T5ForSequenceClassification` exists in `transformers`), not the
  `Seq2SeqLM` head shown on the card. [ASSUMPTION] This adds friction relative to the
  encoder-only candidates.
- Intended use per card/tags: summarization, translation, question answering — generation tasks.
  Sentiment classification is not the designed use. [VERIFIED FACT]
- Known limitations: encoder-decoder complexity, no direct published sentiment-classification
  evidence on the card, and no evidence of a documented word-segmentation contract either way.

### 3.5 `FPTAI/vibert-base-cased` (ViBERT) — exists but undocumented

- Checkpoint: `FPTAI/vibert-base-cased` — https://huggingface.co/FPTAI/vibert-base-cased
  [VERIFIED FACT: the checkpoint exists; HF page reports ~4.9k downloads/month and 12 fine-tuned
  derivatives, tag `bert`]
- Model card: **none** — the page itself says "No model card". No license tag, no parameter count,
  no corpus, no segmentation guidance. [VERIFIED FACT]
- Consequence: it is impossible to do license or preprocessing diligence from primary sources.
  Whether it needs pre-segmented input is [UNVERIFIED]. Community fine-tunes on Vietnamese
  text-classification tasks exist (visible in HF search), which suggests usability but proves
  nothing about licensing. [UNVERIFIED]

### 3.6 `FPTAI/velectra-base-discriminator-cased` (vELECTRA) — exists but undocumented

- Checkpoint: `FPTAI/velectra-base-discriminator-cased` —
  https://huggingface.co/FPTAI/velectra-base-discriminator-cased [VERIFIED FACT: exists; tag
  `electra`, "pretraining"; 16 fine-tuned derivatives; card absent]
- Model card: **none** ("No model card"), no license tag. [VERIFIED FACT]
- Note: a community derivative named `tinh0620/FPTAI_velectra-base-discriminator-cased-uit-vsfc`
  appears in HF search — evidence that practitioners fine-tune this checkpoint on UIT-VSFC —
  but the base model remains undocumented. [VERIFIED FACT for the derivative's existence]
- Same consequence as ViBERT: excluded until license and segmentation requirements can be
  verified from a primary source.

### 3.7 BamiBERT — mentioned on the PhoBERT card, not findable on the Hub

- The current PhoBERT card recommends BamiBERT as a newer alternative trained on a 129GB corpus,
  supporting 2,048-token context and operating directly on raw input "eliminating the need for
  external word segmentation". [VERIFIED FACT — quote from the phobert-base card]
- However, an HF Hub API search for `bambert` returns no models. [VERIFIED FACT — empty result,
  2026-11-07] No usable checkpoint could be identified, so it is out of the shortlist.
  Availability: [UNVERIFIED beyond the empty search].

### 3.8 `xlm-roberta-base` — multilingual comparison baseline

- Checkpoint: `xlm-roberta-base`, canonical redirect to `FacebookAI/xlm-roberta-base` —
  https://huggingface.co/xlm-roberta-base [VERIFIED FACT]
- Architecture: XLM-RoBERTa, multilingual RoBERTa, MLM pretraining. [VERIFIED FACT]
- Parameters: 278,885,778 (safetensors metadata via HF API). [VERIFIED FACT] The original paper
  cited "250M (XLM-Rbase)" (also quoted in the PhoBERT paper); the difference is parameter
  counting, not a different model. [VERIFIED FACT for both statements]
- Pre-training data: 2.5TB filtered CommonCrawl (CC-100), 100 languages. [VERIFIED FACT]
- License: metadata tag `license: mit`. [VERIFIED FACT]
- Tokenizer: SentencePiece-based in `transformers` (requires the `sentencepiece` package).
  [ASSUMPTION — standard implementation detail; the card itself does not name the tokenizer]
- Word segmentation: not required — the card targets raw-text "sequence classification, token
  classification or question answering". [VERIFIED FACT] This is exactly what makes it a useful
  ablation: it isolates the value of (a) Vietnamese-specialized pretraining and (b) the
  segmentation pipeline.
- Max length: card does not state it; XLM-R follows the RoBERTa 512-token convention.
  [ASSUMPTION]
- Known limitations: not Vietnamese-specialized; the PhoBERT paper reports PhoBERT outperforming
  XLM-R on Vietnamese tasks "using far fewer parameters" ("135M (PhoBERTbase) vs. 250M
  (XLM-Rbase)"). [VERIFIED FACT quote]

## 4. Cross-cutting dependency concerns

- `transformers` version: all candidates load via `AutoTokenizer`/`AutoModel*` with current
  transformers (the PhoBERT cards say `pip install -U transformers accelerate`). [VERIFIED FACT]
  PhoBERT's slow tokenizer is in mainline transformers; its fast tokenizer comes from a separate
  branch per the v2 card — plan on the slow tokenizer. [VERIFIED FACT]
- `sentencepiece` is required for ViT5 and (by standard implementation) XLM-R tokenizers.
  [ASSUMPTION for XLM-R]
- `py_vncorenlp` (0.1.4 on PyPI) is required for the PhoBERT preprocessing contract and needs a
  **Java 1.8+ runtime**; see tokenization-design.md for deployment implications and the
  underthesea alternative. [VERIFIED FACT]
- safetensors: `vinai/phobert-base` has no safetensors weights on the Hub; XLM-R does.
  [VERIFIED FACT] Keep `from_tf`/bin-loading compatibility in the Phase 2 loader design.

## 5. Open questions carried into Phase 2 planning

1. Exact hardware envelope for fine-tuning (the Phase 1 CI is CPU-only; base-size fine-tuning on
   CPU is possible but slow). VRAM estimates are collected in
   [model-selection-decision.md](model-selection-decision.md), all labeled [ESTIMATE].
2. Whether `phobert-base-v2`'s extra 120GB of web data actually helps UIT-VSFC-style feedback —
   unknown without an experiment; the AGPL license already demotes it to an optional arm.
3. UIT-VSFC published F1 for PhoBERT-based systems: secondary literature suggests roughly
   89–94% F1 across various PhoBERT-based models, but none of these numbers was verified from a
   primary source for this note. [UNVERIFIED] The PhoBERT paper itself does **not** report
   UIT-VSFC (its tasks are POS tagging, dependency parsing, NER, and NLI). [VERIFIED FACT]
