# Phase 2 Tokenization & Preprocessing Design

Status: research note (Phase 2 readiness only). Phase 2 fine-tuning is **not** authorized yet.
Facts about the Phase 1 pipeline below were read from the Phase 1 source at commit
`95f7687735c785367fa0a6f9b1efbf0e284aac7f` (`git show`, read-only). Model-side facts are labeled
per the evidence discipline; see [model-landscape.md](model-landscape.md) for the model cards.

Evidence labels: **[VERIFIED FACT]**, **[ESTIMATE]**, **[ASSUMPTION]**, **[UNVERIFIED]**.

## 1. Where Phase 1 stands (verified from source)

The Phase 1 classical pipeline processes **raw whitespace text**:

- `src/vfi/data/loading.py` — `normalize_text()` applies Unicode **NFC composition** and collapses
  whitespace runs to single spaces; deliberately no lowercasing, no diacritic stripping, no
  stopword removal, because these carry Vietnamese sentiment signal. [VERIFIED FACT]
- `src/vfi/models/baseline.py` — the TF-IDF + Logistic Regression pipeline tokenizes inside
  `TfidfVectorizer` (`analyzer` word or `char_wb`, `ngram_range` (1,2) or (2,5), `lowercase=True`,
  `min_df=2`, `max_df=0.95`, `sublinear_tf`). There is no separate tokenization stage: scikit-learn
  splits on whitespace/punctuation at fit/transform time. [VERIFIED FACT]
- The whole model is one sklearn `Pipeline`, so training-time and inference-time preprocessing
  cannot drift by construction. [VERIFIED FACT]
- Labels: canonical space `negative=0, neutral=1, positive=2`; raw UIT-VSFC integers
  (`0=positive, 1=negative, 2=neutral`) are remapped once at load time. [VERIFIED FACT]

So Phase 1 "tokenization" = whitespace/punctuation n-gram splitting on NFC-normalized raw text —
**no word segmentation anywhere in Phase 1**.

## 2. What each Phase 2 candidate needs

| Model | Input form | Tokenizer | Segmentation before tokenizer? | Max length |
|-------|-----------|-----------|-------------------------------|------------|
| `vinai/phobert-base` / `-v2` | **Pre-segmented**, underscore-joined syllables | BPE (fastBPE, 64K subword types) | **Yes — mandatory** | 256 |
| `vinai/phobert-large` | Same as phobert-base | Same | Yes — mandatory | 256 |
| `VietAI/vit5-base` | Raw text | SentencePiece, 36K vocab | No (not mentioned anywhere in paper/repo) | 256 or 1024 per training variant (card silent) |
| `xlm-roberta-base` | Raw text | SentencePiece [ASSUMPTION] | No | 512 [ASSUMPTION — RoBERTa convention] |
| `FPTAI/vibert-base-cased`, `FPTAI/velectra-base-discriminator-cased` | Unknown | Unknown | **Unknown** — cards absent | Unknown |

Sources: PhoBERT max length 256 and fastBPE/64K vocab and the segmentation-before-BPE requirement
are quoted in [model-landscape.md](model-landscape.md) (paper + phobert-base-v2 card). [VERIFIED FACT]
ViT5 SentencePiece/36K and "no segmentation mentioned" from the ViT5 paper and repo. [VERIFIED FACT]
Everything about the two FPT.AI checkpoints is [UNVERIFIED] — their cards are empty.

## 3. The exact PhoBERT preprocessing contract

The contract, in order, for every text (train, validation, test, and later production input):

1. **NFC normalization** — reuse `vfi.data.loading.normalize_text` (NFC + whitespace collapse).
   [ASSUMPTION: applying NFC before segmentation is the right order; it matches what the segmenter
   reasonably expects and matches the Phase 1 entry point. The PhoBERT card does not explicitly
   discuss Unicode normalization — UNVERIFIED either way.]
2. **Vietnamese word segmentation BEFORE the BPE tokenizer.** This is not optional: the PhoBERT
   paper states PhoBERT "performs Vietnamese word segmentation to segment syllable-level sentences
   into word tokens before applying BPE", and the v2 model card states verbatim: "INPUT TEXT MUST
   BE ALREADY WORD-SEGMENTED!" Segmentation output must be underscore-joined multi-syllable words,
   e.g. `Chúng_tôi là những nghiên_cứu_viên.` [VERIFIED FACT — card/paper quotes]
3. **BPE tokenization** via `AutoTokenizer.from_pretrained("vinai/phobert-base")` on the segmented
   string. The underscores are part of the surface form the BPE vocabulary was trained on, so they
   must survive into the tokenizer input. [VERIFIED FACT for the requirement; exact BPE behavior on
   underscores — VERIFIED FACT via the card's stated contract, implementation detail ASSUMPTION]
4. **Truncation/padding to the model maximum of 256 subword tokens.** [VERIFIED FACT]

Two hard rules fall out of this contract:

- **Segmentation happens before the HF tokenizer, once per text, always with the same tool and
  version** for every split and every deployment.
- **The Phase 1 `normalize_text` is reused, not bypassed** — the Phase 2 input chain starts exactly
  where the Phase 1 chain starts, which keeps the two phases comparable and keeps NFC handling in
  one audited place.

## 4. Segmentation tools: what exists, licenses, caveats

| Tool | Output for segmentation | License | Runtime caveat |
|------|------------------------|---------|----------------|
| **VnCoreNLP RDRsegmenter** (via `py_vncorenlp`) | `rdrsegmenter.word_segment(text)` returns underscore-joined text, e.g. `'Ông Nguyễn_Khắc_Chúc đang làm_việc tại Đại_học Quốc_gia Hà_Nội .'`; word-segmentation-only mode via `annotators=["wseg"]` | **GPL v3** — the repo's `LICENSE.md` is a GPLv3 notice ("you can redistribute it and/or modify it under the terms of the GNU General Public License, version 3"); the GitHub API reports the license as `NOASSERTION`/"Other" because it is a custom file | Requires **Java 1.8+** (JRE/JDK); ships a 27MB jar + 115MB models folder; models downloaded with `py_vncorenlp.download_model()`; JVM startup adds seconds-scale latency — segment in batches |
| **underthesea** `word_tokenize` | Default returns a **list** of words (`["Chàng trai", "9X", "Quảng Trị", ...]`); `word_tokenize(text, format="text")` returns **underscore-joined text**, e.g. `"Bác_sĩ bây_giờ có_thể thản_nhiên báo_tin bệnh_nhân bị ung_thư"` | **Apache-2.0** (GitHub API; PyPI `license_expression: Apache-2.0`), current release 9.5.0, requires Python >= 3.10 | Pure Python — no Java; simpler to deploy, but it is a **different segmenter** from the one used in PhoBERT pre-training |

Tool facts are [VERIFIED FACT] from: https://github.com/vncorenlp/VnCoreNLP,
https://pypi.org/pypi/py_vncorenlp/json (py_vncorenlp 0.1.4, "A Python Wrapper for VnCoreNLP",
Java 1.8+ prerequisite, `annotators=["wseg"]` example),
https://github.com/undertheseanlp/underthesea, https://pypi.org/pypi/underthesea/json, and
https://underthesea.readthedocs.io/en/latest/api.html (word_tokenize signature
`word_tokenize(sentence, format=None, use_token_normalize=True, fixed_words=[])`).

**Fidelity argument.** The PhoBERT v2 card explicitly recommends using "the same word segmenter"
as during pre-training — RDRSegmenter from VnCoreNLP (via `pip install py_vncorenlp`). [VERIFIED
FACT] That makes `py_vncorenlp` the fidelity-first default. The GPL v3 license and the Java
dependency are acceptable for a research/portfolio artifact (nothing is being distributed as a
proprietary product in Phase 2), but both must be re-examined if Phase 3 ever ships a served
product.

**Ops argument.** `underthesea` is Apache-2.0 and pure Python — the safer dependency for a serving
path — but it is not the segmenter PhoBERT was pre-trained with, so switching tools between train
and serve is exactly the skew risk described below. Pick one tool per model, record it in the
artifact metadata, and never mix.

## 5. Where segmentation fits in a sklearn -> torch pipeline

```text
raw text (UIT-VSFC sentences.txt)
   |
   v
[shared] vfi.data.loading.normalize_text        NFC + whitespace collapse   (Phase 1 code, reused)
   |
   v
[Phase 2 only] word segmenter                   py_vncorenlp wseg (default) or underthesea
   |                                            deterministic function: str -> str
   v
[Phase 2 only] HF tokenizer                     AutoTokenizer, truncation to 256
   |
   v
torch Dataset / DataLoader                      labels via canonical mapping (negative=0, neutral=1, positive=2)
```

Design points:

- **Segmentation is a pure, deterministic `str -> str` function** — it sits between Phase 1
  normalization and the HF tokenizer and must be applied identically to every split. Keep the
  sklearn `Pipeline` object untouched; the segmentation step is *not* inside it. The Phase 2
  equivalent of "pipeline can't drift" is: **persist segmented texts or the segmenter identity
  (tool + version + model dir fingerprint) in the artifact sidecar** (`meta.json` equivalent),
  so any prediction run can prove it used the same preprocessing. [DESIGN RECOMMENDATION]
- **Segment once, cache.** For UIT-VSFC-scale data (~16k short sentences per split [ESTIMATE —
  count UNVERIFIED]), segmenting the whole split up front is cheap; JVM startup cost is amortized
  by batch segmentation.
- **Label mapping stays exactly as Phase 1** — canonical `negative=0, neutral=1, positive=2`,
  remapped once at load time. The HF classifier head is trained against the canonical ids.
  [VERIFIED FACT for Phase 1; direct reuse — DESIGN RECOMMENDATION]

## 6. Risk register

| Risk | Description | Mitigation |
|------|-------------|------------|
| Segmentation error propagation | Wrong joins (e.g., "không tốt" -> "không_tốt") change token semantics and cannot be fixed downstream; errors hit the same words BPE was trained on | Use the segmenter PhoBERT was trained with (RDRsegmenter); spot-check a fixed sample of segmented sentences in CI with synthetic fixtures; report segmentation agreement between tools on a small audit set |
| Train/serve skew | Different segmenter (or version) at inference than at training -> silent quality drop; the model has no way to warn | Record tool name, version, and model fingerprint in artifact metadata; a prediction run loads/validates that record; a CI test asserts identical segmentation on golden sentences across the recorded version |
| Max-length truncation | PhoBERT caps at 256 subword tokens [VERIFIED FACT]; long multi-sentence feedback gets truncated — sentiment cues near the end are lost | Measure the token-length distribution of the real train split before finalizing (truncation side: `longest_first` default); if a meaningful tail exceeds 256, document it as a known limitation rather than silently truncating [ASSUMPTION: UIT-VSFC feedback is mostly short — ESTIMATE, unverified] |
| Segmenter output format mistakes | `underthesea.word_tokenize` defaults to a **list**; feeding a list's `str()` representation into the tokenizer corrupts text | Contract: underscore-joined string only — use `format="text"` (underthesea) or `rdrsegmenter.word_segment` (py_vncorenlp); unit-test the exact join behavior |
| Non-lexical content (URLs, emojis, teencode) | Segmenters may split or mangle URLs/emojis; teencode/spelling variants are a Vietnamese feedback reality | Do not strip anything in preprocessing (Phase 1 principle); let BPE handle it; include representative fixtures in tests; error analysis on validation |
| Java dependency | `py_vncorenlp` needs Java 1.8+; missing JVM = hard failure at train or predict time | Pin in project docs; check availability early in the Phase 2 environment bootstrap; the underthesea path is the documented fallback |
| Licensing | VnCoreNLP is GPL v3 [VERIFIED FACT] | Fine for internal research use; revisit before any Phase 3 distribution/service |
| Unicode mismatch | Text not NFC-normalized before segmentation may segment inconsistently | NFC is step 1 of the chain (reuse `normalize_text`); idempotence unit test |

## 7. Recommended default preprocessing chain (Phase 2, PhoBERT family)

```text
normalize_text (NFC + collapse, from vfi.data.loading)
  -> py_vncorenlp RDRsegmenter word_segment (wseg-only annotators, batch mode)
  -> whitespace re-collapse (defensive, after segmentation)
  -> AutoTokenizer("vinai/phobert-base"), truncation to max 256, padding
  -> canonical label ids (negative=0, neutral=1, positive=2)
```

Rationale: (1) NFC first matches Phase 1 and is idempotent-safe; (2) RDRsegmenter is the segmenter
the PhoBERT cards recommend for PhoBERT-family input [VERIFIED FACT]; (3) the defensive
re-collapse guards against segmenters emitting double spaces; (4) the HF tokenizer owns
subword/truncation logic; (5) the canonical label space is inherited unchanged from Phase 1.

For the XLM-R comparison arm, the chain is simply `normalize_text -> AutoTokenizer` — no
segmentation step — which is precisely the ablation the comparison is for. [DESIGN RECOMMENDATION]

## 8. Tests to write when Phase 2 starts (not before)

- `normalize_text` idempotence and NFC composition fixtures (extends existing Phase 1 tests).
- Segmenter golden-file test: N fixed sentences -> exact expected underscore-joined output for the
  pinned tool version; fails loudly on version change.
- Format contract test: `underthesea` path asserts `format="text"` produces a `str`, not a list.
- Truncation boundary test: sentence at exactly 255/256/257 PhoBERT tokens behaves as specified.
- Round-trip test: tokenize -> decode -> re-segment leaves the segmented string unchanged.
- End-to-end determinism: same seed + same inputs -> identical token ids across two runs.
