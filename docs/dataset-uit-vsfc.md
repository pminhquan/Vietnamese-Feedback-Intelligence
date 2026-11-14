# Dataset Investigation: UIT-VSFC (Vietnamese Students' Feedback Corpus)

Investigation date: 2026-11-07. Investigator: ZCode agent, on behalf of the
repository owner. Verdict first, evidence below.

## Verdict: BLOCKED — do not download yet

The license could not be verified from any authoritative source:

- The Hugging Face mirror tags the dataset **`license: unknown`**, and its
  loading script explicitly sets `_LICENSE = Licenses.UNKNOWN.value`.
- The **original author's distribution page** requires contacting the author
  and **signing a corpus user agreement** before receiving the data.

Per this repository's data-safety rules, we do not silently download data
whose usage terms are unverified. All Phase 1 code is complete and tested on
synthetic fixtures; real experiments start once the owner obtains permission
through the official channel.

## Candidate dataset facts (verified from sources, not invented)

| Field | Value |
| ----- | ----- |
| Name | UIT-VSFC — Vietnamese Students' Feedback Corpus (version 1.0) |
| Paper | Kiet Van Nguyen, Vu Duc Nguyen, Phu Xuan-Vinh Nguyen, Tham Thi-Hong Truong, Ngan Luu-Thuy Nguyen, "UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis", 2018 10th International Conference on Knowledge and Systems Engineering (KSE 2018), Nov 1-3 2018, Ho Chi Minh City, Vietnam (IEEE, pp. 19-24) |
| Language | Vietnamese (`vie`) |
| Content | 16,000+ student feedback sentences from end-of-semester surveys at a Vietnamese university (UIT) |
| Tasks | Sentiment (3 classes) and topic (4 classes) annotation |
| Sentiment label schema | `positive`, `negative`, `neutral` |
| Topic label schema | `lecturer`, `training_program`, `others`, `facility` |
| Raw file format | Line-aligned plain text: `sentences.txt` + `sentiments.txt` (+ `topics.txt`) per split |
| Splits | Official `train` / `validation` (dev) / `test` (three Google Drive file pairs) |
| Raw label encoding | Integer per line; the SEACrowd loader declares `ClassLabel(names=["positive","negative","neutral"])`, i.e. `0=positive, 1=negative, 2=neutral` |
| Quality signals (author-reported) | Inter-annotator agreement > 91% (sentiment), > 71% (topic); Maximum Entropy baseline ≈ 88% sentiment F1 |
| HF mirror | `SEACrowd/uit_vsfc` (loading script only; data fetched from the original Google Drive) |

**Caveat on the label mapping:** the `0=positive, 1=negative, 2=neutral`
order is taken from the SEACrowd loader's `ClassLabel` declaration. When the
owner obtains the corpus, verify a handful of lines against the author's
README before trusting the mapping (the loading code and
`docs/learning-notes.md` both call this out).

## Sources checked (2026-11-07)

1. `https://huggingface.co/datasets/SEACrowd/uit_vsfc` — dataset card:
   license field `Unknown (unknown)`; `tags: [license:unknown, language:vie, ...]`;
   dataset viewer unavailable (requires the `seacrowd` package).
2. `https://huggingface.co/datasets/SEACrowd/uit_vsfc/raw/main/uit_vsfc.py` —
   the loader script: `_LICENSE = Licenses.UNKNOWN.value`; `_HOMEPAGE` points
   to the author's Google Drive folder; direct-download URLs for the six
   data files; split generators for train/test/validation; label classes as
   listed above.
3. `https://github.com/kietnv/VietnameseDatasets` — the dataset author's own
   repository (Mr. Kiet Nguyen, UIT). The UIT-VSFC section states, verbatim:
   > "Please contact us via email: kietnv@uit.edu.vn (Mr. Kiet Nguyen) to
   > sign the corpus user agreement and then receive the corpus."
   (The repository's MIT license covers the repo itself, not the corpora.)
4. The original UIT NLP group homepage (`sites.google.com/uit.edu.vn/kiemnn-nlp`)
   requires a Google login to view; terms could not be read anonymously.

## Privacy / sensitivity review

The corpus contains student opinions about named courses, facilities, and
lecturers. Even with permission to use the data:

- Raw text must never be committed to this repository (enforced by
  `.gitignore`).
- Error-analysis write-ups must quote at most short, truncated fragments and
  prefer paraphrase; never publish full sentences with personal references.
- Reported metrics must be aggregates only.

## How to unblock (owner action)

1. Email `kietnv@uit.edu.vn` (Mr. Kiet Nguyen), introduce yourself as a
   student doing a non-commercial educational/research project, and request
   the corpus user agreement for UIT-VSFC v1.0.
2. Sign and return the agreement; receive the download link from the author.
3. Place the files locally (never commit them) following `data/README.md`.
4. Verify the raw label encoding against the author's README, then run:
   `vfi prepare-data` → `vfi evaluate --preflight-only` → `vfi train` →
   `vfi evaluate --model-id ...` per
   `docs/experiment-protocol.md` and record the real numbers in `reports/`.

Until then, no experiment numbers are produced or claimed in this repo.
