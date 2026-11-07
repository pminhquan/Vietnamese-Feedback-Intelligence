# Label Semantics Evidence — UIT-VSFC Raw Integer Encoding

Investigation date: 2026-11-07 (same-day hardening pass re-verifying the 2026-11-07 morning
findings and adding new primary artifacts). Prepared by: Hardening Agent A (Dataset & Label
Audit), Phase 2 readiness program.

Companion documents:

- Recommendation and owner actions: [dataset-acquisition-plan.md](dataset-acquisition-plan.md)
- License/access comparison: [dataset-license-matrix.md](dataset-license-matrix.md)
- Per-candidate deep dives: [dataset-landscape.md](dataset-landscape.md)
- The audit procedure this evidence feeds into: [experiment-blueprint.md](experiment-blueprint.md) §3

## Verdict (read this first)

**The raw integer encoding of the author-distributed `{split}/sentiments.txt` files remains
UNRESOLVED.** No artifact fetched today — the author's GitHub README, the KSE 2018 paper, or
either Hugging Face loader — documents it explicitly. What this pass established beyond the
morning findings:

1. **The two loader scripts consume the identical raw files.** All nine Google Drive file IDs
   in their `_URLS` blocks match character-for-character, and both pass raw integers through
   unmodified (`int(sentiment.strip())`). Their `ClassLabel` declarations are contradictory —
   so exactly one of the two mirrors mislabels every row.
2. **The authors' own HF org is internally consistent across three independent artifacts**
   (card "Data Fields" prose, the datasets-library loader script embedded in their repo, and
   the parquet features their upload serves): all three say 0=negative, 1=neutral, 2=positive.
3. **SEACrowd's label orders are an unverifiable editorial choice.** Their loader reorders
   *both* label spaces (sentiment and topic) relative to the datasets-library script, and the
   PR/issue trail contains no label-verification discussion whatsoever.

The evidential balance therefore favors **0 = negative** (the authors' own layer) as a
*planning prior*, but the chain of inference from card prose to raw files is not documented by
the authors, so the decision gate is unchanged: **author confirmation + blind audit of the
delivered files before anything trains** (G2 in `00-executive-decision.md`). Neither mirror is
trusted for training in the interim.

## Evidence legend

- **[VERIFIED FACT]** — read directly from the cited primary source on 2026-11-07 (exact URL
  and fetch method given in the Sources table at the end).
- **[ASSUMPTION]** — reasonable inference, not directly evidenced.
- **[UNVERIFIED]** — plausible claim not confirmed from a primary source this session.

---

## 1. The two conflicting sources, re-fetched 2026-11-07

Both sources were re-fetched in this session and are unchanged from the morning findings.

### 1.1 Source A — SEACrowd loader (implies 0=positive)

URL: https://huggingface.co/datasets/SEACrowd/uit_vsfc/raw/main/uit_vsfc.py — fetched
2026-11-07. **[VERIFIED FACT]**

```python
SENTIMENT_LABEL_CLASSES = ["positive", "negative", "neutral"]
...
"sentiment": datasets.ClassLabel(names=self.SENTIMENT_LABEL_CLASSES),
...
_LICENSE = Licenses.UNKNOWN.value
...
"sentiment": int(sentiment.strip()),
```

Under `datasets` semantics this implies raw `0` → `"positive"`, `1` → `"negative"`,
`2` → `"neutral"`. The script contains **no comment, assertion, or test about label
verification**. **[VERIFIED FACT — absence]**

### 1.2 Source B — the authors' own dataset card (states 0=negative)

URL: https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback — fetched 2026-11-07.
The card is published by "The UIT Natural Language Processing Group" (the authors' own HF
organization). **[VERIFIED FACT]**

> ### Data Fields
>
> - `sentence` (str): Text sentence.
> - `sentiment`: Sentiment class, with values 0 (negative), 1 (neutral) and 2 (positive).
> - `topic`: Topic class, with values 0 (lecturer), 1 (training_program), 2 (facility) and 3
>   (others).

Licensing section: "### Licensing Information — Unknown." **[VERIFIED FACT]**

**New corroborating evidence from the same page:** the auto-converted Dataset Viewer preview
renders the actual served data with the class-label metadata baked into the parquet files.
Two paraphrased examples (paraphrased per the project's no-raw-text policy; the originals are
publicly visible on the card):

- a sentence expressing satisfaction with everything renders as **2positive**;
- a sentence calling a lecturer's teaching boring renders as **0negative**.

So the *served data itself* — not just the prose — carries the 0=negative, 1=neutral,
2=positive mapping. **[VERIFIED FACT for the metadata mapping; the valence readings are our
interpretation of paraphrased content]**

### 1.3 New artifact — the datasets-library loader script inside the authors' repo

The `uitnlp` repo contains `vietnamese_students_feedback.py` (the script of the former
canonical `datasets`-library dataset; card Contributions note: "Thanks to @albertvillanova for
adding this dataset"). URL:
https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/raw/main/vietnamese_students_feedback.py
— fetched 2026-11-07. **[VERIFIED FACT]**

```python
"sentiment": datasets.ClassLabel(names=["negative", "neutral", "positive"]),
"topic": datasets.ClassLabel(names=["lecturer", "training_program", "facility", "others"]),
...
yield key, {
    "sentence": sentence.strip(),
    "sentiment": int(sentiment.strip()),
    "topic": int(topic.strip()),
}
```

Three observations. First, this declaration matches the card prose and the served parquet
features — three independent artifacts, one consistent mapping. Second, the script downloads
the **raw author Google Drive files** (its `_URLS` lists the same nine Drive file IDs the
SEACrowd script uses) and forwards the raw integers unmodified into the `ClassLabel` — meaning
its parquet output preserves whatever integers the raw files contain. Third, the script header
reads "Copyright 2020 The HuggingFace Datasets Authors and the current dataset script
contributor" — it is a third-party-maintained *loader*, not author documentation. **[VERIFIED
FACT for the text; see §4 for what this does and does not prove]**

---

## 2. The authors' own documentation — checked for the raw encoding

### 2.1 Author GitHub README: no encoding documented

URL: https://raw.githubusercontent.com/kietnv/VietnameseDatasets/master/README.md — fetched
2026-11-07. **[VERIFIED FACT]**

- The UIT-VSFC section describes the corpus ("consists of over 16,000 sentences which are
  human-annotated with two different tasks: sentiment-based and topic-based classifications")
  and gives the access instruction, verbatim: *"Please contact us via email:
  kietnv@uit.edu.vn (Mr. Kiet Nguyen) to sign the corpus user agreement and then receive the
  corpus."*
- **The README documents no integer label encoding anywhere — for UIT-VSFC or for any other
  dataset in the repository.** No mention of `sentiments.txt`, no 0/1/2 mapping, no file-format
  description. **[VERIFIED FACT — absence]**

### 2.2 The paper (KSE 2018): closed access; abstract silent

- DOI 10.1109/KSE.2018.8573337 ("UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment
  Analysis", KSE 2018, IEEE, pp. 19-24). Semantic Scholar API reports `openAccessPdf.status =
  "CLOSED"` with an empty URL — no open PDF exists there. The IEEE Xplore landing page
  (https://ieeexplore.ieee.org/document/8573337) returned no content to automated fetch
  (blocked). An arXiv API search for "UIT-VSFC" returned zero results — no arXiv version.
  **[VERIFIED FACT]**
- The abstract (retrieved via the Semantic Scholar API, 2026-11-07) describes corpus
  construction, annotation guidelines, IAA (91.20% sentiment, 71.07% topic), and MaxEnt
  baselines (87.94% / 84.03% F1). It contains **no statement about integer label encoding,
  label order, or file formats**. **[VERIFIED FACT for the abstract; the paywalled full text
  could not be examined this session]**
- Bottom line: we cannot cite the paper for any raw-file encoding; nothing in the open record
  suggests it contains a label-encoding table. **[UNVERIFIED whether the full text does]**

### 2.3 Author homepage

The homepage deep-link recorded in the loader scripts
(https://sites.google.com/uit.edu.vn/uit-nlp/datasets-projects) returns HTTP 404 as of
2026-11-07; the site root (https://sites.google.com/uit.edu.vn/uit-nlp) renders dynamically
and could not be checked for label documentation. **[VERIFIED FACT for the 404; UNVERIFIED
whether current site content documents encoding]**

---

## 3. SEACrowd provenance: was the mapping ever verified?

Searched `github.com/SEACrowd/seacrowd-datahub` for `uit_vsfc` (GitHub issues search API,
2026-11-07). Two items exist:

- **Issue #208** — "Create dataset loader for Vietnamese Students' Feedback Corpus (UIT-VSFC)",
  closed, 9 comments. All comments concern a technical naming collision (two tasks sharing one
  schema) and its resolution via config names; **none discuss label encoding or verification**.
  **[VERIFIED FACT — absence]**
- **PR #246** — "Closes #208 | Implement dataloaders for UIT_VSFC", merged 2024-01-16, 2
  comments (a revision ping and an approval). **None discuss label encoding or verification.**
  **[VERIFIED FACT — absence]**

The PR diff (fetched 2026-11-07) adds both label constants **already diverged from the
datasets-library script**:

```diff
+    SENTIMENT_LABEL_CLASSES = ["positive", "negative", "neutral"]
+    TOPIC_LABEL_CLASSES = ["lecturer", "training_program", "others", "facility"]
```

The datasets-library script declares `["negative", "neutral", "positive"]` (sentiment) and
`["lecturer", "training_program", "facility", "others"]` (topic). SEACrowd reordered **both**
— and as of the current master script (fetched 2026-11-07), the topic reorder was never
reverted, while the sentiment reorder matches no known primary artifact. **[VERIFIED FACT]**

Conclusion: there is no evidence — none in the issue, none in the PR, none in the script —
that anyone at SEACrowd checked the raw `sentiments.txt` integers against the declared names.
The reordering is an unverifiable editorial choice. **[VERIFIED FACT for the trail; ASSUMPTION
that no off-GitHub verification occurred]**

---

## 4. Does a `ClassLabel` declaration reflect the raw-file encoding?

**No — not necessarily. A `ClassLabel` declaration is a claim by the loader author, not a fact
about the data.** Mechanics: in Hugging Face `datasets`, `ClassLabel(names=[...])` maps integer
`i` to `names[i]`. The names order is chosen by whoever writes the loader. A loader that reads
a raw file via `int(x.strip())` and wraps it in `ClassLabel(names=[...])` *asserts* — but does
not *verify* — that raw integer `i` means `names[i]`. **[VERIFIED FACT for the library
semantics; well-documented `datasets` behavior]**

Applying this to the two mirrors:

1. **Both consume the same raw files.** The nine Google Drive file IDs in `_URLS` are identical
   between the SEACrowd script and the datasets-library script — including the same six
   sentence/topic IDs and the same three sentiments IDs (the SEACrowd copy even repairs a
   whitespace typo, `"export=down load"`, present in the datasets-library script's train
   sentiments URL — itself evidence that SEACrowd's script was derived from the latter).
   **[VERIFIED FACT]**
2. **Both pass integers through unmodified.** Neither script remaps values; the only mapping
   between raw integers and names is the declaration. **[VERIFIED FACT]**
3. **Therefore the two declarations cannot both be correct.** Given identical inputs and
   identity int handling, if the raw files encode 0=negative then SEACrowd's loader permutes
   every row (0→"positive", 1→"negative", 2→"neutral"); if the raw files encode 0=positive
   then the authors' own card, script, and served data are all wrong in the same way. Exactly
   one mirror mislabels systematically. **[VERIFIED FACT as a logical consequence of 1+2;
   the raw truth itself is UNRESOLVED]**
4. **One caveat:** both scripts were written years apart (datasets-library script ~2020-2022;
   SEACrowd merged 2024-01-16) against a Drive folder neither of us can inspect without
   downloading (out of scope). If the raw files had changed in between, both could be "right
   at the time". Nothing in either card notes a data change. **[ASSUMPTION that the raw files
   are unchanged]**

**What would count as authoritative evidence for layer (c)** — none of it exists publicly
today:

- an author statement mapping integers to sentiments for the distributed files;
- documentation shipped inside the corpus package (README/agreement material);
- a blind audit of the delivered files ([experiment-blueprint.md](experiment-blueprint.md) §3).

---

## 5. The three layers, kept separate

| Layer | Artifact(s) | Sentiment integer claim | Evidential status |
|---|---|---|---|
| **(a) VFI internal canonical label space** | VFI Phase 1/2 configs (arbitrary but fixed project convention) | negative=0, neutral=1, positive=2 | A **definition**, not a claim — nothing external to verify |
| **(b) Mirror/repackager declarations** | SEACrowd loader (`uit_vsfc.py`) | 0=positive, 1=negative, 2=neutral | Third-party re-packager; reorders both label spaces vs the datasets-library script; zero verification trail. **[VERIFIED FACT as their declaration; UNTRUSTED as a reading of the raw files]** |
| **(b) Authors' own HF upload** | `uitnlp` card prose + embedded datasets-library script + served parquet features | 0=negative, 1=neutral, 2=positive | Three mutually consistent artifacts under the authors' own org. Still a loader-derived reading, not explicit author documentation of the raw files. **[VERIFIED FACT as their declaration; best available proxy]** |
| **(c) Original corpus raw encoding** (`{split}/sentiments.txt` as distributed) | — none fetched — | **UNKNOWN** | Documented nowhere: not in the author README, not in the open paper record, not in either loader. **UNRESOLVED** |

Phase 1's loader hard-codes `RAW_SENTIMENT_TO_LABEL = {0: "positive", 1: "negative",
2: "neutral"}` — a layer-(b)-SEACrowd inference, already flagged
[UNVERIFIED AGAINST AUTHOR SOURCE] in `00-executive-decision.md`. Layer (a) is unaffected by
this conflict: whatever the raw files say, VFI's canonical space is fixed by definition, and
the raw-to-canonical mapping is exactly what G2 must pin down.

---

## 6. Status and weighting argument

**Status: UNRESOLVED.** Argued honestly, the two sources are *not* symmetric:

- **Why the authors' own layer outweighs SEACrowd.** (i) Provenance: the `uitnlp` artifacts
  live in the authors' own organization; SEACrowd is a regional re-packager of third-party
  corpora. (ii) Consistency: the authors' layer is coherent across three independent artifacts
  (prose, script, served data); SEACrowd's single script contradicts all three. (iii)
  Process: SEACrowd's own PR trail shows both label orders were changed relative to the
  datasets-library script with no verification discussion; the authors' artifacts at least
  claim the mapping in their own voice. If a planning decision had to be made today with no
  author response, the rational prior is 0=negative, 1=neutral, 2=positive.
- **Why the conclusion is still "resolve with the author + blind audit".** (i) Even the
  authors' card describes the HF *feature* encoding, not the raw `sentiments.txt` files as
  such; the chain "HF upload was generated by this script from the official raw files" is
  [ASSUMPTION] — undocumented by the authors. (ii) The cost of settling it is one email answer
  plus a 135-line audit, trivially cheap against the cost of silently inverting a label space
  across a research project. (iii) **An inverted mapping is invisible in headline metrics**
  (see below), so "it trained fine" would prove nothing — only documentation or the audit
  catches it.

**Blast-radius note (conditional reasoning, not a result).** If the raw encoding turns out to
be 0=negative (H1 below) and Phase 1's SEACrowd-derived mapping were applied to the raw files,
the error would be a *consistent cyclic permutation* (raw 0 read as "positive", 1 as
"negative", 2 as "neutral") applied identically to every split. Consistent permutations
preserve the data partition, so overall accuracy and macro-F1 are invariant — but every
class-specific claim (per-class F1, class distributions, error analysis, any exported mapping)
would name the wrong classes. This is why the conflict cannot be caught by watching metrics.
Phase 1 has not run on real data, so nothing existing is invalidated; this paragraph only
explains why the gate exists.

## 7. Pre-registered audit hypotheses

These plug into the audit procedure in [experiment-blueprint.md](experiment-blueprint.md) §3
(manual audit, acceptance rule, distribution corroboration, audit trail):

- **H1 (expected):** raw `0`=negative, `1`=neutral, `2`=positive — the authors' own layer.
  If the audit supports H1, Phase 1's `RAW_SENTIMENT_TO_LABEL` must be corrected to
  `{0: "negative", 1: "neutral", 2: "positive"}` (mapped into the VFI canonical space) before
  any real-data run, and the correction recorded in the audit report.
- **H2 (alternative):** raw `0`=positive, `1`=negative, `2`=neutral — the SEACrowd layer. If
  the audit supports H2, Phase 1's loader stands as-is and the `uitnlp` card/script/features
  are recorded as the error.
- The audit is decided by author documentation first (step 1 of the protocol), the blind
  reader audit second (steps 2-3); a conflict between author answer and audit triggers
  investigation, not silent acceptance of either.

## 8. Draft question block for the author (owner sends — nobody on this program emails anyone)

Copy-paste into the UIT-VSFC request email (template in
[dataset-acquisition-plan.md](dataset-acquisition-plan.md)), or send as a follow-up in the
same thread:

```text
Three quick technical questions about the UIT-VSFC label files (the sentiments.txt file in
each split folder), so we can configure our data loader exactly to your specification:

1. In the distributed train/validation/test sentiment files, what is the exact integer-to-
   sentiment mapping? Specifically, which integer (0, 1, or 2) denotes "negative", which
   "neutral", and which "positive"?

2. Two public Hugging Face sources appear to encode these labels differently, and we would
   like to know which (if either) matches your raw files:
   - The SEACrowd loader (SEACrowd/uit_vsfc) declares the class order
     ["positive", "negative", "neutral"], which would mean 0=positive, 1=negative,
     2=neutral.
   - Your group's own dataset card (uitnlp/vietnamese_students_feedback) states the values
     are "0 (negative), 1 (neutral) and 2 (positive)".
   Are both of these correct interpretations of your raw sentiments.txt files, or is one of
   them mislabeled?

3. To confirm the access procedure: is signing the corpus user agreement (requested via
   kietnv@uit.edu.vn) the required and sufficient step to obtain UIT-VSFC for non-commercial
   research? And once the agreement is signed, may the data be obtained through your group's
   Hugging Face upload (uitnlp/vietnamese_students_feedback), or is a separate delivery used?
```

## Sources (all fetched 2026-11-07)

| # | Source | URL | What was taken | Method |
|---|---|---|---|---|
| 1 | SEACrowd loader script | https://huggingface.co/datasets/SEACrowd/uit_vsfc/raw/main/uit_vsfc.py | ClassLabel order, license, int passthrough, Drive URLs, absence of verification comments | HTTP GET (raw file) |
| 2 | `uitnlp` dataset card | https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback | "Data Fields" prose, Licensing "Unknown", viewer preview rendering (0negative/1neutral/2positive), Contributions note | HTTP GET (page) |
| 3 | `uitnlp` loader script | https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/raw/main/vietnamese_students_feedback.py | ClassLabel orders, int passthrough, Drive URLs, "down load" typo, copyright header | HTTP GET (raw file) |
| 4 | Author repo README | https://raw.githubusercontent.com/kietnv/VietnameseDatasets/master/README.md | UIT-VSFC section, agreement instruction, absence of any label-encoding documentation | HTTP GET (raw file) |
| 5 | SEACrowd issue #208 + comments | https://api.github.com/repos/SEACrowd/seacrowd-datahub/issues/208/comments | Absence of label discussion (9 comments) | GitHub REST API |
| 6 | SEACrowd PR #246 + comments | https://api.github.com/repos/SEACrowd/seacrowd-datahub/issues/246/comments ; .../pulls/246/files | Merge date 2024-01-16, absence of label discussion, diff showing both label orders added diverged | GitHub REST API |
| 7 | Paper metadata/abstract | https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/KSE.2018.8573337 | Closed-access status, abstract text (no encoding content) | Semantic Scholar API |
| 8 | IEEE landing page | https://ieeexplore.ieee.org/document/8573337 | Blocked; no content retrievable | HTTP GET (failed) |
| 9 | arXiv search | http://export.arxiv.org/api/query?search_query=all:%22UIT-VSFC%22 | Zero results (no arXiv version) | arXiv API |
| 10 | Author homepage deep link | https://sites.google.com/uit.edu.vn/uit-nlp/datasets-projects | HTTP 404 | HTTP GET (failed) |
| 11 | HF datasets API metadata | https://huggingface.co/api/datasets/uitnlp/vietnamese_students_feedback | `gated: False` (openly downloadable), license tag `unknown` | HF Hub API |
