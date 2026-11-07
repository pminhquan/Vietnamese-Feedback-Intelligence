# Dataset Strategy — Dataset-Independent Path for VFI

Investigation date: 2026-11-07. Prepared by: Research Agent R (Dataset Strategy),
Phase 1 hardening branch (`hardening/phase1-integrity-zcode`). This document is
**strategy + evidence**, not a download script. It does not modify the standing
UIT-VSFC investigation (`docs/dataset-uit-vsfc.md` in this branch); it
supersedes UIT-VSFC's role in the *plan* per the owner's decision below.

## Owner decision (fixed constraints, restated so the strategy is auditable)

- The owner will **not** email the UIT-VSFC authors and will **not** sign a
  corpus user agreement. UIT-VSFC is therefore **no longer the mandatory
  dataset**. Its usage conditions are not bypassed and no mirror download is
  performed (see `docs/dataset-uit-vsfc.md`, verdict "BLOCKED — do not
  download yet").
- The project must become **dataset-independent**: (A) legitimately accessible
  public Vietnamese sentiment datasets, and/or (B) an independently created
  dataset with explicit provenance and human annotation.
- Public availability is **not** permission. Every access condition below is
  stated as it actually is.

## Evidence legend

- **[VERIFIED FACT]** — read directly from the cited primary source on
  2026-11-07 (URL given in the text or the Sources section).
- **[ESTIMATE]** — numeric judgment with stated basis, not a primary-source
  quote.
- **[ASSUMPTION]** — reasonable inference, not directly evidenced.
- **[UNVERIFIED]** — plausible claim not confirmed from a primary source this
  session.

---

## 1. Per-candidate verification table (verdict first)

| Dataset | Status for VFI | License (as declared) | Access gate — the exact manual step | One-line risk |
| --- | --- | --- | --- | --- |
| `hung20gg/NEU-ESC` (HF) | **Primary real-data candidate** | Apache-2.0 (self-declared on HF by the paper authors; no license statement in the paper text) | `gated: "auto"` — owner logs in to HF, clicks "agree and access" on the dataset page; approval is automatic, no email, no signature; files are then downloadable with a personal HF token | Publisher is an individual account name-matched to paper author #2, not an institution; data scraped from social media; no IAA anywhere; 69% neutral / 2.56% toxic imbalance; paper's own Table 1 is internally inconsistent by 50 samples |
| `sonlam1102/multilevel_sentiment_vietnamese` (HF) | **Secondary candidate (domain-diverse supplement)** | MIT (self-declared on HF card) | `gated: "auto"` — same click-through step as NEU-ESC; the author also exposes `dataset.zip` publicly on GitHub (no gate at all) | 5,971 reviews only; e-commerce product domain (shift vs student feedback); annotation protocol/IAA undocumented; the GitHub mirror carries **no** license file, so the HF copy is the license-bearing artifact |
| `LuongPhan/UIT-ViSFD` (GitHub, official) | **Optional supplement (no signature needed)** | None declared; README-level condition: "freely available for research purposes"; commercial use requires contacting the author | **No gate.** Public GitHub zip; conditions are: cite the paper, stay non-commercial, or email the author | "Research purposes" is a README sentence, not a formal license; ABSA schema (10 aspects x 3 polarities), not sentence-level 3-class; scraped e-commerce text |
| `VLSP2016-SA` (via vlsp.org.vn) | **Fallback only (conflicts with owner constraints)** | No open license; distributed under VLSP's signed data user agreement | Download `VLSP2016UserAgreement.pdf`, fill it out, **sign it**, email it to `vlsp.resources@gmail.com` — this is exactly the kind of signed agreement the owner has ruled out | Signed-agreement path contradicts the owner's fixed constraint; only 6,450 sentences; 4th label `USELESS` needs a documented disposition |

Community HF datasets matching "vietnamese sentiment" (e.g.
`sepidmnorozy/Vietnamese_sentiment`, `minhtoan/vietnamese-comment-sentiment`,
`PhatNguyen41/vietnamese-social-media-sentiment`, all MIT-tagged) were located
via the HF search API but their provenance and upload rights were **not**
verified this session — [UNVERIFIED], deprioritized. A license tag on HF is a
string the uploader typed; it is not evidence of rights over the underlying
text.

---

## 2. Candidate A1 — `hung20gg/NEU-ESC` (primary)

### Publisher identity vs the paper

- Paper: **"NEU-ESC: A Comprehensive Vietnamese dataset for Educational
  Sentiment analysis and topic Classification toward multitask learning"**,
  arXiv:2506.23524, submitted 2025-06-30. Authors: Phan Quoc Hung Mai, Quang
  Hung Nguyen, Phuong Giang Duong, Hong Hanh Nguyen, Nguyen Tuan Long.
  [VERIFIED FACT — arXiv abs page]
- The abstract itself states: "The dataset is publicly available at:
  https://huggingface.co/datasets/hung20gg/NEU-ESC." The paper therefore
  designates this exact HF repo as the canonical distribution. [VERIFIED
  FACT]
- The HF account `hung20gg` has fullname "Nguyen Quang Hung" — matching paper
  author #2 (Nguyen Quang Hung) — and lists no organization memberships.
  [VERIFIED FACT for the profile fields; the identity match between the HF
  account and the author is a strong [ASSUMPTION] (name match + the paper's
  own link), not cryptographic proof]
- The HF search API for "NEU-ESC" returns **exactly one** dataset — this one.
  There is no institutional `NEU` org mirror to prefer instead. [VERIFIED
  FACT]

### Gate, license, files

- API record (fetched 2026-11-07): `gated: "auto"`, tags include
  `license:apache-2.0`, `language:vi`, `task_categories:text-classification`,
  `size_categories:10K<n<100K`; files `train_set.csv`, `val_set.csv`,
  `test_set.csv` (~2.78 MB total); created 2024-07-19, last modified
  2025-03-17; 43 downloads, 5 likes. [VERIFIED FACT]
- The gate is real: fetching the raw README without authentication returns
  "Access to dataset hung20gg/NEU-ESC is restricted. You must have access to
  it and be authenticated to access it." The dataset page shows the standard
  auto-gate text: "You need to agree to share your Hugging Face information to
  access this dataset". Per the Hub docs, `gated: auto` means **automatic
  approval**: any logged-in user who accepts the terms gets access
  immediately, with no owner review. [VERIFIED FACT]
- **The exact manual step the owner must perform:** (1) sign in to Hugging
  Face (create a free account if needed); (2) open the dataset page and accept
  the access request — this is a click-through, **not** an email to authors
  and **not** a signed corpus agreement, but note plainly that it does share
  the account's HF identity with the publisher; (3) create a read token and
  store it in `.env` (never committed); (4) load via `huggingface_hub` /
  `datasets` with that token. [VERIFIED FACT for the mechanism]
- License: `apache-2.0` exists **only** as the authors' HF declaration; the
  24-page paper text contains no license statement at all (only "publicly
  available"). [VERIFIED FACT — absence]. The underlying text was scraped
  from university Facebook groups, community forums, and — for toxic content —
  unmoderated forums (XamVN, Voz) [VERIFIED FACT, paper §3.1]; an Apache-2.0
  grant over scraped third-party posts is the authors' own claim, and the
  original posters' rights are not addressable by that declaration. This is a
  normal-but-real provenance caveat for the whole domain, not a reason to
  reject; record it in the dataset fact sheet. [ASSUMPTION/ESTIMATE]

### Task, schema, splits, size

- Sentiment task: **four classes**, per paper Table 1 with raw label indices:
  `0=Neutral (22,773, 69.08%)`, `1=Positive (4,148, 12.58%)`,
  `2=Negative (5,250, 15.77%)`, `3=Toxic (845, 2.56%)`. The paper defines
  Toxic as "comments with strong, hateful words or the use of offensive
  language aimed at others". [VERIFIED FACT — paper PDF, Table 1]
- **Internal inconsistency, verified:** the four counts sum to 33,016 while
  the paper's stated total is 32,966 (which the split counts do sum to), and
  5,250/32,966 = 15.93%, not the table's 15.77%; a Negative count of 5,200
  would satisfy both. The delivered CSVs, not the prose, are authoritative
  once access is obtained. [VERIFIED FACT — arithmetic on the paper's own
  numbers]
- Splits: 7:1:2 → train 23,048 / validation 3,305 / test 6,613, matching the
  three CSV filenames on HF. [VERIFIED FACT — paper Table 3 + HF file list]
- Topic task: "nine different groups" per the prose, followed by a list of
  ten names — another paper-internal sloppiness; irrelevant unless we ever use
  topics. [VERIFIED FACT — presence of the inconsistency]

### Annotation protocol

- The paper contains **no annotator identification, no annotation guideline
  reference, and no inter-annotator agreement statistic**: "agreement" and
  "kappa" appear zero times in the full text; every "annotat-" hit refers to
  other datasets or the schema description. [VERIFIED FACT — absence, full
  24-page text extracted]
- The abstract's 83.7% / 79.8% accuracy figures are the paper's own BERT
  benchmarks. They are **not** expectations for VFI and must not be quoted as
  such anywhere. [VERIFIED FACT for what the paper claims; explicitly out of
  scope for us]

### Why primary despite the caveats

Domain match (Vietnamese student/education feedback, ~33k samples), schema
superset of our 3-class target, an author-linked distribution channel with an
automatic click-through gate (compatible with the owner's no-email/no-signature
constraints), and the paper's shortcomings (no IAA, inconsistent table) are
auditable *by us on the actual files* — the same blind-audit discipline the
UIT-VSFC investigation established.

---

## 3. Candidate A2 — `sonlam1102/multilevel_sentiment_vietnamese`

- Publisher: HF user `sonlam1102`, fullname "Son T. Luu", a member of the
  `uitnlp` org (The UIT Natural Language Processing Group) on HF. [VERIFIED
  FACT]. The dataset card cites: Hien D. Nguyen, Thanh Le, Khiem Tran, **Son
  T. Luu**, Suong N. Hoang, Hieu T. Phan, "Multi-Level Sentiment Analysis of
  Product Reviews Based on Grammar Rules", SoMeT'21, IOS Press (FAIA vol. 337,
  p. 444). The uploader name-matches the paper co-author. [VERIFIED FACT for
  the citation string; co-author identity is [ASSUMPTION] by name match]
- Record: `gated: "auto"`, `license:mit` tag and `cardData.license = mit`,
  single file `multi_level_sentiment.csv`, created 2026-01-23, last modified
  2026-01-24. [VERIFIED FACT]
- Schema and size (card description): **5,971 reviews of products on
  e-commerce websites**; labels are 5 levels — 1 Very negative, 2 Negative, 3
  Neutral, 4 Positive, 5 Very positive. [VERIFIED FACT]
- Gate: same auto click-through as NEU-ESC; raw README fetch without auth is
  refused. [VERIFIED FACT]. Additionally, the author's GitHub repo
  (`sonlam1102/multi-level-sentiments`) serves `dataset.zip` publicly (HTTP
  200, ~202 KB — checked by HEAD request only; not downloaded). [VERIFIED
  FACT]
- Reuse restrictions: the card requests citation of the publication; license
  field says MIT. Note the **discrepancy**: the GitHub repo has **no** license
  file (GitHub API `license: null`), so the MIT tag exists only on the HF
  copy. Use the HF-gated artifact as the license-bearing source of truth; do
  not substitute the GitHub zip as the training source. [VERIFIED FACT]
- Annotation protocol: no annotator count, guideline, or IAA is documented in
  the card or the GitHub README. [VERIFIED FACT — absence]
- **Modeling note (documented choice, required by the mission):** mapping
  5-level scores to the canonical 3-class space via 1,2 → negative, 3 →
  neutral, 4,5 → positive is a **modeling choice we document, not a fact of
  the data**. Caveats that must travel with it: (a) the source domain is
  e-commerce product reviews, not student feedback — vocabulary, aspect
  structure and rating behavior differ, so a model built on it is not
  drop-in for VFI's target domain; (b) 5,971 rows is small; (c) collapsed
  class ratios must be recomputed from the CSV, not assumed. **No accuracy
  expectation is claimed or implied.**

---

## 4. Candidate B1 — UIT-ViSFD (official GitHub)

- Publisher: the paper authors themselves — GitHub repo `LuongPhan/UIT-ViSFD`
  (Luc Phan Luong et al., UIT VNUHCM, 2021), containing `README.md` +
  `UIT-ViSFD.zip`. [VERIFIED FACT]
- Paper: "SA2SL: From Aspect-Based Sentiment Analysis to Social Listening
  System for Business Intelligence", KSEM 2021, Springer
  (10.1007/978-3-030-82147-0_53). The abstract states the dataset of 11,122
  human-annotated comments (mobile e-commerce) "is freely available for
  research purposes". [VERIFIED FACT — abstract text as quoted in the repo
  README and the SEACrowd citation record]
- **What the access condition actually is:** no gate and no signature; the
  README-level conditions are (a) cite the paper, and (b) if any company uses
  it commercially, contact the author by email first. That is a de-facto
  custom condition rather than a formal license. [VERIFIED FACT]
- Schema: 10 aspects x 3 polarities (aspect-based), attributes `comment`,
  `n_star`, `data_time`, `label`; official splits 7,786 / 1,112 / 2,224.
  [VERIFIED FACT]
- HF copy: `SEACrowd/uit_visfd` (ungated) exists but is tagged
  `license:unknown` — a third-party mirror, same situation as SEACrowd's
  UIT-VSFC mirror analyzed in the PR #2 evidence (mirrors add editorial
  layers). Prefer the authors' GitHub zip. `sonlam1102` does **not** host a
  ViSFD copy on HF (his datasets are `vithsd`,
  `multilevel_sentiment_vietnamese`, `vispamdetection`, `vispamdetection2`).
  [VERIFIED FACT]
- IAA: not stated in the README; the abstract claims a "strict annotation
  scheme" without a number. [UNVERIFIED beyond that claim]
- Fit caveat: ABSA (aspect + polarity) is a different task shape from
  sentence-level 3-class sentiment; using it would require a documented
  transformation decision (e.g., polarity-only projection with aspect
  explosion) — out of scope for the primary path.

---

## 5. Candidate B2 — VLSP2016-SA (fallback only)

- Distributor: VLSP (Association for Vietnamese Language and Speech
  Processing). The 2016 resources page states: "Sentiment Analysis: 6450
  sentences (training set and test set)" and "In order to get access to VLSP
  2016 datasets, please fill out the form below, **sign it and send it back to
  us** via the following email address: vlsp dot resources at gmail dot com",
  with `VLSP2016UserAgreement.pdf` (16.7 KB) attached to the page. [VERIFIED
  FACT — https://vlsp.org.vn/resources-vlsp2016]
- Schema (per the SEACrowd mirror description): four labels POS, NEG, NEU,
  USELESS (filter-out); sources tinhte.vn, vnexpress.net, Facebook. [VERIFIED
  FACT for the mirror's description; USELESS disposition would need a
  documented decision]
- Mirror caveat: `SEACrowd/vlsp2016_sa` is ungated and tagged
  `license:cc-by-nc-sa-4.0` while its license field is null, and its citation
  block cites pages 295–310 (the NER shared task paper) instead of the
  sentiment paper's 283–294 per the VLSP site. The authoritative access
  condition is VLSP's signed agreement, not the mirror's tag. [VERIFIED FACT
  for the fields; the tag's legal effect is [UNVERIFIED]]
- Publication: "VLSP Shared Task: Sentiment Analysis", Journal of Computer
  Science and Cybernetics, Vol 34, No 4, pp. 283–294, 2018. [VERIFIED FACT —
  VLSP page]
- **Status:** the signed-agreement path is exactly what the owner has ruled
  out. Kept on the list only as a fallback if the primary path fails, and
  only if the owner revisits that constraint. [ASSUMPTION about owner
  willingness: none]

---

## 6. Option (B) — independently created dataset (normative requirements)

The engineering track runs in parallel; this section states what the dataset
must satisfy to be *defensible*, not how to build it:

- **Per-record provenance:** every record carries machine-checkable metadata —
  source identifier/URL (or `source_type: self-written`), collection method
  and timestamp, and the pipeline version that produced the record.
- **Explicit `source_type` taxonomy** (e.g., `survey_response`,
  `public_post`, `synthetic_template`) with privacy rules per type; public
  posts inherit the scraped-provenance caveat and need a lawful basis
  statement.
- **Human annotation with recorded method and status:** guideline version,
  annotator identities (pseudonymized in-repo), adjudication rule for
  disagreements, and a measured agreement statistic (report the metric and
  the number, even if modest — an honest kappa beats no kappa).
- **Versioning:** semantic dataset version, frozen snapshot with a SHA-256
  manifest of data files, and a changelog entry per version; splits frozen at
  a declared version and never re-split silently.
- **Privacy rules:** no PII in released records, a documented takedown
  procedure, and (for survey data) informed-consent wording kept with the
  provenance metadata.

---

## 7. Recommendation and decision table

**Primary strategy (3 lines):**

1. Bring the pipeline up **today** on the existing synthetic fixtures — this
   needs no owner action and no rights decision.
2. Make **NEU-ESC** the primary real-data target: owner performs the HF
   auto-gate click-through (no email, no signature), token in `.env`, then
   ingest under the same blind-audit discipline used for UIT-VSFC label
   semantics (verify the CSV label encoding against actual content before
   anything trains — the raw-index order `0=Neutral, 1=Positive, 2=Negative,
   3=Toxic` must be remapped to the canonical `negative=0, neutral=1,
   positive=2` explicitly, never positionally).
3. Add **sonlam1102/multilevel_sentiment_vietnamese** as a secondary,
  domain-diverse supplement under the documented 5→3 mapping; treat
   UIT-ViSFD as an optional no-signature supplement; keep VLSP2016-SA as
   fallback only; grow the custom dataset (Section 6) as the long-term
   defensible core.

### Decision table

| Option | Starts today (no owner step) | Owner manual step | Risks (honest) |
| --- | --- | --- | --- |
| Synthetic pipeline bring-up | Yes — Phase 1 code already runs on synthetic fixtures | None | No external validity; it validates engineering, not model quality |
| NEU-ESC (primary) | Ingest code, schema mapping, audit scripts can be written today against the documented schema | HF login + click-through agree (auto-approval), token into `.env` | License is the authors' self-declaration over scraped social-media text; no IAA; 69% neutral / 2.56% toxic imbalance; paper Table 1 off by 50 samples — so validate row counts and label encoding on the delivered CSVs, not the paper |
| sonlam1102 multilevel | Same as above | Same HF click-through (or the author's public GitHub zip — but that copy carries no license; use HF) | Small (5,971); e-commerce domain shift; undocumented annotation; 5→3 collapse is a documented modeling choice, not a data fact |
| UIT-ViSFD | Download path is public; transformation design can start | None strictly — download + record the citation obligation; commercial use would require contacting the author (out of scope) | README-level "research purposes" is not a formal license; ABSA shape mismatch; scraped e-commerce text |
| VLSP2016-SA | Nothing beyond reading | Fill out, sign, email the agreement — **conflicts with the owner's fixed constraint** | Only 6,450 sentences; `USELESS` label disposition; mirror metadata is unreliable |
| Custom dataset (B) | Schema, provenance fields, consent wording, versioning rules can be specified today | Owner decisions: sources, annotators, guideline sign-off | Slowest to first real data; quality depends entirely on annotation discipline actually being followed |

### Toxic-class rule (non-negotiable)

NEU-ESC's `Toxic` class (845 samples, 2.56%) is **distinct**: it is defined by
offensive/hateful language, not by dissatisfaction. It must never be silently
merged into `negative` by a positional label mapping. Any of these is
acceptable **only with explicit, written scientific justification**: keep a
4-class head; exclude toxic rows from the 3-class task with the exclusion
recorded; or add a separate toxicity label path. The default position on this
branch is: preserve the 4-class information at ingestion (a `is_toxic` flag or
preserved raw label), decide the task mapping afterwards, and record the
decision in the experiment protocol.

### Cross-links to existing evidence

- The unresolved UIT-VSFC raw-label-encoding conflict (author artifacts say
  0=negative, SEACrowd's loader implies 0=positive) lives in the PR #2 branch:
  `git show FETCH_HEAD:docs/phase2/label-semantics-evidence.md` (FETCH_HEAD =
  b92e7a4, fetched 2026-11-07). **This branch does not contain
  `docs/phase2/`** — the file must be read out of git, not from the working
  tree. That investigation remains open, but under the owner decision it no
  longer gates the primary path; its *method* (never trust a loader's
  `ClassLabel` order over the served bytes; blind-audit before training)
  carries over directly to NEU-ESC ingestion.
- `docs/dataset-uit-vsfc.md` (this branch) keeps its BLOCKED verdict; this
  document changes the strategy, not that verdict.

---

## 8. Sources (all fetched 2026-11-07)

- HF API, `hung20gg/NEU-ESC`: https://huggingface.co/api/datasets/hung20gg/NEU-ESC
- HF search API, "NEU-ESC" (single result): https://huggingface.co/api/datasets?search=NEU-ESC
- HF user API, `hung20gg`: https://huggingface.co/api/users/hung20gg/overview
- HF dataset page (gate text): https://huggingface.co/datasets/hung20gg/NEU-ESC
- arXiv abstract page: https://arxiv.org/abs/2506.23524
- arXiv PDF (full text extracted with pypdf in a disposable venv):
  https://arxiv.org/pdf/2506.23524
- HF Hub gating docs (auto-approval semantics):
  https://huggingface.co/docs/hub/en/datasets-gated
- HF API, `sonlam1102/multilevel_sentiment_vietnamese`:
  https://huggingface.co/api/datasets/sonlam1102/multilevel_sentiment_vietnamese
- HF user API, `sonlam1102`: https://huggingface.co/api/users/sonlam1102/overview
- GitHub repo + README (fetched raw): https://github.com/sonlam1102/multi-level-sentiments
- GitHub repo + README: https://github.com/LuongPhan/UIT-ViSFD
- HF API, `SEACrowd/uit_visfd` and `SEACrowd/vlsp2016_sa`:
  https://huggingface.co/api/datasets/SEACrowd/uit_visfd ,
  https://huggingface.co/api/datasets/SEACrowd/vlsp2016_sa
- VLSP resources index: https://vlsp.org.vn/resources ; VLSP 2016 datasets page
  (agreement form, sizes, citation): https://vlsp.org.vn/resources-vlsp2016
- HF search API sweep for "vietnamese sentiment" (landscape scan):
  https://huggingface.co/api/datasets?search=vietnamese%20sentiment
