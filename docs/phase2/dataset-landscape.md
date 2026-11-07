# Phase 2 Dataset Landscape — Vietnamese 3-Class Sentiment Classification

Investigation date: 2026-11-07. Prepared by: Research Agent A (Dataset & Legal), Phase 2
readiness program. Scope: identify the best *legitimately usable* dataset for Vietnamese
3-class sentiment classification (`negative` / `neutral` / `positive`), student-feedback
domain preferred, as the data foundation for Phase 2. Fine-tuning is **not** authorized in
this phase; this document makes no experiment claims.

Companion documents:

- Comparison table (one row per candidate): [dataset-license-matrix.md](dataset-license-matrix.md)
- Recommendation, contingencies, and owner actions: [dataset-acquisition-plan.md](dataset-acquisition-plan.md)

## Evidence legend and method

Every load-bearing claim carries one of these labels:

- **[VERIFIED FACT]** — read directly from the cited primary source on 2026-11-07 (URL given).
- **[ESTIMATE]** — approximate figure derived from a stated source, not an exact published number.
- **[ASSUMPTION]** — reasonable inference, not directly evidenced.
- **[UNVERIFIED]** — plausible claim from secondary sources; not confirmed from a primary source.

Method: for each candidate the distribution page (GitHub/HF), license file where present,
and the dataset paper's abstract or full text were fetched read-only on 2026-11-07. No
datasets were downloaded; no authors were contacted. Presence on a public mirror was never
treated as evidence of usage rights.

## Decision summary

| # | Candidate | Domain | Sentiment classes | Verdict |
|---|-----------|--------|-------------------|---------|
| 1 | UIT-VSFC | Student feedback (surveys) | 3 (exact match) | **PERMISSION REQUIRED** (primary recommendation) |
| 2 | NEU-ESC | University forums / FB groups | 4 (Neutral/Pos/Neg/Toxic) | **USABLE NOW** (with documented caveats) |
| 3 | sonlam1102/multilevel_sentiment_vietnamese | E-commerce product reviews | 5-level (collapsible to 3) | **USABLE NOW** (small, cross-domain) |
| 4 | UIT-ViSFD | Smartphone feedback (e-commerce) | 3 polarities x 10 aspects (ABSA) | **PERMISSION REQUIRED** (research-use statement, no formal license) |
| 5 | SA-VLSP2016 | Tech forum / news / FB comments | 3 + USELESS | **PERMISSION REQUIRED** (signed user agreement) |
| 6 | UIT-VSMEC | Social media | 6 emotions (no sentiment classes) | **PERMISSION REQUIRED** + poor label fit |
| 7 | UIT-ViOCD | E-commerce product reviews | Complaint vs non-complaint | **PERMISSION REQUIRED** + wrong task |
| 8 | NTC-SCV | Blog/review comments | Binary (no neutral) | **REJECTED** |
| 9 | anotherpolarbear/vietnamese-sentiment-analysis | Phone-retailer reviews (apparent) | 1-5 stars | **REJECTED** |
| 10 | thnhan3/vietnamese-students-feedback-synthetic | Synthetic student feedback | Neutral only | **REJECTED** as a corpus (synthetic derivative) |
| 11 | ViTASA / EduPulse / misc HF uploads | various | various | **NOT EVALUABLE / REJECTED** (see §C) |

---

## A. Candidate deep dives

### A1. UIT-VSFC — Vietnamese Students' Feedback Corpus

**Status: PERMISSION REQUIRED. Primary recommendation for Phase 2.**

- **Publisher / maintainer:** Kiet Van Nguyen and colleagues, UIT NLP group (University of
  Information Technology, VNU-HCM). **[VERIFIED FACT]**
- **Official source:** the author's distribution repository,
  https://github.com/kietnv/VietnameseDatasets. **[VERIFIED FACT]**
- **Citation:** Kiet Van Nguyen, Vu Duc Nguyen, Phu Xuan-Vinh Nguyen, Tham Thi-Hong Truong,
  Ngan Luu-Thuy Nguyen, "UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment
  Analysis", KSE 2018, IEEE, pp. 19-24, DOI 10.1109/KSE.2018.8573337. **[VERIFIED FACT]**
  (consistent on both HF mirrors fetched 2026-11-07)
- **License / user agreement:** The author's repository states, verbatim: *"Please contact
  us via email: kietnv@uit.edu.vn (Mr. Kiet Nguyen) to sign the corpus user agreement and
  then receive the corpus."* **[VERIFIED FACT, re-fetched 2026-11-07]** No license field
  establishes usage rights anywhere. Both HF mirrors tag the license as unknown:
  - `SEACrowd/uit_vsfc`: card shows "License: unknown"; the loader script sets
    `_LICENSE = Licenses.UNKNOWN.value`. **[VERIFIED FACT]**
  - `uitnlp/vietnamese_students_feedback` — an upload by the UIT NLP group's **own HF org**:
    card states "Licensing Information — Unknown." **[VERIFIED FACT]** That the authors' own
    org uploaded the data with license `unknown` does not itself grant usage rights; the
    agreement channel remains the authoritative path. It does, however, strongly corroborate
    the corpus's authenticity, and makes the HF artifact a plausible delivery mechanism to
    ask about when signing the agreement. **[ASSUMPTION]**
- **Access mechanism:** email + signed corpus user agreement, then data is delivered by the
  author (Phase 1 investigation documented the author Google Drive folder as the homepage:
  https://drive.google.com/drive/folders/1HooABJyrddVGzll7fgkJ6VzkG_XuWfRu). **[VERIFIED FACT
  for the Drive link from the SEACrowd card; the Drive content itself was not fetched]**
- **Redistribution rules:** not published; governed by the signed agreement (contents not
  publicly available). **[ASSUMPTION: conservative default is no redistribution]**
- **Commercial vs non-commercial:** agreement terms unknown; the project is non-commercial,
  which matches every known use case in the literature. **[ASSUMPTION]**
- **Task / label schema:** 3-class sentiment (`positive`, `negative`, `neutral`) + 4-class
  topic. **[VERIFIED FACT]**
- **Size and splits:** 16,175 rows total; official splits train 11,426 / validation 1,583 /
  test 3,166 (per the `uitnlp` card). **[VERIFIED FACT]**
- **Label-encoding inconsistency across mirrors (important):** the SEACrowd loader declares
  `SENTIMENT_LABEL_CLASSES = ["positive", "negative", "neutral"]` (0=positive) **[VERIFIED
  FACT]**, while the `uitnlp` card documents sentiment labels as "0 negative, 1 neutral,
  2 positive" **[VERIFIED FACT]**. The two mirrors disagree; Phase 1's caveat that the raw
  `0/1/2` encoding must be checked against the author's README after access is therefore
  even more important. See [dataset-acquisition-plan.md](dataset-acquisition-plan.md).
- **Annotation quality:** inter-annotator agreement > 91% for sentiment and > 71% for topic;
  Maximum Entropy baseline ~88% sentiment F1 — author-reported, as documented in the Phase 1
  investigation (`git show 95f7687:docs/dataset-uit-vsfc.md`). **[VERIFIED FACT as to what
  Phase 1 documented; paper not re-fetched in this pass]**
- **Personal-data concerns:** student opinions referencing courses, facilities, and
  lecturers; the Phase 1 policy (never commit raw text; aggregate metrics; paraphrase in
  error analysis) carries over unchanged. **[ASSUMPTION carried from Phase 1 policy]**
- **Fit with VFI:** exact — same domain (student feedback), exact 3-class schema, official
  splits already wired into the Phase 1 pipeline (`sentences.txt`/`sentiments.txt` layout).
- **Access feasibility:** one email + signature; known active contact; low effort, unknown
  turnaround. **[ESTIMATE]**

### A2. NEU-ESC — Vietnamese Educational Sentiment Classification

**Status: USABLE NOW (with caveats). Best immediate fallback: permissive license + education domain.**

- **Publisher / maintainer:** HF user `hung20gg`, whose handle matches first author Mai Phan
  Quoc Hung; card citation authors match the paper author list (Nguyen Quang Hung, Mai Phan
  Quoc Hung, Nguyen Thi Hong Hanh, Duong Phuong Giang). **[VERIFIED FACT — consistency noted;
  identity not independently confirmed beyond the name match]**
- **Official source:** https://huggingface.co/datasets/hung20gg/NEU-ESC (gated: accept
  conditions while logged in). **[VERIFIED FACT]** The paper states: *"The dataset is
  publicly available at: https://huggingface.co/datasets/hung20gg/NEU-ESC."* **[VERIFIED
  FACT, from the paper full text]**
- **Citation:** Mai Phan Quoc Hung, Quang Hung Nguyen, Phuong Giang Duong, Hong Hanh Nguyen,
  Nguyen Tuan Long, "NEU-ESC: A Vietnamese Dataset for Educational Sentiment Classification
  and Topic Classification", arXiv:2506.23524 (2025, preprint; no peer-reviewed venue
  listed). **[VERIFIED FACT]**
- **License:** HF card license tag: **`apache-2.0`**. **[VERIFIED FACT]** The paper contains
  no separate license statement. **[VERIFIED FACT — absence]** Note Apache-2.0's patent and
  attribution terms; keep the NOTICE/citation.
- **Access mechanism:** gated HF download (accept conditions). **[VERIFIED FACT]**
- **Redistribution:** Apache-2.0 permits redistribution with license/ac NOTICE preserved.
  **[VERIFIED FACT — standard license text; card imposes no extra restriction]**
- **Commercial vs non-commercial:** Apache-2.0 permits both. **[VERIFIED FACT]**
- **Task / label schema:** sentiment 4-class — per paper Table 1: 0 Neutral, 1 Positive,
  2 Negative, 3 Toxic; plus a 10-class topic task. **[VERIFIED FACT]** To use in a 3-class
  setup, Toxic (2.56%) must be dropped or explicitly mapped — this changes the task
  definition and must be documented (see acquisition plan).
- **Size and splits:** paper: 32,966 samples; splits train 23,048 / validation 3,305 /
  test 6,613 (paper Table 3). **[VERIFIED FACT]** The HF card's per-class sentiment counts
  sum to 33,016 — a small discrepancy between card and paper; the card lists no split
  information. **[VERIFIED FACT — discrepancy documented]**
- **Class distribution:** Neutral 69.08%, Negative 15.77%, Positive 12.58%, Toxic 2.56%.
  **[VERIFIED FACT]** This is a heavily neutral-skewed corpus, very different from UIT-VSFC's
  balance (VSFC's neutral share is roughly 4-5% per a third-party derivative card —
  **[ESTIMATE, secondary source thnhan3 card]**; verify after access).
- **Data collection:** university-related Facebook groups and community forums; additionally
  scraped from unmoderated forums (XamVN, Voz) to capture harmful/offensive content; manual
  teen-code mapping dictionary built during curation. **[VERIFIED FACT, paper §3.1]**
- **Annotation quality:** the paper describes collection and curation but (per full-text
  keyword review) contains **no human annotation protocol, no annotator count, and no
  inter-annotator agreement numbers** for NEU-ESC's own labels. **[VERIFIED FACT — absence]**
  This is the dataset's principal quality gap: label provenance is undocumented. **[ASSUMPTION:
  treat label quality as unproven until spot-checked]**
- **Personal-data concerns:** social-media user-generated content; may contain usernames,
  photos references, and offensive text (a dedicated Toxic class). Raw text must never be
  committed; quotes in reports must be truncated/paraphrased; the Toxic class content
  additionally needs handling care in any shared artifact. **[ASSUMPTION: same policy as
  Phase 1]**
- **Fit with VFI:** strong domain overlap (education), but it is forum chatter, not
  end-of-semester feedback surveys; label schema needs a documented 4-to-3 mapping; heavy
  neutral skew changes the difficulty profile. See distribution-shift discussion in
  [dataset-acquisition-plan.md](dataset-acquisition-plan.md).
- **Access feasibility:** immediate (HF account + accept gating). **[VERIFIED FACT]**

### A3. sonlam1102/multilevel_sentiment_vietnamese

**Status: USABLE NOW (small and cross-domain). Cleanest license evidence among alternatives.**

- **Publisher / maintainer:** HF user `sonlam1102`; Son T. Luu is one of the authors of the
  cited paper, i.e., the uploader appears to be a paper author. **[VERIFIED FACT — name
  match; identity beyond that not confirmed]**
- **Official source:** https://huggingface.co/datasets/sonlam1102/multilevel_sentiment_vietnamese
  (gated=auto; accept conditions to download). Companion code:
  https://github.com/sonlam1102/multi-level-sentiments. **[VERIFIED FACT]**
- **Citation:** Hien D. Nguyen, Thanh Le, Khiem Tran, Son T. Luu, Suong N. Hoang, Hieu T.
  Phan, "Multi-Level Sentiment Analysis of Product Reviews Based on Grammar Rules,"
  SoMeT_21, IOS Press, 2021. The card asks users to cite this publication. **[VERIFIED FACT]**
- **License:** HF card license tag: **`mit`**. **[VERIFIED FACT]** No README-level license
  text beyond the tag was observed; the citation request accompanies it.
- **Access mechanism:** gated (HF login + accept conditions). **[VERIFIED FACT]**
- **Redistribution:** MIT permits use, modification, redistribution (retain the license
  notice). **[VERIFIED FACT — standard license semantics]**
- **Commercial vs non-commercial:** permitted under MIT. **[VERIFIED FACT]**
- **Task / label schema:** 5-level sentiment on product reviews: 1 Very negative,
  2 Negative, 3 Neutral, 4 Positive, 5 Very positive. **[VERIFIED FACT]** A documented
  collapse (1+2→negative, 3→neutral, 4+5→positive) yields the VFI 3-class schema.
- **Size:** 5,971 reviews; no official train/dev/test splits on the card (file ~454 kB).
  **[VERIFIED FACT]**
- **Annotation quality:** no IAA on the card; the SoMeT_21 paper was not fetched in this
  pass. **[UNVERIFIED]**
- **Personal-data concerns:** e-commerce product reviews (user-generated); low-moderate PII
  risk; standard no-commit policy applies. **[ASSUMPTION]**
- **Fit with VFI:** cross-domain (product reviews, not student feedback) and small; useful
  as a licensed secondary corpus or sanity corpus, not as the primary.
- **Access feasibility:** immediate. **[VERIFIED FACT]**

### A4. UIT-ViSFD — Vietnamese Smartphone Feedback Dataset

**Status: PERMISSION REQUIRED (research-use statement, but no formal license).**

- **Publisher / maintainer:** Luc Phan Luong et al., UIT. Official repo:
  https://github.com/LuongPhan/UIT-ViSFD. **[VERIFIED FACT]**
- **Correction to the mission brief:** ViSFD is **smartphone** feedback crawled from a
  Vietnamese e-commerce site — *not* student feedback. **[VERIFIED FACT from the repo README]**
- **Citation:** "SA2SL: From Aspect-Based Sentiment Analysis to Social Listening System for
  Business Intelligence", KSEM 2021, Springer, pp. 647-658. **[VERIFIED FACT]**
- **License:** **no LICENSE file** in the repository (only `README.md` and `UIT-ViSFD.zip`).
  **[VERIFIED FACT]** The README states verbatim: *"which is freely available for research
  purposes"* and *"If any company utilizes the dataset for commercial purposes, please
  contact us via email: 18521073@gm.uit.edu.vn"*. **[VERIFIED FACT]** With no formal license
  file, default copyright restrictions technically apply; the README sentence is a
  research-use permission but leaves redistribution undefined. **[ASSUMPTION]**
- **Access mechanism:** download the zip from the official repo. **[VERIFIED FACT]** Note:
  the author-group index repo (kietnv/VietnameseDatasets) also lists UIT-ViSFD behind its
  email-and-agreement process — two channels with slightly different terms; the repo README
  is the more specific, published statement. **[VERIFIED FACT]**
- **Task / label schema:** aspect-based sentiment — 10 aspects x 3 polarities (positive/
  negative/neutral). **[VERIFIED FACT]** Deriving sentence-level 3-class sentiment requires
  a documented reduction (e.g., comment-level polarity if uniform); non-trivial.
- **Size and splits:** 11,122 annotated comments; train 7,786 / dev 1,112 / test 2,224.
  **[VERIFIED FACT from repo README]**
- **Annotation quality:** no IAA on the repo page. **[VERIFIED FACT — absence]**
- **PII:** crawled customer comments; standard no-commit policy. **[ASSUMPTION]**
- **Fit with VFI:** moderate — polarity classes match, domain is e-commerce smartphone
  reviews; aspect structure adds preprocessing work.
- **Feasibility:** immediate download, but rights posture is weaker than A2/A3 (no license
  file). **[VERIFIED FACT]**

### A5. SA-VLSP2016 — VLSP 2016 Shared Task on Sentiment Analysis

**Status: PERMISSION REQUIRED (signed user agreement).**

- **Publisher / maintainer:** VLSP consortium (Vietnamese Language and Speech Processing).
  **[VERIFIED FACT]**
- **Official source / access:** https://vlsp.org.vn/resources-vlsp2016 — *"please fill out
  the form below, sign it and send it back to us via the following email address: vlsp dot
  resources at gmail dot com"* (VLSP2016 User Agreement PDF). **[VERIFIED FACT]** The 2016
  SA reference paper says data are *"accessible for research purpose via the VLSP website"*.
  **[VERIFIED FACT as quoted in the paper abstract surfaced by search; paper not fetched]**
- **Citation:** Nguyen, Huyen TM et al., "VLSP shared task: sentiment analysis", Journal of
  Computer Science and Cybernetics, vol. 34, no. 4, pp. 295-310, 2018. **[VERIFIED FACT from
  the SEACrowd card citation block]**
- **License:** no explicit license on the VLSP page; access controlled by the signed
  agreement (terms inside the PDF, not publicly quoted). **[VERIFIED FACT — absence]** The
  HF mirror `SEACrowd/vlsp2016_sa` carries the tag **`license:cc-by-nc-sa-4.0`**
  **[VERIFIED FACT]**, but the card gives **no basis** for that designation and explicitly
  states the data must be *"obtain[ed] separately"* (it is a loader-only mirror). A mirror
  tag does not override the consortium's signed-agreement channel. **[ASSUMPTION: treat
  CC-BY-NC-SA-4.0 as unconfirmed]**
- **Task / label schema:** sentiment labels POS / NEG / NEU plus USELESS (filter-out).
  **[VERIFIED FACT from the SEACrowd card]** Sources: tinhte.vn, vnexpress.net, Facebook
  comments. **[VERIFIED FACT]**
- **Size:** 6,450 sentences (training and test set) per the VLSP 2016 resources page.
  **[VERIFIED FACT]**
- **Splits:** training and test exist; exact sizes not verified. **[UNVERIFIED]**
- **PII:** social-media comments; standard UGC policy. **[ASSUMPTION]**
- **Fit with VFI:** correct 3-class sentiment schema; domain is tech-news/Facebook
  commentary, not education. Good legitimacy signal (consortium-run shared task), but
  smaller and older.
- **Feasibility:** fill + sign + email a form; low effort, unknown turnaround. **[ESTIMATE]**

### A6. UIT-VSMEC — Vietnamese Social Media Emotion Corpus

**Status: PERMISSION REQUIRED + poor label fit.**

- **Publisher / maintainer:** Vong Anh Ho, Duc-Vu Nguyen, Kiet Van Nguyen, Ngan Luu-Thuy
  Nguyen et al. (UIT). **[VERIFIED FACT from arXiv:1911.09339 abstract]**
- **Official source:** distributed via the author-group repo (kietnv/VietnameseDatasets)
  behind the email + corpus user agreement process (contact kietnv@uit.edu.vn).
  **[VERIFIED FACT]** The paper says the corpus *"is available at our research website"*.
  **[VERIFIED FACT]** The SEACrowd mirror `SEACrowd/uit_vismec` is no longer publicly
  reachable: unauthenticated HF API requests return **401** (vs 200 for `SEACrowd/uit_vsfc`
  fetched in the same session), indicating the mirror was made private or removed.
  **[VERIFIED FACT]**
- **Citation:** Ho, V.A. et al., "Emotion Recognition for Vietnamese Social Media Text",
  PACLING 2019 (arXiv:1911.09339). **[VERIFIED FACT]**
- **License:** none stated publicly; agreement-gated. **[VERIFIED FACT — absence]**
- **Task / label schema:** 6 emotions (sadness, enjoyment, anger, disgust, fear, surprise) —
  no neutral/positive sentiment classes. **[VERIFIED FACT]** Mapping to a 3-class sentiment
  task would be a lossy reinterpretation of the annotation schema. **[ASSUMPTION]**
- **Size:** exactly 6,927 emotion-annotated sentences. **[VERIFIED FACT from the arXiv
  abstract]**
- **Fit with VFI:** weak (wrong label space for sentiment). Included because it is the
  closest UIT social-media corpus and appears in many Vietnamese "sentiment" resource lists.

### A7. UIT-ViOCD — Vietnamese Open-domain Complaint Detection

**Status: PERMISSION REQUIRED + wrong task.**

- **Publisher / maintainer:** Nhung Thi-Hong Nguyen et al. (incl. Kiet Van Nguyen), UIT.
  **[VERIFIED FACT from arXiv:2104.11969]**
- **Official source:** listed in kietnv/VietnameseDatasets — contact
  18521218@gm.uit.edu.vn (Ms. Nhung) to sign the corpus user agreement. **[VERIFIED FACT]**
- **Citation:** "Vietnamese Complaint Detection on E-Commerce Websites", arXiv:2104.11969
  (2021). **[VERIFIED FACT]**
- **Task / labels:** complaint detection on 5,485 human-annotated e-commerce product reviews;
  inter-annotator agreement 87% (per the abstract). **[VERIFIED FACT]** This is complaint
  detection, not 3-class sentiment — a different task with a different label space.
  **[VERIFIED FACT / ASSUMPTION on binary structure — the abstract does not enumerate
  labels]**
- **License:** none public; agreement-gated. **[VERIFIED FACT — absence]**
- **Fit with VFI:** low (task mismatch), despite e-commerce complaints being sentiment-
  adjacent. Listed for completeness since the mission brief mentioned it (and corrected the
  "online course domain" assumption — ViOCD is open-domain e-commerce complaints).

---

## B. Rejected candidates (rights or data quality)

### B1. NTC-SCV (congnghia0609/ntc-scv) — REJECTED

- **What it is:** a GitHub repo containing `data_train.zip` / `data_test.zip` (data files are
  in the repository), plus word-segmentation and word2vec assets; the README states only
  *"ntc-scv is dataset of blogs on website https://streetcodevn.com"*. **[VERIFIED FACT]**
- **License:** a standard Apache License 2.0 text exists at repo root, boilerplate appendix
  reading "Copyright 2016 nghiatc". **[VERIFIED FACT from the raw LICENSE file]** The repo
  owner handle (`congnghia0609`) and the copyright-line handle (`nghiatc`) do not obviously
  match; the 2016 copyright predates the dataset's known usage history. **[VERIFIED FACT for
  the text; ASSUMPTION that the mismatch matters]** Widely-cited descriptions ("~30k reviews,
  1star/0star classes") come from tutorials and secondary repos, not the primary README.
  **[UNVERIFIED]**
- **Why rejected:** (1) binary polarity only — no neutral class, so it cannot train or
  evaluate VFI's 3-class task without a biased redefinition; (2) provenance ambiguity —
  scraped user-generated content whose uploader's right to relicense is not demonstrable,
  and the license-vs-content relationship (2016 copyright line) is unexplained; (3) the
  primary source itself documents nothing about labels or collection. Under the project's
  rule — "a public repo does not establish usage rights" — this is not a defensible primary
  dataset. **[ASSUMPTION: residual risk judged unacceptable for a portfolio project]**

### B2. anotherpolarbear/vietnamese-sentiment-analysis — REJECTED

- HF card: license tag `mit`, 10,010 rows, integer labels 1-5 (star-like), train 7,790 /
  test 2,220; the README "exists but content is empty"; no provenance, citation, or source;
  sample rows appear to be customer reviews of a Vietnamese phone retailer (TGDD).
  **[VERIFIED FACT from the card]**
- **Why rejected:** a permissive license tag applied by an unrelated third party over
  scraped customer reviews is exactly the pattern the data-safety rules exclude — the
  uploader plausibly holds no rights to grant. No provenance to verify. Not usable.

### B3. thnhan3/vietnamese-students-feedback-synthetic — REJECTED as a corpus

- HF card: license `mit`; 3,174 rows, neutral class only; generated with Gemini Flash 2.5
  (paraphrase + generation), stated to be human-in-the-loop validated; explicitly derived
  from `uitnlp/vietnamese_students_feedback`, whose license is `unknown`. **[VERIFIED FACT]**
- **Why rejected:** synthetic derivative of a license-unknown source inherits the unresolved
  rights question, and synthetic text cannot substitute for real student feedback in
  portfolio evidence. Useful only as inspiration for imbalance handling (documented idea,
  not data). **[ASSUMPTION]**

### B4. Minor HF uploads — REJECTED (noise)

`PhatNguyen41/vietnamese-social-media-sentiment` (MIT tag, but 3 rows total with English
sample text) and several unlicensed mirrors (`sepidmnorozy/Vietnamese_sentiment`,
`minhtoan/vietnamese-comment-sentiment`, `pqthinh232/vietnamese-restaurant-review-sentiment-dataset`,
etc.) have either no license field or no provenance; none qualify under the rules.
**[VERIFIED FACT — license tags absent per HF API search sweep]**

---

## C. Not evaluatable at this time

- **ViTASA** (`kh4nh12/ViTASA`, targeted aspect-based sentiment benchmark): HF returns 401
  unauthenticated as of 2026-11-07 — private or removed. **[VERIFIED FACT]** Could not be
  assessed; do not rely on it.
- **EduPulse** (EACL 2026 industry track, LLM-enhanced opinion mining for Vietnamese student
  feedback): no public dataset release found (searches surface only the paper).
  **[VERIFIED FACT — absence per search]** Not a dataset candidate.
- **"SEntiE"**: no Vietnamese sentiment dataset with this name could be found in indexed
  literature; treated as a garbled reference (closest matches: SA-VLSP2016, NEU-ESC).
  **[VERIFIED FACT — absence per search]**
- **Kaggle mirrors of UIT-VSFC and other corpora**: not investigated as candidates — Kaggle
  uploads are third-party re-distributions and cannot establish usage rights.
  **[ASSUMPTION: policy, not absence of evidence]**

## Key findings recap

1. **UIT-VSFC remains the best-fit dataset and remains blocked on its official agreement
   channel** (unchanged from Phase 1; re-verified 2026-11-07). Its status did not improve:
   both HF mirrors still say `license: unknown`, including the authors' own `uitnlp` org
   upload.
2. **Two genuinely licensed, immediately accessible alternatives exist**, but neither is a
   perfect match: **NEU-ESC** (Apache-2.0, education domain, 4-class schema, no documented
   annotation protocol) and **sonlam1102/multilevel_sentiment_vietnamese** (MIT, e-commerce,
   5-level, ~6k rows).
3. **The corpus user agreement is the bottleneck, not discovery** — every high-quality
   Vietnamese sentiment corpus from the UIT/VLSP ecosystem is agreement-gated. The
   recommended strategy (see [dataset-acquisition-plan.md](dataset-acquisition-plan.md)) is
   to run the agreement request and the NEU-ESC fallback in parallel.
