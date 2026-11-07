# Phase 2 Dataset License Matrix

One row per candidate. All sources fetched read-only on 2026-11-07; detailed evidence and
labels ([VERIFIED FACT]/[ESTIMATE]/[ASSUMPTION]/[UNVERIFIED]) are in
[dataset-landscape.md](dataset-landscape.md); recommendation and owner actions in
[dataset-acquisition-plan.md](dataset-acquisition-plan.md).

Legend for **Verdict**: USABLE NOW = license/access established from the cited primary
source; PERMISSION REQUIRED = data is legitimate but access is gated by an explicit
request/agreement step or a non-formal research-use statement; REJECTED = rights or data
quality fail the project's data-safety rules.

| Dataset | License (as stated at source) | Access mechanism | Redistribution | Commercial use | PII / UGC risk | Verdict | Primary source URL |
|---|---|---|---|---|---|---|---|
| **UIT-VSFC** (v1.0, KSE 2018) | None published; mirrors tag `license: unknown` (SEACrowd loader: `_LICENSE = Licenses.UNKNOWN.value`; `uitnlp` org card: "Licensing Information — Unknown") | Email `kietnv@uit.edu.vn`, **sign corpus user agreement**, receive corpus from author (Google Drive) | Governed by signed agreement (not public) — assume no | Unknown (agreement-gated); project is non-commercial | High: student opinions on named courses/lecturers/facilities; never commit raw text | **PERMISSION REQUIRED** (primary recommendation) | https://github.com/kietnv/VietnameseDatasets |
| **NEU-ESC** (arXiv:2506.23524, 2025) | HF tag **`apache-2.0`**; no separate license statement in paper | HF download, gated ("accept the conditions to access its files and content") | Permitted under Apache-2.0 with NOTICE | Permitted under Apache-2.0 | Moderate-high: FB-group/forum UGC incl. XamVN/Voz toxic content; aggregate-only reporting | **USABLE NOW** (caveats: 4-class schema, 69% neutral, no documented annotation protocol) | https://huggingface.co/datasets/hung20gg/NEU-ESC |
| **sonlam1102/multilevel_sentiment_vietnamese** (SoMeT_21) | HF tag **`mit`** + citation request | HF download, gated=auto (login + accept) | Permitted under MIT (retain notice) | Permitted under MIT | Low-moderate: e-commerce product reviews | **USABLE NOW** (small: 5,971 rows; cross-domain; 5-level collapse needed) | https://huggingface.co/datasets/sonlam1102/multilevel_sentiment_vietnamese |
| **UIT-ViSFD** (KSEM 2021) | **No LICENSE file**; README: "freely available for research purposes"; "If any company utilizes the dataset for commercial purposes, please contact us via email: 18521073@gm.uit.edu.vn" | Download `UIT-ViSFD.zip` from official repo | Not specified (default copyright applies) | Requires author contact per README | Moderate: crawled e-commerce customer comments | **PERMISSION REQUIRED** (research-use statement, no formal license; ABSA schema reduction needed) | https://github.com/LuongPhan/UIT-ViSFD |
| **SA-VLSP2016** (VLSP 2016 shared task; JCSC 34(4) 2018) | No explicit license; access controlled by signed VLSP2016 User Agreement. HF mirror `SEACrowd/vlsp2016_sa` tags `license:cc-by-nc-sa-4.0` with no stated basis — treat as unconfirmed | Fill + sign VLSP2016 User Agreement PDF, email to vlsp.resources@gmail.com | Governed by agreement (not public) | Governed by agreement; mirror tag suggests NC but unconfirmed | Moderate: tinhte.vn / vnexpress / Facebook comments | **PERMISSION REQUIRED** | https://vlsp.org.vn/resources-vlsp2016 |
| **UIT-VSMEC** (PACLING 2019) | None published; agreement-gated | Email `kietnv@uit.edu.vn`, sign corpus user agreement | Governed by agreement | Governed by agreement | Moderate: social-media text | **PERMISSION REQUIRED** + poor fit (6 emotions, no sentiment classes; SEACrowd mirror now 401) | https://github.com/kietnv/VietnameseDatasets |
| **UIT-ViOCD** (arXiv:2104.11969) | None published; agreement-gated | Email `18521218@gm.uit.edu.vn` (Ms. Nhung), sign corpus user agreement | Governed by agreement | Governed by agreement | Moderate: e-commerce reviews | **PERMISSION REQUIRED** + wrong task (complaint detection, 5,485 reviews) | https://github.com/kietnv/VietnameseDatasets |
| **NTC-SCV** (congnghia0609/ntc-scv) | Apache-2.0 file in repo ("Copyright 2016 nghiatc") — relationship to the scraped data unexplained | Clone repo (`data_train.zip`/`data_test.zip`) | Ambiguous (uploader's rights over scraped UGC not demonstrable) | Ambiguous | Unknown: blog/review UGC; internal structure/labels unverified | **REJECTED** (binary-only classes + provenance ambiguity) | https://github.com/congnghia0609/ntc-scv |
| **anotherpolarbear/vietnamese-sentiment-analysis** | HF tag `mit` applied by third party; empty README, no provenance | HF download (open) | Not establishable (uploader's rights unknown) | Not establishable | Unknown: apparent retailer customer reviews | **REJECTED** (license tag without rights provenance) | https://huggingface.co/datasets/anotherpolarbear/vietnamese-sentiment-analysis |
| **thnhan3/vietnamese-students-feedback-synthetic** | HF tag `mit`; LLM-generated (Gemini Flash 2.5) derivative of license-`unknown` uitnlp UIT-VSFC | HF download (open) | Derivative inherits unresolved source rights | Derivative inherits unresolved source rights | Low (synthetic) | **REJECTED** as a corpus (synthetic; derivative of unknown-license data) | https://huggingface.co/datasets/thnhan3/vietnamese-students-feedback-synthetic |
| **ViTASA** (`kh4nh12/ViTASA`) | Unknown — HF returns 401 unauthenticated (private/removed) as of 2026-11-07 | Not accessible | n/a | n/a | n/a | **NOT EVALUATABLE** | https://huggingface.co/datasets/kh4nh12/ViTASA |
| **EduPulse** (EACL 2026 industry) | No dataset release found | n/a (paper only) | n/a | n/a | n/a | **NOT A DATASET CANDIDATE** | https://aclanthology.org/2026.eacl-industry.25 |

## Notes

1. **Mirrors never establish rights.** For every row, the "Primary source URL" is the
   distribution channel whose terms were read. HF mirrors of agreement-gated corpora
   (`SEACrowd/uit_vsfc`, `uitnlp/vietnamese_students_feedback`, `SEACrowd/vlsp2016_sa`) do
   not override the original distributors' terms, and the `SEACrowd/vlsp2016_sa`
   `cc-by-nc-sa-4.0` tag has no documented basis on its card.
2. **Gating is not a license.** NEU-ESC and sonlam1102 are HF-gated; the license tags
   (`apache-2.0`, `mit`) are the operative grants. Re-read both tags immediately before any
   download, and keep a NOTICE/citation record.
3. **Two out of three "unknown"-license mirrors disagree on the raw UIT-VSFC label order**
   (SEACrowd loader: `["positive","negative","neutral"]`; `uitnlp` card: "0 negative,
   1 neutral, 2 positive"). Verify against the author's README after access, per
   [dataset-acquisition-plan.md](dataset-acquisition-plan.md).
4. **Commercial use** is a non-issue for VFI (non-commercial educational/portfolio project),
   but is recorded because the portfolio's terms-of-use posture may change.
