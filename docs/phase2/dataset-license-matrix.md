# Phase 2 Dataset License Matrix

One row per candidate. All sources fetched read-only on 2026-11-07; detailed evidence and
labels ([VERIFIED FACT]/[ESTIMATE]/[ASSUMPTION]/[UNVERIFIED]) are in
[dataset-landscape.md](dataset-landscape.md); recommendation and owner actions in
[dataset-acquisition-plan.md](dataset-acquisition-plan.md); UIT-VSFC label-encoding evidence
in [label-semantics-evidence.md](label-semantics-evidence.md).

## Terminology (status labels used in every dataset doc)

The matrix records each candidate's stated access condition. Actual use depends on the
selected release's terms and the owner's documented permission for the intended use.

- **CONDITIONAL CANDIDATE** — the publisher lists a license and an access gate such as
  Hugging Face login and click-accept. Verify the terms and the scope of the publisher's
  grant before use. Rights over the underlying third-party content may remain unclear.
- **PERMISSION REQUIRED (author agreement / stated permission step)** — the data is real and
  relevant, but lawful use requires an explicit human step: signing an agreement, an
  author-approved request, or (for research-use statements without a license file) contact
  with the authors. Not downloadable under project rules until that step is completed.
- **REJECTED** — rights, provenance, or data quality fail the project's data-safety rules;
  do not pursue.
- **NOT EVALUATABLE** — the candidate could not be assessed this session (inaccessible or
  no dataset exists); no verdict is claimed.

| Dataset | License (as stated at source) | Access mechanism | Redistribution | Commercial use | PII / UGC risk | Verdict | Primary source URL |
|---|---|---|---|---|---|---|---|
| **Owner-collected feedback** | Depends on documented collection terms and permission for the intended use | Collect authorized feedback locally and record provenance. Use human annotation and approval before splitting. | Verify separately before sharing any data | Verify separately for the proposed use | Potentially high. Remove unnecessary identifiers and report aggregates. | **PRIMARY PATH**, subject to documented use rights and review | [Local workflow](../../README.md#build-an-independent-dataset) |
| **UIT-VSFC** (v1.0, KSE 2018) | None published. Both HF mirrors mark licensing unknown. | Author distribution requires a signed corpus user agreement | Governed by the agreement | Governed by the agreement | High: surveys may name courses, lecturers and facilities | **PERMISSION REQUIRED**. Optional adapter, inactive under the owner's decision to skip contact and agreement. Original label encoding remains unresolved. | https://github.com/kietnv/VietnameseDatasets |
| **NEU-ESC** (arXiv:2506.23524, 2025) | Uploader's HF tag **`apache-2.0`**. The paper provides no separate license statement. | HF login and click-accept gate recorded in the 2026-11-07 review | Rights over underlying scraped content are unconfirmed | Rights over underlying scraped content are unconfirmed | Moderate-high: social-media content. Report aggregates. | **CONDITIONAL CANDIDATE**. Verify terms, acceptance and label encoding. Toxic needs an explicit policy. | https://huggingface.co/datasets/hung20gg/NEU-ESC |
| **sonlam1102/multilevel_sentiment_vietnamese** (SoMeT_21) | Uploader's HF tag **`mit`** and citation request | HF login and click-accept gate recorded in the 2026-11-07 review | Check terms and rights over source reviews before redistribution | Check terms for the intended use | Low-moderate: e-commerce product reviews | **CONDITIONAL CANDIDATE**. Verify the ranked 5-level semantics and document the collapse to 3 classes. | https://huggingface.co/datasets/sonlam1102/multilevel_sentiment_vietnamese |
| **UIT-ViSFD** (KSEM 2021) | **No LICENSE file**; README: "freely available for research purposes"; "If any company utilizes the dataset for commercial purposes, please contact us via email: 18521073@gm.uit.edu.vn" | Download `UIT-ViSFD.zip` from official repo | Not specified (default copyright applies) | Requires author contact per README | Moderate: crawled e-commerce customer comments | **PERMISSION REQUIRED (author research-use statement, no license file)**; ABSA schema reduction needed | https://github.com/LuongPhan/UIT-ViSFD |
| **SA-VLSP2016** (VLSP 2016 shared task; JCSC 34(4) 2018) | No explicit license; access controlled by signed VLSP2016 User Agreement. HF mirror `SEACrowd/vlsp2016_sa` tags `license:cc-by-nc-sa-4.0` with no stated basis — treat as unconfirmed | Fill + sign VLSP2016 User Agreement PDF, email to vlsp.resources@gmail.com | Governed by agreement (not public) | Governed by agreement; mirror tag suggests NC but unconfirmed | Moderate: tinhte.vn / vnexpress / Facebook comments | **PERMISSION REQUIRED (signed VLSP2016 User Agreement)** | https://vlsp.org.vn/resources-vlsp2016 |
| **UIT-VSMEC** (PACLING 2019) | None published; agreement-gated | Email `kietnv@uit.edu.vn`, sign corpus user agreement | Governed by agreement | Governed by agreement | Moderate: social-media text | **PERMISSION REQUIRED (author agreement)** + poor fit (6 emotions, no sentiment classes; SEACrowd mirror now 401) | https://github.com/kietnv/VietnameseDatasets |
| **UIT-ViOCD** (arXiv:2104.11969) | None published; agreement-gated | Email `18521218@gm.uit.edu.vn` (Ms. Nhung), sign corpus user agreement | Governed by agreement | Governed by agreement | Moderate: e-commerce reviews | **PERMISSION REQUIRED (author agreement)** + wrong task (complaint detection, 5,485 reviews) | https://github.com/kietnv/VietnameseDatasets |
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
2. **Access and permission are separate checks.** NEU-ESC and sonlam1102 were listed as
   automatically gated on 2026-11-07. Check the terms of the chosen release and record the
   owner's acceptance and citation requirements before use.
3. **Underlying rights remain uncertain.** The uploader's license tag does not establish
   permission from the people who contributed the original content. For NEU-ESC, the paper
   describes collection from university Facebook groups and unmoderated forums. The
   [dataset landscape](dataset-landscape.md) records the evidence for review.
4. **The raw UIT-VSFC label order remains UNRESOLVED, now with a full evidence trail**
   ([label-semantics-evidence.md](label-semantics-evidence.md)): both mirrors consume the
   *identical* raw Google Drive files with int passthrough, so their contradictory
   `ClassLabel` declarations cannot both describe the same original encoding. The evidence
   balance favors the authors' own org layer (0=negative, 1=neutral, 2=positive) over
   SEACrowd's unverified reorder. Use neither declaration as the integer mapping for original
   files without verification. This requirement applies only to the optional UIT-VSFC adapter.
5. **Commercial use** is a non-issue for VFI (non-commercial educational/portfolio project),
   but is recorded because the portfolio's terms-of-use posture may change.
