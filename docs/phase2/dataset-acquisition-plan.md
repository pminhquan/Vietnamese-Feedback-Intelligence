# Phase 2 Dataset Acquisition Plan

Date: 2026-11-07 (hardening pass same day: access-condition wording, NEU-ESC gate
re-verification, full label-encoding evidence trail). Evidence and per-candidate detail:
[dataset-landscape.md](dataset-landscape.md). License/access comparison:
[dataset-license-matrix.md](dataset-license-matrix.md). UIT-VSFC label-encoding evidence:
[label-semantics-evidence.md](label-semantics-evidence.md).
This plan proposes owner actions only — nothing has been downloaded, sent, or signed.

## Recommendation

**Primary dataset: UIT-VSFC, obtained through the official corpus user agreement**
(PERMISSION REQUIRED — author agreement via `kietnv@uit.edu.vn`).
**Fallback dataset (USABLE NOW, subject to its HF click-accept gate; start in parallel):
NEU-ESC.**

Rationale for the primary choice:

1. **Exact task and domain match.** UIT-VSFC is the only verified candidate whose domain
   (end-of-semester student feedback at a Vietnamese university) and label schema
   (exactly `positive` / `negative` / `neutral`) match VFI's Phase 1/2 problem statement.
2. **Best annotation-quality evidence.** Inter-annotator agreement > 91% (sentiment),
   author-reported — no alternative documents an annotation protocol at all (NEU-ESC's paper
   contains none) [VERIFIED FACT for the absence; see landscape].
3. **Pipeline continuity.** Official train/validation/test splits (11,426 / 1,583 / 3,166)
   already map onto the Phase 1 loader layout (`sentences.txt` + `sentiments.txt`), so
   Phase 2 comparisons against the Phase 1 classical baselines stay within-dataset.
4. **Portfolio value.** Navigating a real corpus user agreement is itself evidence of
   production-grade data governance.

No fully permissive dataset was found that matches the student-feedback domain; the best
licensed fallback (NEU-ESC, Apache-2.0) covers the same broad education domain but shifts
the sub-domain and label schema (see the distribution-shift paragraph below).

## Blockers (explicit)

| Path | Blocker | Type |
|---|---|---|
| UIT-VSFC (primary) | Signed corpus user agreement required before data delivery; agreement text not public; turnaround unknown | Human action (email + signature); **cannot be engineered around** |
| UIT-VSFC via mirrors | Both mirrors (`SEACrowd/uit_vsfc`, `uitnlp/vietnamese_students_feedback`) tag `license: unknown` — downloading them is not authorized under project rules | Legal; no workaround |
| NEU-ESC (fallback) | HF gate (login + click-accept, auto-approved — re-verified 2026-11-07); no documented annotation protocol; 4-class schema incl. Toxic; 69% neutral skew; minor card-vs-paper size discrepancy (33,016 vs 32,966); residual rights risk: Apache-2.0 tag is the uploader's claim over scraped UGC (accepted for non-commercial, non-redistributing use) | Mitigable (click-accept gate; documented 3-class mapping; never claim label-quality parity with VSFC; no raw-text redistribution) |
| sonlam1102 multilevel (fallback 2) | Cross-domain (product reviews); 5-level collapse; only 5,971 rows; HF gating | Mitigable; use only as secondary corpus |
| SA-VLSP2016 | Signed VLSP2016 User Agreement (form PDF, email) | Human action |
| UIT-ViSFD | No formal license (research-use README statement only) + ABSA schema reduction | Legal ambiguity; not recommended as primary |
| Fine-tuning experiments | Not authorized in this phase (program constraint) | Policy; acquisition proceeds regardless so data is ready |

## Owner action steps (in order)

1. **Send the UIT-VSFC request email** (template below) to `kietnv@uit.edu.vn`.
   Suggested subject: `Request for UIT-VSFC corpus user agreement (non-commercial student research)`.
   Include the three label/access questions (they cost nothing): the authoritative raw
   integer-to-sentiment mapping for the distributed `{split}/sentiments.txt` files; whether
   the two public HF sources (SEACrowd loader vs the authors' own card) interpret those files
   correctly; and confirmation that signing the user agreement is the required and sufficient
   access path (and whether the authors' HF upload is a valid delivery channel). The full
   copy-paste question block is in
   [label-semantics-evidence.md](label-semantics-evidence.md) §8 and is embedded in the
   template below.
2. **Sign and return the agreement**; store the corpus only in `data/raw/uit_vsfc/`
   (gitignored), settle the label encoding per gate G2 — the author's answer plus the blind
   audit of delivered files ([experiment-blueprint.md](experiment-blueprint.md) §3; the two
   pre-registered hypotheses are in [label-semantics-evidence.md](label-semantics-evidence.md)
   §7; note the author's public README documents no encoding, so the audit is mandatory, not
   optional) — then run `vfi prepare-data` → `vfi train` → `vfi evaluate` per Phase 1 protocol.
3. **In parallel, create a HF account and accept the NEU-ESC gate**
   (`hung20gg/NEU-ESC`, Apache-2.0; login + click-accept, auto-approved) so the fallback is on
   disk and ready if the agreement stalls. Record the acceptance date. Do the same for
   `sonlam1102/multilevel_sentiment_vietnamese` (MIT) only if a secondary corpus is actually needed.
4. **Optional (low priority):** request SA-VLSP2016 via the VLSP2016 User Agreement
   (https://vlsp.org.vn/resources-vlsp2016, email `vlsp.resources@gmail.com`) — correct
   3-class schema, different domain.
5. **Record everything** in the experiment metadata (source URL, license tag, acceptance
   date, file sha256 per split) — the Phase 1 provenance fields already support this.

### Email template (suggestion — do not send without personalization)

```text
Subject: Request for UIT-VSFC corpus user agreement (non-commercial student research)

Dear Mr. Kiet Nguyen,

I am a third-year AI student working on a personal, non-commercial portfolio project
called Vietnamese Feedback Intelligence (VFI), which studies Vietnamese 3-class
sentiment classification (positive / neutral / negative) on student feedback. I would
like to request the UIT-VSFC corpus (v1.0) for this research.

Could you please send me the corpus user agreement to sign? If convenient, I would
also appreciate answers to three small questions (so I can configure my data loader
exactly to your specification):

1. In the distributed train/validation/test sentiment files (sentiments.txt), what is
   the exact integer-to-sentiment mapping? Specifically, which integer (0, 1, or 2)
   denotes "negative", which "neutral", and which "positive"?
2. Two public Hugging Face sources appear to encode these labels differently, and I
   would like to know which (if either) matches your raw files: the SEACrowd loader
   (SEACrowd/uit_vsfc) declares the class order ["positive", "negative", "neutral"],
   which would mean 0=positive, 1=negative, 2=neutral; your group's own dataset card
   (uitnlp/vietnamese_students_feedback) states the values are "0 (negative), 1
   (neutral) and 2 (positive)". Are both of these correct interpretations of your raw
   sentiments.txt files, or is one of them mislabeled?
3. Is signing the corpus user agreement (requested via kietnv@uit.edu.vn) the required
   and sufficient step to obtain UIT-VSFC for non-commercial research? And once the
   agreement is signed, may the data be obtained through your group's Hugging Face
   upload (uitnlp/vietnamese_students_feedback), or is a separate delivery used?

I commit to: citing the KSE 2018 UIT-VSFC paper in all derived work; using the corpus
only for this non-commercial educational research; not redistributing the raw data;
and reporting only aggregate statistics and short paraphrased examples.

Thank you very much for maintaining these datasets for the community.

Best regards,
<full name>, <program/university>, <contact>
```

## Distribution shift and comparability with Phase 1 baselines

If Phase 2 proceeds on **NEU-ESC** (or any non-primary dataset) while the UIT-VSFC
agreement is pending, the data no longer matches Phase 1's corpus, and every comparison
with the Phase 1 baselines (majority, TF-IDF + LogReg) changes meaning in three concrete
ways. First, **sub-domain shift**: NEU-ESC is university Facebook-group and forum chatter
(with XamVN/Voz content in its Toxic slice), not end-of-semester survey feedback — text is
longer on average, slang/teen-code density is higher, and the "feedback" framing (course,
lecturer, facility) is absent, so Phase 1's word-bigram vs char n-gram intuitions may
invert. Second, **label-schema shift**: NEU-ESC annotates 4 sentiment classes
(Neutral/Positive/Negative/Toxic) with no neutral-definition alignment to VSFC; the
documented mapping must be (Positive→positive, Negative→negative, Neutral→neutral, Toxic→
excluded), and the excluded 2.56% plus a 69% neutral majority reshape the problem from a
roughly balanced classification into a heavily skewed one — macro-F1 will move for reasons
that have nothing to do with the model. Third, **annotation-provenance shift**: VSFC's
labels carry a >91% author-reported agreement, NEU-ESC's annotation process is undocumented,
so any score difference is confounded by label quality. The operating rules are therefore:
(1) never compare scores across datasets — Phase 1 baseline numbers are only comparable
with Phase 2 numbers produced on the *same* dataset and splits; (2) if Phase 2 runs on a
fallback dataset, **re-run both Phase 1 classical baselines on it** (the pipeline is
dataset-configurable) so there is an honest within-dataset reference; (3) report the
dataset, mapping, and excluded classes in every artifact's metadata; (4) treat cross-dataset
gaps (e.g., "model X does better on NEU-ESC than TF-IDF did on VSFC") as anecdotes for
learning notes, never as results.

## Decision record to fill after execution

- [ ] Date agreement email sent; reply received (yes/no, date)
- [ ] Agreement signed and returned; data received (date, sha256 per split)
- [ ] Label order settled (G2): author's answer recorded AND blind audit passed; which
      pre-registered hypothesis held (label-semantics-evidence.md §7), mapping corrected if H1
- [ ] Fallback activated? (NEU-ESC accepted on HF: date) — used for pipeline bring-up only
- [ ] Final dataset for Phase 2 experiments: ____________
