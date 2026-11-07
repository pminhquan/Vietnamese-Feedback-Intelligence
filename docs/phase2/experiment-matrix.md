# Phase 2 Experiment Matrix

Status: **design only** — Phase 2 is not authorized to start; the UIT-VSFC
dataset is not available (license blocked). This matrix is the honest map of
what can run, where, when, and at what cost. All runtimes and costs are
`[ESTIMATE]` with visible arithmetic; none is measured. Written 2026-11-07.

## 1. Status legend

| Tag | Meaning |
| --- | ------- |
| `RUN-NOW` | Runnable at Phase 2 kickoff with no further prerequisites (no dataset needed). Today, before kickoff, nothing runs. |
| `BLOCKED(dataset license)` | Needs the real UIT-VSFC files; blocked until the corpus user agreement is signed and the integrity gate passes. |
| `BLOCKED(compute)` | Data available but the compute scenario cannot deliver a protocol-valid result in reasonable time. |
| `OPTIONAL` | Valuable but not mandatory; runs only if all mandatory cells are complete **and** the owner explicitly approves extra budget. Default decision: skip. |

## 2. Grounding facts (verified, not assumed)

**Local machine** (read from the system, 2026-11-07): Intel i5-10500H
(6C/12T), 32 GB RAM, NVIDIA RTX 3060 Laptop **6 GB VRAM**, Windows.

**External facts** (verified 2026-11-07):

| Fact | Value | Source |
| ---- | ----- | ------ |
| PhoBERT-base size | ~135M params, hidden 768, 12 layers, vocab 64,001, `max_position_embeddings=258`, ~1.2 GB weights | HF card + `config.json`, `vinai/phobert-base` |
| PhoBERT license | HF card tag `license:mit` — checkpoint itself unblocked; re-verify at kickoff and record in metadata | HF card `vinai/phobert-base` |
| PhoBERT input requirement | Text must be pre-segmented (underscore-joined syllables); RDRSegmenter/VnCoreNLP recommended | HF card |
| VnCoreNLP / py_vncorenlp | Apache-2.0, including pre-trained models | GitHub `vncorenlp/VnCoreNLP` |
| underthesea | GPL-3.0 historically; Apache-2.0 in recent releases only — pin and record version if used | GitHub `undertheseanlp/underthesea` |
| UIT-VSFC scale | ≈16k sentences; commonly cited official split train 11,426 / validation 1,583 / test 3,166 — `[ESTIMATE]`, secondary sources disagree slightly; authoritative numbers = what the delivered files contain | Secondary sources; to verify at acquisition |
| UIT-VSFC license | Blocked: author requires contact + signed corpus user agreement | Phase 1 `docs/dataset-uit-vsfc.md` |

**Runtime arithmetic for one fine-tune run** (`[ESTIMATE]`, fp16, seq 128,
batch 16, RTX 3060 Laptop):

- Steps/epoch ≈ ceil(11,426 / 16) ≈ 715; ≤ 5 epochs ⇒ ≤ 3,575 optimizer steps.
- Transformer fwd+bwd ≈ 6 × 135M × 16 × 128 ≈ 1.7e12 FLOPs/step; at an
  assumed sustained 5-8 TFLOPS (20-30% MFU on this card) ⇒ ≈ 0.3-0.6
  s/step ⇒ ≈ 4-7 min/epoch ⇒ **≈ 15-40 min per run** including per-epoch
  validation inference (1,583 sentences).
- CPU-only (i5-10500H): assume 10-25× slower ⇒ **≈ 4-10 h per run**.

Every downstream cell below cites this arithmetic. The first real run
replaces the estimate.

## 3. Compute scenarios

| Scenario | Hardware | Notes |
| -------- | -------- | ----- |
| **local-CPU** | i5-10500H, 12 threads | The Phase 1 scenario; fine for classical models, insufficient for the fine-tuning protocol |
| **local-GPU** | RTX 3060 Laptop 6 GB | Primary Phase 2 scenario; sequential runs only (no parallel training on 6 GB) |
| **cloud-GPU** | E.g. Colab T4 (16 GB) / Kaggle | Identical protocol; **legal gate first**: the signed corpus agreement may forbid uploading the corpus to third-party services — if so, cloud cells become `BLOCKED(license)` |

## 4. The matrix

`[E]` marks an estimate. "Protocol-valid" means: official splits, integrity
gate passed, selection on validation only, full provenance
(`evaluation-protocol.md` §8).

| # | Experiment | Needs dataset | local-CPU | local-GPU (6 GB) | cloud-GPU |
| - | ---------- | ------------- | --------- | ---------------- | --------- |
| E0 | Phase 2 code smoke tests on synthetic fixtures (loader→segmenter→trainer→metrics wiring; VRAM probe with a hand-written sentence) | No | **RUN-NOW** — minutes `[E]` | **RUN-NOW** (checkpoint download ~1.2 GB + forward/backward probe) — < 30 min `[E]` | Not useful |
| E1 | Dataset integrity gate: alignment, fingerprints, label-semantics audit, distributions, leakage | Yes (real files) | **BLOCKED(dataset license)** — minutes once unblocked `[E]` | n/a | n/a |
| E2 | Majority baseline on real data (control floor) | Yes | **BLOCKED(dataset license)** — seconds-minutes `[E]` | n/a | n/a |
| E3 | TF-IDF+LogReg on real data, Phase 1 protocol (4-candidate search, select on validation) | Yes | **BLOCKED(dataset license)** — ≈ 10-20 min `[E]` (4 fits of a sparse linear model on ≈11.4k docs) | n/a | n/a |
| E3v | Variant `tfidf-logreg-seg`: Phase 1-selected candidate re-run once on segmented text (no re-search) | Yes | **BLOCKED(dataset license)**, OPTIONAL — minutes `[E]` | n/a | n/a |
| E4 | PhoBERT-base fine-tune search: 3 candidates (LR {2e-5, 3e-5, 5e-5}), seed 42, early stopping ≤ 5 epochs | Yes | **BLOCKED(dataset license)** and **BLOCKED(compute)** — ≈ 3 × 4-10 h ≈ 12-30 h `[E]`, no seed repetition possible in reasonable time | **BLOCKED(dataset license)** — ≈ 1-2 h total once unblocked `[E]` (3 × 15-40 min) | **BLOCKED(dataset license)** — ≈ 1-2 h `[E]`; gated by corpus-agreement hosting terms |
| E5 | PhoBERT-base final config, 3-5 seeds {42,43,44(,45,46)} + single test evaluation per seed | Yes | **BLOCKED(dataset license)** + **BLOCKED(compute)** — ≈ 12-50 h `[E]` | **BLOCKED(dataset license)** — ≈ 1-2 h (3 seeds) to 2-3.5 h (5 seeds) `[E]` | **BLOCKED(dataset license)** — same order `[E]`, plus hosting gate |
| E6 | Frozen PhoBERT encoder + logistic head (CPU-feasible fallback) | Yes | **BLOCKED(dataset license)** — ≈ 30-60 min encode + seconds head `[E]` | **BLOCKED(dataset license)** — ≈ 5-10 min `[E]` | Same, gated |
| E7a | OPTIONAL second pretrained model: XLM-R-base (279M, raw-syllable input — no segmenter) | Yes | **BLOCKED(compute)** — ≈ 8-20 h/run `[E]` | **OPTIONAL** — tight on 6 GB (batch 8 likely required); ≈ 30-60 min/run `[E]`; only after E4+E5 + owner approval | **OPTIONAL** — comfortable on 16 GB; hosting gate |
| E7b | OPTIONAL second pretrained model: PhoBERT-large (370M) | Yes | **BLOCKED(compute)** | **BLOCKED(compute)** — optimizer state ≈ 6-7 GB exceeds 6 GB VRAM `[E]` | **OPTIONAL** — fits 16 GB `[E]`; only after E4+E5 + owner approval + hosting gate |
| E8 | Latency benchmarks (batch-1 protocol; classical on CPU; PhoBERT on CPU **and** GPU; segmentation cost reported separately) | Yes (real validation texts as samples) | **BLOCKED(dataset license)** — ≈ 15-30 min `[E]` (PhoBERT CPU inference ≈ 0.3-1 s/call × 110 calls `[E]`) | **BLOCKED(dataset license)** — minutes `[E]` | Same, if cloud used |
| E9 | Error analysis on validation (per-sample dump, cell counts, evidence-gated patterns) | Yes | **BLOCKED(dataset license)** — ≈ 1-2 h including write-up `[E]` | n/a (uses saved predictions) | n/a |

Summary shape: **everything needing real data is `BLOCKED(dataset license)`
today**; E4/E5 are compute-feasible on the local GPU the moment the license
unblocks; the CPU column cannot deliver a protocol-valid fine-tuning
comparison; both OPTIONAL cells default to skip.

### 4.1 The same matrix as model × dataset × compute

The experiment rows above, re-cut along the model dimension (real data =
delivered, integrity-gated UIT-VSFC; synthetic = local fixtures, no dataset
needed). Every cost is `[ESTIMATE]`, per §2's arithmetic.

| Model | Dataset: synthetic fixtures | Dataset: real — local-CPU | Dataset: real — local-GPU (6 GB) | Dataset: real — cloud-GPU |
| ----- | --------------------------- | ------------------------- | -------------------------------- | ------------------------- |
| `majority` | RUN-NOW (smoke test only, never a result) | BLOCKED(dataset license) — seconds `[E]` | n/a | n/a |
| `tfidf-logreg` | RUN-NOW (smoke test only) | BLOCKED(dataset license) — ≈ 10-20 min `[E]` | n/a | n/a |
| `phobert-base-ft` | RUN-NOW (wiring + VRAM probe, < 30 min `[E]`) | BLOCKED(dataset license) + BLOCKED(compute) — ≈ 4-10 h/run `[E]`; protocol needs ≈ 6-8 runs ⇒ 25-40 h | BLOCKED(dataset license) — feasible: search ≈ 1-2 h + seeds ≈ 1-3.5 h `[E]` | BLOCKED(dataset license) — ≈ 2-4 h total `[E]`; gated by corpus hosting terms |
| `phobert-base-frozen` | RUN-NOW (wiring) | BLOCKED(dataset license) — feasible ≈ 30-60 min `[E]` (the CPU fallback) | BLOCKED(dataset license) — ≈ 5-10 min `[E]` | Same, gated |
| `second-pretrained` (XLM-R-base / PhoBERT-large) | RUN-NOW (wiring only, if implemented) | BLOCKED(compute) | XLM-R-base: OPTIONAL (tight, batch 8); PhoBERT-large: BLOCKED(compute) — VRAM `[E]` | OPTIONAL — only after all mandatory cells + owner approval + hosting gate |

Rule the table encodes: synthetic-fixture cells are code-correctness smoke
tests (category C) and are **never** quoted as model results; only real-data
cells can produce reportable metrics, and all of them wait on the license.

## 5. The smallest scientifically valuable experiment

1. After the integrity gate passes, fine-tune PhoBERT-base once — seed 42,
   fixed recipe a priori (LR 3e-5, batch 16, ≤ 5 epochs, early stopping on
   validation macro-F1), no hyperparameter search.
2. Evaluate it and the Phase 1 TF-IDF+LogReg winner on validation, then on
   test **once each**, with the same metric code, splits, and full
   provenance.
3. Report both under single-seed claim phrasing ("in this run", no variance
   language); escalate to the full matrix (E4 search + E5 seeds) only if the
   owner wants the stronger claim.

Why it qualifies: it answers the research question directionally with one
real deep-model number against the strong classical baseline under the full
protocol — ≈ 15-40 min of GPU time `[E]` after data unblocks. What it cannot
support: variance claims, hyperparameter-sensitivity claims, or any
"significantly better" language.

## 6. Objective Phase 2 completion criteria

Phase 2 is complete when all of the following hold — not "when the numbers
look good":

1. **Data**: signed agreement on file; corpus delivered; integrity gate
   (blueprint §3.2) passed; label-semantics audit passed (≥ 90% per raw
   value, no systematic inversion) with the report committed; fingerprints
   recorded.
2. **Mandatory cells executed under protocol**: E2, E3, E4+E5 on the local
   GPU (or E6 with explicit, documented `BLOCKED(compute)` evidence for
   E4/E5 — e.g. sustained hardware failure logs), E8, E9.
3. **Seeds**: ≥ 3 fine-tuning seeds completed and reported, or the
   documented fallback claim phrasing used.
4. **Provenance**: every published number carries the full metadata of
   `evaluation-protocol.md` §8; any number without it is removed or labeled
   anecdotal.
5. **Reporting**: all tables category-labeled (A-E); test-evaluation count
   stated and equal to final-configuration seed count; no test-based
   selection anywhere in the logs.
6. **Claims**: every comparative statement satisfies the phrasing rules of
   `evaluation-protocol.md` §6.
7. **Optional cells decided explicitly**: run (with owner approval) or
   skipped (with reason) — never silently dropped.

## 7. Rules for comparative claims (when may we say A > B)

Model A may be described as **better than** model B only when **all** hold:

1. Same protocol for both: official splits, integrity gate, same metric
   code, same label space, selection on validation only
   (`experiment-blueprint.md` §9).
2. Both results are category-B (final test) numbers, produced after
   selection froze.
3. Variance handled: for the deep side, ≥ 3 seeds; the direction
   (A's mean > B's value) holds in **every** seed, and the gap between
   means exceeds the measured seed std. For classical-vs-classical,
   determinism makes single runs sufficient.
4. If 3 fails — overlapping ranges, or single-seed only — the only allowed
   wording is "comparable within measured variance" / "in this run, X vs Y".
5. No significance language ("significant", "p < 0.05") exists in this
   protocol except the optional descriptive McNemar test on existing paired
   test predictions, which may support an error-pattern direction but never
   a headline.
6. Claims are scoped: "on UIT-VSFC student feedback, under this protocol" —
   never "Vietnamese sentiment analysis" in general, and never presented in
   the same row as author-reported context numbers (category E).

## 8. Escalation and decision rules

- **License unblocks** → E0 already done at kickoff → E1 → E2/E3 same day
  `[E]` → E4 (1-2 h) → E5 (1-3.5 h) → E8/E9 → report. Calendar cost on the
  local GPU: ≈ 3-6 h of compute spread over 1-2 days, `[E]`.
- **Local GPU fails mid-protocol** → finish with E6 (frozen variant), report
  E4/E5 as `BLOCKED(compute)` with evidence, scope claims accordingly.
- **Cloud desired** → first check the signed agreement for hosting
  restrictions; prohibited ⇒ cloud cells `BLOCKED(license)`; permitted ⇒
  cost `[ESTIMATE]` $0 (free tiers) to ≈ $10-20 (Colab Pro compute units)
  for the whole Phase 2 — a few GPU-hours total. Budget approval is the
  owner's decision.
- **Optional cells** → only after mandatory cells complete and the owner
  explicitly approves the extra budget; default is skip, and skipping is a
  documented decision, not an omission.
