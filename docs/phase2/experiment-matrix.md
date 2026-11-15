# Phase 2 Experiment Matrix

Status: **design only**. Real model runs need a selected, authorized dataset,
verified mapping, a passing integrity preflight, and a separate holdout.
Use independent human-approved JSONL or an authorized local CSV adapter.
UIT-VSFC remains optional and agreement-gated. All runtimes and costs below
are planning estimates. **M1 is the three-model pilot** (majority,
TF-IDF+LogReg, and one fixed-recipe PhoBERT run at seed 42), as described in
[pilot-experiment-plan.md](pilot-experiment-plan.md). E4 search and E5 extra
seeds are M2 escalation. XLM-R and PhoBERT-large are optional.

## 1. Status legend

| Tag | Meaning |
| --- | ------- |
| `RUN-NOW` | Synthetic software checks that need no real dataset. Checkpoint-dependent smoke tests still need checkpoint and stack approval. |
| `BLOCKED(data readiness)` | Needs the selected authorized dataset, its verified label mapping, and an integrity-checked holdout. |
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
| VnCoreNLP / py_vncorenlp | VnCoreNLP tool + pre-trained models: **GPLv3** (`LICENSE.md`) — acceptable for local research use, re-examine before Phase 3 serving (`tokenization-design.md` §4) | GitHub `vncorenlp/VnCoreNLP` |
| underthesea | **Apache-2.0** on the main repo (`LICENSE`) — the license-safe default; pin and record version if used | GitHub `undertheseanlp/underthesea` |
| Historical UIT-VSFC scale | ≈16k sentences; train 11,426 / validation 1,583 / test 3,166 per the authors' `uitnlp` card `[VERIFIED FACT]`, fetched 2026-11-07; used only for the planning arithmetic below | `uitnlp` card |
| Optional UIT-VSFC adapter | Requires the signed agreement and verified mapping when selected | [Phase 1 dataset notes](../dataset-uit-vsfc.md) |

**Historical-size arithmetic for one fine-tune run** (`[ESTIMATE]`, fp16, seq 256,
micro-batch 8 × grad-accum 4 = effective batch 32, RTX 3060 Laptop — the M1
pilot geometry from `pilot-experiment-plan.md` §5):

- Micro fwd/bwd steps/epoch ≈ ceil(11,426 / 8) ≈ 1,429; optimizer updates
  ≈ ceil(11,426 / 32) ≈ 357; ≤ 5 epochs ⇒ ≤ 1,785 optimizer updates.
- Transformer fwd+bwd ≈ 6 × 135M × 8 × 256 ≈ 1.7e12 FLOPs per micro-step;
  at an assumed sustained 5-8 TFLOPS (20-30% MFU on this card) ⇒ ≈ 0.2-0.35
  s/micro-step ⇒ ≈ 5-9 min/epoch (the independent bottom-up estimate in
  `compute-feasibility.md` §3.2, ≈40-70 samples/s, brackets the same range)
  ⇒ **≈ 20-45 min per run** including per-epoch validation inference
  (1,583 sentences). Uncertainty ≈ 2× (thermals, background load).
- CPU-only (i5-10500H): assume 10-25× slower ⇒ **≈ 3.5-19 h per run**.

The downstream estimates assume this historical size. Recalculate time
and step counts using the selected dataset before planning a real run.

## 3. Compute scenarios

| Scenario | Hardware | Notes |
| -------- | -------- | ----- |
| **local-CPU** | i5-10500H, 12 threads | The Phase 1 scenario; fine for classical models, insufficient for the fine-tuning protocol |
| **local-GPU** | RTX 3060 Laptop 6 GB | Primary Phase 2 scenario; sequential runs only (no parallel training on 6 GB) |
| **cloud-GPU** | E.g. Colab T4 (16 GB) / Kaggle | Identical protocol; **legal gate first**: verified terms for the selected dataset and explicit owner approval must allow third-party hosting. Otherwise, cloud cells become `BLOCKED(license)` |

## 4. The matrix

`[E]` marks an estimate. "Protocol-valid" means the same frozen split fingerprints and adapter
provenance for all arms, a passing preflight before fit, selection on
validation only, and complete reproducibility evidence
(`evaluation-protocol.md` §8).

| # | Experiment | Needs dataset | local-CPU | local-GPU (6 GB) | cloud-GPU |
| - | ---------- | ------------- | --------- | ---------------- | --------- |
| E0 | **(M1)** Phase 2 code smoke tests on synthetic fixtures (loader→segmenter→trainer→metrics wiring; VRAM probe with a hand-written sentence) | No | **RUN-NOW** (sklearn-only parts) — minutes `[E]`; torch parts gated on G3+G4 | **RUN-NOW** once G3+G4 close (checkpoint download ~1.2 GB + forward/backward probe) — < 30 min `[E]` | Not useful |
| E1 | **(M1)** Dataset integrity preflight: verify mapping, schema, effective and source fingerprints, and leakage | Yes (real files) | **BLOCKED(data readiness)** — minutes once unblocked `[E]` | n/a | n/a |
| E2 | **(M1)** Majority baseline on real data (control floor) | Yes | **BLOCKED(data readiness)** — seconds-minutes `[E]` | n/a | n/a |
| E3 | **(M1)** TF-IDF+LogReg on real data, Phase 1 protocol (4 candidates, select on validation). Freeze both classical artifacts and real validation reports before E4p. | Yes | **BLOCKED(data readiness)** — ≈ 10-20 min `[E]` (4 fits of a sparse linear model on ≈11.4k docs) | n/a | n/a |
| E3v | **(M1, OPTIONAL variant)** `tfidf-logreg-seg`: Phase 1-selected candidate re-run once on segmented text (no re-search) | Yes | **BLOCKED(data readiness)**, OPTIONAL — minutes `[E]` | n/a | n/a |
| E4p | **(M1 — the pilot deep run)** PhoBERT-base fixed pre-declared recipe (LR 3e-5, effective batch 32 = micro 8 × accum 4, seq 256, ≤ 5 epochs, AMP, seed 42), after E2/E3 validation evidence; select best on validation; single test evaluation after all final artifacts are frozen | Yes | **BLOCKED(data readiness)** + **BLOCKED(compute)** — ≈ 3.5-19 h `[E]`, emergency fallback only | **BLOCKED(data readiness)** — ≈ 20-45 min once unblocked `[E]` | **BLOCKED(data readiness)** — ≈ 20-45 min `[E]`; gated by dataset-specific hosting terms |
| E4 | **(M2)** PhoBERT-base fine-tune search: 3 candidates (LR {2e-5, 3e-5, 5e-5}), seed 42, early stopping ≤ 5 epochs | Yes | **BLOCKED(data readiness)** and **BLOCKED(compute)** — ≈ 3 × 3.5-19 h ≈ 10-57 h `[E]`, no seed repetition possible in reasonable time | **BLOCKED(data readiness)** — ≈ 1-2.5 h total once unblocked `[E]` (3 × 20-45 min) | **BLOCKED(data readiness)** — ≈ 1-2.5 h `[E]`; gated by dataset-specific hosting terms |
| E5 | **(M2)** PhoBERT-base winning config, 3 seeds {42, 7, 2026} + single test evaluation per seed | Yes | **BLOCKED(data readiness)** + **BLOCKED(compute)** — ≈ 10-57 h `[E]` | **BLOCKED(data readiness)** — ≈ 1-2.5 h (3 seeds) `[E]` | **BLOCKED(data readiness)** — same order `[E]`, plus hosting gate |
| E6 | **(fallback)** Frozen PhoBERT encoder + logistic head (CPU-feasible fallback) | Yes | **BLOCKED(data readiness)** — ≈ 30-60 min encode + seconds head `[E]` | **BLOCKED(data readiness)** — ≈ 5-10 min `[E]` | Same, gated |
| E7a | **(OPTIONAL)** XLM-R-base (279M, raw-syllable input). Additional M2+ model or declared M1 fallback only with owner approval (`pilot-experiment-plan.md` §1.2) | Yes | **BLOCKED(compute)** — ≈ 8-20 h/run `[E]` | **OPTIONAL** — tight on 6 GB (batch 8 likely required); ≈ 30-60 min/run `[E]` | **OPTIONAL** — comfortable on 16 GB; hosting gate |
| E7b | **(OPTIONAL, M2+)** Second pretrained model: PhoBERT-large (370M) | Yes | **BLOCKED(compute)** | **BLOCKED(compute)** — optimizer state ≈ 6-7 GB exceeds 6 GB VRAM `[E]` | **OPTIONAL** — fits 16 GB `[E]`; only after E4+E5 + owner approval + hosting gate |
| E8 | **(M1)** Latency benchmarks (batch-1 protocol; classical on CPU; PhoBERT on CPU **and** GPU; segmentation cost reported separately) | Yes (real validation texts as samples) | **BLOCKED(data readiness)** — ≈ 15-30 min `[E]` (PhoBERT CPU inference ≈ 0.3-1 s/call × 110 calls `[E]`) | **BLOCKED(data readiness)** — minutes `[E]` | Same, if cloud used |
| E9 | **(M1)** Error analysis on validation (per-sample dump, cell counts, evidence-gated patterns) | Yes | **BLOCKED(data readiness)** — ≈ 1-2 h including write-up `[E]` | n/a (uses saved predictions) | n/a |

Real-data cells are blocked pending data readiness. The local GPU is the
planned pilot device once both classical baselines exist and the compute
stack is approved. The CPU deep run is an emergency option. M2 and optional
models require the owner's later approval.

### 4.1 The same matrix as model × dataset × compute

The experiment rows above, re-cut along the model dimension (real data =
a selected, legitimate dataset with a passed preflight and a separate
holdout; synthetic = local fixtures). Every cost is `[ESTIMATE]`, per §2's arithmetic.

| Model | Dataset: synthetic fixtures | Dataset: real — local-CPU | Dataset: real — local-GPU (6 GB) | Dataset: real — cloud-GPU |
| ----- | --------------------------- | ------------------------- | -------------------------------- | ------------------------- |
| `majority` | RUN-NOW (smoke test only, never a result) | BLOCKED(data readiness) — seconds `[E]` | n/a | n/a |
| `tfidf-logreg` | RUN-NOW (smoke test only) | BLOCKED(data readiness) — ≈ 10-20 min `[E]` | n/a | n/a |
| `phobert-base-ft` | RUN-NOW (wiring + VRAM probe, < 30 min `[E]`; torch parts gated on G3+G4) | BLOCKED(data readiness) + BLOCKED(compute) — ≈ 3.5-19 h/run `[E]`; M1 needs 1 run, M2 needs 6 ⇒ M1 ≈ 3.5-19 h, full protocol ≈ 25-90 h | BLOCKED(data readiness) — feasible: M1 single run ≈ 20-45 min `[E]`; M2 search + seeds ≈ 2-5 h `[E]` | BLOCKED(data readiness) — M1 ≈ 20-45 min, full ≈ 2-5 h `[E]`; gated by dataset hosting terms |
| `phobert-base-frozen` | RUN-NOW (wiring) | BLOCKED(data readiness) — feasible ≈ 30-60 min `[E]` (the CPU fallback) | BLOCKED(data readiness) — ≈ 5-10 min `[E]` | Same, gated |
| `second-pretrained` (XLM-R-base / PhoBERT-large) | RUN-NOW (wiring only, if implemented) | BLOCKED(compute) | XLM-R-base: OPTIONAL (tight, batch 8); PhoBERT-large: BLOCKED(compute) — VRAM `[E]` | OPTIONAL — only after all mandatory cells + owner approval + hosting gate |

Rule the table encodes: synthetic-fixture cells are code-correctness smoke
tests (category C) and are **never** quoted as model results; only real-data
cells can produce benchmark metrics, and those wait on data readiness.

### 4.2 Implementable now vs gated (short form)

What can be coded and tested **today** with synthetic fixtures in this
readiness branch using the released Phase 1 code versus what stays gated. Full step-level table
with evidence labels: `pilot-experiment-plan.md` §14.

| Pilot step | Can start today? | Gate |
| ---------- | ---------------- | ---- |
| Data-contract module + schema-violation tests (synthetic fixture files, Phase 1 loader reuse) | **Yes — codeable + executable now** (sklearn-only) | — |
| Training-log JSONL writer + log-hygiene (no-raw-text) tests | **Yes — codeable + executable now** | — |
| `meta.json` provenance extension + completeness test | **Yes — codeable + executable now** | — |
| Classical-arm wiring on synthetic records (Phase 1 `run_small_search`/`save_model`) | **Yes — codeable + executable now** | — |
| Seed utility + determinism test; tokenization chain + golden tests; PhoBERT training-loop code + OOM ladder | Codeable now (import-guarded) | Executable after **G4** (pinned stack); checkpoint-dependent code also **G3** |
| VRAM probe (replaces §2 estimates with measurements) | No | **G3 + G4** |
| E1 preflight and E2/E3 real classical fits | No | Selected authorized dataset, verified mapping and holdout, passing preflight before fitting |
| E4p/E8/E9 deep-arm work on real data | No | Real E2/E3 validation evidence and frozen artifacts, applicable release gates, and approved compute stack |

Honest reading: the scaffolding layer (data contract, logging, metadata,
tests, and even the training-loop code against synthetic tensors) is real
work available now; every number waits on the gates.

## 5. The smallest scientifically valuable experiment = milestone M1

This is the pilot formalized in [pilot-experiment-plan.md](pilot-experiment-plan.md):

1. Select one authorized dataset, verify labels and pass the all-split
   preflight. Fit the actual majority and TF-IDF+LogReg baselines, retain
   their validation metrics and provenance, and freeze both artifacts.
2. Fine-tune PhoBERT-base once after the baseline evidence exists, using the
   fixed recipe (seed 42, LR 3e-5, effective batch 32, sequence length 256,
   at most five epochs). Choose its checkpoint on validation macro-F1.
3. Once all final configurations are frozen, score the common holdout once
   per artifact (three test evaluations total) with the same metric code.
   Use single-seed phrasing for M1. M2 search and additional seeds require
   the owner's approval and a protocol that prevents prior test results from
   influencing new selection.

Why it qualifies: it answers the research question directionally with one
real deep-model number against the classical control floor and strong
baseline under the full protocol — ≈ 20-45 min of GPU time `[E]` after the
gates unblock. What it cannot support: variance claims,
hyperparameter-sensitivity claims, or any "significantly better" language.

## 6. Objective Phase 2 completion criteria

Phase 2 is complete when all of the following hold — not "when the numbers
look good":

1. **Data**: the selected dataset is locally available and authorized,
   its verified label mapping is documented, and a separate holdout is bound
   to a passing preflight. Retain the effective and source fingerprints and
   actual validation results of both Phase 1 baselines.
2. **Milestone M1 cells executed under protocol**: E1–E3 collect data and
   classical evidence locally on CPU. E4p uses local GPU, with E8/E9 on
   the devices specified in the matrix (or use E6 as an explicit, documented
   `BLOCKED(compute)` evidence for E4p — e.g. OOM-at-micro-batch-1 logs).
   **Phase 2 in full** additionally requires the M2 escalation: E4 search +
   E5 seeds, or the documented decision to stop at M1.
3. **Seeds**: M1 = single seed 42 with the single-seed claim phrasing; the
   variance-supported claim requires the M2 escalation (3 seeds
   {42, 7, 2026}) completed and reported.
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

1. Same protocol for both: identical authorized dataset version, verified
   mapping, preflight-bound splits, metric code, and validation-only selection
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
6. Name the selected dataset, version, domain, and split in each result
   claim. Keep author-reported context numbers in category E.

## 8. Escalation and decision rules

- **Data and baselines ready**: E1 preflight comes before E2/E3 fits.
  Retain both validation reports before E4p, which also needs the readiness
  gates and owner approval. Freeze all configurations before scoring.
  Complete E8/E9 and report M1. M2 needs owner approval and an evaluation
  plan that protects selection from any scores released during M1. Use a
  fresh holdout and matching baselines if prior test results informed
  selection.
- **Local GPU fails mid-protocol** → finish with E6 (frozen variant), report
  E4p/E4/E5 as `BLOCKED(compute)` with evidence, scope claims accordingly.
- **Cloud desired** → first verify that the selected dataset's terms and the owner
  permit hosting; prohibited ⇒ cloud cells `BLOCKED(license)`; permitted ⇒
  cost `[ESTIMATE]` $0 (free tiers) to ≈ $10-20 (Colab Pro compute units)
  for the whole Phase 2 — a few GPU-hours total. Budget approval is the
  owner's decision.
- **Optional cells** → only after mandatory cells complete and the owner
  explicitly approves the extra budget; default is skip, and skipping is a
  documented decision, not an omission.
