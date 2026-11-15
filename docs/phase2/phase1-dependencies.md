# Phase 1 Dependencies — What Phase 2 Reuses, Changes, and Must Fix

Prepared by: Research Agent E (Security, Privacy & Failure-Mode Review), Phase 2 readiness program.
Date: 2026-11-07.
Reviewed Phase 1 ref: `95f7687735c785367fa0a6f9b1efbf0e284aac7f` (head of
`phase1/classical-ml-baselines`, Draft PR #1: OPEN, MERGEABLE, CI green — verified via
`gh pr view 1` at the time). Historical file:line citations refer to that commit.

Current baseline as of 2026-11-15: released `main` at
`db9b921791832337ac0536ba5b1ae03f664ba2d0`. The released implementation
supersedes the historical findings below where the integrated code already
addresses them.

Companion document: `risk-register.md` (Phase 2 threat model).

## 1. What Phase 2 REUSES from Phase 1 (as-is or near-as-is)

### 1.1 Data layer — reuse directly

| Asset | Location | Reuse notes |
| ----- | -------- | ----------- |
| `Record` dataclass, `SPLITS` | `loading.py:23, 39-45` | The universal data currency; Phase 2 dataset builders must consume `Record`s, not raw files. |
| `normalize_text` (NFC + whitespace) | `loading.py:48-59` | Reuse unchanged as the shared normalization for fingerprinting/leakage comparisons; it is deliberately Vietnamese-safe (no lowercasing/diacritic stripping). |
| `load_dataset_for_config` | [`adapters.py`](../../src/vfi/data/adapters.py) | Use the selected local adapter. Default calls open train and validation only. Evaluation requests the holdout explicitly. |
| `CANONICAL_LABELS` / `LABEL_TO_ID` | [`loading.py`](../../src/vfi/data/loading.py) | Derive canonical IDs here. The adapter supplies verified source-specific mapping and policy provenance. |
| `validate_records` | `validation.py:16-29` | Pre-training schema gate; call on every split before building a HF `Dataset`. |
| `class_distribution` | `validation.py:62-65` | Sanity reporting per split. |
| `fingerprint` | [`validation.py`](../../src/vfi/data/validation.py) | Order-sensitive SHA-256 of effective text and label pairs. Retain it alongside adapter identity and source digests. |
| `find_duplicates` / `find_leakage` | `validation.py` | Exact-match and contradiction checks. The bounded Phase 1 similarity sample is advisory. |
| `stratified_split` | [`splitting.py`](../../src/vfi/data/splitting.py) | Seeded grouped fallback. Identical normalized texts stay together. Review other grouping units before freezing real splits. |

### 1.2 Evaluation layer — reuse directly

| Asset | Location | Reuse notes |
| ----- | -------- | ----------- |
| `compute_metrics` | `metrics.py:22-60` | Exact metric contract for the Phase 1↔Phase 2 comparison: accuracy, macro-F1 primary, per-class P/R/F1/support, 3x3 confusion matrix in canonical order, `zero_division=0` explicit. Phase 2 must use this unchanged so numbers are comparable (`docs/roadmap.md:22-23`). |
| `measure_latency` | `metrics.py:63-105` | Documented batch-1 latency protocol. Reusable for transformer inference; consider raising `timed_calls` (p95 over 100 samples has 5-sample resolution) for the slower model. |

### 1.3 Config pattern — reuse the pattern, extend the schema

| Asset | Location | Reuse notes |
| ----- | -------- | ----------- |
| Frozen dataclasses + YAML `safe_load` + explicit validation | `config.py:20-165` | The pattern (typed config, range checks, `ConfigError`, `config_hash` short sha256 at `config.py:69-72`) is exactly right for Phase 2's `TrainingConfig` (model id/revision, epochs, lr, batch size, fp16, seed, device policy). Additive change; do not fork the loader. |
| `configs/baseline.yaml` shape | `configs/baseline.yaml` | Additive sections (`models.phobert:` etc.) keep one-file-per-experiment convention. |

### 1.4 CLI and artifact conventions — reuse conventions, new subcommands

| Asset | Location | Reuse notes |
| ----- | -------- | ----------- |
| argparse subcommand pattern, exit codes (0/1/2), `_fail`, JSON-to-stdout | `cli.py:37-47, 255-291` | Phase 2 adds subcommands (e.g. `vfi train-transformer`, `vfi evaluate-transformer`) following the same shape. |
| Report-writing convention: JSON to `reports/artifacts/` (gitignored), committed prose to `reports/` | `cli.py:96-99, 169-184`; `reports/README.md` | Phase 2 evaluation reports follow the same split; per-example error analysis stays in the gitignored dir (risk R-05). |
| Metadata sidecar provenance | `baseline.py:226-243` | `build_training_metadata` pattern (created_at, config, config_hash, dataset fingerprints, library versions, seed) extends naturally with torch/transformers versions, checkpoint revision SHA, and segmentation tool version (risks R-01/R-10/R-12). |
| Preflight and training provenance | [`cli.py`](../../src/vfi/cli.py) | Bind the dataset and test fingerprint to a passing pre-fit receipt. Keep test out of fitting and selection. |

### 1.5 Verification gate and repo hygiene — reuse, then harden

| Asset | Location | Reuse notes |
| ----- | -------- | ----------- |
| `scripts/verify.py` (ruff + format + pytest + structural checks) | [`verify.py`](../../scripts/verify.py) | Verify both suites and tracked-file guards. The CI workflow also runs the structural gate. |
| `.gitignore` coverage and data layout | [`.gitignore`](../../.gitignore), [data layout](../../data/README.md) | Phase 1 now covers torch-era weight extensions, local datasets, reports and secret files. |
| Honesty rules / AGENTS boundaries | `docs/experiment-protocol.md:73-79`; `AGENTS.md:8-34` | Binding on all Phase 2 agents; note rule 8 (`AGENTS.md:29-30`) currently *forbids* PyTorch/Transformers in Phase 1 work — the Phase 2 stack lands on Phase 2 branches only. |

## 2. What Phase 2 must CHANGE or ADD

### 2.1 Dependencies (new)

- `torch` (pin exact; ≥2.6 so `torch.load` defaults to `weights_only=True` — risk R-01),
  `transformers` (pin exact), `accelerate` (pin exact), `safetensors`.
- Segmentation: `underthesea` (pip-delivered) or VnCoreNLP (Java + jar download — prefer to
  avoid; see risk R-12). Confirm against the chosen checkpoint's model card whether segmentation
  is needed at all before adding it [UNVERIFIED].
- Optional: `datasets` is *not* needed (script-based mirror would fail on `datasets>=3.0`
  anyway; local-file loading is the established path — risk R-02).
- Everything recorded in `build_training_metadata`-style provenance (risk R-10).

### 2.2 Configuration (new `TrainingConfig` section)

- GPU-aware fields: device policy (cpu/cuda/auto), fp16/bf16, batch size, gradient accumulation,
  epochs, learning rate, warmup, weight decay, early-stopping patience — with Phase 1-style
  range validation in `config.py` fashion.
- Roadmap constraint: GPU feasibility must be assessed first; CPU fallback (smaller model or
  frozen embeddings) must remain expressible in config (`docs/roadmap.md:24-25`).

### 2.3 Data pipeline additions

- Record a near-duplicate audit across all declared split pairs before fitting.
  The bounded Phase 1 train/validation sample does not cover this requirement.
- Use the same frozen grouped splits and mapping as the actual Phase 1 baselines.
- Retain adapter provenance and dataset-specific access references. An agreement
  reference applies only where the chosen source requires one.

### 2.4 Artifact types (new)

- HF-style checkpoint directories (`config.json`, `model.safetensors`, tokenizer files) under
  `models/<model_id>/` following the existing `save_model` layout (`baseline.py:160-183`) —
  but serialized with safetensors, not joblib/pickle (risks R-01/R-03).
- Confirm new outputs remain covered by the released structural guard before
  any checkpoint is acquired (risk R-08).
- Test-set evaluation ledger (`reports/artifacts/test_ledger.jsonl`) — new, enforces risk R-09.

### 2.5 Metrics (no change)

- `compute_metrics` is the contract; only latency parameters may need tuning. No new metric
  should displace macro-F1 for model selection.

## 3. Historical findings from the pre-release Phase 1 review

Severity scale: Critical / High / Medium / Low (same qualitative model as the risk register).
None of these are correctness bugs in Phase 1's own scope; they are gaps that matter for
Phase 2. Listed in priority order at the time of the review. The released
code adds checksum and config validation, privacy-safe reports, grouped
splitting, explicit UIT-VSFC mapping, split provenance and Git/CI protection.

| # | Location | Severity | Issue | Why it matters | Suggested direction (Codex-anti owns the change) |
| --- | -------- | -------- | ----- | -------------- | ------------------------------------------------ |
| 1 | `baseline.py:222` (with `:178`) | High | `load_model` deserializes `model.joblib` (pickle) behind advisory-only checks | Pickle load = arbitrary code execution if `models/` is ever poisoned; the "trusted-artifact policy" (`:189-198`) cannot detect a re-stamped sidecar + swapped pickle | Add artifact integrity: sha256 of `model.joblib` recorded at save, verified at load, hard fail on mismatch. Keep joblib strictly for sklearn pipelines |
| 2 | `validation.py:43-59` (+ `loading.py:48-59`) | High | Leakage/duplicate detection is exact-match after NFC+whitespace only; near-duplicates undetected (acknowledged at `:52-53` and `project-spec.md:85-86`) | Fine-tuned transformers memorize near-duplicates; test metrics will be inflated invisibly (risk R-06) | Add a separate near-duplicate audit function (n-gram/MinHash) + report rates; keep exact-match behavior and tests intact |
| 3 | `baseline.py:212-213` | Medium | `load_model` checks only that `config_hash` is *present*, never that it matches the current config | An artifact trained under different hyperparameters loads silently; predictions attributed to the wrong config | Recompute `config_hash` from the loaded config (or pass the expected hash) and fail on mismatch |
| 4 | `baseline.py:160-183` | Medium | `save_model` writes no integrity hash of the artifact into `meta.json` | Load-time tamper detection impossible; foundation for issue #1 | Record `sha256(model.joblib)` + file size in metadata at save time |
| 5 | `cli.py:83-91` (written `:99`, printed `:101`) | Medium | `data_report.json` embeds the literal text of every cross-split duplicate (and prints to stdout) | Raw student feedback lands in a JSON artifact + console; normalized privacy-exposure pattern for Phase 2's heavier reporting (risk R-05) | Report counts + text-hash prefixes (e.g. 12 hex of sha256) instead of raw strings |
| 6 | `.gitignore:33-37` | Medium | No ignore rules for `*.bin`, `*.pt`, `*.ckpt`, `*.h5`, `*.onnx`; `/models/` is root-anchored only | HF `pytorch_model.bin` / PEFT `adapter_model.bin` are committable anywhere in the tree; the exact Phase 2 artifact formats are uncovered (risk R-08) | Add the extensions (plus any intentional-fixture negations); commit before the first HF download |
| 7 | `verify.py:56-60` | Medium | Structural gate checks only 3 hardcoded paths and tracked-patterns `*.joblib`/`*.pkl` | `*.safetensors`, `*.bin`, `*.pt`, `.env` tracked-file drift undetectable; gate silently weaker than the gitignore intent | Extend `check_no_tracked` to `*.safetensors`, `*.bin`, `*.pt`, `*.ckpt`, `.env*` |
| 8 | `splitting.py:41-46` | Medium | Fallback `stratified_split` does not deduplicate (or group duplicates) before splitting | Duplicate texts can straddle train/validation in the `use_official_splits: false` path, corrupting selection signals | Group records by normalized text before splitting; split by group, then expand |
| 9 | `loading.py:26` (+ `dataset-uit-vsfc.md:37-41`, `data/README.md:29-30`) | Medium | Raw label mapping `0=positive, 1=negative, 2=neutral` is sourced from the SEACrowd `ClassLabel` declaration and is unverified against the corpus author's README | Single silent-inversion point for all labels (risk R-07); inverted mapping yields plausible-looking but wrong metrics | Not a code fix: owner must verify on data receipt; add a recorded verification note + (optionally) a prepare-data distribution sanity assertion |
| 10 | `pyproject.toml:13-19` | Low (High if replicated in Phase 2) | Lower-bound-only pins (`scikit-learn>=1.4`, etc.), no lockfile | Environment drift changes predictions/reproducibility; sklearn itself warns about cross-version prediction changes (`baseline.py:214-221`) | Pin exact versions when adding the Phase 2 stack; consider a lockfile; extend metadata with all library versions (risk R-10) |
| 11 | `baseline.py:214-221` | Low | sklearn version check warns on ANY version difference while the message talks about "major versions" (noisy on patch diffs, ignorable at warn level) | Warning fatigue weakens the one drift signal that exists | Keep strict comparison but make the message reflect the actual comparison; consider escalating to a hard error on major-version mismatch |
| 12 | `cli.py:65-102` | Low | `cmd_prepare_data` always exits 0, even with leakage present | No mechanical gate for automation/CI to key on; leakage is report-only | Add `--fail-on-leakage` (default off) or a dedicated gate command for Phase 2 CI use |
| 13 | `.github/workflows/ci.yml:9-32` | Low | No structural-check step in CI (verify.py is local-only); no `permissions:` block; actions pinned by tag (`@v4`/`@v5`), not SHA | The artifact-exclusion gate does not run where merges happen; broad default token permissions; mutable action tags (risk R-08/R-13) | Add a structural-checks step; add `permissions: contents: read`; optionally SHA-pin actions |
| 14 | `cli.py:223-252` | Low | `vfi predict` echoes full input text to stdout and has no input length cap | Minor privacy surface (console logs); unbounded input is a Phase 3 serving concern (roadmap) | Document; length cap belongs to Phase 3 input limits — no Phase 1 change needed |

### Review-scope notes

- Git history across **all** refs was scanned for ever-committed data/artifact/secrets files
  (`git log --all --name-only` filtered on artifact extensions): clean [VERIFIED FACT].
- Phase 1 has **no real-data metrics anywhere** (dataset access blocked; synthetic fixtures
  only — `HANDOFF.md:10-12`, `project-spec.md:73-74`), so the "misleading accuracy claims"
  exposure is currently process-level only (risk R-11, Medium).
- `AGENTS.md:29-30` explicitly bars PyTorch/Transformers from Phase 1 work: the dependency
  additions in §2.1 must land on Phase 2 branches, not as part of PR #1.

## 4. Blocking items for Phase 2 start

1. **Dataset access and real baseline evidence:** select an authorized source,
   document mapping or human annotation, create physical splits, pass preflight,
   and fit both classical baselines before beginning the deep arm.
2. **Artifact-format checklist:** confirm the planned paths match the released
   ignore, structural and CI protections before a checkpoint is acquired.
3. **Checkpoint policy decision:** which pretrained checkpoint(s), pinned revision, safetensors
   availability (risk R-01); owner approval required.
4. **GPU feasibility assessment** before committing to a fine-tuning plan (`docs/roadmap.md:24-25`).
5. **Near-duplicate audit** planned as a prerequisite before the first test-set evaluation
   (issue #2, risk R-06) — can be prepared on synthetic data in advance.
