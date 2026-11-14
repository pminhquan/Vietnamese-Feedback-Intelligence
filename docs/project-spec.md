# VFI Project Specification — Phase 1

## Problem

Given a Vietnamese feedback sentence, predict its sentiment class:
`negative`, `neutral`, or `positive`. Predictions must be reproducible,
identify the model that produced them, and report class probabilities when
the model supports them.

## Scope

### In scope (Phase 1)

1. Dataset foundations: verified loading, validation, leakage checks, and
   fingerprints for the UIT-VSFC format.
2. Classical baselines: majority-class classifier and TF-IDF + Logistic
   Regression.
3. Correct classification metrics with explicit zero-division behavior;
   macro-F1 as the primary model-selection metric.
4. Split discipline: preflight checks split integrity through `vfi evaluate`.
   Test scoring follows final model selection.
5. A minimal local inference CLI (`vfi predict`).
6. Focused tests (synthetic fixtures only) and CPU-only CI.

### Explicitly out of scope (Phase 1)

- PhoBERT / any deep learning (Phase 2).
- FastAPI serving, dashboards, batch prediction, Docker (Phase 3).
- Topic classification and aspect-based sentiment (Phase 4, only if
  trustworthy topic labels are licensed for use).
- No public API, no web UI.

## Architecture

```text
src/vfi/
├── config.py               # dataclass configs + YAML loading/validation
├── data/
│   ├── loading.py          # UIT-VSFC file format -> Record objects; normalize_text
│   ├── validation.py       # duplicates, cross-split leakage, distribution, fingerprints
│   └── splitting.py        # deterministic stratified fallback split
├── models/
│   └── baseline.py         # MajorityBaseline, TF-IDF+LogReg pipeline, small search,
│                           # artifact save/load with metadata sidecar
├── evaluation/
│   └── metrics.py          # metrics + documented latency protocol
└── cli.py                  # prepare-data | train | evaluate | predict
```

Design principles:

- One canonical label space (`negative=0, neutral=1, positive=2`); the raw
  UIT-VSFC integer mapping (`0=positive, 1=negative, 2=neutral`) is converted
  at load time in one documented place.
- Minimal, Vietnamese-safe normalization: Unicode NFC + whitespace collapse.
  No lowercasing in the pipeline input (TF-IDF handles it), no stopword
  removal, no diacritic stripping — those carry Vietnamese sentiment signal.
- Models are single sklearn `Pipeline` objects (vectorizer + classifier), so
  training-time preprocessing and inference-time preprocessing cannot drift.
- Artifacts are `models/<model_id>/model.joblib` plus `meta.json` with full
  provenance (config hash, dataset fingerprints, library versions, seed).
  Loading checks the sidecar and checksum before deserializing. Load only your
  own local artifacts because Joblib can execute code during deserialization.

## Acceptance criteria (Phase 1)

- [x] Repository safely bootstrapped; main + feature branch + Draft PR.
- [x] Reproducible Python package (`pip install -e ".[dev]"`).
- [x] Dataset candidate evaluated for license/provenance with a precise,
      documented blocker (see `docs/dataset-uit-vsfc.md`).
- [x] Data pipeline with automated regression tests (synthetic fixtures).
- [x] Both classical baselines implemented and tested.
- [x] Metrics correctness verified against hand-computed examples.
- [ ] Real experiment results — **blocked** by dataset license; no numbers
      are reported until data access is legitimately obtained.
- [x] CLI inference works against a trusted local artifact with current metadata.
- [x] CI green on the Draft PR head.
- [x] README/HANDOFF sufficient for another developer to reproduce.
- [x] No dataset files, model weights, or secrets committed.
- [x] Owner learning exercises present (`docs/learning-notes.md`).

## Known limitations

- No real-data experiment results yet; the pipeline is validated on
  synthetic fixtures only. Unit tests passing does NOT measure model quality.
- Cross-split leakage detection is exact-match after normalization;
  near-duplicate leakage is not detected.
- ` MajorityBaseline` probabilities are degenerate one-hot values, reported
  as-is for honesty.
- The hyperparameter search space is intentionally small (4 candidates);
  it is a demonstration of disciplined search, not exhaustive tuning.
