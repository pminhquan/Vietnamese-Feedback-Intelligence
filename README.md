# Vietnamese Feedback Intelligence (VFI)

Production-inspired, reproducible Vietnamese NLP system that analyzes
customer/student feedback. Given Vietnamese feedback text, VFI predicts the
sentiment class (`positive` / `neutral` / `negative`) with class probabilities
when supported, plus model identity and proper error handling.

## Scope

This repository currently targets **Phase 1** only:

- A real, license-verified Vietnamese feedback dataset (candidate: UIT-VSFC).
- A reproducible data pipeline (loading, validation, splitting, leakage checks).
- Classical ML baselines: majority-class classifier and TF-IDF + Logistic
  Regression.
- Correct classification metrics (macro-F1 as the primary selection metric).
- A minimal local inference CLI.
- Focused tests and CI.

Deep-learning models (PhoBERT), API serving, and dashboards are explicitly
**out of scope** for Phase 1 and will be introduced in later phases.

## Status

Work in progress. Phase 1 is being developed on a feature branch and will land
via a reviewed pull request. See `docs/` for the project specification,
experiment protocol, and roadmap.

## Requirements

- Python 3.12
- Windows PowerShell (setup instructions provided), but any OS works

## Development

Setup and usage instructions will be finalized together with Phase 1.
The authoritative workflow is described in `HANDOFF.md` (added on the feature
branch).
