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

## Dataset status (important)

**The project is dataset-independent** (owner decision: no author email, no
corpus agreement for UIT-VSFC). Datasets are selected through
`data.dataset_adapter` — the historical UIT-VSFC layout, five-level CSV
exports (documented `1,2→neg / 3→neu / 4,5→pos` mapping), NEU-ESC-style
4-class CSV (Toxic stays a DISTINCT class; exclusion/relabeling requires an
explicit written justification), or the **independent custom dataset** built
with `vfi dataset init/add-synthetic/annotate/report` (human-annotation
honesty invariants: synthetic labels can never be approved without personal
human review). Verified landscape and recommendation:
[`docs/dataset-strategy.md`](docs/dataset-strategy.md).

The UIT-VSFC raw-label encoding remains UNRESOLVED (two public mirrors
disagree); training through the `uit_vsfc` adapter is therefore refused
until an explicit, evidenced label mapping is configured. Its license still
forbids casual download — see [`docs/dataset-uit-vsfc.md`](docs/dataset-uit-vsfc.md).

**Consequently, no model metrics are reported in this repository yet.** The
entire pipeline is implemented and tested on deterministic synthetic
fixtures; real experiments run locally once a legitimately licensed dataset
is in place.

## Setup (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Verify (lint + format + tests + structural checks)

```powershell
python scripts\verify.py
```

## CLI

```powershell
# Validate raw data and report duplicates/leakage/distribution
vfi prepare-data --config configs\baseline.yaml

# Train baselines (fit on train, select on validation; test untouched)
vfi train --config configs\baseline.yaml

# Evaluate a trained artifact on the official test split
vfi evaluate --model-id tfidf-logreg --config configs\baseline.yaml

# Single-text prediction from a locally trained artifact
vfi predict --text "Giảng viên giải thích rất dễ hiểu"
```

## Documentation

| Document | Purpose |
| --- | --- |
| [`docs/project-spec.md`](docs/project-spec.md) | Scope, architecture, acceptance criteria |
| [`docs/experiment-protocol.md`](docs/experiment-protocol.md) | Split discipline, metrics, latency protocol, reproducibility |
| [`docs/dataset-uit-vsfc.md`](docs/dataset-uit-vsfc.md) | Dataset investigation, license findings, unblocking steps |
| [`docs/roadmap.md`](docs/roadmap.md) | Phase 1-4 plan (Phase 2+ not started) |
| [`docs/learning-notes.md`](docs/learning-notes.md) | Owner learning notes in Vietnamese + exercises |
| [`HANDOFF.md`](HANDOFF.md) | Reproduction instructions for developers |
| [`AGENTS.md`](AGENTS.md) | Ground rules for AI assistants |

## Development principles

- Real evaluation evidence over visual complexity; no invented numbers.
- Train/validation/test discipline; TF-IDF vocabulary learned from training
  data only.
- Vietnamese-safe preprocessing: Unicode NFC + whitespace collapse; no
  stopword removal, no diacritic stripping.
- Artifacts carry provenance metadata; loading refuses untrusted artifacts.
