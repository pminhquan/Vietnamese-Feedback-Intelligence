# Vietnamese Feedback Intelligence (VFI)

Production-inspired, reproducible Vietnamese NLP system that analyzes
customer/student feedback. Given Vietnamese feedback text, VFI predicts the
sentiment class (`positive` / `neutral` / `negative`) with class probabilities
when supported, plus model identity and proper error handling.

## Scope

This repository currently targets **Phase 1** only:

- Local adapters for authorized Vietnamese sentiment datasets and feedback
  collected and labeled by the owner.
- A reproducible data pipeline (loading, validation, splitting, leakage checks).
- Classical ML baselines: majority-class classifier and TF-IDF + Logistic
  Regression.
- Correct classification metrics (macro-F1 as the primary selection metric).
- A minimal local inference CLI.
- Focused tests and CI.

Deep-learning models (PhoBERT), API serving, and dashboards are explicitly
**out of scope** for Phase 1 and will be introduced in later phases.

## Dataset choices

The default configuration expects an independent, human-approved JSONL dataset.
You can also use an authorized five-level sentiment CSV or a local NEU-ESC CSV
export. UIT-VSFC remains an optional adapter. It requires a signed corpus
agreement and independently confirmed integer-label mapping before use.
NEU-ESC requires the publisher's access acceptance and a review of permitted
usage before any real files are used. Set `data.rights_confirmed: true` only
after verifying your rights. Adapters read local files.

There are no real benchmark results yet. Synthetic tests check integration.

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

## Build an independent dataset

Collect feedback you have permission to use. Remove unnecessary personal
details before entry, and use short non-sensitive dataset identifiers. The
annotator displays record text locally when reviewing a row. Avoid recording or
sharing that terminal. Store all datasets under an ignored local path.

```powershell
$dataset = "data/datasets/owner_feedback/v1"
vfi dataset init --root $dataset --dataset-version v1
vfi dataset add --root $dataset --dataset-version v1
vfi dataset annotate --root $dataset --dataset-version v1
vfi dataset approve --root $dataset --dataset-version v1
vfi dataset validate --root $dataset --dataset-version v1
vfi dataset split --root $dataset --dataset-version v1 --out "$dataset/splits"
```

Repeat data entry, annotation and review until there are enough distinct
examples of each class for three splits. CLI help lists record selection
options. Approval follows human labeling. A correction requires renewed
approval. The split command preserves exact duplicate groups, and exports files
for fit, selection and holdout under a versioned directory. Synthetic candidates
stay marked synthetic after review. The explicit fixture mode is for smoke tests.

## Select and run an adapter

Set `raw_dir`, `dataset_adapter`, `dataset_id`, and `dataset_version` in the
`data` section of [`configs/baseline.yaml`](configs/baseline.yaml). Set
`rights_confirmed: true` when that dataset is authorized. Use separate files for
train, validation and test. See [file layouts and label policies](data/README.md).
False `use_official_splits` derives validation from the physical train source.
A test benchmark requires an independently declared holdout.

```powershell
# Summarize train/validation and check all split pairs before fitting
vfi prepare-data --config configs\baseline.yaml
vfi evaluate --preflight-only --config configs\baseline.yaml

# Train baselines (fit on train, select on validation; test untouched)
vfi train --config configs\baseline.yaml

# Evaluate a trained artifact after finalizing validation-only selection
vfi evaluate --model-id tfidf-logreg --config configs\baseline.yaml

# Single-text prediction from a locally trained artifact
vfi predict --text "Giảng viên giải thích rất dễ hiểu"
```

The evaluation report's macro-F1 weights each of the three sentiment classes
equally on the declared holdout. To compare runs, retain the test fingerprint
and dataset mapping, policy and version. A changed snapshot requires a new
preflight and run. The [experiment protocol](docs/experiment-protocol.md)
explains split checks, provenance and model loading.

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
- Load Joblib artifacts only from your own local training runs. Metadata and
  checksums help detect mismatches, and can be forged. Retrain older artifacts
  to use the benchmark provenance checks.
