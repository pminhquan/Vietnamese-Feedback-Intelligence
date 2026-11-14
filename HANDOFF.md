# HANDOFF — Reproducing and continuing VFI Phase 1

Last updated: 2026-11-14 (Phase 1 on branch `phase1/classical-ml-baselines`).

## Current state

- `main` contains only the bootstrap commit (metadata, README, .gitignore,
  pyproject).
- Phase 1 work lives on `phase1/classical-ml-baselines` behind a Draft PR.
- **Dataset blocker:** UIT-VSFC requires a signed user agreement
  (`docs/dataset-uit-vsfc.md`). All code is done and tested on synthetic
  fixtures; real experiments are pending data access. No metrics are claimed.

## One-time setup (Windows PowerShell)

```powershell
cd "D:\project-personal\Vietnamese Feedback Intelligence"
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Verify everything:

```powershell
python scripts\verify.py
```

This runs ruff check, ruff format --check, pytest, and structural checks
(gitignore coverage for data/models artifacts, no tracked artifacts).

## Daily workflow

```powershell
# Lint + tests (fast loop)
ruff check . ; ruff format . ; pytest

# CLI smoke test (requires a trained artifact; see below)
vfi predict --text "Giảng viên giải thích rất dễ hiểu"
```

## Running real experiments (once data access is granted)

1. Follow `docs/dataset-uit-vsfc.md` to obtain and place the corpus under
   `data/raw/uit_vsfc/` (layout in `data/README.md`). The directory is
   gitignored; never commit it.
2. Validate train/validation and record a clean preflight before training:

   ```powershell
   vfi prepare-data --config configs\baseline.yaml
   vfi evaluate --preflight-only --config configs\baseline.yaml
   ```

3. Train + select on validation:

   ```powershell
   vfi train --config configs\baseline.yaml
   ```

4. Only after finalizing selection, evaluate on test:

   ```powershell
   vfi evaluate --model-id tfidf-logreg --config configs\baseline.yaml
   vfi evaluate --model-id majority --config configs\baseline.yaml
   ```

5. Write the milestone Markdown report in `reports/` from the JSON artifacts
   in `reports/artifacts/` (gitignored). Quote no raw student text.

Use the same configuration throughout. Evaluation checks the original preflight
and current split fingerprints before scoring. Retrain legacy artifacts with the
current workflow. Load Joblib files only from your own local training runs.

## Repo map

```text
configs/baseline.yaml      # experiment configuration
src/vfi/                   # package (data, models, evaluation, cli)
tests/                     # pytest suite (synthetic fixtures only)
scripts/verify.py          # local gate: lint + format + tests + structure
docs/                      # spec, protocol, dataset findings, roadmap, learning notes
data/                      # local-only datasets (gitignored except README)
models/                    # local-only trained artifacts (gitignored)
reports/                   # committed milestone reports (+ gitignored artifacts/)
```

## Conventions

- Commit messages: conventional, English, focused (`feat(data): ...`).
- Branches: `phase1/...`, future `phase2/...`.
- CI: GitHub Actions runs ruff + pytest on push/PR to `main` (no data
  downloads, CPU-only).
- PRs: Draft until reviewed; merge requires the owner's explicit approval.
