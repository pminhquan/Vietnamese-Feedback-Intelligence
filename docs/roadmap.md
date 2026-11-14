# Roadmap

Status legend: `[ ]` not started · `[~]` blocked/partial · `[x]` done.

## Phase 1 — Classical ML foundation (current)

- [x] Bootstrap repository, environment, CI.
- [x] Dataset investigation and local adapters. Optional UIT-VSFC access
      requires a signed agreement and verified raw label mapping.
- [x] Data pipeline (loading/validation/splitting) + tests.
- [x] Majority baseline + TF-IDF + Logistic Regression + tests.
- [x] Metrics, latency protocol, reproducible experiment records.
- [x] Minimal inference CLI.
- [~] Real experiments await authorized data and a defensible test holdout.

## Phase 2 — Deep learning (NOT started; do not begin without approval)

- Verify a suitable PhoBERT checkpoint and its license.
- PyTorch fundamentals; Vietnamese word segmentation (e.g., VnCoreNLP/underthesea)
  and how tokenization changes the pipeline.
- Fine-tune on the same benchmark and splits; compare against Phase 1
  baselines honestly (same metrics, same protocol).
- GPU feasibility assessment first — training may need a GPU; do not assume
  one exists. CPU-only fallback: smaller model or frozen-embedding approach.
- Learning-rate experiments, early stopping, checkpoint management,
  ablations, error analysis.

## Phase 3 — Productization (NOT started)

- FastAPI serving endpoint with input limits and health checks.
- Batch prediction, simple interactive dashboard, model comparison view.
- Docker deployment, reproducible inference, model version display.

## Phase 4 — Optional advanced extensions (evidence-gated)

- Topic classification (only if topic labels are licensed for use).
- Aspect-based sentiment analysis, explainability, drift monitoring,
  model optimization.

## Immediate next actions for the owner

1. Collect and human-review an independent dataset or choose an authorized
   source using the README's adapter workflow.
2. Complete the OWNER LEARNING TASKS in `docs/learning-notes.md`.
3. Place the files under an ignored local path and run the verification gate.
   Follow the preflight, training and evaluation protocol on the selected
   dataset. Record aggregate results in `reports/`.
