# AGENTS.md — Ground rules for AI assistants working on VFI

This file sets hard boundaries for any AI coding agent (ZCode, Copilot, ...)
contributing to this repository. The owner is a third-year AI student; the
repository doubles as an internship portfolio, so honesty and learning value
outrank feature count.

## Non-negotiable rules

1. **Never invent results.** No fabricated accuracy/F1 numbers, no invented
   benchmark comparisons, no "expected results" presented as real ones.
   Unit tests passing is not model quality.
2. **Never commit raw datasets, model weights, or secrets.** `data/raw/`,
   `models/`, `reports/artifacts/`, `.env*`, `*.joblib`, `*.pkl` stay out of
   Git (enforced by `.gitignore`; verify before every commit).
3. **Respect split discipline.** Train fits on train only; validation is for
   selection; test is touched only by `vfi evaluate`. Never merge splits.
4. **Never force-push, never rewrite pushed commits, never merge a PR
   without the owner's explicit authorization.**
5. **Keep the dataset blocker honest.** UIT-VSFC usage requires a signed
   user agreement (see `docs/dataset-uit-vsfc.md`). Do not download it, do
   not add download scripts that bypass the agreement, until the owner says
   permission was granted.
6. **Language split:** code, comments, docs' technical content, commit
   messages, and PR descriptions in English; owner-facing explanations in
   Vietnamese (`docs/learning-notes.md`).
7. **Minimal diffs.** Match surrounding style; no drive-by refactors; no
   empty placeholder modules; no unrequested dependencies.
8. **No scope creep into Phase 2/3.** No PyTorch/Transformers, no API
   server, no dashboard in Phase 1 work.
9. **Do not disable security prompts** (UAC, Defender, firewall) or ZCode
   permission gates, ever.
10. **Run the gate before declaring done:** `python scripts/verify.py`
    (ruff check + ruff format --check + pytest + structural checks).

## Useful commands (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python scripts/verify.py
vfi predict --text "Giảng viên giải thích rất dễ hiểu"
```

## Definition of done for any change

- Focused tests added/updated for the behavior changed.
- `python scripts/verify.py` green.
- Docs updated when behavior or protocol changes.
- Commit message describes why, not just what.
