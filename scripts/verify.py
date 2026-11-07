#!/usr/bin/env python
"""Local verification gate for VFI.

Runs, in order:
1. ruff check            (lint)
2. ruff format --check   (formatting)
3. pytest tests          (unit suite, synthetic fixtures only)
4. structural checks     (gitignore really excludes data/models artifacts)

Exit code 0 means the repo is ready to push. No dataset downloads, no model
training: CI-safe and CPU-only by design.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_step(title: str, cmd: list[str]) -> bool:
    print(f"==> {title}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    ok = result.returncode == 0
    print(f"<== {title}: {'PASS' if ok else 'FAIL'}")
    return ok


def structural_checks() -> list[str]:
    """Fail fast if ignore rules would let data/artifacts into Git."""
    failures: list[str] = []

    def check_ignored(path: str) -> None:
        result = subprocess.run(
            ["git", "check-ignore", "-q", path],
            cwd=ROOT,
            check=False,
        )
        if result.returncode != 0:
            failures.append(f"'{path}' is NOT covered by .gitignore")

    def check_no_tracked(pattern: str) -> None:
        result = subprocess.run(
            ["git", "ls-files", pattern],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        tracked = result.stdout.strip()
        if tracked:
            failures.append(f"Tracked files match artifact pattern '{pattern}': {tracked}")

    check_ignored("data/raw/uit_vsfc/train/sentences.txt")
    check_ignored("models/majority_v1/model.joblib")
    check_ignored("reports/artifacts/data_report.json")
    check_no_tracked("*.joblib")
    check_no_tracked("*.pkl")
    return failures


def main() -> int:
    python = sys.executable
    steps = [
        ("ruff check", [python, "-m", "ruff", "check", "."]),
        ("ruff format --check", [python, "-m", "ruff", "format", "--check", "."]),
        ("pytest", [python, "-m", "pytest", "tests", "-q"]),
    ]
    failures: list[str] = []
    for title, cmd in steps:
        if not run_step(title, cmd):
            failures.append(title)

    structure_failures = structural_checks()
    if structure_failures:
        print("\nStructural check failures:")
        failures.extend(structure_failures)
        for failure in structure_failures:
            print(f"  - {failure}")

    if failures:
        print(f"\nVerification FAILED: {', '.join(failures)}")
        return 1
    print("\nAll verification checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
