#!/usr/bin/env python
"""Local verification gate for VFI.

Runs, in order:
1. ruff check            (lint)
2. ruff format --check   (formatting)
3. pytest tests          (unit suite, synthetic fixtures only)
4. structural checks     (gitignore really excludes data/models artifacts,
                         and no artifact files are already tracked)

``--structure-only`` runs just the structural checks (used by CI so the
structure gate executes there too without duplicating ruff/pytest steps).

Exit code 0 means the repo is ready to push. No dataset downloads, no model
training: CI-safe and CPU-only by design.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Representative paths that MUST be covered by .gitignore (risk R-08:
#: accidental weight/dataset commits). Paths need not exist for
#: ``git check-ignore`` to evaluate the ignore rules.
MUST_BE_IGNORED = (
    "data/raw/uit_vsfc/train/sentences.txt",
    "models/majority_v1/model.joblib",
    "reports/artifacts/data_report.json",
    "checkpoints/epoch3.pt",
    "artifacts/export/model.onnx",
    "checkpoints/weights.h5",
    "pytorch_model.bin",
    ".env",
)

#: Tracked files matching these patterns are a structural failure:
#: .gitignore alone does not protect already-tracked files.
TRACKED_ARTIFACT_PATTERNS = (
    "*.joblib",
    "*.pkl",
    "*.bin",
    "*.pt",
    "*.ckpt",
    "*.onnx",
    "*.h5",
    "*.safetensors",
    "data/raw/*",
)


def run_step(title: str, cmd: list[str]) -> bool:
    print(f"==> {title}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    ok = result.returncode == 0
    print(f"<== {title}: {'PASS' if ok else 'FAIL'}")
    return ok


def _matches_tracked_patterns(path: str) -> str | None:
    """Return the matching pattern for a tracked path, or None if clean."""
    for pattern in TRACKED_ARTIFACT_PATTERNS:
        if pattern.endswith("/*"):
            prefix = pattern[:-2]
            if path == prefix or path.startswith(prefix + "/"):
                return pattern
            continue
        if path.endswith(pattern[1:]) and pattern.startswith("*"):
            return pattern
    return None


def scan_tracked_files(tracked_files: list[str]) -> list[str]:
    """Pure scan over an explicit file list (unit-testable without git)."""
    failures: list[str] = []
    for path in tracked_files:
        pattern = _matches_tracked_patterns(path)
        if pattern is not None:
            failures.append(f"tracked file matches artifact pattern '{pattern}': {path}")
    return failures


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

    for path in MUST_BE_IGNORED:
        check_ignored(path)

    listed = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    tracked = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
    failures.extend(scan_tracked_files(tracked))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="VFI local verification gate")
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="run only the structural checks (used by CI)",
    )
    args = parser.parse_args()

    failures: list[str] = []
    if not args.structure_only:
        python = sys.executable
        steps = [
            ("ruff check", [python, "-m", "ruff", "check", "."]),
            ("ruff format --check", [python, "-m", "ruff", "format", "--check", "."]),
            ("pytest", [python, "-m", "pytest", "tests", "-q"]),
        ]
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
