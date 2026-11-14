#!/usr/bin/env python
"""Local verification gate for VFI.

Runs, in order:
1. ruff check            (lint)
2. ruff format --check   (formatting)
3. pytest tests          (unit suite, synthetic fixtures only)
4. structural checks     (ignore coverage and tracked sensitive paths)

Use ``--structure-only`` to run the repository checks separately in CI.

Exit code 0 means the repo is ready to push. No dataset downloads, no model
training: CI-safe and CPU-only by design.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROTECTED_DIRECTORIES = (
    "data/raw/",
    "data/processed/",
    "data/external/",
    "data/datasets/",
    "models/",
    "reports/artifacts/",
)
DATA_SUFFIXES = (".csv", ".tsv", ".jsonl", ".parquet", ".zip")
ARTIFACT_SUFFIXES = (".joblib", ".pkl", ".safetensors", ".bin", ".pt", ".ckpt", ".onnx", ".h5")

MUST_BE_IGNORED = (
    "data/raw/uit_vsfc/train/sentences.txt",
    "data/processed/train.json",
    "data/external/corpus.txt",
    "data/datasets/custom/dataset.json",
    "models/majority_v1/meta.json",
    "reports/artifacts/data_report.json",
    "configs/.env",
    "configs/.env.local",
    "configs/.envrc",
    "configs/signing.key",
    *(f"data/dataset{suffix}" for suffix in DATA_SUFFIXES),
    *(f"checkpoints/model{suffix}" for suffix in ARTIFACT_SUFFIXES),
)


def run_step(title: str, cmd: list[str]) -> bool:
    print(f"==> {title}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    ok = result.returncode == 0
    print(f"<== {title}: {'PASS' if ok else 'FAIL'}")
    return ok


def _is_protected_path(path: str) -> bool:
    path = path.casefold()
    parent, _, name = path.rpartition("/")
    return (
        path.startswith(PROTECTED_DIRECTORIES)
        or (parent == "data" and name.endswith(DATA_SUFFIXES))
        or name.endswith((*ARTIFACT_SUFFIXES, ".key"))
        or name.startswith(".env")
    )


def _covered_paths(output: bytes) -> set[str]:
    """Read check-ignore's NUL-separated source, line, pattern, path records."""
    fields = output.decode("utf-8", errors="surrogateescape").split("\0")[:-1]
    covered = set()
    for start in range(0, len(fields), 4):
        source, _, pattern, path = fields[start : start + 4]
        if source == ".gitignore" and not pattern.startswith("!"):
            covered.add(path)
    return covered


def structural_checks() -> list[str]:
    """Require repository ignore rules and reject protected paths in the index."""
    try:
        ignored = subprocess.run(
            ["git", "check-ignore", "--no-index", "--verbose", "-z", "--stdin"],
            cwd=ROOT,
            input=("\0".join(MUST_BE_IGNORED) + "\0").encode("utf-8"),
            check=False,
            capture_output=True,
        )
        if ignored.returncode not in (0, 1):
            return ["Git could not check .gitignore coverage"]
        tracked = subprocess.run(
            ["git", "ls-files", "--cached", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return [f"Git structural check failed: {exc}"]

    covered = _covered_paths(ignored.stdout)
    failures = [
        f"{path!r} is NOT covered by .gitignore" for path in MUST_BE_IGNORED if path not in covered
    ]
    failures.extend(
        f"Tracked dataset, artifact, or secret path: {path!r}"
        for path in tracked.stdout.decode("utf-8", errors="surrogateescape").split("\0")
        if _is_protected_path(path)
    )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="VFI local verification gate")
    parser.add_argument(
        "--structure-only", action="store_true", help="run only the Git structural checks"
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
