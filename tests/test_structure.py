"""Exercise the structural gate on temporary Git repositories."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from scripts import verify

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is required")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def guard_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_bytes((verify.ROOT / ".gitignore").read_bytes())
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    return tmp_path


def test_structure_allows_project_sources_and_docs(guard_repo: Path) -> None:
    for relative in ("data/README.md", "reports/README.md", "src/vfi/models/baseline.py"):
        path = guard_repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic file", encoding="utf-8")
        _git(guard_repo, "add", "--", relative)

    assert verify.structural_checks() == []


def test_structure_detects_protected_files_even_when_staged_forcefully(guard_repo: Path) -> None:
    protected = (
        "data/raw/uit_vsfc/train/sentences.txt",
        "data/processed/labeled_feedback.jsonl",
        "data/external/corpus.zip",
        "data/datasets/custom/dataset.json",
        "data/feedback.tsv",
        "models/majority_v1/meta.json",
        "reports/artifacts/training_report.json",
        "checkpoints/trọng số.bin",
        "checkpoints/model.pt",
        "checkpoints/model.ckpt",
        "export/model.onnx",
        "export/model.h5",
        "export/model.safetensors",
        "export/model.joblib",
        "export/model.pkl",
        "config/.env.local",
        "config/.envrc",
        "config/signing.key",
    )
    for relative in protected:
        path = guard_repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic fixture", encoding="utf-8")
    _git(guard_repo, "add", "-f", "--", *protected)

    failures = verify.structural_checks()
    assert all(any(path in failure for failure in failures) for path in protected)
    assert len(failures) == len(protected)


def test_structure_detects_missing_protected_ignore_rule(guard_repo: Path) -> None:
    ignore = guard_repo / ".gitignore"
    ignore.write_text(ignore.read_text(encoding="utf-8").replace("*.onnx\n", ""), encoding="utf-8")

    assert any("model.onnx" in error for error in verify.structural_checks())
