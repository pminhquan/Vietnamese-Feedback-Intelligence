"""Structural guard tests: artifact files must never be tracked in Git.

Uses throwaway git repositories in tmp_path so the scan logic is tested
end-to-end without touching this repository. Skips cleanly when git is
unavailable.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
from scripts.verify import MUST_BE_IGNORED, scan_tracked_files

GIT_AVAILABLE = shutil.which("git") is not None

pytestmark = pytest.mark.skipif(not GIT_AVAILABLE, reason="git not available")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    return repo


class TestScanTrackedFiles:
    def test_clean_file_list_passes(self) -> None:
        assert scan_tracked_files(["src/vfi/cli.py", "README.md", "configs/baseline.yaml"]) == []

    @pytest.mark.parametrize(
        "tracked_path",
        [
            "models/majority_v1/model.joblib",
            "checkpoints/epoch3.pt",
            "pytorch_model.bin",
            "export/model.onnx",
            "weights.h5",
            "anything.safetensors",
            "notebook.pkl",
            "data/raw/train/sentences.txt",
        ],
    )
    def test_artifact_files_flagged(self, tracked_path: str) -> None:
        failures = scan_tracked_files([tracked_path])
        assert len(failures) == 1
        assert tracked_path in failures[0]

    def test_non_artifact_bin_prefixed_name_passes(self) -> None:
        # "binary.txt" does not end with ".bin"; the scan matches extensions.
        assert scan_tracked_files(["docs/binary.txt"]) == []

    def test_data_raw_prefix_matches(self) -> None:
        failures = scan_tracked_files(["data/raw/uit_vsfc/train/sentences.txt"])
        assert failures


class TestMustBeIgnoredList:
    def test_weight_formats_represented(self) -> None:
        for suffix in (".bin", ".pt", ".onnx", ".h5", ".joblib", ".env"):
            assert any(path.endswith(suffix) for path in MUST_BE_IGNORED), suffix


@pytest.mark.skipif(not GIT_AVAILABLE, reason="git not available")
class TestThrowawayRepoScan:
    def test_tracked_weight_file_is_detected(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "weights.bin").write_bytes(b"\x00\x01")
        _git(repo, "add", "weights.bin")
        _git(repo, "commit", "-q", "-m", "accidental artifact")
        listed = subprocess.run(
            ["git", "ls-files"], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.split()
        failures = scan_tracked_files([p for p in listed if p])
        assert failures and "weights.bin" in failures[0]

    def test_clean_repo_passes(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "README.md").write_text("clean")
        _git(repo, "add", "README.md")
        _git(repo, "commit", "-q", "-m", "clean")
        listed = subprocess.run(
            ["git", "ls-files"], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.split()
        assert scan_tracked_files([p for p in listed if p]) == []
