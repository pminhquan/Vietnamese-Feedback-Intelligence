"""Saved models must not silently benchmark changed data (synthetic fixtures)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.test_cli import project as project

from vfi.cli import main
from vfi.data.loading import load_dataset
from vfi.data.validation import fingerprint


def evaluate(project: dict[str, Path]) -> int:
    return main(["evaluate", "--model-id", "majority", "--config", str(project["config"])])


def train(project: dict[str, Path]) -> None:
    assert main(["train", "--config", str(project["config"])]) == 0


def preflight(project: dict[str, Path]) -> None:
    assert main(["evaluate", "--preflight-only", "--config", str(project["config"])]) == 0


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_evaluation_refuses_changed_dataset(
    project: dict[str, Path], split: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vfi.models.baseline import MajorityBaseline

    preflight(project)
    train(project)
    path = project["raw"] / split / "sentences.txt"
    path.write_text(path.read_text(encoding="utf-8") + "SYNTHETIC_NEW_RECORD\n", encoding="utf-8")
    with (project["raw"] / split / "sentiments.txt").open("a", encoding="utf-8") as stream:
        stream.write("0\n")

    def forbidden_predict(*args: object) -> None:
        pytest.fail("Evaluation predicted before rejecting changed provenance")

    monkeypatch.setattr(MajorityBaseline, "predict", forbidden_predict)
    assert evaluate(project) == 1
    assert not (project["reports"] / "eval_majority.json").exists()


def test_evaluation_requires_preflight_before_training(project: dict[str, Path]) -> None:
    train(project)
    assert evaluate(project) == 1
    preflight(project)
    assert evaluate(project) == 1
    train(project)
    assert evaluate(project) == 0


def test_repeated_identical_preflight_preserves_original_binding(project: dict[str, Path]) -> None:
    preflight(project)
    path = project["reports"] / "evaluation_preflight.json"
    first = json.loads(path.read_text())
    train(project)
    preflight(project)
    assert json.loads(path.read_text())["created_at"] == first["created_at"]
    assert evaluate(project) == 0


def test_evaluation_records_test_and_training_provenance(project: dict[str, Path]) -> None:
    preflight(project)
    train(project)
    assert evaluate(project) == 0
    report = json.loads((project["reports"] / "eval_majority.json").read_text())
    data = load_dataset(project["raw"])
    assert report["dataset_fingerprint_test"] == fingerprint(data["test"])
    assert report["training_provenance"]["dataset_fingerprints"] == {
        split: fingerprint(data[split]) for split in ("train", "validation")
    }
    assert "artifact_sha256" in report["training_provenance"]
    assert report["preflight_created_at"]


def test_preflight_contains_no_test_scores_or_label_distribution(project: dict[str, Path]) -> None:
    preflight(project)
    report = json.loads((project["reports"] / "evaluation_preflight.json").read_text())
    assert "class_distribution" not in report["test"]
    assert "test_metrics" not in report
    assert not project["models"].exists()
    assert not (project["reports"] / "eval_majority.json").exists()


def test_refreshed_preflight_cannot_rebind_old_model(project: dict[str, Path]) -> None:
    preflight(project)
    train(project)
    path = project["raw"] / "test" / "sentiments.txt"
    path.write_text("1\n" + "\n".join(path.read_text().splitlines()[1:]) + "\n")
    preflight(project)
    assert evaluate(project) == 1


@pytest.mark.parametrize("payload", ["[]", "{}", "invalid json"])
def test_evaluation_refuses_malformed_preflight(project: dict[str, Path], payload: str) -> None:
    preflight(project)
    train(project)
    (project["reports"] / "evaluation_preflight.json").write_text(payload)
    assert evaluate(project) == 1
