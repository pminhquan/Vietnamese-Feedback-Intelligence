"""Regression probes for privacy and effective split integrity; synthetic only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from tests.test_cli import project as project

from vfi.cli import main
from vfi.data.loading import load_dataset
from vfi.data.splitting import stratified_split
from vfi.data.validation import fingerprint

MARKER = "FAKE_STUDENT_PRIVATE_ID_730519"


@pytest.mark.parametrize("bad_label", [MARKER, "7305195550123"])
def test_prepare_error_omits_invalid_label_contents(
    project: dict[str, Path], capsys: pytest.CaptureFixture[str], bad_label: str
) -> None:
    path = project["raw"] / "train" / "sentiments.txt"
    _, _, remaining = path.read_text(encoding="utf-8").partition("\n")
    path.write_text(f"{bad_label}\n{remaining}", encoding="utf-8")
    assert main(["prepare-data", "--config", str(project["config"])]) == 1
    error = capsys.readouterr().err
    assert "line 1" in error
    assert bad_label not in error


def append_record(project: dict[str, Path], split: str, text: str, raw_label: int = 0) -> None:
    for filename, line in (("sentences.txt", text), ("sentiments.txt", str(raw_label))):
        with (project["raw"] / split / filename).open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def read_report(project: dict[str, Path], name: str) -> dict:
    return json.loads((project["reports"] / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("other_split", ["validation", "test"])
def test_preflight_does_not_disclose_duplicate_text(
    project: dict[str, Path], capsys: pytest.CaptureFixture[str], other_split: str
) -> None:
    sentence = f"Giảng viên giúp sinh viên {MARKER} hiểu bài."
    append_record(project, "train", sentence)
    append_record(project, other_split, "  " + sentence + "  ")
    if other_split == "validation":
        command, report_name = ["prepare-data"], "data_report.json"
    else:
        command, report_name = ["evaluate", "--preflight-only"], "evaluation_preflight.json"
    code = main([*command, "--config", str(project["config"])])
    captured = capsys.readouterr()
    saved = (project["reports"] / report_name).read_text(encoding="utf-8")
    assert MARKER not in captured.out + captured.err + saved
    report = json.loads(saved)
    assert report["leakage"]["cross_split_duplicate_texts"] == 1
    assert report["leakage"]["split_pair_counts"][f"train/{other_split}"] == 1
    assert report["passed"] is False
    assert code == 1


def test_preflight_blocks_validation_test_conflict(project: dict[str, Path]) -> None:
    append_record(project, "validation", MARKER)
    append_record(project, "test", MARKER)
    assert main(["evaluate", "--preflight-only", "--config", str(project["config"])]) == 1
    report = read_report(project, "evaluation_preflight.json")
    assert report["leakage"]["split_pair_counts"]["validation/test"] == 1


def test_train_blocks_leakage_before_model_fit(
    project: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import vfi.cli as cli

    append_record(project, "train", MARKER)
    append_record(project, "validation", MARKER)

    def forbidden_fit(*args: object) -> None:
        pytest.fail("Fitting started despite train/validation leakage")

    monkeypatch.setattr(cli.MajorityBaseline, "fit", forbidden_fit)
    assert main(["train", "--config", str(project["config"])]) == 1
    assert MARKER not in capsys.readouterr().err
    assert not project["models"].exists()


def test_train_never_requests_test_split(
    project: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import vfi.cli as cli

    requested = []

    def guarded_load(raw_dir: str, splits: tuple[str, ...] = ("train", "validation", "test")):
        assert "test" not in splits, "Training requested test features/labels"
        requested.append(splits)
        return load_dataset(raw_dir, splits=splits)

    monkeypatch.setattr(cli, "load_dataset", guarded_load)
    assert main(["train", "--config", str(project["config"])]) == 0
    assert requested == [("train", "validation")]
    report = read_report(project, "training_report.json")
    assert set(report["dataset_fingerprints"]) == {"train", "validation"}
    assert set(report["split_sizes"]) == {"train", "validation"}


@pytest.mark.parametrize("command", ["prepare-data", "train"])
def test_train_and_prepare_work_without_test_files(project: dict[str, Path], command: str) -> None:
    (project["raw"] / "test").rename(project["tmp"] / "sealed_test")
    assert main([command, "--config", str(project["config"])]) == 0


def test_fallback_without_validation_files_has_effective_fingerprints(
    project: dict[str, Path],
) -> None:
    config = yaml.safe_load(project["config"].read_text(encoding="utf-8"))
    config["data"]["use_official_splits"] = False
    config["data"]["validation_size"] = 0.25
    project["config"].write_text(yaml.safe_dump(config), encoding="utf-8")
    (project["raw"] / "validation").rename(project["tmp"] / "unused_validation")
    source = load_dataset(project["raw"], splits=("train",))["train"]
    train, validation = stratified_split(source, validation_size=0.25, seed=42)
    assert main(["prepare-data", "--config", str(project["config"])]) == 0
    assert main(["evaluate", "--preflight-only", "--config", str(project["config"])]) == 0
    fingerprints = {"train": fingerprint(train), "validation": fingerprint(validation)}
    assert read_report(project, "data_report.json")["dataset_fingerprints"] == fingerprints
    assert main(["train", "--config", str(project["config"])]) == 0
    first = read_report(project, "training_report.json")
    assert first["dataset_fingerprints"] == fingerprints
    assert all(record.split == "validation" for record in validation)
    assert main(["train", "--config", str(project["config"])]) == 0
    assert read_report(project, "training_report.json")["dataset_fingerprints"] == fingerprints
    for model_id in ("majority_v1", "tfidf_logreg_v1"):
        metadata = json.loads((project["models"] / model_id / "meta.json").read_text())
        assert metadata["dataset_fingerprints"] == fingerprints
    assert main(["evaluate", "--model-id", "majority", "--config", str(project["config"])]) == 0


def test_training_rejects_missing_sentiment_class(project: dict[str, Path]) -> None:
    for filename in ("sentences.txt", "sentiments.txt"):
        path = project["raw"] / "train" / filename
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text(
            "\n".join(line for i, line in enumerate(lines) if i % 3 != 2) + "\n", encoding="utf-8"
        )
    assert main(["train", "--config", str(project["config"])]) == 1
    assert not project["models"].exists()
