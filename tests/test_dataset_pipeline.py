"""End-to-end integration against synthetic files in temporary directories."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from tests.conftest import make_records

from vfi.cli import main
from vfi.config import AppConfig, ConfigError, config_hash, load_config
from vfi.models.baseline import MODEL_IDS, MajorityBaseline, build_training_metadata, save_model


def write_config(tmp_path: Path, adapter: str, **overrides) -> Path:
    config = {
        "data": {
            "raw_dir": str(tmp_path / "splits"),
            "dataset_adapter": adapter,
            "dataset_id": "synthetic_test_fixture",
            "dataset_version": "fixture-v1",
            "synthetic_fixture": True,
            **overrides,
        },
        "models": {"tfidf_logreg": {"min_df": 1, "max_df": 1.0, "class_weight": None}},
        "output": {
            "model_dir": str(tmp_path / "models"),
            "reports_dir": str(tmp_path / "reports"),
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def run(config: Path, *args: str) -> int:
    return main([*args, "--config", str(config)])


def read_report(tmp_path: Path, filename: str) -> dict:
    return json.loads((tmp_path / "reports" / filename).read_text(encoding="utf-8"))


def write_csv(tmp_path: Path, adapter: str, *, with_test: bool = True) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    mapping = {"negative": "1", "neutral": "3", "positive": "5"}
    splits = ("train", "validation", "test") if with_test else ("train", "validation")
    for index, split in enumerate(splits):
        with (split_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["text", "label"])
            for record in make_records(5, split=split, offset=100 * index):
                label = mapping[record.label] if adapter == "five_level_csv" else record.label
                writer.writerow([record.text, label])


@pytest.mark.parametrize("adapter", ["five_level_csv", "neu_esc"])
def test_csv_only_opens_holdout_during_evaluation(
    tmp_path: Path, monkeypatch, adapter: str
) -> None:
    config = write_config(tmp_path, adapter)
    write_csv(tmp_path, adapter)
    original = Path.open
    seen = []
    forbidden = {"test.csv", "evaluation_preflight.json"}

    def guard(path, *args, **kwargs):
        assert path.name not in forbidden
        if path.suffix == ".csv":
            seen.append(path.name)
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guard)
        assert run(config, "train") == 0
    assert set(seen) == {"train.csv", "validation.csv"}
    assert run(config, "evaluate", "--preflight-only") == 0
    assert run(config, "evaluate", "--model-id", "majority") == 1
    assert run(config, "train") == 0
    assert run(config, "evaluate", "--model-id", "majority") == 0
    result = read_report(tmp_path, "eval_majority.json")
    assert result["result_scope"] == "synthetic_fixture"
    assert "test" not in result["training_provenance"]["dataset_provenance"]["split_provenance"]
    holdout = tmp_path / "splits" / "test.csv"
    # Change the raw representation while preserving normalized text and labels.
    # Source provenance still changes.
    holdout.write_bytes(b"\xef\xbb\xbf" + holdout.read_bytes())
    assert run(config, "evaluate", "--model-id", "majority") == 1
    assert run(config, "evaluate", "--preflight-only") == 0
    assert run(config, "evaluate", "--model-id", "majority") == 1


def test_no_holdout_can_train_but_has_no_test_benchmark(tmp_path: Path, capsys) -> None:
    config = write_config(tmp_path, "five_level_csv")
    write_csv(tmp_path, "five_level_csv", with_test=False)
    assert run(config, "train") == 0
    assert run(config, "evaluate", "--preflight-only") == 1
    assert run(config, "evaluate", "--model-id", "majority") == 1
    assert "test" in capsys.readouterr().err
    assert not (tmp_path / "reports" / "eval_majority.json").exists()


def test_contradictory_labels_block_before_fitting(tmp_path: Path) -> None:
    config = write_config(tmp_path, "five_level_csv")
    write_csv(tmp_path, "five_level_csv")
    text = make_records(5)[0].text
    with (tmp_path / "splits" / "train.csv").open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow([text, "1"])
    assert run(config, "prepare-data", "--near-duplicates") == 1
    report = read_report(tmp_path, "data_report.json")
    assert report["contradictions"]["count"] == 1
    assert report["near_duplicate_audit"]["records_checked"] > 0
    assert text not in json.dumps(report, ensure_ascii=False)
    assert run(config, "train") == 1
    assert not (tmp_path / "models").exists()


def test_neu_toxic_keep_is_audit_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = write_config(tmp_path, "neu_esc", toxic_policy="keep", toxic_justification="audit")
    write_csv(tmp_path, "neu_esc")
    with (tmp_path / "splits" / "train.csv").open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(["synthetic audit-only feedback", "Toxic"])
    assert run(config, "prepare-data") == 0
    assert read_report(tmp_path, "data_report.json")["train"]["class_distribution"]["toxic"] == 1
    assert run(config, "train") == 1
    assert run(config, "evaluate", "--preflight-only") == 1
    assert "audits only" in capsys.readouterr().err
    assert not (tmp_path / "models").exists()


def test_verified_legacy_artifact_predicts_but_cannot_score(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, "five_level_csv")
    write_csv(tmp_path, "five_level_csv")
    assert run(config, "evaluate", "--preflight-only") == 0
    model_id = MODEL_IDS["majority"]
    metadata = build_training_metadata(AppConfig(), {}, {"model_id": model_id})
    # The audited PR #1 format predates the adapter configuration fields.
    metadata["config"]["data"] = {
        "raw_dir": "data/raw/uit_vsfc",
        "use_official_splits": True,
        "validation_size": 0.2,
        "seed": 42,
    }
    payload = json.dumps(metadata["config"], sort_keys=True).encode("utf-8")
    metadata["config_hash"] = hashlib.sha256(payload).hexdigest()[:16]
    save_model(tmp_path / "models", model_id, MajorityBaseline().fit(make_records()), metadata)
    capsys.readouterr()
    assert (
        main(
            [
                "predict",
                "--model-id",
                "majority",
                "--model-dir",
                str(tmp_path / "models"),
                "--text",
                "fixture feedback",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["model_id"] == model_id
    assert run(config, "evaluate", "--model-id", "majority") == 1
    assert not (tmp_path / "reports" / "eval_majority.json").exists()


def test_policy_evidence_is_hashed_in_metadata_and_config_identity(tmp_path: Path) -> None:
    private_marker = "FAKE_PRIVATE_POLICY_REFERENCE"
    first = write_config(
        tmp_path, "neu_esc", toxic_policy="exclude", toxic_justification=private_marker
    )
    first_config = load_config(first)
    second = write_config(tmp_path, "neu_esc", toxic_policy="exclude", toxic_justification="other")
    assert config_hash(first_config) != config_hash(load_config(second))
    assert private_marker not in json.dumps(build_training_metadata(first_config, {}))


@pytest.mark.parametrize(
    "section",
    [
        "unexpected: 42",
        "data: {unexpected: 42}",
        "models: {unexpected: 42}",
        "models: {tfidf_logreg: {unexpected: 42}}",
        "output: {unexpected: 42}",
        "data: {rights_confirmed: 'false'}",
        "data: {synthetic_fixture: 'true'}",
        "data: {dataset_adapter: five_level_csv, label_mapping: '0:positive'}",
    ],
)
def test_rejects_unknown_or_ambiguous_config_keys(tmp_path: Path, section: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(section, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_duplicate_access_declaration_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "data:\n  rights_confirmed: false\n  rights_confirmed: true\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="Duplicate"):
        load_config(path)
