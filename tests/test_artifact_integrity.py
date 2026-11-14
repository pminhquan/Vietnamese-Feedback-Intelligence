"""Malformed metadata is rejected before trusted-local joblib deserialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.conftest import make_records

from vfi.cli import main
from vfi.config import AppConfig, TfidfLogRegConfig
from vfi.models.baseline import (
    MODEL_IDS,
    MajorityBaseline,
    build_tfidf_logreg_pipeline,
    build_training_metadata,
    load_model,
    run_small_search,
    save_model,
)


def test_binary_artifact_probability_columns_use_fitted_classes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = [r for r in make_records(4) if r.label != "neutral"]
    model = build_tfidf_logreg_pipeline(TfidfLogRegConfig(min_df=1, max_df=1.0))
    model.fit([r.text for r in records], [r.label for r in records])
    model_id = MODEL_IDS["tfidf-logreg"]
    save_model(
        tmp_path, model_id, model, build_training_metadata(AppConfig(), {}, {"model_id": model_id})
    )
    text = "Giảng viên dạy tốt"
    assert main(["predict", "--text", text, "--model-dir", str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    expected = dict(zip(model.classes_, model.predict_proba([text])[0], strict=True))
    assert result["probabilities"] == pytest.approx(expected)
    assert result["untrained_labels"] == ["neutral"]
    assert result["label"] == model.predict([text])[0]
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_search_rejects_missing_class_before_fitting() -> None:
    train = [r for r in make_records(3) if r.label != "neutral"]
    with pytest.raises(ValueError, match="all canonical classes"):
        run_small_search(train, make_records(2, offset=20), TfidfLogRegConfig(min_df=1, max_df=1.0))


@pytest.mark.parametrize("payload", ["[]", "null", "invalid JSON", '{"config_hash": []}'])
def test_bad_metadata_never_reaches_joblib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: str
) -> None:
    model_id = MODEL_IDS["majority"]
    model = MajorityBaseline().fit(make_records(2))
    save_model(
        tmp_path, model_id, model, build_training_metadata(AppConfig(), {}, {"model_id": model_id})
    )
    (tmp_path / model_id / "meta.json").write_text(payload, encoding="utf-8")

    def forbidden_load(*args: object) -> None:
        pytest.fail("Malformed metadata reached joblib.load")

    monkeypatch.setattr("vfi.models.baseline.joblib.load", forbidden_load)
    with pytest.raises(FileNotFoundError, match="metadata"):
        load_model(tmp_path, model_id)


def test_artifact_mutation_is_rejected_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_id = MODEL_IDS["majority"]
    save_model(
        tmp_path,
        model_id,
        MajorityBaseline().fit(make_records(2)),
        build_training_metadata(AppConfig(), {}, {"model_id": model_id}),
    )
    (tmp_path / model_id / "model.joblib").write_bytes(b"synthetic corrupted artifact")

    def forbidden_load(*args: object) -> None:
        pytest.fail("Changed artifact reached joblib.load")

    monkeypatch.setattr("vfi.models.baseline.joblib.load", forbidden_load)
    with pytest.raises(FileNotFoundError, match="checksum"):
        load_model(tmp_path, model_id)


def test_missing_python_version_is_rejected_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_id = MODEL_IDS["majority"]
    metadata = build_training_metadata(AppConfig(), {}, {"model_id": model_id})
    del metadata["python_version"]
    save_model(tmp_path, model_id, MajorityBaseline().fit(make_records(2)), metadata)

    def forbidden_load(*args: object) -> None:
        pytest.fail("Incomplete metadata reached joblib.load")

    monkeypatch.setattr("vfi.models.baseline.joblib.load", forbidden_load)
    with pytest.raises(FileNotFoundError, match="metadata"):
        load_model(tmp_path, model_id)
