"""Tests for baseline models, hyperparameter search, and artifact handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sklearn
from sklearn.pipeline import Pipeline
from tests.conftest import make_records

from vfi.config import AppConfig, TfidfLogRegConfig
from vfi.data.loading import Record
from vfi.evaluation.metrics import compute_metrics
from vfi.models.baseline import (
    MODEL_IDS,
    SEARCH_SPACE,
    MajorityBaseline,
    build_tfidf_logreg_pipeline,
    build_training_metadata,
    load_model,
    run_small_search,
    save_model,
)


class TestMajorityBaseline:
    def test_predicts_most_frequent(self) -> None:
        model = MajorityBaseline().fit(make_records(3))  # 3 per class -> tie
        # Tie-break must be deterministic: lowest canonical id = "negative".
        assert model.predict(["anything"]) == ["negative"]

    def test_clear_majority_wins(self) -> None:
        records = make_records(1) + [
            Record("thêm tích cực", "positive", "train"),
            Record("rất tích cực", "positive", "train"),
        ]
        model = MajorityBaseline().fit(records)
        assert model.majority_label == "positive"
        assert model.predict(["a", "b"]) == ["positive", "positive"]

    def test_unfitted_predict_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fitted"):
            MajorityBaseline().predict(["x"])

    def test_empty_fit_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            MajorityBaseline().fit([])

    def test_predict_proba_one_hot(self) -> None:
        model = MajorityBaseline().fit(make_records(2))
        probs = model.predict_proba(["x", "y"])
        assert len(probs) == 2
        for row in probs:
            assert row.index(1.0) == 0  # "negative" is canonical id 0
            assert sum(row) == pytest.approx(1.0)


class TestTfidfLogRegPipeline:
    def test_vocabulary_learned_from_training_data_only(self) -> None:
        train = make_records(4)
        train.append(Record("từ hiếm xyzzy xuất hiện", "positive", "train"))
        # "qqqtest" deliberately never appears in the training texts below;
        # it exists only in hypothetical test data, so it must stay out of
        # the fitted vocabulary.
        pipeline = build_tfidf_logreg_pipeline(TfidfLogRegConfig(min_df=1, max_df=1.0))
        pipeline.fit([r.text for r in train], [r.label for r in train])
        vocabulary = pipeline.named_steps["tfidf"].vocabulary_
        assert "xyzzy" in vocabulary
        assert "qqqtest" not in vocabulary

    def test_same_seed_same_predictions(self) -> None:
        train = make_records(4)
        x_train = [r.text for r in train]
        y_train = [r.label for r in train]
        probe = [r.text for r in make_records(2, offset=10)]
        preds = []
        for _ in range(2):
            pipeline = build_tfidf_logreg_pipeline(TfidfLogRegConfig(min_df=1, max_df=1.0))
            pipeline.fit(x_train, y_train)
            preds.append(list(pipeline.predict(probe)))
        assert preds[0] == preds[1]

    def test_pipeline_structure(self) -> None:
        pipeline = build_tfidf_logreg_pipeline(TfidfLogRegConfig())
        assert isinstance(pipeline, Pipeline)
        assert [name for name, _ in pipeline.steps] == ["tfidf", "clf"]

    def test_learns_synthetic_task(self) -> None:
        # Sanity check on clearly separable synthetic data: the model must
        # reach perfect validation accuracy (unit-test success is NOT a claim
        # of real-world model quality).
        train = make_records(4)
        pipeline = build_tfidf_logreg_pipeline(TfidfLogRegConfig(min_df=1, max_df=1.0))
        pipeline.fit([r.text for r in train], [r.label for r in train])
        validation = make_records(2, offset=20)
        metrics = compute_metrics(
            [r.label for r in validation], list(pipeline.predict([r.text for r in validation]))
        )
        assert metrics["accuracy"] == pytest.approx(1.0)


class TestSmallSearch:
    def test_runs_all_candidates_and_selects_best(self) -> None:
        train = make_records(4)
        validation = make_records(2, split="validation", offset=20)
        base = TfidfLogRegConfig(min_df=1, max_df=1.0)
        pipeline, trials = run_small_search(train, validation, base)
        assert len(trials) == len(SEARCH_SPACE) == 4
        best_f1 = max(trial.val_macro_f1 for trial in trials)
        val_metrics = compute_metrics(
            [r.label for r in validation], list(pipeline.predict([r.text for r in validation]))
        )
        # Refit best pipeline must be at least as good as the best trial.
        assert val_metrics["macro_f1"] == pytest.approx(best_f1)

    def test_char_analyzer_trial_present(self) -> None:
        train = make_records(4)
        validation = make_records(2, split="validation", offset=20)
        _, trials = run_small_search(train, validation, TfidfLogRegConfig(min_df=1, max_df=1.0))
        analyzers = {trial.params["analyzer"] for trial in trials}
        assert analyzers == {"word", "char_wb"}


class TestArtifacts:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        model = MajorityBaseline().fit(make_records(2))
        metadata = build_training_metadata(
            AppConfig(), {"train": "a" * 64}, extra={"model_id": MODEL_IDS["majority"]}
        )
        save_model(tmp_path, MODEL_IDS["majority"], model, metadata)
        loaded, loaded_meta = load_model(tmp_path, MODEL_IDS["majority"])
        assert loaded.predict(["a", "b"]) == model.predict(["a", "b"])
        assert loaded_meta["config_hash"] == metadata["config_hash"]
        assert loaded_meta["dataset_fingerprints"] == {"train": "a" * 64}

    def test_load_missing_model_raises_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="vfi train"):
            load_model(tmp_path, MODEL_IDS["tfidf-logreg"])

    def test_load_unknown_model_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Unknown model_id"):
            load_model(tmp_path, "mystery_model")

    def test_load_rejects_metadata_without_config_hash(self, tmp_path: Path) -> None:
        model = MajorityBaseline().fit(make_records(2))
        save_model(tmp_path, MODEL_IDS["majority"], model, {"note": "hand-written"})
        with pytest.raises(FileNotFoundError, match="config_hash"):
            load_model(tmp_path, MODEL_IDS["majority"])

    def test_load_warns_on_sklearn_mismatch(self, tmp_path: Path) -> None:
        model = MajorityBaseline().fit(make_records(2))
        metadata = build_training_metadata(
            AppConfig(), {}, extra={"model_id": MODEL_IDS["majority"]}
        )
        metadata["sklearn_version"] = "0.0.0-fake"
        save_model(tmp_path, MODEL_IDS["majority"], model, metadata)
        with pytest.warns(UserWarning, match="scikit-learn"):
            load_model(tmp_path, MODEL_IDS["majority"])

    def test_save_rejects_unknown_model_id(self, tmp_path: Path) -> None:
        model = MajorityBaseline().fit(make_records(2))
        with pytest.raises(ValueError, match="Unknown model_id"):
            save_model(tmp_path, "mystery_model", model, {})

    def test_meta_json_written_and_valid(self, tmp_path: Path) -> None:
        model = MajorityBaseline().fit(make_records(2))
        metadata = build_training_metadata(
            AppConfig(), {"train": "f" * 64}, extra={"model_id": MODEL_IDS["majority"]}
        )
        out_dir = save_model(tmp_path, MODEL_IDS["majority"], model, metadata)
        assert (out_dir / "model.joblib").exists()
        loaded = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        assert loaded["sklearn_version"] == sklearn.__version__
        assert loaded["seed"] == AppConfig().data.seed
        assert loaded["dataset_fingerprints"] == {"train": "f" * 64}
