"""Tests for the CLI: input validation, artifact handling, output consistency."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.conftest import make_records, write_test_config

from vfi.cli import main


@pytest.fixture()
def project(tmp_path: Path) -> dict[str, Path]:
    """A tiny synthetic VFI project: raw data, config, model and report dirs."""
    raw_dir = tmp_path / "raw"
    model_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    write_vsfc = {
        "train": make_records(4, split="train"),
        "validation": make_records(2, split="validation", offset=20),
        "test": make_records(2, split="test", offset=40),
    }
    from tests.conftest import write_vsfc_files

    write_vsfc_files(raw_dir, write_vsfc)
    config_path = tmp_path / "config.yaml"
    write_test_config(config_path, raw_dir, model_dir, reports_dir)
    return {
        "tmp": tmp_path,
        "raw": raw_dir,
        "models": model_dir,
        "reports": reports_dir,
        "config": config_path,
    }


class TestPrepareData:
    def test_reports_written(
        self, project: dict[str, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["prepare-data", "--config", str(project["config"])])
        assert exit_code == 0
        report_path = project["reports"] / "data_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        # The report is keyed by split under "splits" and is privacy-safe.
        assert report["splits"]["train"]["n_records"] == 12
        assert report["splits"]["train"]["class_distribution"] == {
            "negative": 4,
            "neutral": 4,
            "positive": 4,
        }
        assert "fingerprint" in report["splits"]["train"]
        # No raw feedback text anywhere in the report (privacy regression).
        from tests.conftest import make_records

        for record in make_records(2, split="train"):
            assert record.text not in report_path.read_text(encoding="utf-8")

    def test_missing_data_fails_cleanly(self, project: dict[str, Path]) -> None:
        project["raw"].rename(project["tmp"] / "moved_away")
        exit_code = main(["prepare-data", "--config", str(project["config"])])
        assert exit_code == 1

    def test_missing_config_usage_error(self, tmp_path: Path) -> None:
        assert main(["prepare-data", "--config", str(tmp_path / "nope.yaml")]) == 2


class TestTrainEvaluate:
    def test_train_saves_artifacts_and_untouched_test(
        self, project: dict[str, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["train", "--config", str(project["config"])])
        assert exit_code == 0
        for model_id in ("majority_v1", "tfidf_logreg_v1"):
            assert (project["models"] / model_id / "model.joblib").exists()
            assert (project["models"] / model_id / "meta.json").exists()
        training_report = json.loads(
            (project["reports"] / "training_report.json").read_text(encoding="utf-8")
        )
        # Test records are never loaded by the training command.
        assert training_report["test_not_loaded_by_training"] is True
        assert training_report["split_sizes_used"] == {"train": 12, "validation": 6}
        assert training_report["benchmark_clean"] is True
        assert len(training_report["validation_report"]["search_trials"]) == 4

    def test_evaluate_on_test_split(
        self, project: dict[str, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["train", "--config", str(project["config"])])
        exit_code = main(["evaluate", "--model-id", "majority", "--config", str(project["config"])])
        assert exit_code == 0
        evaluation = json.loads(
            (project["reports"] / "eval_majority.json").read_text(encoding="utf-8")
        )
        assert evaluation["test_metrics"]["n_samples"] == 6
        assert "latency" in evaluation

    def test_evaluate_without_model_fails(self, project: dict[str, Path]) -> None:
        exit_code = main(
            ["evaluate", "--model-id", "tfidf-logreg", "--config", str(project["config"])]
        )
        assert exit_code == 1

    def test_evaluate_unknown_model_id_usage_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            main(["evaluate", "--model-id", "bogus"])


class TestPredict:
    def test_predict_after_train_outputs_consistent_json(
        self, project: dict[str, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["train", "--config", str(project["config"])])
        capsys.readouterr()  # discard training report output
        exit_code = main(
            [
                "predict",
                "--text",
                "Giảng viên giảng dạy rất dễ hiểu và nhiệt tình.",
                "--model-id",
                "tfidf-logreg",
                "--model-dir",
                str(project["models"]),
            ]
        )
        assert exit_code == 0
        result = json.loads(capsys.readouterr().out)
        assert result["label"] in {"negative", "neutral", "positive"}
        assert result["model_id"] == "tfidf_logreg_v1"
        assert result["input_chars"] > 0
        probs = result["probabilities"]
        assert probs is not None
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_predict_majority_constant(
        self, project: dict[str, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["train", "--config", str(project["config"])])
        capsys.readouterr()
        exit_code = main(
            [
                "predict",
                "--text",
                "môn học bất kỳ",
                "--model-id",
                "majority",
                "--model-dir",
                str(project["models"]),
            ]
        )
        assert exit_code == 0
        result = json.loads(capsys.readouterr().out)
        assert result["probabilities"] is not None

    def test_predict_empty_text_usage_error(self) -> None:
        assert main(["predict", "--text", "   "]) == 2

    def test_predict_binary_model_aligns_probabilities_to_fitted_classes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Regression: a partial-class model must not crash or mislabel probabilities."""
        from tests.conftest import write_test_config, write_vsfc_files

        from vfi.cli import main as cli_main
        from vfi.data.loading import Record

        raw, models = tmp_path / "raw", tmp_path / "models"
        neg = [Record(f"khong tot lam {i}", "negative", "train") for i in range(6)]
        pos = [Record(f"rat tot va hay {i}", "positive", "train") for i in range(6)]
        negv = [Record(f"khong tot lam val {i}", "negative", "validation") for i in range(3)]
        posv = [Record(f"rat tot val {i}", "positive", "validation") for i in range(3)]
        write_vsfc_files(
            raw,
            {
                "train": neg + pos,
                "validation": negv + posv,
                "test": [Record("t1", "negative", "test"), Record("t2", "positive", "test")],
            },
        )
        config = tmp_path / "config.yaml"
        write_test_config(config, raw, models, tmp_path / "reports")
        assert cli_main(["train", "--config", str(config)]) == 0
        capsys.readouterr()
        assert (
            cli_main(
                [
                    "predict",
                    "--text",
                    "rat tot va hay",
                    "--model-dir",
                    str(models),
                ]
            )
            == 0
        )
        result = json.loads(capsys.readouterr().out)
        # Columns follow the FITTED classes (2 classes, canonical order here),
        # not an assumed 3-wide layout.
        assert sorted(result["probabilities"]) == ["negative", "positive"]
        assert sum(result["probabilities"].values()) == pytest.approx(1.0)
        assert result["classes_missing_from_training"] == ["neutral"]

    def test_predict_missing_model_runtime_error(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "no_models"
        empty_dir.mkdir()
        assert main(["predict", "--text", "một câu", "--model-dir", str(empty_dir)]) == 1

    def test_predict_normalizes_input_consistently(
        self, project: dict[str, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["train", "--config", str(project["config"])])
        capsys.readouterr()
        results = []
        for text in ("  môn học hay   ", "môn học hay"):
            main(["predict", "--text", text, "--model-dir", str(project["models"])])
            results.append(json.loads(capsys.readouterr().out))
        assert results[0]["label"] == results[1]["label"]
        assert results[0]["input_chars"] == results[1]["input_chars"]
