"""Command-line interface for VFI.

Subcommands
-----------
``vfi prepare-data``  validate raw files, report duplicates/leakage/distribution.
``vfi train``         train baselines on train, select on validation, save artifacts.
``vfi evaluate``      evaluate a saved artifact on the official test split.
``vfi predict``       single-text prediction from a locally trained artifact.

Exit codes: 0 success; 1 runtime error (missing data/model); 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vfi import __version__
from vfi.config import AppConfig, ConfigError, config_hash, load_config
from vfi.data.loading import (
    CANONICAL_LABELS,
    DataLoadingError,
    Record,
    load_dataset,
    normalize_text,
)
from vfi.data.splitting import stratified_split
from vfi.data.validation import (
    DataValidationError,
    class_distribution,
    find_duplicates,
    find_leakage,
    fingerprint,
    leakage_summary,
    validate_records,
)
from vfi.evaluation.metrics import compute_metrics, measure_latency
from vfi.models.baseline import (
    MODEL_IDS,
    MajorityBaseline,
    build_training_metadata,
    load_model,
    run_small_search,
    save_model,
)

DEFAULT_CONFIG_PATH = "configs/baseline.yaml"
DEFAULT_MODEL_DIR = "models"
PREFLIGHT_REPORT = "evaluation_preflight.json"


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _fail(message: str, code: int) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


def _load_data(config: AppConfig, *, include_test: bool = False) -> dict[str, list[Record]]:
    """Load effective splits; training never opens the official test files."""
    splits = ("train", "validation") if config.data.use_official_splits else ("train",)
    if include_test:
        splits += ("test",)
    data = load_dataset(config.data.raw_dir, splits=splits)
    train, validation = _resolve_train_validation(data, config)
    effective = {"train": train, "validation": validation}
    if include_test:
        effective["test"] = data["test"]
    for name, records in effective.items():
        if not records:
            raise DataValidationError(f"Split {name!r} must not be empty")
        validate_records(records)
    return effective


def _resolve_train_validation(
    data: dict[str, list[Record]], config: AppConfig
) -> tuple[list[Record], list[Record]]:
    """Apply the documented split policy (official splits by default)."""
    if config.data.use_official_splits:
        return data["train"], data["validation"]
    return stratified_split(
        data["train"], validation_size=config.data.validation_size, seed=config.data.seed
    )


def cmd_prepare_data(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        return _fail(str(exc), 2)
    try:
        data = _load_data(config)
    except (DataLoadingError, DataValidationError, ValueError) as exc:
        return _fail(str(exc), 1)
    return _write_data_report(config, data, "data_report.json")


def _write_data_report(config: AppConfig, data: dict[str, list[Record]], filename: str) -> int:
    fingerprints = {split: fingerprint(records) for split, records in data.items()}
    report: dict[str, Any] = {}
    for split, records in data.items():
        report[split] = {
            "n_records": len(records),
            "duplicate_texts": len(find_duplicates(records)),
            "fingerprint": fingerprints[split],
        }
        if split != "test":
            report[split]["class_distribution"] = class_distribution(records)
    report["schema_version"] = 1
    report["config_hash"] = config_hash(config)
    report["dataset_fingerprints"] = fingerprints
    report["leakage"] = leakage_summary(*data.items())
    report["passed"] = report["leakage"]["cross_split_duplicate_texts"] == 0
    report["created_at"] = datetime.now(UTC).isoformat()
    report["note"] = "Exact-match conflicts after shared normalization. All records are preserved."
    reports_dir = Path(config.output.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / filename
    # Rechecking the same data must not move its pre-training timestamp.
    if filename == PREFLIGHT_REPORT and report_path.exists():
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                isinstance(previous, dict)
                and previous.get("schema_version") == report["schema_version"]
                and previous.get("config_hash") == report["config_hash"]
                and previous.get("dataset_fingerprints") == report["dataset_fingerprints"]
                and previous.get("passed") is True
                and report["passed"]
            ):
                timestamp = datetime.fromisoformat(previous["created_at"])
                if timestamp.tzinfo is not None:
                    report["created_at"] = previous["created_at"]
        except (OSError, ValueError, KeyError, TypeError):
            pass  # Replace a malformed local report with a fresh preflight.
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    _print_json(report)
    if not report["passed"]:
        return _fail("Cross-split duplicates found; review split integrity.", 1)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        return _fail(str(exc), 2)
    try:
        data = _load_data(config)
    except (DataLoadingError, DataValidationError, ValueError) as exc:
        return _fail(str(exc), 1)

    train_records, validation_records = data["train"], data["validation"]
    conflicts = find_leakage(("train", train_records), ("validation", validation_records))
    if conflicts:
        return _fail(
            f"Train/validation leakage: {len(conflicts)} duplicate texts; "
            "review the split protocol. No records were removed and no model was fitted.",
            1,
        )
    if {record.label for record in train_records} != set(CANONICAL_LABELS):
        return _fail("Training requires all canonical classes: negative, neutral, positive", 1)
    fingerprints = {split: fingerprint(records) for split, records in data.items()}

    training_started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    majority = MajorityBaseline().fit(train_records)
    majority_fit_seconds = time.perf_counter() - started

    try:
        best_pipeline, trials = run_small_search(
            train_records, validation_records, config.tfidf_logreg
        )
    except (DataValidationError, ValueError) as exc:
        return _fail(str(exc), 1)

    # Validation metrics for every candidate (selection evidence). The test
    # split is deliberately NOT touched here; see 'vfi evaluate'.
    def _val_metrics(model: Any) -> dict[str, Any]:
        texts = [record.text for record in validation_records]
        labels = [record.label for record in validation_records]
        return compute_metrics(labels, model.predict(texts))

    val_report = {
        "majority": _val_metrics(majority),
        "tfidf_logreg_selected": _val_metrics(best_pipeline),
        "search_trials": [
            {
                "params": trial.params,
                "val_macro_f1": trial.val_macro_f1,
                "fit_seconds": trial.fit_seconds,
            }
            for trial in trials
        ],
    }
    model_dir = Path(config.output.model_dir)
    majority_meta = build_training_metadata(
        config,
        fingerprints,
        extra={
            "model_id": MODEL_IDS["majority"],
            "training_started_at": training_started_at,
            "fit_seconds": majority_fit_seconds,
            "validation_metrics": val_report["majority"],
            "majority_label": majority.majority_label,
        },
    )
    save_model(model_dir, MODEL_IDS["majority"], majority, majority_meta)

    tfidf_meta = build_training_metadata(
        config,
        fingerprints,
        extra={
            "model_id": MODEL_IDS["tfidf-logreg"],
            "training_started_at": training_started_at,
            "validation_metrics": val_report["tfidf_logreg_selected"],
            "search_trials": val_report["search_trials"],
            "selection_metric": "val_macro_f1",
        },
    )
    save_model(model_dir, MODEL_IDS["tfidf-logreg"], best_pipeline, tfidf_meta)

    reports_dir = Path(config.output.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    training_report = {
        "training_started_at": training_started_at,
        "split_sizes": {
            "train": len(train_records),
            "validation": len(validation_records),
        },
        "dataset_fingerprints": fingerprints,
        "train_class_distribution": class_distribution(train_records),
        "validation_report": val_report,
        "total_train_seconds": time.perf_counter() - started,
    }
    (reports_dir / "training_report.json").write_text(
        json.dumps(training_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _print_json(training_report)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        return _fail(str(exc), 2)
    if args.preflight_only:
        try:
            data = _load_data(config, include_test=True)
        except (DataLoadingError, DataValidationError, ValueError) as exc:
            return _fail(str(exc), 1)
        return _write_data_report(config, data, PREFLIGHT_REPORT)
    if args.model_id not in MODEL_IDS:
        return _fail(f"--model-id must be one of {sorted(MODEL_IDS)}", 2)
    try:
        model, metadata = load_model(config.output.model_dir, MODEL_IDS[args.model_id])
        data = _load_data(config, include_test=True)
        preflight = _verify_evaluation_provenance(config, metadata, data)
    except (FileNotFoundError, DataLoadingError, DataValidationError, ValueError) as exc:
        return _fail(str(exc), 1)

    test_records = data["test"]
    texts = [record.text for record in test_records]
    labels = [record.label for record in test_records]
    metrics = compute_metrics(labels, model.predict(texts))
    latency = measure_latency(model.predict, texts, seed=config.data.seed)
    evaluation = {
        "model_id": MODEL_IDS[args.model_id],
        "created_at": metadata.get("created_at"),
        "config_hash": metadata.get("config_hash"),
        "dataset_fingerprint_test": preflight["dataset_fingerprints"]["test"],
        "training_provenance": {
            key: metadata[key]
            for key in (
                "created_at",
                "training_started_at",
                "config_hash",
                "dataset_fingerprints",
                "artifact_sha256",
                "seed",
                "python_version",
                "sklearn_version",
            )
        },
        "preflight_created_at": preflight["created_at"],
        "test_metrics": metrics,
        "latency": latency,
    }
    reports_dir = Path(config.output.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"eval_{args.model_id}.json"
    out_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    evaluation["report_path"] = str(out_path)
    _print_json(evaluation)
    return 0


def _verify_evaluation_provenance(
    config: AppConfig, metadata: dict[str, Any], data: dict[str, list[Record]]
) -> dict[str, Any]:
    """Bind benchmark data to a clean preflight that predates this model.

    This runs only during evaluation. Training neither loads this all-split
    report nor consumes test fingerprints, features, labels or statistics.
    Local reports are provenance records, not a tamper-proof security boundary.
    """
    path = Path(config.output.reports_dir) / PREFLIGHT_REPORT
    try:
        preflight = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(preflight, dict)
            or preflight.get("schema_version") != 1
            or preflight.get("passed") is not True
            or preflight.get("config_hash") != config_hash(config)
            or metadata.get("config_hash") != config_hash(config)
        ):
            raise ValueError("Missing, failed or incompatible preflight/configuration")
        current = {split: fingerprint(records) for split, records in data.items()}
        training = {split: current[split] for split in ("train", "validation")}
        if metadata.get("dataset_fingerprints") != training:
            raise ValueError("Training/validation dataset differs from saved model provenance")
        if preflight.get("dataset_fingerprints") != current:
            raise ValueError("Dataset differs from the preflight snapshot")
        checked_at = datetime.fromisoformat(preflight["created_at"])
        trained_at = datetime.fromisoformat(metadata["training_started_at"])
        if checked_at.tzinfo is None or trained_at.tzinfo is None or checked_at > trained_at:
            raise ValueError("Clean preflight must predate model training")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise DataValidationError(
            f"Evaluation provenance rejected: {exc}. "
            "Run 'vfi evaluate --preflight-only', review the report, then retrain. "
            "changed datasets are not comparable benchmarks."
        ) from exc
    return preflight


def cmd_predict(args: argparse.Namespace) -> int:
    text = normalize_text(args.text)
    if not text:
        return _fail("--text must not be empty", 2)
    if args.model_id not in MODEL_IDS:
        return _fail(f"--model-id must be one of {sorted(MODEL_IDS)}", 2)
    model_dir = args.model_dir or DEFAULT_MODEL_DIR
    try:
        model, metadata = load_model(model_dir, MODEL_IDS[args.model_id])
    except FileNotFoundError as exc:
        return _fail(str(exc), 1)

    prediction = model.predict([text])[0]
    result: dict[str, Any] = {
        "model_id": MODEL_IDS[args.model_id],
        "model_version": metadata.get("created_at"),
        "input_chars": len(text),
        "label": prediction,
        "probabilities": None,
        "untrained_labels": [],
    }
    proba_fn = getattr(model, "predict_proba", None)
    if proba_fn is not None:
        row = proba_fn([text])[0]
        classes = list(getattr(model, "classes_", ()))
        if (
            len(row) != len(classes)
            or len(set(classes)) != len(classes)
            or (not set(classes) <= set(CANONICAL_LABELS) or prediction not in classes)
        ):
            return _fail("Model probability columns do not match its fitted classes", 1)
        result["probabilities"] = {
            label: float(prob) for label, prob in zip(classes, row, strict=True)
        }
        result["untrained_labels"] = [label for label in CANONICAL_LABELS if label not in classes]
    _print_json(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vfi",
        description="Vietnamese Feedback Intelligence: classical sentiment baselines.",
    )
    parser.add_argument("--version", action="version", version=f"vfi {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Validate and summarize train/validation")
    prepare.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    prepare.set_defaults(func=cmd_prepare_data)

    train = subparsers.add_parser("train", help="Train baselines (train/validation splits only)")
    train.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a saved model on the test split")
    evaluation_mode = evaluate.add_mutually_exclusive_group(required=True)
    evaluation_mode.add_argument("--model-id", choices=sorted(MODEL_IDS))
    evaluation_mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check split integrity without scoring a model",
    )
    evaluate.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    evaluate.set_defaults(func=cmd_evaluate)

    predict = subparsers.add_parser("predict", help="Predict sentiment for one text")
    predict.add_argument("--text", required=True)
    predict.add_argument("--model-id", default="tfidf-logreg", choices=sorted(MODEL_IDS))
    predict.add_argument("--model-dir", default=None, help="Directory holding trained artifacts")
    predict.set_defaults(func=cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
