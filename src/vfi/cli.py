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
from pathlib import Path
from typing import Any

from vfi import __version__
from vfi.config import AppConfig, ConfigError, load_config
from vfi.data.loading import DataLoadingError, Record, load_dataset, normalize_text
from vfi.data.splitting import stratified_split
from vfi.data.validation import class_distribution, find_duplicates, find_leakage, fingerprint
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


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _fail(message: str, code: int) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


def _load_data(config: AppConfig) -> dict[str, list[Record]]:
    return load_dataset(config.data.raw_dir)


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
    except DataLoadingError as exc:
        return _fail(str(exc), 1)

    report: dict[str, Any] = {}
    for split, records in data.items():
        report[split] = {
            "n_records": len(records),
            "class_distribution": class_distribution(records),
            "duplicate_texts": len(find_duplicates(records)),
            "fingerprint": fingerprint(records),
        }
    report["leakage"] = {
        "cross_split_duplicate_texts": sorted(
            find_leakage(
                ("train", data["train"]),
                ("validation", data["validation"]),
                ("test", data["test"]),
            )
        )
    }
    report["note"] = (
        "Duplicates are reported, not removed (short feedback can legitimately repeat). "
        "Leakage is exact-match after shared normalization."
    )
    reports_dir = Path(config.output.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "data_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    _print_json(report)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        return _fail(str(exc), 2)
    try:
        data = _load_data(config)
    except DataLoadingError as exc:
        return _fail(str(exc), 1)

    train_records, validation_records = _resolve_train_validation(data, config)
    test_records = data["test"]
    fingerprints = {split: fingerprint(records) for split, records in data.items()}

    started = time.perf_counter()
    majority = MajorityBaseline().fit(train_records)
    majority_fit_seconds = time.perf_counter() - started

    best_pipeline, trials = run_small_search(train_records, validation_records, config.tfidf_logreg)

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
            "validation_metrics": val_report["tfidf_logreg_selected"],
            "search_trials": val_report["search_trials"],
            "selection_metric": "val_macro_f1",
        },
    )
    save_model(model_dir, MODEL_IDS["tfidf-logreg"], best_pipeline, tfidf_meta)

    reports_dir = Path(config.output.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    training_report = {
        "split_sizes": {
            "train": len(train_records),
            "validation": len(validation_records),
            "test_untouched": len(test_records),
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
    if args.model_id not in MODEL_IDS:
        return _fail(f"--model-id must be one of {sorted(MODEL_IDS)}", 2)
    try:
        model, metadata = load_model(config.output.model_dir, MODEL_IDS[args.model_id])
        test_records = load_dataset(config.data.raw_dir, splits=("test",))["test"]
    except (FileNotFoundError, DataLoadingError) as exc:
        return _fail(str(exc), 1)

    texts = [record.text for record in test_records]
    labels = [record.label for record in test_records]
    metrics = compute_metrics(labels, model.predict(texts))
    latency = measure_latency(model.predict, texts, seed=config.data.seed)
    evaluation = {
        "model_id": MODEL_IDS[args.model_id],
        "created_at": metadata.get("created_at"),
        "config_hash": metadata.get("config_hash"),
        "dataset_fingerprint_test": fingerprint(test_records),
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
    }
    proba_fn = getattr(model, "predict_proba", None)
    if proba_fn is not None:
        from vfi.data.loading import CANONICAL_LABELS

        row = proba_fn([text])[0]
        result["probabilities"] = {
            label: float(prob) for label, prob in zip(CANONICAL_LABELS, row, strict=True)
        }
    _print_json(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vfi",
        description="Vietnamese Feedback Intelligence: classical sentiment baselines.",
    )
    parser.add_argument("--version", action="version", version=f"vfi {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Validate raw data and report statistics")
    prepare.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    prepare.set_defaults(func=cmd_prepare_data)

    train = subparsers.add_parser("train", help="Train baselines (train/validation splits only)")
    train.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a saved model on the test split")
    evaluate.add_argument("--model-id", required=True, choices=sorted(MODEL_IDS))
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
