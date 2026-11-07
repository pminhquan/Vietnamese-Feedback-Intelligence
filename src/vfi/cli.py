"""Command-line interface for VFI.

Subcommands
-----------
``vfi prepare-data``   adapter-based integrity inspection (privacy-safe report).
``vfi train``          train baselines on train, select on validation; refuses
                       unresolved train/validation leakage; never uses test.
``vfi evaluate``       evaluate a saved artifact on the test split (single touch).
``vfi predict``        single-text prediction from a locally trained artifact.
``vfi dataset ...``    independent custom-dataset management (init, add-synthetic,
                       annotate, report, validate).

Datasets are selected through ``data.dataset_adapter`` (UIT-VSFC layout,
five-level CSV, NEU-ESC 4-class CSV, or the custom JSONL store) so no
dataset is hardcoded into the pipeline. Exit codes: 0 success; 1 runtime
error (missing data/model); 2 usage error.
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
from vfi.custom_dataset import (
    ANNOTATION_METHOD_HUMAN,
    AnnotationDecision,
    CustomDatasetError,
    CustomDatasetStore,
    generate_synthetic_draft,
)
from vfi.data.adapters import (
    CSV_FILENAME,
    JSONL_FILENAME,
    AdapterError,
    AdapterResult,
    CustomJsonlAdapter,
    DatasetAdapter,
    FiveLevelCsvAdapter,
    NeuEscAdapter,
    UitVsfcAdapter,
)
from vfi.data.loading import (
    CANONICAL_LABELS,
    LABEL_SOURCE_OWNER_CONFIRMED,
    LABEL_SOURCE_SYNTHETIC,
    RawLabelMapping,
    normalize_text,
    parse_label_mapping,
)
from vfi.data.validation import class_distribution, fingerprint, leakage_summary
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

_ANNOTATION_SHORTCUTS = {"n": "negative", "e": "neutral", "p": "positive"}


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _fail(message: str, code: int) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


def _build_adapter(config: AppConfig) -> DatasetAdapter:
    """Instantiate the configured dataset adapter (pure local loading)."""
    seed = config.data.seed
    fraction = config.data.validation_size
    filename = config.data.filename or None
    adapter_id = config.data.dataset_adapter
    if adapter_id == "uit_vsfc":
        mapping = None
        if config.data.label_mapping_source:
            source = (
                LABEL_SOURCE_SYNTHETIC
                if config.data.label_mapping_source == "synthetic_fixture"
                else LABEL_SOURCE_OWNER_CONFIRMED
            )
            mapping = RawLabelMapping(
                parse_label_mapping(config.data.label_mapping),
                source,
                config.data.label_mapping_evidence,
            )
        return UitVsfcAdapter(label_mapping=mapping)
    if adapter_id == "five_level_csv":
        return FiveLevelCsvAdapter(
            validation_fraction=fraction,
            seed=seed,
            filename=filename or CSV_FILENAME,
        )
    if adapter_id == "neu_esc":
        return NeuEscAdapter(
            toxic_policy=config.data.toxic_policy,
            justification=config.data.toxic_justification or None,
            validation_fraction=fraction,
            seed=seed,
            filename=filename or CSV_FILENAME,
        )
    if adapter_id == "custom_jsonl":
        return CustomJsonlAdapter(
            validation_fraction=fraction,
            seed=seed,
            filename=filename or JSONL_FILENAME,
        )
    raise ConfigError(f"unsupported dataset_adapter {adapter_id!r}")  # pragma: no cover


def _load_via_adapter(config: AppConfig) -> AdapterResult:
    adapter = _build_adapter(config)
    return adapter.load(Path(config.data.raw_dir))


def _label_gate_refusal(config: AppConfig) -> str | None:
    """UIT-VSFC training on an unverified raw-label interpretation is refused."""
    if config.data.dataset_adapter != "uit_vsfc" or config.data.label_mapping_source:
        return None
    return (
        "training through the uit_vsfc adapter is refused: the raw integer label "
        "encoding is UNRESOLVED (SEACrowd says 0=positive..., the authors' own card "
        "says 0=negative...). Set data.label_mapping_source to 'synthetic_fixture' "
        "(tests with self-defined labels) or 'owner_confirmed' together with "
        "data.label_mapping and data.label_mapping_evidence once the encoding is "
        "verified, or switch data.dataset_adapter to a dataset with verified labels. "
        "See docs/dataset-strategy.md"
    )


def cmd_prepare_data(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        result = _load_via_adapter(config)
    except (ConfigError, AdapterError) as exc:
        return _fail(str(exc), 2 if isinstance(exc, ConfigError) else 1)

    # The adapter's integrity block is privacy-safe by construction
    # (counts, line indices, fingerprints - never raw feedback text).
    report: dict[str, Any] = {
        "dataset_id": result.provenance.get("dataset_id"),
        "adapter": result.provenance.get("adapter"),
        "source": result.provenance.get("source"),
        "label_mapping": result.provenance.get("label_mapping"),
        "splits": {
            split: {
                "n_records": len(records),
                "class_distribution": class_distribution(records),
                "fingerprint": fingerprint(records),
            }
            for split, records in sorted(result.records_by_split.items())
        },
        "integrity": result.provenance.get("integrity"),
        "note": (
            "Integrity figures are counts and indices only; raw feedback text is "
            "never written to reports or stdout. Label-mapping provenance is "
            "recorded verbatim."
        ),
    }
    reports_dir = Path(config.output.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "data_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_json(report)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        return _fail(str(exc), 2)
    gate = _label_gate_refusal(config)
    if gate is not None:
        return _fail(gate, 2)
    try:
        result = _load_via_adapter(config)
    except AdapterError as exc:
        return _fail(str(exc), 1)

    # The adapter may inspect every split (integrity preflight), but training
    # itself only ever touches train/validation: drop test records here.
    data = {split: records for split, records in result.records_by_split.items() if split != "test"}
    if "train" not in data or "validation" not in data:
        return _fail(
            f"adapter produced no train/validation pair (splits found: {sorted(data)}); "
            "provide a split column or enough distinct texts per class",
            1,
        )
    train_records, validation_records = data["train"], data["validation"]

    leakage = leakage_summary(("train", train_records), ("validation", validation_records))
    leakage_override = bool(args.allow_leakage_override)
    if leakage["count"] > 0 and not leakage_override:
        return _fail(
            f"unresolved train/validation leakage: {leakage['count']} text(s) appear in "
            f"both splits ({leakage['affected_records']} records, line indices "
            f"{leakage['line_indices']}). Resolve or group them before training; "
            "pass --allow-leakage-override only if you accept a benchmark that is "
            "NOT clean (the report will say so)",
            1,
        )

    fingerprints = {split: fingerprint(records) for split, records in sorted(data.items())}
    started = time.perf_counter()
    majority = MajorityBaseline().fit(train_records)
    majority_fit_seconds = time.perf_counter() - started
    best_pipeline, trials = run_small_search(train_records, validation_records, config.tfidf_logreg)

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
    adapter_provenance = {
        key: result.provenance[key]
        for key in ("dataset_id", "adapter", "label_mapping", "label_mapping_source", "integrity")
        if key in result.provenance
    }
    shared_metadata = {
        "dataset_provenance": adapter_provenance,
        "effective_fingerprints": fingerprints,
        "leakage_override": leakage_override,
        "benchmark_clean": leakage["count"] == 0,
        "preflight_note": (
            "The adapter's integrity preflight may inspect all split texts; training "
            "and model selection used only the train/validation records recorded in "
            "effective_fingerprints. Test records were not loaded by this command."
        ),
    }
    majority_meta = build_training_metadata(
        config,
        fingerprints,
        extra={
            "model_id": MODEL_IDS["majority"],
            "fit_seconds": majority_fit_seconds,
            "validation_metrics": val_report["majority"],
            "majority_label": majority.majority_label,
            **shared_metadata,
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
            **shared_metadata,
        },
    )
    save_model(model_dir, MODEL_IDS["tfidf-logreg"], best_pipeline, tfidf_meta)

    reports_dir = Path(config.output.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    training_report = {
        "split_sizes_used": {split: len(records) for split, records in sorted(data.items())},
        "test_not_loaded_by_training": True,
        "effective_fingerprints": fingerprints,
        "train_class_distribution": class_distribution(train_records),
        "leakage_train_validation": leakage,
        "leakage_override": leakage_override,
        "benchmark_clean": leakage["count"] == 0,
        "dataset_provenance": adapter_provenance,
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
        result = _load_via_adapter(config)
    except (FileNotFoundError, AdapterError) as exc:
        return _fail(str(exc), 1)
    if "test" not in result.records_by_split:
        return _fail(
            f"adapter produced no test split (splits: {sorted(result.records_by_split)})", 1
        )
    test_records = result.records_by_split["test"]

    test_fingerprint = fingerprint(test_records)
    recorded_fingerprint = (
        metadata.get("dataset_provenance", {}),
        metadata.get("effective_fingerprints", {}),
    )
    recorded_test = recorded_fingerprint[1].get("test", "")
    dataset_mismatch = bool(recorded_test) and recorded_test != test_fingerprint
    texts = [record.text for record in test_records]
    labels = [record.label for record in test_records]
    metrics = compute_metrics(labels, model.predict(texts))
    latency = measure_latency(model.predict, texts, seed=config.data.seed)
    evaluation = {
        "model_id": MODEL_IDS[args.model_id],
        "created_at": metadata.get("created_at"),
        "config_hash": metadata.get("config_hash"),
        "dataset_fingerprint_test": test_fingerprint,
        "recorded_test_fingerprint": recorded_test,
        "dataset_mismatch": dataset_mismatch,
        "dataset_mismatch_policy": (
            "a mismatch means the artifact was trained on different data than this "
            "test split; numbers are reported for transparency but are NOT a clean "
            "benchmark - retrain or restore the original split"
        )
        if dataset_mismatch
        else "match",
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
        # Probability columns follow the FITTED class order (the classifier's
        # classes_), never an assumed canonical order - a model trained on a
        # subset of classes has fewer columns, and a different training-label
        # order would silently mislabel probabilities.
        classifier = getattr(model, "named_steps", {}).get("clf")
        classes = getattr(classifier, "classes_", None) if classifier is not None else None
        if classes is None:
            classes = getattr(model, "classes_", None)
        if classes is None or len(classes) != len(proba_fn([text])[0]):
            return _fail(
                "cannot align probabilities: the model does not expose its fitted "
                "class order (classes_); retrain with this project's train command",
                1,
            )
        classes = [str(label) for label in classes]
        unknown = [label for label in classes if label not in CANONICAL_LABELS]
        if unknown:
            return _fail(
                f"model was fitted with label(s) outside the canonical space: {unknown}",
                1,
            )
        row = proba_fn([text])[0]
        result["probabilities"] = {
            label: float(prob) for label, prob in zip(classes, row, strict=True)
        }
        if len(classes) < len(CANONICAL_LABELS):
            # Partial-class model: report the gap honestly instead of padding.
            result["classes_missing_from_training"] = sorted(set(CANONICAL_LABELS) - set(classes))
    _print_json(result)
    return 0


# ---------------------------------------------------------------------------
# vfi dataset ... - independent custom-dataset management
# ---------------------------------------------------------------------------


def _open_store(args: argparse.Namespace) -> CustomDatasetStore:
    return CustomDatasetStore(args.root, args.dataset_version)


def cmd_dataset_init(args: argparse.Namespace) -> int:
    store = _open_store(args)
    manifest = store.write_manifest()
    print(f"initialized dataset v{args.dataset_version} at {store.root}")
    print(f"manifest: {manifest}")
    return 0


def cmd_dataset_add_synthetic(args: argparse.Namespace) -> int:
    store = _open_store(args)
    drafts = generate_synthetic_draft(
        args.count_per_class,
        dataset_version=args.dataset_version,
        run_id=args.run_id,
        seed=args.seed,
    )
    try:
        added = store.append_records(drafts)
    except CustomDatasetError as exc:
        return _fail(str(exc), 1)
    store.write_manifest()
    print(
        f"added {added} SYNTHETIC draft records (source_type='synthetic', "
        "annotation_method='synthetic_generator', status='draft'); they are "
        "annotation-queue candidates, never real user feedback, and cannot be "
        "approved without your personal human annotation"
    )
    return 0


def cmd_dataset_annotate(args: argparse.Namespace) -> int:
    store = _open_store(args)
    try:
        records = store.load_records()
    except CustomDatasetError as exc:
        return _fail(str(exc), 1)
    pending = [
        record
        for record in records
        if record["annotation_status"] != "approved" or not record["sentiment_label"]
    ]
    if not pending:
        print("nothing pending: every record is labeled and approved")
        return 0
    print(
        "annotation session (human): enter n/e/p (or negative/neutral/positive), "
        "'s' to skip, 'q' to quit; approving requires a label and is a human action"
    )
    for record in pending:
        header = (
            f"\n[{record['record_id']}] "
            f"({record['source_type']}, status={record['annotation_status']})"
        )
        print(header)
        print(record["text"])
        raw = input("label [n/e/p/s/q/a=approve]: ").strip().lower()
        if raw == "q":
            print("session ended")
            break
        if raw == "s":
            continue
        if raw == "a":
            try:
                store.apply_annotation(
                    AnnotationDecision(record["record_id"], "approved", None, None, "", "")
                )
            except CustomDatasetError as exc:
                print(f"cannot approve: {exc}")
            continue
        if raw not in _ANNOTATION_SHORTCUTS and raw not in CANONICAL_LABELS:
            print(f"unrecognized input {raw!r}; skipped")
            continue
        label = _ANNOTATION_SHORTCUTS.get(raw, raw)
        action = "corrected" if record["sentiment_label"] else "labeled"
        try:
            store.apply_annotation(
                AnnotationDecision(
                    record["record_id"],
                    action,
                    record["sentiment_label"],
                    label,
                    ANNOTATION_METHOD_HUMAN,
                    "",
                )
            )
        except CustomDatasetError as exc:
            print(f"error: {exc}")
            continue
        print(f"labeled '{record['record_id']}' as {label} (annotation_method=human)")
    store.write_manifest()
    return 0


def cmd_dataset_report(args: argparse.Namespace) -> int:
    store = _open_store(args)
    try:
        summary = store.dataset_summary()
    except CustomDatasetError as exc:
        return _fail(str(exc), 1)
    manifest_path = store.write_manifest()
    summary["manifest_path"] = str(manifest_path)
    _print_json(summary)
    return 0


def cmd_dataset_validate(args: argparse.Namespace) -> int:
    store = _open_store(args)
    try:
        records = store.load_records()
    except CustomDatasetError as exc:
        return _fail(str(exc), 1)
    print(f"schema OK: {len(records)} record(s) valid")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vfi",
        description="Vietnamese Feedback Intelligence: classical sentiment baselines.",
    )
    parser.add_argument("--version", action="version", version=f"vfi {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Inspect dataset integrity (privacy-safe)")
    prepare.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    prepare.set_defaults(func=cmd_prepare_data)

    train = subparsers.add_parser("train", help="Train baselines (train/validation only)")
    train.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    train.add_argument(
        "--allow-leakage-override",
        action="store_true",
        help="explicitly proceed despite train/validation leakage (the report marks "
        "the run benchmark_clean=false)",
    )
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

    dataset = subparsers.add_parser("dataset", help="Independent custom-dataset management")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)

    def _add_store_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--root", required=True, help="dataset directory (local, git-ignored)")
        sub.add_argument("--dataset-version", required=True)

    init = dataset_sub.add_parser("init", help="Initialize an empty dataset store")
    _add_store_args(init)
    init.set_defaults(func=cmd_dataset_init)

    add_syn = dataset_sub.add_parser(
        "add-synthetic", help="Add clearly-marked SYNTHETIC draft records to the annotation queue"
    )
    _add_store_args(add_syn)
    add_syn.add_argument("--count-per-class", type=int, default=3)
    add_syn.add_argument("--run-id", required=True, help="generator run id for provenance")
    add_syn.add_argument("--seed", type=int, default=42)
    add_syn.set_defaults(func=cmd_dataset_add_synthetic)

    annotate = dataset_sub.add_parser(
        "annotate", help="Interactive human annotation loop (label, correct, approve)"
    )
    _add_store_args(annotate)
    annotate.set_defaults(func=cmd_dataset_annotate)

    report = dataset_sub.add_parser("report", help="Privacy-safe dataset summary + manifest")
    _add_store_args(report)
    report.set_defaults(func=cmd_dataset_report)

    validate = dataset_sub.add_parser("validate", help="Validate the store schema")
    _add_store_args(validate)
    validate.set_defaults(func=cmd_dataset_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
