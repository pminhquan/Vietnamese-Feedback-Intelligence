"""Local feedback collection, human review, and reproducible JSONL export."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from random import Random
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any
from uuid import uuid4

from vfi.data.loading import CANONICAL_LABELS, SPLITS, Record, normalize_text
from vfi.data.splitting import stratified_split
from vfi.data.validation import class_distribution

CUSTOM_DATASET_FIELDS = (
    "record_id",
    "text",
    "sentiment_label",
    "source_type",
    "provenance_reference",
    "annotation_method",
    "annotation_status",
    "dataset_version",
)
SOURCE_TYPE_SYNTHETIC = "synthetic"
SOURCE_TYPE_OWNER_COLLECTED = "owner_collected"
SOURCE_TYPE_THIRD_PARTY_IMPORT = "third_party_import"
SOURCE_TYPES = (
    SOURCE_TYPE_SYNTHETIC,
    SOURCE_TYPE_OWNER_COLLECTED,
    SOURCE_TYPE_THIRD_PARTY_IMPORT,
)
ANNOTATION_METHOD_HUMAN = "human"
ANNOTATION_METHOD_SYNTHETIC = "synthetic_generator"
ANNOTATION_METHODS = (ANNOTATION_METHOD_HUMAN, ANNOTATION_METHOD_SYNTHETIC)
ANNOTATION_STATUS_DRAFT = "draft"
ANNOTATION_STATUS_ANNOTATED = "annotated"
ANNOTATION_STATUS_APPROVED = "approved"
ANNOTATION_STATUSES = (
    ANNOTATION_STATUS_DRAFT,
    ANNOTATION_STATUS_ANNOTATED,
    ANNOTATION_STATUS_APPROVED,
)
RECORDS_FILENAME = "records.jsonl"
MANIFEST_FILENAME = "dataset_manifest.json"


class CustomDatasetError(Exception):
    """Invalid schema, annotation transition, or dataset store operation."""


def _validate_version(version: Any) -> None:
    if not isinstance(version, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", version
    ):
        raise CustomDatasetError("dataset_version must be a version such as v1 or 1.0.0")


def validate_record(record: Any, *, line_no: int | None = None) -> None:
    """Validate fields without putting record contents in error messages."""
    where = f"Line {line_no}" if line_no is not None else "Record"
    if not isinstance(record, dict):
        raise CustomDatasetError(f"{where}: expected a JSON object")
    if set(record) - set(CUSTOM_DATASET_FIELDS) - {"split"}:
        raise CustomDatasetError(f"{where}: unknown field")
    missing = [field for field in CUSTOM_DATASET_FIELDS if field not in record]
    if missing:
        raise CustomDatasetError(f"{where}: missing required fields ({', '.join(missing)})")
    for field in ("record_id", "text", "provenance_reference"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise CustomDatasetError(f"{where}: {field} must be a non-empty string")
        try:
            record[field].encode("utf-8")
        except UnicodeError:
            raise CustomDatasetError(f"{where}: {field} must contain valid Unicode") from None
    try:
        _validate_version(record["dataset_version"])
    except CustomDatasetError as exc:
        raise CustomDatasetError(f"{where}: {exc}") from None
    if "split" in record and record["split"] not in SPLITS:
        raise CustomDatasetError(f"{where}: split must be train, validation, or test")
    for field, allowed in (
        ("source_type", SOURCE_TYPES),
        ("annotation_method", ("", *ANNOTATION_METHODS)),
        ("annotation_status", ANNOTATION_STATUSES),
        ("sentiment_label", ("", *CANONICAL_LABELS)),
    ):
        if record[field] not in allowed:
            raise CustomDatasetError(f"{where}: invalid {field}")
    label = record["sentiment_label"]
    method = record["annotation_method"]
    status = record["annotation_status"]
    if bool(label) != bool(method) or (status != ANNOTATION_STATUS_DRAFT and not label):
        raise CustomDatasetError(f"{where}: labeled records require both label and method")
    if status == ANNOTATION_STATUS_APPROVED and method != ANNOTATION_METHOD_HUMAN:
        raise CustomDatasetError(f"{where}: approval requires a human annotation")


def _validated_rows(
    rows: Iterable[tuple[int, Any]], expected_version: str | None = None
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}
    version = expected_version
    for line_no, row in rows:
        validate_record(row, line_no=line_no)
        if version is None:
            version = row["dataset_version"]
        if row["dataset_version"] != version:
            raise CustomDatasetError(f"Line {line_no}: dataset_version does not match the store")
        record_id = row["record_id"]
        if record_id in seen_ids:
            raise CustomDatasetError(
                f"Line {line_no}: duplicate record_id (first seen on line {seen_ids[record_id]})"
            )
        seen_ids[record_id] = line_no
        records.append({**row, "text": normalize_text(row["text"])})
    return records


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CustomDatasetError("duplicate JSON object key")
        result[key] = value
    return result


def read_records(path: Path, *, expected_version: str | None = None) -> list[dict[str, Any]]:
    """Read one physical JSONL source and validate its version and annotations."""
    try:
        contents = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        raise CustomDatasetError("Cannot read the JSONL source as UTF-8") from None
    rows: list[tuple[int, Any]] = []
    for line_no, line in enumerate(contents.split("\n"), 1):
        if not line.strip():
            continue
        try:
            rows.append((line_no, json.loads(line, object_pairs_hook=_unique_json_object)))
        except CustomDatasetError as exc:
            raise CustomDatasetError(f"Line {line_no}: {exc}") from None
        except (ValueError, RecursionError):
            raise CustomDatasetError(f"Line {line_no}: invalid JSON") from None
    return _validated_rows(rows, expected_version)


def _jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )


def record_fingerprint(records: list[dict[str, Any]]) -> str:
    """Hash text, labels, source, version, and annotation provenance together."""
    return hashlib.sha256(_jsonl(records).encode("utf-8")).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, mode="w", encoding="utf-8", newline="\n", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _store_write_lock(root: Path) -> Iterator[None]:
    """Serialize local writers on a stable file that is never replaced or removed."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".dataset.lock").open("a+b") as handle:
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            except OSError:
                raise CustomDatasetError(
                    "Cannot acquire the dataset write lock. Retry shortly."
                ) from None
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def make_record(
    *,
    record_id: str,
    text: str,
    source_type: str,
    provenance_reference: str,
    dataset_version: str,
    sentiment_label: str = "",
    annotation_method: str = "",
    annotation_status: str = ANNOTATION_STATUS_DRAFT,
) -> dict[str, Any]:
    record = {
        "record_id": record_id,
        "text": text,
        "sentiment_label": sentiment_label,
        "source_type": source_type,
        "provenance_reference": provenance_reference,
        "annotation_method": annotation_method,
        "annotation_status": annotation_status,
        "dataset_version": dataset_version,
    }
    validate_record(record)
    record["text"] = normalize_text(text)
    return record


def generate_synthetic_draft(
    count_per_class: int, *, dataset_version: str, run_id: str, seed: int = 42
) -> list[dict[str, Any]]:
    """Create deterministic test candidates with machine labels and draft status."""
    if count_per_class < 1 or not run_id.strip():
        raise CustomDatasetError("Generation requires a positive count and non-empty run ID")
    templates = {
        "positive": (
            "Giảng viên giải thích rõ ràng và rất nhiệt tình.",
            "Môn học này rất hữu ích cho sinh viên.",
            "Nội dung thú vị và tài liệu đầy đủ.",
        ),
        "negative": (
            "Giảng viên giảng quá nhanh, sinh viên không theo kịp.",
            "Bài tập khó hiểu và thiếu hướng dẫn.",
            "Phòng học nóng, máy chiếu thường xuyên bị hỏng.",
        ),
        "neutral": (
            "Môn học diễn ra vào thứ Hai hàng tuần.",
            "Giảng viên dạy tại phòng A101 vào buổi sáng.",
            "Lớp có ba mươi sinh viên đăng ký.",
        ),
    }
    rng = Random(seed)
    run_key = hashlib.sha256(f"{run_id}:{seed}".encode()).hexdigest()[:16]
    return [
        make_record(
            record_id=f"syn-{run_key}-{index}-{label}",
            text=f"{options[index % len(options)]} (ví dụ {index + 1}, mã {rng.randrange(10000)})",
            source_type=SOURCE_TYPE_SYNTHETIC,
            provenance_reference=f"synthetic draft run {run_id} (seed {seed})",
            dataset_version=dataset_version,
            sentiment_label=label,
            annotation_method=ANNOTATION_METHOD_SYNTHETIC,
        )
        for index in range(count_per_class)
        for label, options in templates.items()
    ]


class CustomDatasetStore:
    """Store source and review state locally in versioned JSONL records."""

    def __init__(self, root: str | Path, dataset_version: str) -> None:
        _validate_version(dataset_version)
        self.root = Path(root)
        self.dataset_version = dataset_version
        self.records_path = self.root / RECORDS_FILENAME
        self.manifest_path = self.root / MANIFEST_FILENAME

    def initialize(self) -> None:
        with _store_write_lock(self.root):
            self._check_manifest_version()
            if not self.records_path.exists():
                self.records_path.touch(exist_ok=False)
            self._write_manifest()

    def _check_manifest_version(self) -> None:
        if self.manifest_path.exists():
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise CustomDatasetError("Cannot read dataset manifest") from None
            if (
                not isinstance(manifest, dict)
                or manifest.get("dataset_version") != self.dataset_version
            ):
                raise CustomDatasetError("dataset_version does not match the store manifest")

    def load_records(self) -> list[dict[str, Any]]:
        self._check_manifest_version()
        return read_records(self.records_path, expected_version=self.dataset_version)

    def append_records(self, new_records: list[dict[str, Any]]) -> int:
        # Validate the complete batch before replacing the records file.
        with _store_write_lock(self.root):
            rows = (
                self.load_records()
                if self.records_path.exists() or self.manifest_path.exists()
                else []
            )
            validated = _validated_rows(enumerate([*rows, *new_records], 1), self.dataset_version)
            _atomic_write(self.records_path, _jsonl(validated))
        return len(new_records)

    def _record_for_update(
        self, record_id: str, expected_record: dict[str, Any] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        records = self.load_records()
        for record in records:
            if record["record_id"] == record_id:
                if expected_record is not None and expected_record != record:
                    raise CustomDatasetError("Record changed during review. Review it again.")
                return records, record
        raise CustomDatasetError("Unknown record_id")

    def annotate(
        self, record_id: str, label: str, *, expected_record: dict[str, Any] | None = None
    ) -> None:
        if label not in CANONICAL_LABELS:
            raise CustomDatasetError("Choose a canonical sentiment label")
        with _store_write_lock(self.root):
            records, record = self._record_for_update(record_id, expected_record)
            record.update(
                sentiment_label=label,
                annotation_method=ANNOTATION_METHOD_HUMAN,
                annotation_status=ANNOTATION_STATUS_ANNOTATED,
            )
            _atomic_write(self.records_path, _jsonl(records))

    def approve(self, record_id: str, *, expected_record: dict[str, Any] | None = None) -> None:
        with _store_write_lock(self.root):
            records, record = self._record_for_update(record_id, expected_record)
            if (
                record["annotation_method"] != ANNOTATION_METHOD_HUMAN
                or not record["sentiment_label"]
            ):
                raise CustomDatasetError("Enter a human annotation before approving the record")
            record["annotation_status"] = ANNOTATION_STATUS_APPROVED
            _atomic_write(self.records_path, _jsonl(records))

    def dataset_summary(self) -> dict[str, Any]:
        records = self.load_records()
        approved = [r for r in records if r["annotation_status"] == ANNOTATION_STATUS_APPROVED]
        labels_by_text: defaultdict[str, set[str]] = defaultdict(set)
        approved_labels_by_text: defaultdict[str, set[str]] = defaultdict(set)
        for record in records:
            if record["sentiment_label"]:
                labels_by_text[record["text"]].add(record["sentiment_label"])
            if record["annotation_status"] == ANNOTATION_STATUS_APPROVED:
                approved_labels_by_text[record["text"]].add(record["sentiment_label"])
        return {
            "dataset_version": self.dataset_version,
            "total_records": len(records),
            "labeled_records": sum(bool(record["sentiment_label"]) for record in records),
            "approved_records": len(approved),
            "records_by_source_type": dict(Counter(record["source_type"] for record in records)),
            "approved_by_source_type": dict(Counter(record["source_type"] for record in approved)),
            "records_by_annotation_method": dict(Counter(r["annotation_method"] for r in records)),
            "class_distribution_approved": class_distribution(
                [Record(r["text"], r["sentiment_label"], "train") for r in approved]
            ),
            "duplicate_text_count": sum(
                count > 1 for count in Counter(r["text"] for r in records).values()
            ),
            "contradictory_text_count": sum(len(labels) > 1 for labels in labels_by_text.values()),
            "contradictory_approved_text_count": sum(
                len(labels) > 1 for labels in approved_labels_by_text.values()
            ),
            "fingerprint": record_fingerprint(records),
        }

    def write_manifest(self) -> Path:
        with _store_write_lock(self.root):
            return self._write_manifest()

    def _write_manifest(self) -> Path:
        _atomic_write(self.manifest_path, json.dumps(self.dataset_summary(), indent=2) + "\n")
        return self.manifest_path

    def export_splits(
        self,
        out: Path,
        *,
        seed: int = 42,
        validation_size: float = 0.2,
        test_size: float = 0.2,
        include_synthetic: bool = False,
    ) -> dict[str, Any]:
        """Materialize a new holdout directory from reviewed records."""
        if out.exists():
            raise CustomDatasetError("Choose a new output directory for the split export")
        if not (0 < validation_size < 1 and 0 < test_size < 1 - validation_size):
            raise CustomDatasetError(
                "Validation and test fractions must be positive and sum to less than 1"
            )
        rows = self.load_records()
        approved = [row for row in rows if row["annotation_status"] == ANNOTATION_STATUS_APPROVED]
        selected = [
            row
            for row in approved
            if (row["source_type"] == SOURCE_TYPE_SYNTHETIC) == include_synthetic
        ]
        if not selected:
            raise CustomDatasetError("No approved records for the selected real or synthetic mode")
        labels: dict[str, str] = {}
        for row in selected:
            if labels.setdefault(row["text"], row["sentiment_label"]) != row["sentiment_label"]:
                raise CustomDatasetError(
                    "Resolve contradictory labels among approved records before splitting"
                )
        groups = [Record(text, label, "train") for text, label in labels.items()]
        try:
            train_val, test = stratified_split(groups, validation_size=test_size, seed=seed)
            train, validation = stratified_split(
                train_val, validation_size=validation_size / (1 - test_size), seed=seed
            )
        except ValueError:
            raise CustomDatasetError(
                "Cannot form stratified splits. Add distinct approved texts per class "
                "or adjust the fractions."
            ) from None
        partitions = dict(zip(SPLITS, (train, validation, test), strict=True))
        if any({r.label for r in group} != set(CANONICAL_LABELS) for group in partitions.values()):
            raise CustomDatasetError(
                "Every split needs all canonical labels. Add distinct texts per class."
            )
        assigned = {record.text: split for split, group in partitions.items() for record in group}
        exported: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
        for row in selected:
            split = assigned[row["text"]]
            exported[split].append({**row, "split": split})
        manifest = {
            "dataset_version": self.dataset_version,
            "data_kind": "synthetic_test_fixture" if include_synthetic else "real_feedback",
            "seed": seed,
            "validation_size": validation_size,
            "test_size": test_size,
            "approved_other_source_count": len(approved) - len(selected),
            "unapproved_count": len(rows) - len(approved),
            "split_counts": {split: len(split_rows) for split, split_rows in exported.items()},
            "split_fingerprints": {
                split: record_fingerprint(split_rows) for split, split_rows in exported.items()
            },
        }
        # Stage the complete export beside its destination before publishing it.
        out.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=out.parent) as temp_path:
            temp = Path(temp_path)
            for split, split_rows in exported.items():
                (temp / f"{split}.jsonl").write_text(
                    _jsonl(split_rows), encoding="utf-8", newline="\n"
                )
            (temp / MANIFEST_FILENAME).write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            temp.rename(out)
        return manifest


def _read_local_prompt(prompt: str) -> str:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise CustomDatasetError("Human entry and review require a local interactive terminal")
    return input(prompt)


def _review_record(store: CustomDatasetStore, args: argparse.Namespace) -> str:
    records = store.load_records()
    wanted = (
        ANNOTATION_STATUS_DRAFT
        if args.dataset_command == "annotate"
        else ANNOTATION_STATUS_ANNOTATED
    )
    row = next(
        (
            record
            for record in records
            if (
                record["record_id"] == args.record_id
                if args.record_id is not None
                else record["annotation_status"] == wanted
            )
        ),
        None,
    )
    if row is None:
        raise CustomDatasetError("No matching record to review")
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise CustomDatasetError("Human entry and review require a local interactive terminal")
    print(f"Local review of record {json.dumps(row['record_id'], ensure_ascii=False)}")
    print(f"Source: {row['source_type']}. Current label: {row['sentiment_label'] or '(none)'}.")
    # JSON escaping makes terminal controls in imported text visible rather
    # than executing them. Text is shown only in this interactive review.
    print(json.dumps(row["text"], ensure_ascii=False))
    if args.dataset_command == "annotate":
        label = _read_local_prompt("Enter negative, neutral, positive, or blank to skip: ").strip()
        if not label:
            return "skipped"
        store.annotate(row["record_id"], label, expected_record=row)
        return "annotated"
    if _read_local_prompt("Type approve to confirm the human label: ").strip() != "approve":
        return "skipped"
    store.approve(row["record_id"], expected_record=row)
    return "approved"


def _dataset_command(args: argparse.Namespace) -> int:
    try:
        store = CustomDatasetStore(args.root, args.dataset_version)
        command = args.dataset_command
        if command == "init":
            store.initialize()
            result = {"status": "initialized", "dataset_version": store.dataset_version}
        elif command in ("add", "add-synthetic"):
            if command == "add":
                records = [
                    make_record(
                        record_id=uuid4().hex,
                        text=_read_local_prompt("Enter feedback from an authorized source: "),
                        provenance_reference=_read_local_prompt(
                            "Enter a local provenance reference (avoid private identifiers): "
                        ),
                        source_type=args.source_type,
                        dataset_version=store.dataset_version,
                    )
                ]
            else:
                records = generate_synthetic_draft(
                    args.count_per_class,
                    dataset_version=store.dataset_version,
                    run_id=args.run_id,
                    seed=args.seed,
                )
            count = store.append_records(records)
            store.write_manifest()
            result = {"drafts_added": count, "source_type": records[0]["source_type"]}
        elif command in ("annotate", "approve"):
            result = {"status": _review_record(store, args)}
            store.write_manifest()
        elif command == "validate":
            result = {
                **store.dataset_summary(),
                "schema_valid": True,
                "benchmark_ready": False,
                "validation_scope": "Record integrity only. A valid split preflight is required.",
            }
        elif command == "report":
            result = store.dataset_summary()
        else:
            result = store.export_splits(
                Path(args.out),
                seed=args.seed,
                validation_size=args.validation_size,
                test_size=args.test_size,
                include_synthetic=args.include_synthetic,
            )
    except CustomDatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, UnicodeError, EOFError):
        print("error: dataset I/O could not be completed", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    if command == "validate" and result["contradictory_approved_text_count"]:
        print("error: conflicting labels among approved records", file=sys.stderr)
        return 1
    return 0


def add_dataset_parser(subparsers: Any) -> None:
    """Register local custom-dataset management on the main CLI."""
    parser = subparsers.add_parser("dataset", help="Manage and export local annotated feedback")
    commands = parser.add_subparsers(dest="dataset_command", required=True)

    def command(name: str, description: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=description)
        sub.add_argument("--root", required=True)
        sub.add_argument("--dataset-version", required=True)
        sub.set_defaults(func=_dataset_command)
        return sub

    command("init", "Initialize a versioned JSONL store")
    add = command("add", "Enter an authorized feedback record in a local terminal")
    add.add_argument(
        "--source-type",
        choices=(SOURCE_TYPE_OWNER_COLLECTED, SOURCE_TYPE_THIRD_PARTY_IMPORT),
        default=SOURCE_TYPE_OWNER_COLLECTED,
    )
    synth = command("add-synthetic", "Generate synthetic draft records for testing")
    synth.add_argument("--count-per-class", type=int, default=3)
    synth.add_argument("--run-id", required=True)
    synth.add_argument("--seed", type=int, default=42)
    annotate = command("annotate", "Label a draft or correct an existing record locally")
    annotate.add_argument("--record-id", help="Select a specific record for labeling or correction")
    approve = command("approve", "Approve a human annotation after local review")
    approve.add_argument(
        "--record-id", help="Select a specific human-annotated record for approval"
    )
    command("validate", "Validate schema and version without displaying feedback")
    command("report", "Report aggregate counts and dataset fingerprint")
    split = command("split", "Write new grouped train/validation/test JSONL files")
    split.add_argument("--out", required=True)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--validation-size", type=float, default=0.2)
    split.add_argument("--test-size", type=float, default=0.2)
    split.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Export only approved synthetic test fixtures",
    )
