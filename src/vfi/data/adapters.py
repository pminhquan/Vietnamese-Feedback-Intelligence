"""Offline dataset adapters sharing the experiment split and provenance rules."""

from __future__ import annotations

import csv
import hashlib
import io
import stat
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vfi.data.loading import (
    CANONICAL_LABELS,
    DataLoadingError,
    Record,
    load_split_snapshot,
    normalize_text,
    parse_label_mapping,
)
from vfi.data.splitting import stratified_split
from vfi.data.validation import (
    duplicate_summary,
    find_contradictory_labels,
    fingerprint,
    leakage_summary,
)

if TYPE_CHECKING:
    from vfi.config import DataConfig

ADAPTERS_VERSION = "2.0.0"
FIVE_LEVEL_TO_CANONICAL = {
    "1": "negative",
    "2": "negative",
    "3": "neutral",
    "4": "positive",
    "5": "positive",
}
TOXIC_LABEL = "toxic"
NEU_ESC_LABELS = (*CANONICAL_LABELS, TOXIC_LABEL)


class AdapterError(DataLoadingError):
    """The selected local dataset or its access policy is invalid."""


@dataclass(frozen=True)
class AdapterResult:
    """Effective records and privacy-safe, deterministic provenance.

    ``identity`` is independent of which splits were requested.
    ``split_provenance`` has one entry per effective split. The source digest
    binds the physical input, while the fingerprint covers model-facing rows.
    ``integrity`` holds aggregate duplicate, leakage, and contradiction counts.
    """

    records_by_split: dict[str, list[Record]]
    provenance: dict[str, Any]


@dataclass
class _SourceSplit:
    records: list[Record]
    digest: str
    source_types: list[str]
    record_ids: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _dataset_identity(data: DataConfig) -> tuple[dict[str, Any], dict[int, str] | None]:
    if data.synthetic_fixture is not True and data.rights_confirmed is not True:
        raise AdapterError("Confirm the dataset's permitted use before loading it")
    adapter = data.dataset_adapter
    identity: dict[str, Any] = {
        "dataset_id": data.dataset_id,
        "dataset_version": data.dataset_version,
        "adapter": adapter,
        "adapter_version": ADAPTERS_VERSION,
        "synthetic_fixture": data.synthetic_fixture,
        "rights_confirmed": data.rights_confirmed,
        "split_policy": "physical" if data.use_official_splits else "grouped_validation",
    }
    if not data.use_official_splits:
        identity.update({"validation_size": data.validation_size, "seed": data.seed})
    if adapter == "uit_vsfc":
        expected = "synthetic_fixture" if data.synthetic_fixture else "owner_confirmed"
        if data.label_mapping_source != expected or not data.label_mapping_evidence.strip():
            raise AdapterError(
                "UIT-VSFC requires an explicit label mapping and matching provenance evidence"
            )
        mapping = parse_label_mapping(data.label_mapping)
        identity.update(
            label_mapping={str(raw): label for raw, label in sorted(mapping.items())},
            label_mapping_source=expected,
            label_mapping_evidence_sha256=_hash_text(data.label_mapping_evidence),
        )
        return identity, mapping
    if adapter == "five_level_csv":
        identity["label_mapping"] = dict(FIVE_LEVEL_TO_CANONICAL)
        identity["label_mapping_policy"] = "ordinal_five_to_three"
    elif adapter == "neu_esc":
        if data.toxic_policy not in ("error", "exclude", "keep"):
            raise AdapterError("NEU-ESC Toxic policy must be error, exclude, or keep")
        if data.toxic_policy != "error" and not data.toxic_justification.strip():
            raise AdapterError("NEU-ESC exclusion or retention of Toxic requires justification")
        identity["label_mapping"] = {label: label for label in NEU_ESC_LABELS}
        identity["toxic_policy"] = data.toxic_policy
        if data.toxic_justification:
            identity["toxic_justification_sha256"] = _hash_text(data.toxic_justification)
    elif adapter == "custom_jsonl":
        identity["label_mapping"] = {label: label for label in CANONICAL_LABELS}
        identity["annotation_policy"] = "human_approved_only"
    else:
        raise AdapterError("Unknown dataset adapter")
    return identity, None


def _check_source(path: Path, split: str) -> None:
    """Reject aliases before opening a requested physical split source."""
    absolute = path.absolute()
    try:
        for candidate in reversed((absolute, *absolute.parents)):
            info = candidate.lstat()
            reparse = getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
            if stat.S_ISLNK(info.st_mode) or reparse:
                raise AdapterError(
                    f"Split {split!r}: source path must not use links or reparse points"
                )
        if not stat.S_ISREG(info.st_mode) or info.st_nlink > 1:
            raise AdapterError(f"Split {split!r}: source must be a separate regular file")
    except FileNotFoundError:
        detail = " Create a separate test holdout before benchmarking." if split == "test" else ""
        raise AdapterError(
            f"Split {split!r}: required source {path.name} is missing.{detail}"
        ) from None
    except OSError:
        raise AdapterError(f"Split {split!r}: cannot inspect the local source") from None


def _read_csv(path: Path, split: str, data: DataConfig) -> _SourceSplit:
    try:
        raw_bytes = path.read_bytes()
        stream = io.StringIO(raw_bytes.decode("utf-8-sig"), newline="")
    except (OSError, UnicodeError):
        raise AdapterError(f"Split {split!r}: cannot read UTF-8 CSV source") from None
    reader = csv.DictReader(stream, strict=True)
    records: list[Record] = []
    toxic_count = 0
    try:
        cols = reader.fieldnames or []
        if len(cols) != len(set(cols)) or set(cols) not in (
            {"text", "label"},
            {"text", "label", "split"},
        ):
            raise AdapterError(
                f"Split {split!r}: CSV requires text,label and optional split columns"
            )
        for row in reader:
            where = f"Split {split!r} line {reader.line_num}"
            if None in row or any(value is None for value in row.values()):
                raise AdapterError(f"{where}: CSV row has the wrong number of fields")
            if "split" in row and row["split"].strip() != split:
                raise AdapterError(f"{where}: split field must match its physical file")
            text = normalize_text(row["text"])
            if not text:
                raise AdapterError(f"{where}: empty text")
            label = row["label"].strip().lower()
            if data.dataset_adapter == "five_level_csv":
                if label not in FIVE_LEVEL_TO_CANONICAL:
                    raise AdapterError(f"{where}: expected a sentiment level from 1 to 5")
                label = FIVE_LEVEL_TO_CANONICAL[label]
            elif label not in NEU_ESC_LABELS:
                raise AdapterError(f"{where}: expected a named NEU-ESC sentiment label")
            if label == TOXIC_LABEL:
                toxic_count += 1
                if data.toxic_policy == "error":
                    raise AdapterError(f"{where}: Toxic requires an explicit handling policy")
                if data.toxic_policy == "exclude":
                    continue
            records.append(Record(text, label, split))
    except csv.Error:
        raise AdapterError(f"Split {split!r} line {reader.line_num}: malformed CSV") from None
    counts = {}
    if data.dataset_adapter == "neu_esc":
        counts = {
            "toxic_count_in_source": toxic_count,
            "toxic_excluded": toxic_count if data.toxic_policy == "exclude" else 0,
        }
    origin = "synthetic" if data.synthetic_fixture else "external"
    return _SourceSplit(
        records, hashlib.sha256(raw_bytes).hexdigest(), [origin] * len(records), counts=counts
    )


def _read_custom(path: Path, split: str, data: DataConfig) -> _SourceSplit:
    from vfi.custom_dataset import (
        ANNOTATION_STATUS_APPROVED,
        SOURCE_TYPE_SYNTHETIC,
        CustomDatasetError,
        read_records,
        record_fingerprint,
    )

    try:
        rows = read_records(path, expected_version=data.dataset_version)
        digest = record_fingerprint(rows)
    except CustomDatasetError as exc:
        # Store validation errors contain only safe field names and positions.
        raise AdapterError(f"Split {split!r}: {exc}") from None
    except (OSError, UnicodeError):
        raise AdapterError(
            f"Split {split!r}: invalid custom JSONL schema, version or annotation; "
            "validate the local dataset and export again"
        ) from None
    records: list[Record] = []
    source_types: list[str] = []
    for index, row in enumerate(rows, start=1):
        if row.get("split", split) != split:
            raise AdapterError(
                f"Split {split!r} record {index}: split must match its physical file"
            )
        if row["annotation_status"] != ANNOTATION_STATUS_APPROVED:
            continue
        is_synthetic = row["source_type"] == SOURCE_TYPE_SYNTHETIC
        if is_synthetic != data.synthetic_fixture:
            raise AdapterError(
                f"Split {split!r} record {index}: source type conflicts with fixture mode"
            )
        records.append(Record(normalize_text(row["text"]), row["sentiment_label"], split))
        source_types.append(row["source_type"])
    return _SourceSplit(
        records,
        digest,
        source_types,
        [row["record_id"] for row in rows],
        {"unapproved_records_in_source": len(rows) - len(records)},
    )


def _read_split(
    root: Path, split: str, data: DataConfig, mapping: dict[int, str] | None
) -> _SourceSplit:
    if data.dataset_adapter == "uit_vsfc":
        for name in ("sentences.txt", "sentiments.txt"):
            _check_source(root / split / name, split)
        records, digest = load_split_snapshot(root, split, mapping)
        origin = "synthetic" if data.synthetic_fixture else "external"
        return _SourceSplit(records, digest, [origin] * len(records))
    is_custom = data.dataset_adapter == "custom_jsonl"
    path = root / f"{split}.{'jsonl' if is_custom else 'csv'}"
    _check_source(path, split)
    return _read_custom(path, split, data) if is_custom else _read_csv(path, split, data)


def _split_provenance(
    records: list[Record], source: _SourceSplit, *, source_split: str
) -> dict[str, Any]:
    # Grouped fallback never splits equal texts, so text membership selects
    # the originating rows (including all repeated copies) unambiguously.
    effective_texts = {record.text for record in records}
    counts = Counter(
        origin
        for record, origin in zip(source.records, source.source_types, strict=True)
        if record.text in effective_texts
    )
    return {
        "fingerprint": fingerprint(records),
        "source_split": source_split,
        "source_content_sha256": source.digest,
        "source_type_counts": dict(sorted(counts.items())),
        **source.counts,
    }


def load_dataset_for_config(data: DataConfig, *, include_test: bool = False) -> AdapterResult:
    """Load only authorized physical sources for the requested experiment stage.

    CSV and custom JSONL use ``train``, ``validation`` and ``test`` filenames
    with their format extension. Training and ordinary preparation do not
    even inspect the test path. A grouped fallback reads train only and never
    generates a test holdout.
    """
    identity, mapping = _dataset_identity(data)
    root = Path(data.raw_dir)
    requested = ("train", "validation") if data.use_official_splits else ("train",)
    if include_test:
        requested += ("test",)
    sources = {split: _read_split(root, split, data, mapping) for split in requested}
    seen_ids: set[str] = set()
    for split, source in sources.items():
        if seen_ids.intersection(source.record_ids):
            raise AdapterError(f"Split {split!r}: a record ID is repeated across loaded splits")
        seen_ids.update(source.record_ids)
        if not source.records:
            raise AdapterError(f"Split {split!r}: no eligible records under the selected policy")
    records = {split: source.records for split, source in sources.items()}
    if not data.use_official_splits:
        try:
            train, validation = stratified_split(
                records["train"], validation_size=data.validation_size, seed=data.seed
            )
        except ValueError as exc:
            raise AdapterError(f"Grouped validation split failed: {exc}") from None
        records = {
            "train": train,
            "validation": validation,
            **({"test": records["test"]} if include_test else {}),
        }
    split_provenance = {}
    for split, effective in records.items():
        source_split = "train" if split == "validation" and not data.use_official_splits else split
        split_provenance[split] = _split_provenance(
            effective, sources[source_split], source_split=source_split
        )
    return AdapterResult(
        records,
        {
            "identity": identity,
            "split_provenance": split_provenance,
            "integrity": {
                "duplicates_by_split": {
                    split: duplicate_summary(values) for split, values in records.items()
                },
                "contradictions": find_contradictory_labels(*records.items()),
                "leakage": leakage_summary(*records.items()),
            },
        },
    )
