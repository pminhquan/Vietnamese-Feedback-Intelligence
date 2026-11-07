"""Independent custom-dataset management: records, annotation, versioning.

This module implements Workstream 2 of the dataset-independence plan: an
independently created Vietnamese feedback dataset with explicit provenance
and human annotation, so the project never depends on UIT-VSFC.

Schema (one JSON object per line, JSONL UTF-8(-sig)):

- ``record_id``: unique, non-empty string (stable identity across versions).
- ``text``: the feedback sentence (normalized on write; original preserved
  verbatim in ``text_raw`` when it differs after normalization).
- ``sentiment_label``: ``negative`` | ``neutral`` | ``positive`` (or empty
  while unannotated).
- ``source_type``: ``synthetic`` | ``owner_collected`` | ``third_party_import``.
- ``provenance_reference``: where the text came from (generator run id,
  collection date/context, source URL). Non-empty.
- ``annotation_method``: ``synthetic_generator`` | ``human`` | ```` (none yet).
- ``annotation_status``: ``draft`` | ``annotated`` | ``approved``.
- ``dataset_version``: version string the record was written under.

HONESTY INVARIANTS (enforced by :class:`CustomDatasetStore`):

- A record can reach ``approved`` only with ``annotation_method="human"`` —
  synthetic-generator labels stay ``draft``/``annotated`` at best.
- ``annotation_method="synthetic_generator"`` marks AI-generated labels; the
  owner may CORRECT such a record through the annotation CLI, which switches
  the method to ``human`` and keeps ``source_type`` unchanged (a human
  annotation of a synthetic text is a human annotation of synthetic data).
- Counts by ``source_type`` are reported in every dataset summary so
  synthetic records can never silently pass as real user feedback.
- No annotation agreement, annotator count, or dataset quality is ever
  inferred or invented here.

Storage layout (all local, git-ignored dataset directory):
``<root>/records.jsonl`` (append-friendly store) and
``<root>/dataset_manifest.json`` (version + counters + fingerprints).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Re-exported single source of truth shared with the JSONL adapter.
from vfi.data.adapters import (
    ANNOTATION_METHOD_HUMAN,
    ANNOTATION_METHOD_SYNTHETIC,
    ANNOTATION_METHODS,
    ANNOTATION_STATUS_ANNOTATED,
    ANNOTATION_STATUS_APPROVED,
    ANNOTATION_STATUS_DRAFT,
    ANNOTATION_STATUSES,
    CUSTOM_DATASET_FIELDS,
    SOURCE_TYPE_SYNTHETIC,
    SOURCE_TYPES,
)
from vfi.data.loading import CANONICAL_LABELS, normalize_text
from vfi.data.validation import (
    class_distribution,
    duplicate_summary,
    find_contradictory_labels,
    fingerprint,
    near_duplicate_candidates,
)

RECORDS_FILENAME = "records.jsonl"
MANIFEST_FILENAME = "dataset_manifest.json"


class CustomDatasetError(Exception):
    """Raised when a dataset store operation violates the schema or invariants."""


@dataclass(frozen=True)
class AnnotationDecision:
    """Result of one annotation action (used by the CLI annotate loop)."""

    record_id: str
    action: str  # "labeled" | "corrected" | "approved" | "skipped" | "noop"
    old_label: str | None
    new_label: str | None
    method: str
    status: str


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
    """Build a schema-valid record dict (raises on invariant violations)."""
    record: dict[str, Any] = {
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
    record["text"] = normalize_text(record["text"])
    return record


def validate_record(record: dict[str, Any], *, line_no: int | None = None) -> None:
    """Validate one record against the schema and honesty invariants.

    Raises:
        CustomDatasetError: naming the offending field (and line when given).
    """
    where = f"record '{record.get('record_id', '?')}'" + (
        f" (line {line_no})" if line_no is not None else ""
    )
    unknown = sorted(set(record) - set(CUSTOM_DATASET_FIELDS) - {"text_raw", "split"})
    if unknown:
        raise CustomDatasetError(f"{where}: unknown field(s) {unknown}")
    missing = [field for field in CUSTOM_DATASET_FIELDS if field not in record]
    if missing:
        raise CustomDatasetError(f"{where}: missing field(s) {missing}")
    for field in ("record_id", "provenance_reference", "dataset_version"):
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            raise CustomDatasetError(f"{where}: {field} must be a non-empty string")
    if not isinstance(record["text"], str) or not normalize_text(record["text"]):
        raise CustomDatasetError(f"{where}: text must be a non-empty string")
    label = record["sentiment_label"]
    if label != "" and label not in CANONICAL_LABELS:
        raise CustomDatasetError(
            f"{where}: sentiment_label {label!r}; expected one of {list(CANONICAL_LABELS)} or ''"
        )
    if record["source_type"] not in SOURCE_TYPES:
        raise CustomDatasetError(
            f"{where}: source_type {record['source_type']!r}; expected one of {SOURCE_TYPES}"
        )
    if record["annotation_status"] not in ANNOTATION_STATUSES:
        raise CustomDatasetError(
            f"{where}: annotation_status {record['annotation_status']!r}; "
            f"expected one of {ANNOTATION_STATUSES}"
        )
    method = record["annotation_method"]
    if method not in ("", *ANNOTATION_METHODS):
        raise CustomDatasetError(
            f"{where}: annotation_method {method!r}; expected one of {['', *ANNOTATION_METHODS]}"
        )
    if record["annotation_status"] == ANNOTATION_STATUS_APPROVED:
        if not label:
            raise CustomDatasetError(f"{where}: approved records must carry a sentiment_label")
        if method != ANNOTATION_METHOD_HUMAN:
            raise CustomDatasetError(
                f"{where}: only human annotations may be approved (annotation_method="
                f"{method!r} cannot reach 'approved'; synthetic labels stay draft/annotated)"
            )


class CustomDatasetStore:
    """Local, versioned store for the independently built dataset.

    The store is a JSONL file plus a manifest. Writes are explicit and
    append-friendly: existing lines are never rewritten except through
    :meth:`apply_annotation` (same-line replacement of the annotated record).
    """

    def __init__(self, root: str | Path, dataset_version: str) -> None:
        self.root = Path(root)
        self.dataset_version = dataset_version
        self.records_path = self.root / RECORDS_FILENAME
        self.manifest_path = self.root / MANIFEST_FILENAME

    # -- reading ----------------------------------------------------------

    def load_records(self) -> list[dict[str, Any]]:
        if not self.records_path.exists():
            return []
        records: list[dict[str, Any]] = []
        seen_ids: dict[str, int] = {}
        for line_no, line in enumerate(
            self.records_path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CustomDatasetError(
                    f"{self.records_path} line {line_no}: invalid JSON ({exc.msg})"
                ) from exc
            try:
                validate_record(payload, line_no=line_no)
            except CustomDatasetError as exc:
                raise CustomDatasetError(str(exc)) from exc
            if payload["record_id"] in seen_ids:
                first_line = seen_ids[payload["record_id"]]
                raise CustomDatasetError(
                    f"{self.records_path} line {line_no}: duplicate record_id "
                    f"'{payload['record_id']}' (first seen on line {first_line})"
                )
            seen_ids[payload["record_id"]] = line_no
            records.append(payload)
        return records

    # -- writing ----------------------------------------------------------

    def append_records(self, new_records: list[dict[str, Any]]) -> int:
        """Append schema-valid records; rejects duplicate record_ids."""
        existing = self.load_records()
        existing_ids = {record["record_id"] for record in existing}
        self.root.mkdir(parents=True, exist_ok=True)
        added = 0
        with self.records_path.open("a", encoding="utf-8") as stream:
            for record in new_records:
                validate_record(record)
                normalized = dict(record)
                normalized["text"] = normalize_text(record["text"])
                if normalized["record_id"] in existing_ids:
                    raise CustomDatasetError(
                        f"duplicate record_id '{normalized['record_id']}' already in the store"
                    )
                existing_ids.add(normalized["record_id"])
                stream.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                added += 1
        return added

    def apply_annotation(self, decision: AnnotationDecision) -> dict[str, Any]:
        """Apply one annotation decision in place (same-line rewrite)."""
        records = self.load_records()
        target = None
        for index, record in enumerate(records):
            if record["record_id"] == decision.record_id:
                target = (index, record)
                break
        if target is None:
            raise CustomDatasetError(f"unknown record_id '{decision.record_id}'")
        index, record = target
        if decision.action in ("labeled", "corrected"):
            record["sentiment_label"] = decision.new_label or ""
            record["annotation_method"] = decision.method
            record["annotation_status"] = (
                ANNOTATION_STATUS_ANNOTATED
                if record["annotation_status"] != ANNOTATION_STATUS_APPROVED
                else record["annotation_status"]
            )
        elif decision.action == "approved":
            # Honesty invariant: approval is a human act on a HUMAN annotation.
            # A synthetic_generator label must be personally reviewed/corrected
            # through a human annotation before it can reach 'approved'.
            if record["annotation_method"] != ANNOTATION_METHOD_HUMAN:
                raise CustomDatasetError(
                    f"record '{decision.record_id}' carries a "
                    f"'{record['annotation_method'] or 'missing'}' annotation; the owner "
                    "must personally label or correct it (method='human') before approval"
                )
            if not record["sentiment_label"]:
                raise CustomDatasetError(
                    f"record '{decision.record_id}' has no label; label it before approving"
                )
            record["annotation_status"] = ANNOTATION_STATUS_APPROVED
        validate_record(record)
        lines = self.records_path.read_text(encoding="utf-8-sig").splitlines()
        # records preserve store order; line index == record index because
        # blank lines are skipped on read, so rewrite by re-serializing all.
        rewritten = [json.dumps(rec, ensure_ascii=False) for rec in records]
        self.records_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
        del index, lines
        return record

    # -- reporting --------------------------------------------------------

    def dataset_summary(self) -> dict[str, Any]:
        """Aggregate, privacy-safe dataset report (no raw text)."""
        records = self.load_records()
        labeled = [record for record in records if record["sentiment_label"]]
        approved = [record for record in records if record["annotation_status"] == "approved"]
        by_source: dict[str, int] = {}
        for record in records:
            by_source[record["source_type"]] = by_source.get(record["source_type"], 0) + 1
        by_method: dict[str, int] = {}
        for record in records:
            method = record["annotation_method"] or "(none)"
            by_method[method] = by_method.get(method, 0) + 1

        def _as_fake(record: dict[str, Any]) -> Any:
            return type(
                "R",
                (),
                {"text": record["text"], "label": record["sentiment_label"], "split": "train"},
            )()

        # Data-quality signals run over ALL records (they describe the store);
        # class distribution and fingerprints describe the training-eligible
        # approved subset only.
        all_as = [_as_fake(record) for record in records]
        approved_as = [_as_fake(record) for record in approved]
        contradictory = find_contradictory_labels(("store", all_as))
        near_dups = near_duplicate_candidates(all_as) if len(all_as) >= 2 else []
        return {
            "dataset_version": self.dataset_version,
            "total_records": len(records),
            "labeled_records": len(labeled),
            "approved_records": len(approved),
            "records_by_source_type": by_source,
            "records_by_annotation_method": by_method,
            "class_distribution_approved": class_distribution(approved_as) if approved else {},
            "duplicate_summary": duplicate_summary(all_as),
            "contradictory_labels": contradictory,
            "near_duplicate_candidates": near_dups,
            "fingerprint_approved": fingerprint(approved_as) if approved else "",
            "honesty_note": (
                "source_type counts distinguish synthetic from real records; synthetic "
                "labels can never be 'approved'; no annotation agreement or quality is "
                "inferred here"
            ),
        }

    def write_manifest(self) -> Path:
        """Write the dataset manifest (version, summary, fingerprints)."""
        summary = self.dataset_summary()
        manifest = {
            "dataset_version": self.dataset_version,
            "generated_at": datetime.now(UTC).isoformat(),
            **summary,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.manifest_path


def generate_synthetic_draft(
    count_per_class: int,
    *,
    dataset_version: str,
    run_id: str,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Draft SYNTHETIC records for pipeline tests and annotation queues.

    Every record is marked ``source_type="synthetic"`` and
    ``annotation_method="synthetic_generator"`` with status ``draft``:
    these labels are machine-generated candidates for the owner to review,
    correct, and (only via human action) approve. They are never real user
    feedback and are never counted as such.
    """
    import random

    rng = random.Random(seed)
    templates = {
        "positive": [
            "Giảng viên giảng dạy rất dễ hiểu và nhiệt tình.",
            "Môn học này rất hữu ích cho sinh viên.",
            "Nội dung môn học thú vị, tài liệu rõ ràng.",
        ],
        "negative": [
            "Giảng viên giảng dạy quá nhanh, sinh viên không theo kịp.",
            "Môn học này không hữu ích, tài liệu cũ.",
            "Học phí tăng nhưng chất lượng phòng học không đổi.",
        ],
        "neutral": [
            "Môn học này được học vào thứ Hai hàng tuần.",
            "Giảng viên giảng dạy tại phòng A101 vào buổi sáng.",
            "Lớp học có ba mươi sinh viên đăng ký.",
        ],
    }
    records: list[dict[str, Any]] = []
    for index in range(count_per_class):
        for label, pool in templates.items():
            template = pool[index % len(pool)]
            text = template + f" (mã {rng.randint(1000, 9999)})"
            records.append(
                make_record(
                    record_id=f"syn-{seed}-{index}-{label}",
                    text=text,
                    source_type=SOURCE_TYPE_SYNTHETIC,
                    provenance_reference=f"synthetic draft run {run_id} (seed {seed})",
                    dataset_version=dataset_version,
                    sentiment_label=label,
                    annotation_method=ANNOTATION_METHOD_SYNTHETIC,
                    annotation_status=ANNOTATION_STATUS_DRAFT,
                )
            )
    return records
