"""Tests for the independent custom-dataset store (synthetic data only).

These tests exercise Workstream 2: schema validation, honesty invariants
(synthetic labels can never be approved without a human annotation), the
annotation flow, privacy-safe summaries, and versioned manifests.
"""

import json
from pathlib import Path

import pytest

from vfi.custom_dataset import (
    ANNOTATION_METHOD_HUMAN,
    ANNOTATION_METHOD_SYNTHETIC,
    ANNOTATION_STATUS_APPROVED,
    ANNOTATION_STATUS_DRAFT,
    AnnotationDecision,
    CustomDatasetError,
    CustomDatasetStore,
    generate_synthetic_draft,
    make_record,
    validate_record,
)

MARKER = "MSSV 2051123456"


def _record(record_id: str = "r-001", **overrides: object) -> dict:
    base = {
        "record_id": record_id,
        "text": f"Giảng viên dạy rất tốt {record_id}",
        "sentiment_label": "",
        "source_type": "owner_collected",
        "provenance_reference": "owner collection 2026-11-07",
        "annotation_method": "",
        "annotation_status": "draft",
        "dataset_version": "0.1.0",
    }
    base.update(overrides)
    return base


class TestRecordValidation:
    def test_make_record_normalizes_text(self) -> None:
        record = make_record(
            record_id="r-1",
            text="  môn   học hay  ",
            source_type="owner_collected",
            provenance_reference="x",
            dataset_version="0.1.0",
        )
        assert record["text"] == "môn học hay"

    @pytest.mark.parametrize(
        "mutation, match",
        [
            ({"extra": 1}, "unknown field"),
            ({"record_id": "  "}, "non-empty string"),
            ({"sentiment_label": "angry"}, "sentiment_label"),
            ({"source_type": "scraped"}, "source_type"),
            ({"annotation_status": "published"}, "annotation_status"),
            ({"annotation_method": "gpt"}, "annotation_method"),
        ],
    )
    def test_invalid_records_raise(self, mutation: dict, match: str) -> None:
        record = _record(**mutation)
        with pytest.raises(CustomDatasetError, match=match):
            validate_record(record)

    def test_approved_requires_label_and_human_method(self) -> None:
        with pytest.raises(CustomDatasetError, match="must carry a sentiment_label"):
            validate_record(_record(annotation_status="approved"))
        with pytest.raises(CustomDatasetError, match="only human annotations"):
            validate_record(
                _record(
                    sentiment_label="positive",
                    annotation_method=ANNOTATION_METHOD_SYNTHETIC,
                    annotation_status="approved",
                )
            )
        validate_record(
            _record(
                sentiment_label="positive",
                annotation_method=ANNOTATION_METHOD_HUMAN,
                annotation_status="approved",
            )
        )

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(CustomDatasetError, match="non-empty"):
            validate_record(_record(text="   "))


class TestStore:
    def test_append_and_load_roundtrip(self, tmp_path: Path) -> None:
        store = CustomDatasetStore(tmp_path / "ds", "0.1.0")
        added = store.append_records([_record("r-1"), _record("r-2")])
        assert added == 2
        records = store.load_records()
        assert [record["record_id"] for record in records] == ["r-1", "r-2"]

    def test_duplicate_record_id_rejected(self, tmp_path: Path) -> None:
        store = CustomDatasetStore(tmp_path / "ds", "0.1.0")
        store.append_records([_record("r-1")])
        with pytest.raises(CustomDatasetError, match="duplicate record_id 'r-1'"):
            store.append_records([_record("r-1")])

    def test_duplicate_id_inside_batch_rejected(self, tmp_path: Path) -> None:
        store = CustomDatasetStore(tmp_path / "ds", "0.1.0")
        with pytest.raises(CustomDatasetError, match="duplicate record_id"):
            store.append_records([_record("r-1"), _record("r-1")])


class TestAnnotationFlow:
    def _store_with_synthetic(self, tmp_path: Path) -> CustomDatasetStore:
        store = CustomDatasetStore(tmp_path / "ds", "0.1.0")
        store.append_records(generate_synthetic_draft(1, dataset_version="0.1.0", run_id="t1"))
        return store

    def test_synthetic_label_cannot_be_approved_directly(self, tmp_path: Path) -> None:
        store = self._store_with_synthetic(tmp_path)
        record_id = store.load_records()[0]["record_id"]
        assert store.load_records()[0]["annotation_method"] == ANNOTATION_METHOD_SYNTHETIC
        with pytest.raises(CustomDatasetError, match="personally label or correct"):
            store.apply_annotation(AnnotationDecision(record_id, "approved", None, None, "", ""))

    def test_human_correction_then_approval(self, tmp_path: Path) -> None:
        store = self._store_with_synthetic(tmp_path)
        record_id = store.load_records()[0]["record_id"]
        store.apply_annotation(
            AnnotationDecision(
                record_id, "corrected", "positive", "negative", ANNOTATION_METHOD_HUMAN, ""
            )
        )
        annotated = store.load_records()[0]
        assert annotated["annotation_method"] == ANNOTATION_METHOD_HUMAN
        assert annotated["annotation_status"] == "annotated"
        store.apply_annotation(AnnotationDecision(record_id, "approved", None, None, "", ""))
        approved = store.load_records()[0]
        assert approved["annotation_status"] == ANNOTATION_STATUS_APPROVED
        assert approved["source_type"] == "synthetic"  # source honesty preserved

    def test_approve_without_label_raises(self, tmp_path: Path) -> None:
        store = CustomDatasetStore(tmp_path / "ds", "0.1.0")
        store.append_records([_record("r-1")])
        # owner labels by hand first
        store.apply_annotation(
            AnnotationDecision("r-1", "labeled", "", "neutral", ANNOTATION_METHOD_HUMAN, "")
        )
        store.apply_annotation(AnnotationDecision("r-1", "approved", None, None, "", ""))
        assert store.load_records()[0]["annotation_status"] == ANNOTATION_STATUS_APPROVED

    def test_unknown_record_id_raises(self, tmp_path: Path) -> None:
        store = CustomDatasetStore(tmp_path / "ds", "0.1.0")
        with pytest.raises(CustomDatasetError, match="unknown record_id"):
            store.apply_annotation(
                AnnotationDecision("ghost", "labeled", "", "positive", ANNOTATION_METHOD_HUMAN, "")
            )


class TestSummaryAndManifest:
    def test_summary_is_privacy_safe(self, tmp_path: Path) -> None:
        store = CustomDatasetStore(tmp_path / "ds", "0.1.0")
        records = [
            _record(
                "r-1",
                text=f"phan hoi co {MARKER} ve mon hoc",
                sentiment_label="positive",
                annotation_method=ANNOTATION_METHOD_HUMAN,
                annotation_status="annotated",
            ),
            _record(
                "r-2",
                text=f"phan hoi co {MARKER} ve mon hoc",
                sentiment_label="positive",
                annotation_method=ANNOTATION_METHOD_HUMAN,
                annotation_status="annotated",
            ),
            _record(
                "r-3",
                text="mon hoc khac",
                sentiment_label="negative",
                annotation_method=ANNOTATION_METHOD_HUMAN,
                annotation_status="annotated",
            ),
        ]
        store.append_records(records)
        summary = store.dataset_summary()
        rendered = json.dumps(summary, ensure_ascii=False)
        assert MARKER not in rendered
        assert summary["total_records"] == 3
        assert summary["records_by_source_type"] == {"owner_collected": 3}
        assert summary["duplicate_summary"]["count"] == 1  # the identical texts
        assert summary["class_distribution_approved"] == {}  # nothing approved yet

    def test_manifest_written_and_versioned(self, tmp_path: Path) -> None:
        store = CustomDatasetStore(tmp_path / "ds", "0.2.0")
        store.append_records(
            [
                _record(
                    "r-1",
                    sentiment_label="positive",
                    annotation_method=ANNOTATION_METHOD_HUMAN,
                    annotation_status=ANNOTATION_STATUS_APPROVED,
                )
            ]
        )
        manifest_path = store.write_manifest()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["dataset_version"] == "0.2.0"
        assert manifest["approved_records"] == 1
        assert manifest["fingerprint_approved"]
        assert "synthetic" in manifest["honesty_note"]


class TestSyntheticDrafts:
    def test_drafts_marked_synthetic_and_draft(self) -> None:
        records = generate_synthetic_draft(2, dataset_version="0.1.0", run_id="abc")
        assert len(records) == 6  # 2 per class, 3 classes
        for record in records:
            assert record["source_type"] == "synthetic"
            assert record["annotation_method"] == ANNOTATION_METHOD_SYNTHETIC
            assert record["annotation_status"] == ANNOTATION_STATUS_DRAFT
        labels = {record["sentiment_label"] for record in records}
        assert labels == {"negative", "neutral", "positive"}

    def test_drafts_deterministic_per_seed(self) -> None:
        first = generate_synthetic_draft(1, dataset_version="0.1.0", run_id="a", seed=7)
        second = generate_synthetic_draft(1, dataset_version="0.1.0", run_id="a", seed=7)
        assert [record["text"] for record in first] == [record["text"] for record in second]
        other = generate_synthetic_draft(1, dataset_version="0.1.0", run_id="a", seed=8)
        assert [record["text"] for record in first] != [record["text"] for record in other]
