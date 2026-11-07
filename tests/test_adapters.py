"""Tests for dataset adapters: offline, deterministic, privacy-safe loading.

Every fixture is synthetic and written to ``tmp_path``; adapters only read
local files (no network access exists in the adapter module by construction).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from tests.conftest import make_records, write_vsfc_files

from vfi.data.adapters import (
    ADAPTERS_VERSION,
    ANNOTATION_METHOD_SYNTHETIC,
    ANNOTATION_STATUS_ANNOTATED,
    ANNOTATION_STATUS_APPROVED,
    ANNOTATION_STATUS_DRAFT,
    FIVE_LEVEL_LABEL_MAPPING,
    SOURCE_TYPE_SYNTHETIC,
    TOXIC_POLICY_EXCLUDE,
    TOXIC_POLICY_KEEP,
    AdapterError,
    CustomJsonlAdapter,
    DatasetAdapter,
    FiveLevelCsvAdapter,
    NeuEscAdapter,
    UitVsfcAdapter,
)
from vfi.data.loading import (
    LABEL_SOURCE_OWNER_CONFIRMED,
    LABEL_SOURCE_SEACROWD_UNVERIFIED,
    RawLabelMapping,
    Record,
)

SENTINEL = "SENTINEL-TEXT-XYZ"


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_jsonl(path: Path, payloads: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload) + "\n")


def all_records(result) -> list[Record]:
    return [record for records in result.records_by_split.values() for record in records]


def assert_provenance_is_text_free(result) -> None:
    """No record text may appear anywhere in the provenance payload."""
    blob = json.dumps(result.provenance)
    for record in all_records(result):
        assert record.text not in blob


def five_level_fraction_rows() -> list[list[str]]:
    """Three distinct texts per canonical class, each duplicated in-group."""
    texts = {
        "1": [
            "Tài liệu cũ, khó theo kịp.",
            "Học phí tăng, chất lượng không đổi.",
            "Giảng dạy quá nhanh.",
        ],
        "3": [
            "Môn học diễn ra vào thứ Hai.",
            "Lớp học có ba mươi sinh viên.",
            "Phòng học ở tòa B.",
        ],
        "5": [
            "Giảng viên giải thích rất dễ hiểu.",
            "Nội dung môn học thú vị.",
            "Tài liệu rõ ràng.",
        ],
    }
    rows = []
    for label, group in texts.items():
        for text in group:
            rows.append([text, label])
            rows.append([text + "  ", label])  # whitespace variant, same duplicate group
    return rows


def neu_esc_rows() -> list[list[str]]:
    """Two distinct texts per NEU-ESC class, toxic rows in the middle."""
    return [
        ["Tài liệu cũ, khó theo kịp.", "negative"],
        ["Học phí tăng, chất lượng không đổi.", "negative"],
        ["Môn học diễn ra vào thứ Hai.", "neutral"],
        ["Lớp học có ba mươi sinh viên.", "neutral"],
        [f"Nội dung xúc phạm giảng viên {SENTINEL}.", "toxic"],
        ["Bình luận mang tính công kích.", "toxic"],
        ["Giảng viên giải thích rất dễ hiểu.", "positive"],
        ["Nội dung môn học thú vị.", "positive"],
    ]


def custom_payload(**overrides: object) -> dict:
    payload = {
        "record_id": "r-001",
        "text": "Giảng viên giải thích rất dễ hiểu.",
        "sentiment_label": "positive",
        "source_type": SOURCE_TYPE_SYNTHETIC,
        "provenance_reference": "generator-v1",
        "annotation_method": ANNOTATION_METHOD_SYNTHETIC,
        "annotation_status": ANNOTATION_STATUS_APPROVED,
        "dataset_version": "2026.11.07",
    }
    payload.update(overrides)
    return payload


class TestFiveLevelCsvAdapter:
    def test_split_column_happy_path(self, tmp_path: Path) -> None:
        rows = [
            ["Giảng viên giải thích rất dễ hiểu.", "5", "train"],
            ["Nội dung môn học thú vị.", "4", "train"],
            ["Môn học diễn ra vào thứ Hai.", "3", "validation"],
            ["Lớp học có ba mươi sinh viên.", "3", "validation"],
            ["Tài liệu cũ, khó theo kịp.", "1", "test"],
            ["Học phí tăng, chất lượng không đổi.", "2", "test"],
        ]
        write_csv(tmp_path / "data.csv", ["text", "label", "split"], rows)
        result = FiveLevelCsvAdapter().load(tmp_path)
        assert set(result.records_by_split) == {"train", "validation", "test"}
        assert [r.label for r in result.records_by_split["train"]] == ["positive", "positive"]
        assert [r.label for r in result.records_by_split["validation"]] == ["neutral", "neutral"]
        assert [r.label for r in result.records_by_split["test"]] == ["negative", "negative"]
        provenance = result.provenance
        assert provenance["dataset_id"] == "five_level_csv"
        assert provenance["adapter"] == {"name": "five_level_csv", "version": ADAPTERS_VERSION}
        assert provenance["label_mapping"] == FIVE_LEVEL_LABEL_MAPPING
        assert provenance["splits_created_by"] == "split_column"
        assert set(provenance["sha256"]) == {"train", "validation", "test"}
        assert provenance["integrity"]["leakage"]["count"] == 0
        assert_provenance_is_text_free(result)

    def test_fraction_split_is_grouped_and_deterministic(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "data.csv", ["text", "label"], five_level_fraction_rows())
        first = FiveLevelCsvAdapter(validation_fraction=0.25, seed=42).load(tmp_path)
        second = FiveLevelCsvAdapter(validation_fraction=0.25, seed=42).load(tmp_path)
        assert first.provenance["sha256"] == second.provenance["sha256"]
        assert set(first.records_by_split) == {"train", "validation"}
        assert first.provenance["splits_created_by"] == "validation_fraction"
        # Whitespace-variant copies share one duplicate group: no leakage.
        integrity = first.provenance["integrity"]
        assert integrity["leakage"]["count"] == 0
        duplicates = integrity["duplicates_by_split"]
        total_groups = sum(split["count"] for split in duplicates.values())
        assert total_groups == 9  # 3 texts per class x 3 classes, all duplicated
        assert sum(split["affected_records"] for split in duplicates.values()) == 18
        assert integrity["contradictions"]["count"] == 0

    def test_unknown_labels_list_every_offending_line(self, tmp_path: Path) -> None:
        rows = [
            ["câu ổn", "3"],  # file line 2
            ["câu lỗi một", "9"],  # file line 3
            ["câu ổn hai", "4"],  # file line 4
            ["câu lỗi hai", "abc"],  # file line 5
        ]
        write_csv(tmp_path / "data.csv", ["text", "label"], rows)
        with pytest.raises(AdapterError) as excinfo:
            FiveLevelCsvAdapter().load(tmp_path)
        message = str(excinfo.value)
        assert "line 3 (label '9')" in message
        assert "line 5 (label 'abc')" in message
        assert "['1', '2', '3', '4', '5']" in message

    def test_empty_text_error_is_privacy_safe(self, tmp_path: Path) -> None:
        rows = [["   ", "3"], [SENTINEL, "9"]]
        write_csv(tmp_path / "data.csv", ["text", "label"], rows)
        with pytest.raises(AdapterError) as excinfo:
            FiveLevelCsvAdapter().load(tmp_path)
        message = str(excinfo.value)
        assert "empty text at line(s): 2" in message
        assert SENTINEL not in message  # no record text in errors

    def test_unexpected_column_rejected(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "data.csv", ["text", "label", "note"], [["a", "3", "x"]])
        with pytest.raises(AdapterError, match="unexpected column"):
            FiveLevelCsvAdapter().load(tmp_path)

    def test_unknown_split_value_rejected(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "data.csv", ["text", "label", "split"], [["a", "3", "dev"]])
        with pytest.raises(AdapterError, match="unknown split values"):
            FiveLevelCsvAdapter().load(tmp_path)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(AdapterError, match="missing"):
            FiveLevelCsvAdapter().load(tmp_path)

    def test_fingerprint_is_deterministic_and_sensitive(self, tmp_path: Path) -> None:
        write_csv(
            tmp_path / "data.csv",
            ["text", "label", "split"],
            [
                ["Giảng viên giải thích rất dễ hiểu.", "5", "train"],
                ["Môn học diễn ra vào thứ Hai.", "3", "validation"],
            ],
        )
        first = FiveLevelCsvAdapter().load(tmp_path)
        second = FiveLevelCsvAdapter().load(tmp_path)
        assert first.provenance["sha256"] == second.provenance["sha256"]
        rows = [
            ["Giảng viên giải thích rất dễ hiểu.", "1", "train"],
            ["Môn học diễn ra vào thứ Hai.", "3", "validation"],
        ]
        write_csv(tmp_path / "data.csv", ["text", "label", "split"], rows)
        changed = FiveLevelCsvAdapter().load(tmp_path)
        assert changed.provenance["sha256"] != first.provenance["sha256"]


class TestNeuEscAdapter:
    def test_default_policy_refuses_toxic_without_leaking_text(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "data.csv", ["text", "label"], neu_esc_rows())
        with pytest.raises(AdapterError) as excinfo:
            NeuEscAdapter().load(tmp_path)
        message = str(excinfo.value)
        assert "toxic" in message
        assert "toxic_policy='exclude'" in message and "toxic_policy='keep'" in message
        assert "line 6, 7" in message  # toxic rows sit on file lines 6 and 7
        assert SENTINEL not in message  # the toxic text itself is never quoted

    def test_error_policy_loads_file_without_toxic(self, tmp_path: Path) -> None:
        rows = [row for row in neu_esc_rows() if row[1] != "toxic"]
        write_csv(tmp_path / "data.csv", ["text", "label"], rows)
        # Tiny fixture: 2 groups per class needs a large validation fraction.
        result = NeuEscAdapter(validation_fraction=0.5).load(tmp_path)
        assert result.provenance["toxic_policy"] == "error"
        assert result.provenance["toxic_count"] == 0
        assert result.provenance["toxic_line_numbers"] == []
        assert {r.label for r in all_records(result)} == {"negative", "neutral", "positive"}

    def test_policies_other_than_error_require_justification(self) -> None:
        for policy in (TOXIC_POLICY_EXCLUDE, TOXIC_POLICY_KEEP):
            with pytest.raises(ValueError, match="justification"):
                NeuEscAdapter(toxic_policy=policy)
            with pytest.raises(ValueError, match="justification"):
                NeuEscAdapter(toxic_policy=policy, justification="   ")
        with pytest.raises(ValueError, match="toxic_policy"):
            NeuEscAdapter(toxic_policy="drop")

    def test_exclude_drops_toxic_and_reports(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "data.csv", ["text", "label"], neu_esc_rows())
        adapter = NeuEscAdapter(
            toxic_policy=TOXIC_POLICY_EXCLUDE,
            justification="3-class pilot: toxic rows excluded per acquisition plan",
            validation_fraction=0.5,
        )
        result = adapter.load(tmp_path)
        labels = [r.label for r in all_records(result)]
        assert "toxic" not in labels
        assert len(labels) == 6  # 3 classes x 2 texts
        provenance = result.provenance
        assert provenance["toxic_policy"] == "exclude"
        assert provenance["toxic_count"] == 2
        assert provenance["toxic_line_numbers"] == [6, 7]
        assert provenance["justification"].startswith("3-class pilot")
        assert_provenance_is_text_free(result)

    def test_keep_retains_toxic_as_distinct_label(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "data.csv", ["text", "label"], neu_esc_rows())
        adapter = NeuEscAdapter(
            toxic_policy=TOXIC_POLICY_KEEP,
            justification="4-class feasibility check",
            validation_fraction=0.5,
        )
        result = adapter.load(tmp_path)
        labels = [r.label for r in all_records(result)]
        assert labels.count("toxic") == 2
        assert labels.count("negative") == 2  # never remapped into 3-class labels
        assert labels.count("neutral") == 2
        assert labels.count("positive") == 2
        provenance = result.provenance
        assert provenance["toxic_policy"] == "keep"
        assert provenance["toxic_count"] == 2
        assert provenance["justification"] == "4-class feasibility check"
        assert_provenance_is_text_free(result)

    def test_unknown_label_rejected(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "data.csv", ["text", "label"], [["câu nào đó", "angry"]])
        with pytest.raises(AdapterError, match="line 2 \\(label 'angry'\\)"):
            NeuEscAdapter().load(tmp_path)

    def test_split_column_respected(self, tmp_path: Path) -> None:
        rows = [
            ["Giảng viên giải thích rất dễ hiểu.", "positive", "train"],
            ["Bình luận mang tính công kích.", "toxic", "train"],
            ["Môn học diễn ra vào thứ Hai.", "neutral", "validation"],
        ]
        write_csv(tmp_path / "data.csv", ["text", "label", "split"], rows)
        adapter = NeuEscAdapter(toxic_policy=TOXIC_POLICY_KEEP, justification="split-column check")
        result = adapter.load(tmp_path)
        assert set(result.records_by_split) == {"train", "validation"}
        assert any(r.label == "toxic" for r in result.records_by_split["train"])
        assert all(len(value) == 64 for value in result.provenance["sha256"].values())
        assert set(result.provenance["sha256"]) == {"train", "validation"}


class TestCustomJsonlAdapter:
    def six_approved(self) -> list[dict]:
        texts = {
            "positive": ["Giảng viên giải thích rất dễ hiểu.", "Nội dung môn học thú vị."],
            "neutral": ["Môn học diễn ra vào thứ Hai.", "Lớp học có ba mươi sinh viên."],
            "negative": ["Tài liệu cũ, khó theo kịp.", "Học phí tăng, chất lượng không đổi."],
        }
        payloads = []
        counter = 0
        for label, group in texts.items():
            for text in group:
                counter += 1
                payloads.append(
                    custom_payload(record_id=f"r-{counter:03d}", text=text, sentiment_label=label)
                )
        return payloads

    def test_only_approved_records_are_loaded(self, tmp_path: Path) -> None:
        payloads = self.six_approved()
        payloads.append(
            custom_payload(record_id="d-001", annotation_status=ANNOTATION_STATUS_DRAFT)
        )
        payloads.append(
            custom_payload(record_id="a-001", annotation_status=ANNOTATION_STATUS_ANNOTATED)
        )
        write_jsonl(tmp_path / "data.jsonl", payloads)
        result = CustomJsonlAdapter(validation_fraction=0.5).load(tmp_path)
        assert len(all_records(result)) == 6
        provenance = result.provenance
        assert provenance["annotation_status_counts"] == {
            "draft": 1,
            "annotated": 1,
            "approved": 6,
        }
        assert provenance["not_approved_counts"] == {"draft": 1, "annotated": 1}
        assert provenance["not_approved_total"] == 2
        assert provenance["approved_count"] == 6
        assert provenance["adapter"] == {"name": "custom_jsonl", "version": ADAPTERS_VERSION}
        assert provenance["splits_created_by"] == "validation_fraction"
        assert set(result.records_by_split) == {"train", "validation"}
        assert_provenance_is_text_free(result)

    def test_split_field_used_when_every_approved_record_declares_it(self, tmp_path: Path) -> None:
        payloads = [
            custom_payload(record_id="r-001", split="train"),
            custom_payload(
                record_id="r-002",
                text="Môn học diễn ra vào thứ Hai.",
                sentiment_label="neutral",
                split="validation",
            ),
            custom_payload(
                record_id="r-003",
                text="Tài liệu cũ, khó theo kịp.",
                sentiment_label="negative",
                split="test",
            ),
        ]
        write_jsonl(tmp_path / "data.jsonl", payloads)
        result = CustomJsonlAdapter().load(tmp_path)
        assert set(result.records_by_split) == {"train", "validation", "test"}
        assert result.provenance["splits_created_by"] == "split_field"
        assert result.records_by_split["test"][0].label == "negative"

    def test_mixed_split_declaration_rejected(self, tmp_path: Path) -> None:
        payloads = [
            custom_payload(record_id="r-001", split="train"),
            custom_payload(
                record_id="r-002", text="Môn học diễn ra vào thứ Hai.", sentiment_label="neutral"
            ),
        ]
        write_jsonl(tmp_path / "data.jsonl", payloads)
        with pytest.raises(AdapterError, match="mixing declared and undeclared"):
            CustomJsonlAdapter().load(tmp_path)

    def test_unknown_split_value_rejected(self, tmp_path: Path) -> None:
        payloads = [custom_payload(record_id="r-001", split="dev")]
        write_jsonl(tmp_path / "data.jsonl", payloads)
        with pytest.raises(AdapterError, match="split 'dev'"):
            CustomJsonlAdapter().load(tmp_path)

    def test_schema_violations_name_the_field_and_line(self, tmp_path: Path) -> None:
        payloads = [
            custom_payload(record_id="r-001"),
            custom_payload(record_id="r-002", notes="extra"),  # unknown field
            {
                key: value
                for key, value in custom_payload(record_id="r-003").items()
                if key != "annotation_status"
            },  # missing field
            custom_payload(record_id="r-004", source_type="scraped"),  # invalid enum
            custom_payload(record_id="r-005", sentiment_label="angry"),  # invalid label
            custom_payload(record_id="r-006", text="   "),  # empty text
            custom_payload(record_id="r-001"),  # duplicate id
        ]
        write_jsonl(tmp_path / "data.jsonl", payloads)
        with pytest.raises(AdapterError) as excinfo:
            CustomJsonlAdapter().load(tmp_path)
        message = str(excinfo.value)
        # Line numbers are physical file lines (line 1 = first JSON record).
        assert "line 2: unknown field(s) ['notes']" in message
        assert "line 3: missing field(s) ['annotation_status']" in message
        assert "line 4: source_type 'scraped'" in message
        assert "line 5: sentiment_label 'angry'" in message
        assert "line 6: text must be a non-empty string" in message
        assert "line 7: duplicate record_id 'r-001' (first seen on line 1)" in message

    def test_malformed_lines_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        payloads = [custom_payload(record_id="r-001")]
        write_jsonl(path, payloads)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("[1, 2]\n")  # not an object
            handle.write("{oops\n")  # not JSON at all
            handle.write("\n")  # blank line
        with pytest.raises(AdapterError) as excinfo:
            CustomJsonlAdapter().load(tmp_path)
        message = str(excinfo.value)
        assert "line 2: expected a JSON object, got list" in message
        assert "line 3: invalid JSON" in message
        assert "line 4: blank line" in message

    def test_all_draft_file_yields_empty_result(self, tmp_path: Path) -> None:
        payloads = [
            custom_payload(record_id="d-001", annotation_status=ANNOTATION_STATUS_DRAFT),
            custom_payload(record_id="d-002", annotation_status=ANNOTATION_STATUS_DRAFT),
        ]
        write_jsonl(tmp_path / "data.jsonl", payloads)
        result = CustomJsonlAdapter().load(tmp_path)
        assert result.records_by_split == {}
        assert result.provenance["not_approved_total"] == 2
        assert result.provenance["sha256"] == {}

    def test_fingerprint_deterministic(self, tmp_path: Path) -> None:
        write_jsonl(tmp_path / "data.jsonl", self.six_approved())
        first = CustomJsonlAdapter(validation_fraction=0.5).load(tmp_path)
        second = CustomJsonlAdapter(validation_fraction=0.5).load(tmp_path)
        assert first.provenance["sha256"] == second.provenance["sha256"]


class TestUitVsfcAdapter:
    def _write(self, tmp_path: Path) -> None:
        write_vsfc_files(
            tmp_path,
            {
                "train": make_records(1, split="train"),
                "validation": make_records(1, split="validation", offset=50),
                "test": make_records(1, split="test", offset=100),
            },
        )

    def test_passthrough_with_default_mapping(self, tmp_path: Path) -> None:
        self._write(tmp_path)
        result = UitVsfcAdapter().load(tmp_path)
        assert set(result.records_by_split) == {"train", "validation", "test"}
        assert [r.label for r in result.records_by_split["train"]] == [
            "positive",
            "negative",
            "neutral",
        ]
        provenance = result.provenance
        assert provenance["dataset_id"] == "uit_vsfc"
        assert provenance["label_mapping_source"] == LABEL_SOURCE_SEACROWD_UNVERIFIED
        assert provenance["label_mapping"]["raw_to_canonical"] == {
            "0": "positive",
            "1": "negative",
            "2": "neutral",
        }
        assert provenance["integrity"]["leakage"]["count"] == 0
        assert_provenance_is_text_free(result)

    def test_passthrough_with_explicit_mapping(self, tmp_path: Path) -> None:
        self._write(tmp_path)
        mapping = RawLabelMapping(
            {0: "negative", 1: "neutral", 2: "positive"},
            LABEL_SOURCE_OWNER_CONFIRMED,
            "owner confirmed the encoding on 2026-11-07",
        )
        result = UitVsfcAdapter(label_mapping=mapping).load(tmp_path)
        assert [r.label for r in result.records_by_split["train"]] == [
            "negative",
            "neutral",
            "positive",
        ]
        provenance = result.provenance
        assert provenance["label_mapping_source"] == LABEL_SOURCE_OWNER_CONFIRMED
        assert (
            provenance["label_mapping"]["evidence"] == "owner confirmed the encoding on 2026-11-07"
        )

    def test_missing_files_surface_as_adapter_error(self, tmp_path: Path) -> None:
        with pytest.raises(AdapterError, match="UIT-VSFC load failed"):
            UitVsfcAdapter().load(tmp_path)


class TestAdapterProtocol:
    @pytest.mark.parametrize(
        "adapter",
        [
            FiveLevelCsvAdapter(),
            NeuEscAdapter(),
            NeuEscAdapter(toxic_policy=TOXIC_POLICY_KEEP, justification="protocol check"),
            CustomJsonlAdapter(),
            UitVsfcAdapter(),
        ],
        ids=["five_level", "neu_esc_default", "neu_esc_keep", "custom_jsonl", "uit_vsfc"],
    )
    def test_adapters_satisfy_the_protocol(self, adapter: DatasetAdapter) -> None:
        assert isinstance(adapter, DatasetAdapter)
        assert isinstance(adapter.name, str)
