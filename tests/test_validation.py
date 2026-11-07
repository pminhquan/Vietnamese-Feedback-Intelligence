"""Tests for dataset validation: duplicates, leakage, distribution, fingerprints."""

from __future__ import annotations

import pytest
from tests.conftest import make_records

from vfi.data.loading import Record
from vfi.data.validation import (
    DataValidationError,
    class_distribution,
    find_duplicates,
    find_leakage,
    fingerprint,
    validate_records,
)


class TestValidateRecords:
    def test_valid_records_pass(self) -> None:
        validate_records(make_records(2))

    def test_unknown_label_raises(self) -> None:
        records = [Record("text", "excited", "train")]
        with pytest.raises(DataValidationError, match="unknown label"):
            validate_records(records)

    def test_empty_text_raises(self) -> None:
        records = [Record("   ", "positive", "train")]
        with pytest.raises(DataValidationError, match="empty text"):
            validate_records(records)


class TestDuplicates:
    def test_duplicate_text_found(self) -> None:
        records = make_records(1) + [Record("giảng viên dạy tốt", "positive", "train")]
        records.append(Record("giảng viên dạy tốt", "negative", "train"))
        duplicates = find_duplicates(records)
        assert duplicates == ["giảng viên dạy tốt"]

    def test_no_false_duplicates(self) -> None:
        assert find_duplicates(make_records(4)) == []

    def test_duplicates_normalized_before_compare(self) -> None:
        records = [
            Record("môn học  hay", "positive", "train"),
            Record("môn học hay ", "positive", "train"),
        ]
        assert find_duplicates(records) == ["môn học hay"]


class TestLeakage:
    def test_cross_split_text_reported(self) -> None:
        shared = "trường tổ chức hội thảo"
        train = [Record(shared, "positive", "train"), Record("a", "negative", "train")]
        test = [Record(shared, "neutral", "test")]
        leakage = find_leakage(("train", train), ("test", test))
        assert leakage == {shared: ["test", "train"]}

    def test_clean_splits_pass(self) -> None:
        leakage = find_leakage(
            ("train", make_records(2, split="train")),
            ("test", make_records(2, split="test", offset=50)),
        )
        assert leakage == {}

    def test_malformed_pair_raises(self) -> None:
        # A malformed (split_name, records) pair fails when unpacked.
        with pytest.raises(ValueError, match="not enough values to unpack"):
            find_leakage(("train",))  # type: ignore[arg-type]


class TestDistributionAndFingerprint:
    def test_distribution_counts_all_labels(self) -> None:
        records = make_records(2)  # 2 per class
        assert class_distribution(records) == {"negative": 2, "neutral": 2, "positive": 2}

    def test_distribution_missing_label_is_zero(self) -> None:
        records = [Record("a", "positive", "train")]
        assert class_distribution(records)["negative"] == 0

    def test_fingerprint_stable(self) -> None:
        records = make_records(3)
        assert fingerprint(records) == fingerprint(list(records))

    def test_fingerprint_changes_when_label_changes(self) -> None:
        records = make_records(2)
        relabeled = [
            Record(r.text, "neutral" if r.label == "positive" else r.label, r.split)
            for r in records
        ]
        assert fingerprint(records) != fingerprint(relabeled)

    def test_fingerprint_changes_when_order_changes(self) -> None:
        records = make_records(3)
        assert fingerprint(records) != fingerprint(list(reversed(records)))

    def test_fingerprint_is_sha256_hex(self) -> None:
        value = fingerprint(make_records(1))
        assert len(value) == 64
        int(value, 16)  # must parse as hexadecimal
