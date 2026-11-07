"""Tests for dataset validation: duplicates, leakage, distribution, fingerprints."""

from __future__ import annotations

import pytest
from tests.conftest import make_records

from vfi.data.loading import Record
from vfi.data.validation import (
    DataValidationError,
    class_distribution,
    duplicate_summary,
    find_contradictory_labels,
    find_duplicates,
    find_leakage,
    fingerprint,
    leakage_summary,
    near_duplicate_candidates,
    validate_records,
)


def assert_text_free(report: object) -> None:
    """Privacy guard: a summary must contain no strings that could carry text.

    Allowed string keys are structural names (split names, pair keys); every
    value must be an int or a list/dict of ints.
    """
    assert isinstance(report, dict)
    for value in report.values():
        if isinstance(value, int):
            continue
        if isinstance(value, list):
            assert all(isinstance(item, int) for item in value)
        elif isinstance(value, dict):
            for item in value.values():
                if isinstance(item, int):
                    continue
                assert isinstance(item, list)
                assert all(isinstance(index, int) for index in item)
        else:
            pytest.fail(f"unexpected non-privacy-safe value: {value!r}")


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


class TestDuplicateSummary:
    def test_counts_and_indices_without_text(self) -> None:
        records = [
            Record("giảng viên dạy tốt", "positive", "train"),
            Record("môn học khó theo", "negative", "train"),
            Record("giảng viên dạy tốt ", "negative", "train"),
        ]
        summary = duplicate_summary(records)
        assert summary == {
            "count": 1,
            "affected_records": 2,
            "max_group_size": 2,
            "line_indices": [0, 2],
        }
        assert_text_free(summary)

    def test_clean_records_report_zero(self) -> None:
        summary = duplicate_summary(make_records(3))
        assert summary == {
            "count": 0,
            "affected_records": 0,
            "max_group_size": 0,
            "line_indices": [],
        }

    def test_matches_find_duplicates_groups(self) -> None:
        records = make_records(1) + [
            Record("giảng viên dạy tốt", "positive", "train"),
            Record("giảng viên dạy tốt", "negative", "train"),
            Record("giảng viên dạy tốt", "neutral", "train"),
        ]
        summary = duplicate_summary(records)
        assert summary["count"] == len(find_duplicates(records))
        assert summary["max_group_size"] == 3


class TestLeakageSummary:
    def test_cross_split_report_without_text(self) -> None:
        shared = "trường tổ chức hội thảo"
        train = [Record(shared, "positive", "train"), Record("a", "negative", "train")]
        test = [Record(shared, "neutral", "test"), Record(shared, "positive", "test")]
        summary = leakage_summary(("train", train), ("test", test))
        assert summary["count"] == 1
        assert summary["affected_records"] == 3  # one train occurrence, two test
        assert summary["pairs"] == {"test|train": 1}
        assert summary["line_indices"] == {"test": [0, 1], "train": [0]}
        assert_text_free(summary)

    def test_clean_splits_report_zero(self) -> None:
        summary = leakage_summary(
            ("train", make_records(2, split="train")),
            ("test", make_records(2, split="test", offset=50)),
        )
        assert summary["count"] == 0
        assert summary["affected_records"] == 0
        assert summary["pairs"] == {}
        assert summary["line_indices"] == {}

    def test_pairs_count_shared_texts_per_split_pair(self) -> None:
        train = [
            Record("câu một", "positive", "train"),
            Record("câu hai", "negative", "train"),
        ]
        validation = [
            Record("câu một ", "neutral", "validation"),
            Record("câu hai", "neutral", "validation"),
        ]
        summary = leakage_summary(("train", train), ("validation", validation))
        assert summary["pairs"] == {"train|validation": 2}
        assert summary["count"] == 2


class TestFindContradictoryLabels:
    def test_within_split_contradiction(self) -> None:
        records = [
            Record("giảng viên dạy tốt", "positive", "train"),
            Record("giảng viên dạy tốt ", "negative", "train"),
            Record("môn học ổn", "neutral", "train"),
        ]
        report = find_contradictory_labels(("train", records))
        assert report == {
            "count": 1,
            "affected_records": 2,
            "line_indices": {"train": [0, 1]},
        }
        assert_text_free(report)

    def test_cross_split_contradiction(self) -> None:
        train = [Record("câu một", "positive", "train")]
        test = [Record("câu một", "negative", "test")]
        report = find_contradictory_labels(("train", train), ("test", test))
        assert report["count"] == 1
        assert report["line_indices"] == {"test": [0], "train": [0]}

    def test_consistent_repeats_not_reported(self) -> None:
        train = [Record("câu nhất quán", "positive", "train")]
        test = [Record("câu nhất quán ", "positive", "test")]
        report = find_contradictory_labels(("train", train), ("test", test))
        assert report == {"count": 0, "affected_records": 0, "line_indices": {}}


class TestNearDuplicateCandidates:
    def test_exact_duplicates_surface_at_similarity_one(self) -> None:
        records = [
            Record("giảng viên dạy rất tốt", "positive", "train"),
            Record("môn học khó theo kịp", "negative", "train"),
            Record("  giảng viên dạy rất tốt ", "negative", "train"),
        ]
        candidates = near_duplicate_candidates(records)
        assert candidates[0]["similarity"] >= 0.99
        assert {candidates[0]["index_a"], candidates[0]["index_b"]} == {0, 2}

    def test_distinct_texts_not_flagged(self) -> None:
        records = [
            Record("Học phí tăng nhưng chất lượng phòng học không đổi.", "negative", "train"),
            Record("Lớp học có ba mươi sinh viên đăng ký môn này.", "neutral", "train"),
        ]
        assert near_duplicate_candidates(records) == []

    def test_paraphrase_found_at_lower_threshold(self) -> None:
        records = [
            Record("giảng viên dạy rất tốt và dễ hiểu", "positive", "train"),
            Record("giảng viên dạy khá tốt và dễ hiểu", "positive", "train"),
        ]
        assert near_duplicate_candidates(records, threshold=0.5) != []
        assert near_duplicate_candidates(records, threshold=0.9999) == []

    def test_pairs_are_index_ordered_without_self_pairs(self) -> None:
        records = [
            Record("câu lặp một", "neutral", "train"),
            Record("câu lặp một", "neutral", "train"),
            Record("câu lặp một", "neutral", "train"),
            Record("câu lặp hai", "neutral", "train"),
            Record("câu lặp hai ", "neutral", "train"),
        ]
        candidates = near_duplicate_candidates(records, threshold=0.5)
        assert candidates
        for pair in candidates:
            assert pair["index_a"] < pair["index_b"]
        # Ties (all similarity 1.0 here) break by ascending indices.
        similarities = [pair["similarity"] for pair in candidates]
        assert similarities == sorted(similarities, reverse=True)
        assert (candidates[0]["index_a"], candidates[0]["index_b"]) == (0, 1)

    def test_max_candidates_caps_output(self) -> None:
        records = [Record(f"câu lặp {index % 3}", "neutral", "train") for index in range(12)]
        candidates = near_duplicate_candidates(records, threshold=0.5, max_candidates=5)
        assert len(candidates) == 5

    def test_single_record_and_empty_vocabulary_return_empty(self) -> None:
        assert near_duplicate_candidates(make_records(1)[:1]) == []
        whitespace = [Record("   ", "neutral", "train"), Record("\t", "neutral", "train")]
        assert near_duplicate_candidates(whitespace) == []

    def test_candidates_contain_no_text(self) -> None:
        records = make_records(2)
        candidates = near_duplicate_candidates(records, threshold=0.0)
        assert all(set(pair) == {"index_a", "index_b", "similarity"} for pair in candidates)


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
