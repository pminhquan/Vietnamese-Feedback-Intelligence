"""Tests for deterministic stratified splitting."""

from __future__ import annotations

import pytest
from tests.conftest import make_records

from vfi.data.loading import Record, normalize_text
from vfi.data.splitting import stratified_split


def test_same_seed_same_split() -> None:
    records = make_records(6)
    train_a, val_a = stratified_split(records, validation_size=0.25, seed=42)
    train_b, val_b = stratified_split(records, validation_size=0.25, seed=42)
    assert [r.text for r in train_a] == [r.text for r in train_b]
    assert [r.text for r in val_a] == [r.text for r in val_b]


def test_different_seed_different_split() -> None:
    records = make_records(8)
    train_a, val_a = stratified_split(records, validation_size=0.25, seed=42)
    train_b, val_b = stratified_split(records, validation_size=0.25, seed=43)
    assert [r.text for r in val_a] != [r.text for r in val_b]


def test_split_is_stratified() -> None:
    records = make_records(10)  # 10 per class -> 30 records
    train, validation = stratified_split(records, validation_size=0.2, seed=42)
    assert len(train) == 24
    assert len(validation) == 6
    for label in ("negative", "neutral", "positive"):
        val_count = sum(1 for record in validation if record.label == label)
        assert val_count == 2


def test_no_record_lost_or_duplicated() -> None:
    records = make_records(5)
    train, validation = stratified_split(records, validation_size=0.2, seed=42)
    texts = [r.text for r in train] + [r.text for r in validation]
    assert sorted(texts) == sorted(r.text for r in records)


def test_empty_records_raise() -> None:
    with pytest.raises(ValueError, match="empty"):
        stratified_split([])


def test_invalid_size_raises() -> None:
    with pytest.raises(ValueError, match="validation_size"):
        stratified_split(make_records(2), validation_size=1.0)


def test_rare_class_raises() -> None:
    # Semantics change (grouped splitter): stratification units are duplicate
    # GROUPS, not records, so the rarity check counts groups per majority
    # label. A class carried by a single group is unstratifiable even when it
    # spans many duplicate records; the error reports group counts.
    records = make_records(2) + [Record("một mình", "neutral", "train")]
    records = [r for r in records if r.label != "neutral"] + [
        Record("lớp duy nhất", "neutral", "train"),
        Record("lớp duy nhất  ", "neutral", "train"),  # same group after normalization
    ]
    assert len(records) > 2  # enough records, but only ONE neutral group
    with pytest.raises(ValueError, match="2 groups per class"):
        stratified_split(records)


class TestGroupedSplitting:
    def _side(self, record: Record, train: list[Record], validation: list[Record]) -> str:
        normalized = normalize_text(record.text)
        train_texts = {normalize_text(r.text) for r in train}
        if normalized in train_texts:
            return "train"
        return "validation"

    def test_identical_normalized_texts_stay_on_one_side(self) -> None:
        # Two spelling variants of one text normalize to the same group and
        # must never straddle train/validation for any seed.
        records = make_records(3) + [
            Record("giảng viên dạy rất tốt", "positive", "train"),
            Record("  giảng viên dạy rất tốt ", "negative", "train"),
        ]
        for seed in range(20):
            train, validation = stratified_split(records, validation_size=0.25, seed=seed)
            copies = [records[-2], records[-1]]
            sides = {self._side(record, train, validation) for record in copies}
            assert len(sides) == 1, f"seed {seed} split one duplicate group"

    def test_mixed_label_group_keeps_own_labels(self) -> None:
        records = make_records(3) + [
            Record("chấm điểm công bằng", "positive", "train"),
            Record("chấm điểm công bằng ", "negative", "train"),
        ]
        train, validation = stratified_split(records, validation_size=0.25, seed=42)
        by_text = {}
        for record in train + validation:
            by_text.setdefault(normalize_text(record.text), []).append(record)
        group = by_text["chấm điểm công bằng"]
        assert {record.label for record in group} == {"positive", "negative"}

    def test_grouped_stratification_is_deterministic(self) -> None:
        records = make_records(3) + [
            Record("một câu lặp", "neutral", "train"),
            Record("một câu lặp ", "neutral", "train"),
        ]
        train_a, val_a = stratified_split(records, validation_size=0.25, seed=7)
        train_b, val_b = stratified_split(records, validation_size=0.25, seed=7)
        assert [(r.text, r.label) for r in train_a] == [(r.text, r.label) for r in train_b]
        assert [(r.text, r.label) for r in val_a] == [(r.text, r.label) for r in val_b]
