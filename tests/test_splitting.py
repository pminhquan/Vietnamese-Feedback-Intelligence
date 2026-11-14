"""Tests for deterministic stratified splitting."""

from __future__ import annotations

import pytest
from tests.conftest import make_records

from vfi.data.loading import Record
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
    records = make_records(2) + [Record("một mình", "neutral", "train")]
    # neutral has 3 records; craft a class with a single sample instead
    records = [r for r in records if r.label != "neutral"] + [
        Record("lớp duy nhất", "neutral", "train")
    ]
    with pytest.raises(ValueError, match="2 samples per class"):
        stratified_split(records)
