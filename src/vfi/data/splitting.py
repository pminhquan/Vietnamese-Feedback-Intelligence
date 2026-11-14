"""Deterministic, duplicate-grouped train/validation splitting.

A seeded validation fallback splits a physical train pool. Normalized texts
that repeat stay together, even when their labels disagree. Test is always
a separate holdout, never produced by this helper.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import replace

from sklearn.model_selection import train_test_split

from vfi.data.loading import LABEL_TO_ID, Record, normalize_text


def stratified_split(
    records: Sequence[Record],
    validation_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[Record], list[Record]]:
    """Split groups of identical normalized text and preserve their records.

    Stratification uses each group's majority label, with canonical label
    order breaking ties. Sizes and proportions target groups, not individual
    rows, so large repeated groups may produce uneven row counts.

    Raises:
        ValueError: on an empty input, out-of-range size, or a class with
            fewer than 2 samples (stratification impossible).
    """
    if not records:
        raise ValueError("Cannot split an empty record list")
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be strictly between 0 and 1")
    groups: defaultdict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[normalize_text(record.text)].append(record)
    group_list = list(groups.values())
    group_labels = [_majority_label(group) for group in group_list]
    label_counts = Counter(group_labels)
    rare = {record.label for record in records if label_counts[record.label] < 2}
    if rare:
        raise ValueError("Stratified split needs >= 2 samples per class in distinct text groups")
    train, validation = train_test_split(
        group_list,
        test_size=validation_size,
        random_state=seed,
        stratify=group_labels,
    )
    return (
        [replace(record, split="train") for group in train for record in group],
        [replace(record, split="validation") for group in validation for record in group],
    )


def _majority_label(records: Sequence[Record]) -> str:
    counts = Counter(record.label for record in records)
    return min(counts, key=lambda label: (-counts[label], LABEL_TO_ID.get(label, 3), label))
