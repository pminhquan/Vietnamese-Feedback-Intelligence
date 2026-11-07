"""Deterministic stratified splitting.

Phase 1 uses the official UIT-VSFC train/validation/test files by default.
This module exists for the documented fallback path
(``use_official_splits: false``) where a validation split must be carved out
of the training data in a reproducible, stratified way. The test split is
never produced here — it always comes from the official test file.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from sklearn.model_selection import train_test_split

from vfi.data.loading import Record


def stratified_split(
    records: Sequence[Record],
    validation_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[Record], list[Record]]:
    """Split records into (train, validation) preserving label proportions.

    Same inputs and seed always produce the same split (checked in tests).

    Raises:
        ValueError: on an empty input, out-of-range size, or a class with
            fewer than 2 samples (stratification impossible).
    """
    if not records:
        raise ValueError("Cannot split an empty record list")
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be strictly between 0 and 1")
    label_counts = Counter(record.label for record in records)
    rare = {label: count for label, count in label_counts.items() if count < 2}
    if rare:
        raise ValueError(f"Stratified split needs >= 2 samples per class, found: {rare}")
    train, validation = train_test_split(
        list(records),
        test_size=validation_size,
        random_state=seed,
        stratify=[record.label for record in records],
    )
    return list(train), list(validation)
