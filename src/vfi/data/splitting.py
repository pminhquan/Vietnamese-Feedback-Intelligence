"""Deterministic, duplicate-grouped stratified splitting.

Phase 1 uses the official UIT-VSFC train/validation/test files by default.
This module exists for the documented fallback path
(``use_official_splits: false``) where a validation split must be carved out
of the training data in a reproducible, stratified way. The test split is
never produced here — it always comes from the official test file.

Records whose *normalized* text is identical (see
:func:`vfi.data.loading.normalize_text`) are treated as one group and are
never separated across the two sides: letting duplicate copies straddle
train/validation would leak labels between the splits. Groups are stratified
as units; a group's stratification label is its majority label (ties broken
by the lowest canonical label id), and records inside a mixed-label group
keep their own labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

from sklearn.model_selection import train_test_split

from vfi.data.loading import LABEL_TO_ID, Record, normalize_text


def _group_label(labels: Counter[str]) -> str:
    """Majority label of a duplicate group; ties go to the lowest canonical id.

    Labels outside the canonical three (e.g. NEU-ESC's distinct ``toxic``
    class) sort after the canonical labels and then alphabetically, so the
    tie-break stays deterministic for any label space.
    """
    top = max(labels.values())
    return min(
        (label for label, count in labels.items() if count == top),
        key=lambda label: (LABEL_TO_ID.get(label, len(LABEL_TO_ID)), label),
    )


def stratified_split(
    records: Sequence[Record],
    validation_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[Record], list[Record]]:
    """Split records into (train, validation), duplicate-grouped and stratified.

    Records are first grouped by normalized text so identical texts always
    land on the same side of the split, whatever their labels (a group may
    be mixed-label). The groups are then split with ``train_test_split``
    stratified by each group's majority label (ties broken by the lowest
    canonical label id), preserving label proportions at the group level.
    Same inputs and seed always produce the same split (checked in tests).

    Raises:
        ValueError: on an empty input, out-of-range size, or a majority
            label with fewer than 2 groups (stratification impossible).
    """
    if not records:
        raise ValueError("Cannot split an empty record list")
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be strictly between 0 and 1")
    groups: defaultdict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[normalize_text(record.text)].append(record)
    # Dicts preserve insertion order, so groups are ordered by first
    # appearance - a deterministic input to train_test_split.
    group_list = list(groups.values())
    group_labels = [_group_label(Counter(r.label for r in group)) for group in group_list]
    rare = {label: count for label, count in Counter(group_labels).items() if count < 2}
    if rare:
        raise ValueError(f"Stratified split needs >= 2 groups per class, found: {rare}")
    train_groups, validation_groups = train_test_split(
        group_list,
        test_size=validation_size,
        random_state=seed,
        stratify=group_labels,
    )
    train = [record for group in train_groups for record in group]
    validation = [record for group in validation_groups for record in group]
    return train, validation
