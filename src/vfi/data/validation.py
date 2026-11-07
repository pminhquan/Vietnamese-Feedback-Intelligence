"""Dataset validation: schema checks, duplicates, leakage, fingerprints."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Sequence

from vfi.data.loading import CANONICAL_LABELS, Record, normalize_text


class DataValidationError(Exception):
    """Raised when records violate the expected dataset schema."""


def validate_records(records: Sequence[Record]) -> None:
    """Assert every record has a known label and non-empty normalized text.

    Raises:
        DataValidationError: naming the first offending record index.
    """
    for index, record in enumerate(records):
        if record.label not in CANONICAL_LABELS:
            raise DataValidationError(
                f"Record {index}: unknown label {record.label!r}; "
                f"expected one of {CANONICAL_LABELS}"
            )
        if not normalize_text(record.text):
            raise DataValidationError(f"Record {index}: empty text")


def find_duplicates(records: Sequence[Record]) -> list[str]:
    """Return normalized texts that occur more than once within the records.

    Duplicates are reported, not silently dropped: short feedback such as
    "Cảm ơn thầy!" can legitimately repeat, so removal is a documented
    decision rather than an automatic step.
    """
    counts: Counter[str] = Counter(normalize_text(record.text) for record in records)
    return sorted(text for text, count in counts.items() if count > 1)


def find_leakage(*split_records: tuple[str, Sequence[Record]]) -> dict[str, list[str]]:
    """Find normalized texts that appear in more than one named split.

    Args:
        *split_records: ``(split_name, records)`` pairs, e.g.
            ``find_leakage(("train", train), ("test", test))``.

    Returns:
        Mapping ``text -> [split names containing it]`` for texts present in
        two or more splits. Exact-match after shared normalization; near-
        duplicate leakage is out of scope for Phase 1 and documented.
    """
    seen: dict[str, set[str]] = defaultdict(set)
    for split_name, records in split_records:
        for record in records:
            seen[normalize_text(record.text)].add(split_name)
    return {text: sorted(splits) for text, splits in sorted(seen.items()) if len(splits) > 1}


def class_distribution(records: Sequence[Record]) -> dict[str, int]:
    """Count records per canonical label (all labels present as keys)."""
    counts: Counter[str] = Counter(record.label for record in records)
    return {label: counts.get(label, 0) for label in CANONICAL_LABELS}


def fingerprint(records: Sequence[Record]) -> str:
    """Deterministic sha256 fingerprint of line-aligned (text, label) pairs.

    Order-sensitive: shuffling records changes the fingerprint, which makes
    it usable as a dataset version identifier in experiment metadata.
    """
    digest = hashlib.sha256()
    for record in records:
        digest.update(normalize_text(record.text).encode("utf-8"))
        digest.update(b"\t")
        digest.update(record.label.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
