"""Dataset validation: schema checks, duplicates, leakage, fingerprints."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from itertools import combinations

from sklearn.feature_extraction.text import TfidfVectorizer

from vfi.data.loading import CANONICAL_LABELS, Record, normalize_text

NEAR_DUPLICATE_MAX_RECORDS = 500


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
                f"Record {index}: unknown label; expected one of {CANONICAL_LABELS}"
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


def leakage_summary(*split_records: tuple[str, Sequence[Record]]) -> dict[str, object]:
    """Aggregate exact-match conflicts without exposing text or per-record IDs.

    A text occurring in three splits contributes once to the total and once
    to each of the three split pairs. Labels do not affect duplicate detection.
    ``find_leakage`` returns private text for in-memory use only.
    """
    leakage = find_leakage(*split_records)
    pairs = combinations([name for name, _ in split_records], 2)
    return {
        "cross_split_duplicate_texts": len(leakage),
        "split_pair_counts": {
            f"{left}/{right}": sum(
                left in splits and right in splits for splits in leakage.values()
            )
            for left, right in pairs
        },
    }


def duplicate_summary(records: Sequence[Record]) -> dict[str, int]:
    """Count repeated normalized texts without exposing text or identifiers."""
    sizes = Counter(normalize_text(record.text) for record in records)
    repeated = [size for size in sizes.values() if size > 1]
    return {
        "count": len(repeated),
        "affected_records": sum(repeated),
        "max_group_size": max(repeated, default=0),
    }


def find_contradictory_labels(*split_records: tuple[str, Sequence[Record]]) -> dict[str, int]:
    """Count same-text label conflicts, including within individual splits."""
    sizes: Counter[str] = Counter()
    labels: defaultdict[str, set[str]] = defaultdict(set)
    for _, records in split_records:
        for record in records:
            if record.label:
                text = normalize_text(record.text)
                sizes[text] += 1
                labels[text].add(record.label)
    conflicted = [sizes[text] for text, group in labels.items() if len(group) > 1]
    return {"count": len(conflicted), "affected_records": sum(conflicted)}


def near_duplicate_candidates(
    records: Sequence[Record], *, threshold: float = 0.9, max_candidates: int = 50
) -> list[dict[str, int | float]]:
    """Audit at most the first 500 records for character TF-IDF similarity.

    Pair indices refer to the supplied order. Candidates are a local review
    signal, not a leakage decision. Default public reports can use counts.
    """
    if not 0 <= threshold <= 1 or max_candidates < 0:
        raise ValueError("Invalid near-duplicate threshold or candidate limit")
    texts = [normalize_text(record.text) for record in records[:NEAR_DUPLICATE_MAX_RECORDS]]
    if len(texts) < 2 or not max_candidates:
        return []
    try:
        tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5)).fit_transform(texts)
    except ValueError:
        return []  # No character n-grams in the sample.
    similarity = (tfidf @ tfidf.T).tocoo()
    pairs = sorted(
        (-float(score), int(left), int(right))
        for left, right, score in zip(similarity.row, similarity.col, similarity.data, strict=True)
        if left < right and score > threshold
    )
    return [
        {"index_a": left, "index_b": right, "similarity": -score}
        for score, left, right in pairs[:max_candidates]
    ]


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
