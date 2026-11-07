"""Dataset validation: schema checks, duplicates, leakage, fingerprints.

Two families of duplicate/leakage primitives live here:

- ``find_duplicates`` / ``find_leakage`` return raw normalized text. They are
  for interactive debugging and tests only - never embed their output in
  machine-readable reports, CLI stdout, or logs (student feedback is private;
  see the Phase 2 risk register, R-05).
- ``duplicate_summary`` / ``leakage_summary`` / ``find_contradictory_labels``
  are the privacy-safe reporting primitives: counts, pair keys, and record
  indices only, no text.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer

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

    Warning:
        Returns raw dataset text. For interactive debugging and tests only -
        NOT for machine-readable reports, CLI stdout, or logs. Use
        :func:`duplicate_summary` for privacy-safe reporting.
    """
    counts: Counter[str] = Counter(normalize_text(record.text) for record in records)
    return sorted(text for text, count in counts.items() if count > 1)


def duplicate_summary(records: Sequence[Record]) -> dict:
    """Privacy-safe duplicate report: counts and record indices, never text.

    Reports the same duplicate groups as :func:`find_duplicates` without
    returning any text, so the result is safe to embed in machine-readable
    reports and CLI output.

    Returns:
        Dict with ``count`` (number of duplicated normalized texts),
        ``affected_records`` (total records belonging to duplicate groups),
        ``max_group_size`` (largest duplicate group), and ``line_indices``
        (sorted 0-based record indices involved in any duplicate group).
    """
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[normalize_text(record.text)].append(index)
    duplicated = [indices for indices in groups.values() if len(indices) > 1]
    return {
        "count": len(duplicated),
        "affected_records": sum(len(indices) for indices in duplicated),
        "max_group_size": max((len(indices) for indices in duplicated), default=0),
        "line_indices": sorted(index for indices in duplicated for index in indices),
    }


def find_leakage(*split_records: tuple[str, Sequence[Record]]) -> dict[str, list[str]]:
    """Find normalized texts that appear in more than one named split.

    Args:
        *split_records: ``(split_name, records)`` pairs, e.g.
            ``find_leakage(("train", train), ("test", test))``.

    Returns:
        Mapping ``text -> [split names containing it]`` for texts present in
        two or more splits. Exact-match after shared normalization; near-
        duplicate leakage is out of scope for Phase 1 and documented.

    Warning:
        Returns raw dataset text keyed by split. For interactive debugging
        and tests only - NOT for machine-readable reports, CLI stdout, or
        logs. Use :func:`leakage_summary` for privacy-safe reporting.
    """
    seen: dict[str, set[str]] = defaultdict(set)
    for split_name, records in split_records:
        for record in records:
            seen[normalize_text(record.text)].add(split_name)
    return {text: sorted(splits) for text, splits in sorted(seen.items()) if len(splits) > 1}


def leakage_summary(*split_records: tuple[str, Sequence[Record]]) -> dict:
    """Privacy-safe leakage report: counts, pair keys, and indices, never text.

    Reports the same cross-split duplicates as :func:`find_leakage` without
    returning any text, so the result is safe to embed in machine-readable
    reports and CLI output.

    Args:
        *split_records: ``(split_name, records)`` pairs, e.g.
            ``leakage_summary(("train", train), ("test", test))``.

    Returns:
        Dict with ``count`` (number of normalized texts appearing in more
        than one split), ``affected_records`` (total occurrences of those
        texts across all splits), ``pairs`` (mapping of
        ``"<split_a>|<split_b>"`` keys - split names sorted, joined with
        ``|`` - to the number of distinct texts shared by that pair), and
        ``line_indices`` (mapping of split name to sorted 0-based indices of
        records carrying a leaked text; only splits with leaks appear).
    """
    splits_by_text: defaultdict[str, set[str]] = defaultdict(set)
    indices_by_text: defaultdict[str, defaultdict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for split_name, records in split_records:
        for index, record in enumerate(records):
            text = normalize_text(record.text)
            splits_by_text[text].add(split_name)
            indices_by_text[text][split_name].append(index)
    leaked = {text: splits for text, splits in splits_by_text.items() if len(splits) > 1}
    pairs: Counter[str] = Counter()
    for splits in leaked.values():
        ordered = sorted(splits)
        pairs.update(
            f"{ordered[left]}|{ordered[right]}"
            for left in range(len(ordered))
            for right in range(left + 1, len(ordered))
        )
    line_indices: defaultdict[str, list[int]] = defaultdict(list)
    for text in leaked:
        for split_name, indices in indices_by_text[text].items():
            line_indices[split_name].extend(indices)
    return {
        "count": len(leaked),
        "affected_records": sum(
            len(indices_by_text[text][split_name]) for text in leaked for split_name in leaked[text]
        ),
        "pairs": dict(sorted(pairs.items())),
        "line_indices": {split: sorted(ix) for split, ix in sorted(line_indices.items())},
    }


def find_contradictory_labels(*split_records: tuple[str, Sequence[Record]]) -> dict:
    """Find normalized texts labeled inconsistently, without returning text.

    Both within-split and cross-split contradictions count: the same
    normalized text carrying two different labels inside one split is
    annotation noise, and the same text carrying different labels in
    different splits is a labeling inconsistency. Texts labeled consistently
    everywhere (even when they appear in several splits) are not reported.

    Args:
        *split_records: ``(split_name, records)`` pairs, e.g.
            ``find_contradictory_labels(("train", train), ("test", test))``.

    Returns:
        Dict with ``count`` (number of contradicted normalized texts),
        ``affected_records`` (total occurrences of those texts across all
        splits), and ``line_indices`` (mapping of split name to sorted
        0-based indices of records carrying a contradicted text; only splits
        with contradictions appear). Contains no text.
    """
    labels_by_text: defaultdict[str, set[str]] = defaultdict(set)
    indices_by_text: defaultdict[str, defaultdict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for split_name, records in split_records:
        for index, record in enumerate(records):
            text = normalize_text(record.text)
            labels_by_text[text].add(record.label)
            indices_by_text[text][split_name].append(index)
    contradicted = {text for text, labels in labels_by_text.items() if len(labels) > 1}
    line_indices: defaultdict[str, list[int]] = defaultdict(list)
    for text in contradicted:
        for split_name, indices in indices_by_text[text].items():
            line_indices[split_name].extend(indices)
    return {
        "count": len(contradicted),
        "affected_records": sum(
            len(indices_by_text[text][split_name])
            for text in contradicted
            for split_name in indices_by_text[text]
        ),
        "line_indices": {split: sorted(ix) for split, ix in sorted(line_indices.items())},
    }


def near_duplicate_candidates(
    records: Sequence[Record],
    *,
    threshold: float = 0.9,
    max_candidates: int = 50,
) -> list[dict]:
    """Rank plausible near-duplicate pairs by character TF-IDF similarity.

    The similarity is a REVIEW SIGNAL only, not proof of leakage: paraphrased
    feedback about the same topic can legitimately score high, and heavily
    reworded true duplicates can score low. Transformers memorize
    near-duplicates, so audit flagged pairs by hand before training on them
    (Phase 2 risk register, R-06). Exact duplicates also surface here (they
    score 1.0) but are reported precisely by :func:`duplicate_summary`.

    Args:
        records: Records to compare pairwise.
        threshold: Minimum cosine similarity (exclusive) for a pair to be
            reported.
        max_candidates: Cap on the number of returned pairs.

    Returns:
        List of ``{"index_a": int, "index_b": int, "similarity": float}``
        dicts with ``index_a < index_b`` (self-pairs excluded), sorted by
        similarity descending, ties broken by ascending indices. No text is
        included. Returns ``[]`` for fewer than two records or when no
        character n-grams can be extracted at all (empty vocabulary).
    """
    texts = [normalize_text(record.text) for record in records]
    if len(texts) < 2:
        return []
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
    try:
        tfidf = vectorizer.fit_transform(texts)
    except ValueError:
        # Empty vocabulary: the corpus yields no character n-grams (e.g. all
        # texts are whitespace after normalization).
        return []
    # TfidfVectorizer L2-normalizes rows by default, so the sparse self-dot
    # product is exactly the pairwise cosine similarity - and stays sparse,
    # unlike sklearn.metrics.pairwise.cosine_similarity, which densifies.
    similarity = (tfidf @ tfidf.T).tocoo()
    candidates = [
        {"index_a": int(row), "index_b": int(col), "similarity": float(value)}
        for row, col, value in zip(similarity.row, similarity.col, similarity.data, strict=True)
        if row < col and value > threshold
    ]
    candidates.sort(key=lambda pair: (-pair["similarity"], pair["index_a"], pair["index_b"]))
    return candidates[:max_candidates]


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
