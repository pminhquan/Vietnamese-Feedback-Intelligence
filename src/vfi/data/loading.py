"""Loading utilities for the UIT-VSFC dataset format.

The corpus is distributed as line-aligned plain-text files: one sentence per
line in ``sentences.txt`` and one integer sentiment label per line in
``sentiments.txt``. Lines are aligned by position within each split directory
(``train``, ``validation``, ``test``).

The historical low-level default follows the SEACrowd loader and remains
available as a synthetic-format fixture convenience. The authors' dataset card
describes a different ordering. The experiment adapter requires an explicit
mapping and provenance before reading any UIT-VSFC files.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

SPLITS: tuple[str, ...] = ("train", "validation", "test")

#: Raw UIT-VSFC sentiment integers mapped to canonical label names.
RAW_SENTIMENT_TO_LABEL: dict[int, str] = {0: "positive", 1: "negative", 2: "neutral"}

#: Canonical label space used throughout VFI (alphabetical order).
CANONICAL_LABELS: tuple[str, ...] = ("negative", "neutral", "positive")

#: Canonical label -> integer id.
LABEL_TO_ID: dict[str, int] = {label: idx for idx, label in enumerate(CANONICAL_LABELS)}


class DataLoadingError(Exception):
    """Raised when raw dataset files are missing, misaligned, or invalid."""


@dataclass(frozen=True)
class Record:
    """One labeled feedback text assigned to a dataset split."""

    text: str
    label: str
    split: str


def parse_label_mapping(spec: str) -> dict[int, str]:
    """Parse raw UIT-VSFC integers onto a permutation of canonical labels."""
    mapping: dict[int, str] = {}
    for pair in spec.split(","):
        raw, sep, label = pair.partition(":")
        label = label.strip().lower()
        raw = raw.strip()
        if not sep or raw not in {"0", "1", "2"} or label not in CANONICAL_LABELS:
            raise DataLoadingError("Invalid label mapping: use 0, 1 and 2 with canonical labels")
        key = int(raw)
        if key in mapping:
            raise DataLoadingError("Invalid label mapping: raw integers must be unique")
        mapping[key] = label
    _validate_label_mapping(mapping)
    return mapping


def _validate_label_mapping(mapping: Mapping[int, str]) -> None:
    if (
        len(mapping) != len(CANONICAL_LABELS)
        or any(type(raw) is not int or raw not in {0, 1, 2} for raw in mapping)
        or any(not isinstance(label, str) for label in mapping.values())
        or set(mapping.values()) != set(CANONICAL_LABELS)
    ):
        raise DataLoadingError("Invalid label mapping: each raw integer needs a distinct label")


def normalize_text(text: str) -> str:
    """Minimal, Vietnamese-safe text normalization.

    Applies Unicode NFC composition and collapses whitespace runs to single
    spaces. Deliberately does NOT lowercase, strip diacritics, remove
    punctuation, or remove stopwords: those carry Vietnamese sentiment signal
    (negation words such as "không", tone diacritics, emphasis punctuation).
    TF-IDF lowercasing is handled inside the vectorizer and recorded in the
    model configuration instead.
    """
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def _read_lines(path: Path, split: str) -> tuple[list[str], str]:
    try:
        contents = path.read_bytes()
        # utf-8-sig tolerates a byte-order mark written by Windows editors.
        return contents.decode("utf-8-sig").splitlines(), hashlib.sha256(contents).hexdigest()
    except FileNotFoundError:
        raise DataLoadingError(f"Split {split!r}: {path.name} is missing") from None
    except (OSError, UnicodeError):
        raise DataLoadingError(f"Split {split!r}: cannot read UTF-8 {path.name}") from None


def load_split(
    raw_dir: str | Path, split: str, label_mapping: Mapping[int, str] | None = None
) -> list[Record]:
    """Load one split from ``<raw_dir>/<split>/{sentences,sentiments}.txt``.

    Raises:
        DataLoadingError: on a missing file, line-count mismatch, empty
            sentence, or an unknown/non-integer label (message includes the
            offending line number).
    """
    records, _ = load_split_snapshot(raw_dir, split, label_mapping)
    return records


def load_split_snapshot(
    raw_dir: str | Path, split: str, label_mapping: Mapping[int, str] | None = None
) -> tuple[list[Record], str]:
    """Read one split, binding its source fingerprint to the parsed bytes."""
    if split not in SPLITS:
        raise DataLoadingError(f"Unknown split; expected one of {SPLITS}")
    mapping = RAW_SENTIMENT_TO_LABEL if label_mapping is None else label_mapping
    _validate_label_mapping(mapping)
    split_dir = Path(raw_dir) / split
    sentences, text_digest = _read_lines(split_dir / "sentences.txt", split)
    sentiments, label_digest = _read_lines(split_dir / "sentiments.txt", split)
    if len(sentences) != len(sentiments):
        raise DataLoadingError(
            f"Split {split!r} is misaligned: {len(sentences)} sentences vs "
            f"{len(sentiments)} sentiment labels"
        )
    records: list[Record] = []
    for line_no, (sentence, sentiment) in enumerate(
        zip(sentences, sentiments, strict=True), start=1
    ):
        text = normalize_text(sentence)
        if not text:
            raise DataLoadingError(f"Split {split!r} line {line_no}: empty sentence text")
        try:
            raw_label = int(sentiment.strip())
        except ValueError:
            raise DataLoadingError(
                f"Split {split!r} line {line_no}: non-integer sentiment label"
            ) from None
        if raw_label not in mapping:
            raise DataLoadingError(
                f"Split {split!r} line {line_no}: unknown sentiment label; "
                f"expected one of {sorted(mapping)}"
            )
        records.append(Record(text=text, label=mapping[raw_label], split=split))
    digest = hashlib.sha256((text_digest + label_digest).encode("ascii")).hexdigest()
    return records, digest


def load_dataset(
    raw_dir: str | Path,
    splits: tuple[str, ...] = SPLITS,
    label_mapping: Mapping[int, str] | None = None,
) -> dict[str, list[Record]]:
    """Load the requested splits into ``{split_name: [Record, ...]}``."""
    return {split: load_split(raw_dir, split, label_mapping) for split in splits}
