"""Loading utilities for the UIT-VSFC dataset format.

The corpus is distributed as line-aligned plain-text files: one sentence per
line in ``sentences.txt`` and one integer sentiment label per line in
``sentiments.txt``. Lines are aligned by position within each split directory
(``train``, ``validation``, ``test``).

Label mapping source: the SEACrowd loader script declares
``ClassLabel(names=["positive", "negative", "neutral"])`` for the raw
sentiment integers, so ``0=positive, 1=negative, 2=neutral``. VFI remaps
these into its own canonical, alphabetically ordered label space
(``negative=0, neutral=1, positive=2``) so all downstream code deals with a
single, documented mapping.
"""

from __future__ import annotations

import re
import unicodedata
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


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise DataLoadingError(f"Expected dataset file is missing: {path}")
    # utf-8-sig tolerates a byte-order mark written by Windows editors.
    return path.read_text(encoding="utf-8-sig").splitlines()


def load_split(raw_dir: str | Path, split: str) -> list[Record]:
    """Load one split from ``<raw_dir>/<split>/{sentences,sentiments}.txt``.

    Raises:
        DataLoadingError: on a missing file, line-count mismatch, empty
            sentence, or an unknown/non-integer label (message includes the
            offending line number).
    """
    if split not in SPLITS:
        raise DataLoadingError(f"Unknown split {split!r}; expected one of {SPLITS}")
    split_dir = Path(raw_dir) / split
    sentences = _read_lines(split_dir / "sentences.txt")
    sentiments = _read_lines(split_dir / "sentiments.txt")
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
        except ValueError as exc:
            raise DataLoadingError(
                f"Split {split!r} line {line_no}: non-integer sentiment label {sentiment!r}"
            ) from exc
        if raw_label not in RAW_SENTIMENT_TO_LABEL:
            raise DataLoadingError(
                f"Split {split!r} line {line_no}: unknown sentiment label {raw_label}; "
                f"expected one of {sorted(RAW_SENTIMENT_TO_LABEL)}"
            )
        records.append(Record(text=text, label=RAW_SENTIMENT_TO_LABEL[raw_label], split=split))
    return records


def load_dataset(raw_dir: str | Path, splits: tuple[str, ...] = SPLITS) -> dict[str, list[Record]]:
    """Load the requested splits into ``{split_name: [Record, ...]}``."""
    return {split: load_split(raw_dir, split) for split in splits}
