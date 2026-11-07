"""Loading utilities for the UIT-VSFC dataset format.

The corpus is distributed as line-aligned plain-text files: one sentence per
line in ``sentences.txt`` and one integer sentiment label per line in
``sentiments.txt``. Lines are aligned by position within each split directory
(``train``, ``validation``, ``test``).

The semantics of the raw sentiment integers are an OPEN question, not an
established fact. Two public mirrors of UIT-VSFC disagree:

- the SEACrowd loader declares ``ClassLabel(names=["positive", "negative",
  "neutral"])``, implying ``0=positive, 1=negative, 2=neutral``;
- the authors' own Hugging Face organization card states values are
  ``0 (negative), 1 (neutral) and 2 (positive)``.

Neither claim is verified against the raw files as of 2026-11-07; see
``docs/phase2/label-semantics-evidence.md``. Loading therefore never applies
an interpretation silently: every load goes through an explicit
:class:`RawLabelMapping` that carries its own provenance. The default mapping
keeps the historical SEACrowd reading for backwards compatibility and is
marked ``seacrowd_unverified``; re-declare it with
``LABEL_SOURCE_OWNER_CONFIRMED`` once the corpus authors (or a documented
blind audit) confirm the encoding. Whatever mapping is used, loaded labels
are remapped into VFI's canonical, alphabetically ordered label space
(``negative=0, neutral=1, positive=2``) so all downstream code deals with a
single, documented mapping.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

SPLITS: tuple[str, ...] = ("train", "validation", "test")

#: Canonical label space used throughout VFI (alphabetical order).
CANONICAL_LABELS: tuple[str, ...] = ("negative", "neutral", "positive")

#: Canonical label -> integer id.
LABEL_TO_ID: dict[str, int] = {label: idx for idx, label in enumerate(CANONICAL_LABELS)}

#: Provenance source: mapping defined by a synthetic test fixture only.
LABEL_SOURCE_SYNTHETIC = "synthetic_fixture"

#: Provenance source: mapping confirmed by the corpus authors in writing.
LABEL_SOURCE_OWNER_CONFIRMED = "owner_confirmed"

#: Provenance source: mapping taken from SEACrowd without verification.
LABEL_SOURCE_SEACROWD_UNVERIFIED = "seacrowd_unverified"

#: Raw integers a valid mapping must cover (the raw files use 0/1/2).
_RAW_LABEL_KEYS: frozenset[int] = frozenset({0, 1, 2})


@dataclass(frozen=True)
class RawLabelMapping:
    """Raw UIT-VSFC sentiment integer -> canonical label name, with provenance.

    Construction validates that ``raw_to_canonical`` maps exactly the raw
    integers 0/1/2 onto a permutation of :data:`CANONICAL_LABELS` and raises
    ``ValueError`` otherwise. The input dict is copied so later mutation of
    the caller's dict cannot change a mapping that reports itself as frozen.

    Attributes:
        raw_to_canonical: Maps each raw sentiment integer (0, 1, 2) to a
            distinct canonical label name.
        source: Where the mapping comes from; use the ``LABEL_SOURCE_*``
            module constants.
        evidence: Free-text justification, e.g. a document reference or the
            author confirmation date. Required so a mapping can never exist
            without a stated provenance.
    """

    raw_to_canonical: dict[int, str]
    source: str
    evidence: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_to_canonical", dict(self.raw_to_canonical))
        mapping = self.raw_to_canonical
        if set(mapping) != set(_RAW_LABEL_KEYS):
            raise ValueError(
                f"RawLabelMapping must map exactly the raw integers "
                f"{sorted(_RAW_LABEL_KEYS)}; got keys {sorted(mapping)}"
            )
        if set(mapping.values()) != set(CANONICAL_LABELS):
            raise ValueError(
                f"RawLabelMapping values must be a permutation of {CANONICAL_LABELS}; "
                f"got {sorted(set(mapping.values()))}"
            )


#: Default raw mapping: the historical SEACrowd reading, kept so existing
#: callers behave unchanged. Marked unverified on purpose - see the module
#: docstring and docs/phase2/label-semantics-evidence.md.
DEFAULT_RAW_LABEL_MAPPING = RawLabelMapping(
    {0: "positive", 1: "negative", 2: "neutral"},
    LABEL_SOURCE_SEACROWD_UNVERIFIED,
    "SEACrowd loader ClassLabel declaration; contradicted by the authors' org card "
    "- UNRESOLVED, see docs/phase2/label-semantics-evidence.md",
)

#: Historical raw-mapping constant, kept for backwards compatibility with
#: code (and tests) written before mappings became explicit. Derived from
#: :data:`DEFAULT_RAW_LABEL_MAPPING` so the two cannot drift apart.
RAW_SENTIMENT_TO_LABEL: dict[int, str] = dict(DEFAULT_RAW_LABEL_MAPPING.raw_to_canonical)


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


def parse_label_mapping(spec: str) -> dict[int, str]:
    """Parse a compact raw-label mapping spec: ``"0:positive,1:negative,2:neutral"``.

    Case-insensitive labels; must cover raw integers 0, 1, 2 exactly once and
    map onto a permutation of the canonical labels. Used by the config-driven
    label gate so the owner can declare the confirmed encoding without
    editing source constants.

    Raises:
        DataLoadingError: on a malformed or non-permutation spec.
    """
    mapping: dict[int, str] = {}
    if not spec.strip():
        raise DataLoadingError("label mapping spec is empty")
    for part in spec.split(","):
        if ":" not in part:
            raise DataLoadingError(
                f"malformed label mapping part {part!r}; expected 'raw:label' pairs "
                "like '0:positive,1:negative,2:neutral'"
            )
        raw_text, _, label_text = part.partition(":")
        try:
            raw = int(raw_text.strip())
        except ValueError as exc:
            raise DataLoadingError(f"malformed raw label {raw_text!r} in {spec!r}") from exc
        label = label_text.strip().lower()
        if label not in CANONICAL_LABELS:
            raise DataLoadingError(
                f"unknown canonical label {label_text!r} in {spec!r}; expected one of "
                f"{list(CANONICAL_LABELS)}"
            )
        if raw in mapping:
            raise DataLoadingError(f"raw label {raw} declared twice in {spec!r}")
        mapping[raw] = label
    if set(mapping) != {0, 1, 2}:
        raise DataLoadingError(
            f"label mapping {spec!r} must cover raw integers 0, 1, 2 exactly "
            f"once; got {sorted(mapping)}"
        )
    if sorted(mapping.values()) != sorted(CANONICAL_LABELS):
        raise DataLoadingError(
            f"label mapping {spec!r} must map onto a permutation of {list(CANONICAL_LABELS)}"
        )
    return mapping


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise DataLoadingError(f"Expected dataset file is missing: {path}")
    # utf-8-sig tolerates a byte-order mark written by Windows editors.
    return path.read_text(encoding="utf-8-sig").splitlines()


def load_split(
    raw_dir: str | Path,
    split: str,
    label_mapping: RawLabelMapping | None = None,
) -> list[Record]:
    """Load one split from ``<raw_dir>/<split>/{sentences,sentiments}.txt``.

    Args:
        raw_dir: Directory containing the ``<split>/`` subdirectories.
        split: One of :data:`SPLITS`.
        label_mapping: Raw-integer-to-canonical-label mapping with provenance.
            Defaults to :data:`DEFAULT_RAW_LABEL_MAPPING` (the SEACrowd
            reading, marked ``seacrowd_unverified``). Pass an explicit
            :class:`RawLabelMapping` once the raw encoding is confirmed.

    Raises:
        DataLoadingError: on a missing file, line-count mismatch, empty
            sentence, or an unknown/non-integer label (message includes the
            offending line number).
    """
    if label_mapping is None:
        label_mapping = DEFAULT_RAW_LABEL_MAPPING
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
        if raw_label not in label_mapping.raw_to_canonical:
            raise DataLoadingError(
                f"Split {split!r} line {line_no}: unknown sentiment label {raw_label}; "
                f"expected one of {sorted(label_mapping.raw_to_canonical)}"
            )
        records.append(
            Record(text=text, label=label_mapping.raw_to_canonical[raw_label], split=split)
        )
    return records


def load_dataset(
    raw_dir: str | Path,
    splits: tuple[str, ...] = SPLITS,
    label_mapping: RawLabelMapping | None = None,
) -> dict[str, list[Record]]:
    """Load the requested splits into ``{split_name: [Record, ...]}``.

    ``label_mapping`` is passed through to :func:`load_split` for every split.
    """
    return {split: load_split(raw_dir, split, label_mapping) for split in splits}
