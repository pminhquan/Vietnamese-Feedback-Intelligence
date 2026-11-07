"""Dataset adapters: a canonical, offline loading layer for labeled datasets.

Every adapter turns a LOCAL file layout the owner placed on disk (no network
access, no downloads, ever) into the same :class:`AdapterResult` shape:
:class:`Record` lists keyed by split, plus a provenance dict documenting
where the data came from and how labels were interpreted.

Background: the project does NOT depend on UIT-VSFC (no author agreement is
signed), so any verified-label dataset must feed the pipeline through these
adapters instead. Two hard rules hold for all of them:

- **Offline only.** Adapters read files; they never fetch anything.
- **Privacy-safe errors and provenance.** Error messages and reports carry
  counts, file line numbers, and record indices - never record text (Phase 2
  risk register, R-05). Duplicate/conflict/leakage findings are reported in
  ``provenance["integrity"]`` via the privacy-safe validation primitives and
  are never silently dropped or silently fixed.

Adapters are deterministic: the same files plus the same constructor
arguments always produce the same :class:`AdapterResult`, including split
assignment (the grouped stratified splitter is seeded).

Split assignment (used by the CSV and JSONL adapters): either the file
carries an explicit ``split`` column/field with values from
:data:`vfi.data.loading.SPLITS`, or the adapter carves out a validation side
with the duplicate-grouped stratified splitter from :mod:`vfi.data.splitting`
at ``validation_fraction`` (only ``train`` and ``validation`` are produced;
an official test file must be provided separately with explicit splits).

The custom-dataset JSONL schema constants in this module are the single
source of truth shared with the dataset writer/annotator (Workstream 2): the
writer imports ``CUSTOM_DATASET_FIELDS`` and the ``ANNOTATION_*`` /
``SOURCE_TYPE_*`` constants from here instead of re-declaring them.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from vfi.data.loading import (
    CANONICAL_LABELS,
    DEFAULT_RAW_LABEL_MAPPING,
    SPLITS,
    DataLoadingError,
    RawLabelMapping,
    Record,
    load_dataset,
    normalize_text,
)
from vfi.data.splitting import stratified_split
from vfi.data.validation import (
    duplicate_summary,
    find_contradictory_labels,
    fingerprint,
    leakage_summary,
)

#: Version recorded in every adapter's provenance; bump on behavior changes.
ADAPTERS_VERSION = "1.0.0"

#: Default filename the CSV adapters look for under the root directory.
CSV_FILENAME = "data.csv"

#: Default filename the JSONL adapter looks for under the root directory.
JSONL_FILENAME = "data.jsonl"

FIVE_LEVEL_COLUMNS = ("text", "label")
FIVE_LEVEL_OPTIONAL_COLUMNS = ("split",)
FIVE_LEVEL_LABEL_MAPPING = "1,2->negative; 3->neutral; 4,5->positive"
FIVE_LEVEL_TO_CANONICAL: dict[str, str] = {
    "1": "negative",
    "2": "negative",
    "3": "neutral",
    "4": "positive",
    "5": "positive",
}

#: NEU-ESC sentiment schema (paper Table 1: Neutral/Positive/Negative/Toxic),
#: expressed as the exact lowercase tokens the local export must use.
NEU_ESC_COLUMNS = ("text", "label")
NEU_ESC_LABELS = ("negative", "neutral", "positive", "toxic")
NEU_ESC_LABEL_MAPPING = "negative/neutral/positive kept canonical; toxic kept distinct"
TOXIC_LABEL = "toxic"
TOXIC_POLICY_ERROR = "error"
TOXIC_POLICY_EXCLUDE = "exclude"
TOXIC_POLICY_KEEP = "keep"
TOXIC_POLICIES = (TOXIC_POLICY_ERROR, TOXIC_POLICY_EXCLUDE, TOXIC_POLICY_KEEP)

#: Custom-dataset JSONL schema (single source of truth for the writer).
CUSTOM_DATASET_FIELDS = (
    "record_id",
    "text",
    "sentiment_label",
    "source_type",
    "provenance_reference",
    "annotation_method",
    "annotation_status",
    "dataset_version",
)
CUSTOM_DATASET_OPTIONAL_FIELDS = ("split",)
ANNOTATION_METHOD_HUMAN = "human"
ANNOTATION_METHOD_SYNTHETIC = "synthetic_generator"
ANNOTATION_METHODS = (ANNOTATION_METHOD_HUMAN, ANNOTATION_METHOD_SYNTHETIC)
ANNOTATION_STATUS_DRAFT = "draft"
ANNOTATION_STATUS_ANNOTATED = "annotated"
ANNOTATION_STATUS_APPROVED = "approved"
ANNOTATION_STATUSES = (
    ANNOTATION_STATUS_DRAFT,
    ANNOTATION_STATUS_ANNOTATED,
    ANNOTATION_STATUS_APPROVED,
)
SOURCE_TYPE_SYNTHETIC = "synthetic"
SOURCE_TYPE_OWNER_COLLECTED = "owner_collected"
SOURCE_TYPE_THIRD_PARTY_IMPORT = "third_party_import"
SOURCE_TYPES = (SOURCE_TYPE_SYNTHETIC, SOURCE_TYPE_OWNER_COLLECTED, SOURCE_TYPE_THIRD_PARTY_IMPORT)

UIT_VSFC_DATASET_ID = "uit_vsfc"


class AdapterError(Exception):
    """Raised when an adapter cannot load its local input files.

    Messages name files, line numbers, and expected schemas - never record
    text.
    """


@dataclass(frozen=True)
class AdapterResult:
    """What every adapter returns: split-keyed records plus provenance.

    Attributes:
        records_by_split: Loaded records keyed by split name (only non-empty
            splits appear).
        provenance: Machine-readable metadata - dataset id, adapter name and
            version, source path(s), per-split sha256 fingerprints, the label
            mapping description, adapter-specific policy decisions, and the
            privacy-safe ``integrity`` block (duplicates, leakage,
            contradictory labels). Contains no record text.
    """

    records_by_split: dict[str, list[Record]]
    provenance: dict


@runtime_checkable
class DatasetAdapter(Protocol):
    """Interface every dataset adapter implements (pure local loading)."""

    name: str

    def load(self, root: Path) -> AdapterResult:
        """Load the dataset from a local directory into split-keyed records."""
        ...


def _integrity_summary(records_by_split: dict[str, list[Record]]) -> dict:
    """Privacy-safe integrity block shared by all adapters (no text)."""
    split_pairs = tuple(records_by_split.items())
    return {
        "duplicates_by_split": {
            split: duplicate_summary(records) for split, records in sorted(records_by_split.items())
        },
        "leakage": leakage_summary(*split_pairs),
        "contradictions": find_contradictory_labels(*split_pairs),
    }


def _sha256_by_split(records_by_split: dict[str, list[Record]]) -> dict[str, str]:
    return {split: fingerprint(records) for split, records in sorted(records_by_split.items())}


def _validate_fraction(validation_fraction: float) -> None:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            f"validation_fraction must be strictly between 0 and 1; got {validation_fraction}"
        )


def _assign_splits(
    records: list[Record],
    *,
    declared: list[str] | None,
    validation_fraction: float,
    seed: int,
) -> dict[str, list[Record]]:
    """Group records into splits: explicit assignments or a seeded grouped split.

    ``declared`` must either be ``None`` (fraction-based grouped stratified
    split) or align with ``records`` position for position and contain only
    values from :data:`vfi.data.loading.SPLITS`.
    """
    if not records:
        return {}
    if declared is not None:
        grouped: dict[str, list[Record]] = {split: [] for split in SPLITS}
        for record, split in zip(records, declared, strict=True):
            grouped[split].append(record)
        return {split: grouped[split] for split in SPLITS if grouped[split]}
    try:
        train_side, validation_side = stratified_split(
            records,
            validation_size=validation_fraction,
            seed=seed,
        )
    except ValueError as exc:
        raise AdapterError(
            f"Grouped stratified validation split failed ({exc}); add more distinct "
            "texts per label or provide explicit split assignments in the file"
        ) from exc
    return {
        "train": [Record(r.text, r.label, "train") for r in train_side],
        "validation": [Record(r.text, r.label, "validation") for r in validation_side],
    }


def _read_csv_rows(
    path: Path,
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
) -> list[tuple[int, dict[str, str]]]:
    """Read a UTF-8(-sig) CSV into ``(file_line_number, row)`` pairs.

    Validates the header strictly (required columns present, nothing outside
    ``required + optional``) and per-row arity (too few or too many fields is
    an error). Line numbers are 1-based file lines including the header, so
    they match what the owner sees in a text editor.
    """
    if not path.exists():
        raise AdapterError(f"Expected dataset file is missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        if header is None:
            raise AdapterError(f"{path}: file is empty (no header row)")
        missing = [column for column in required if column not in header]
        unexpected = [column for column in header if column not in required + optional]
        problems = []
        if missing:
            problems.append(f"missing required column(s) {missing}")
        if unexpected:
            problems.append(
                f"unexpected column(s) {unexpected}; allowed: {list(required + optional)}"
            )
        if problems:
            raise AdapterError(f"{path}: invalid header ({'; '.join(problems)})")
        rows: list[tuple[int, dict[str, str]]] = []
        for row in reader:
            line_no = reader.line_num
            extra = row.get(None)
            if extra is not None:
                raise AdapterError(
                    f"{path} line {line_no}: row has {len(extra)} extra field(s); "
                    "check delimiter and quoting"
                )
            absent = [column for column in required if row.get(column) is None]
            if absent:
                raise AdapterError(
                    f"{path} line {line_no}: missing value(s) for column(s) {absent}"
                )
            rows.append((line_no, {key: value for key, value in row.items() if key is not None}))
    return rows


def _line_list(lines: list[int]) -> str:
    return ", ".join(str(line) for line in lines)


class FiveLevelCsvAdapter:
    """Adapter for a five-level CSV export (labels 1-5 -> 3 canonical classes).

    Expects a UTF-8(-sig) CSV at ``<root>/data.csv`` (override with
    ``filename``) with columns ``text,label`` and an optional ``split``
    column. Labels ``1``/``2`` map to ``negative``, ``3`` to ``neutral``, and
    ``4``/``5`` to ``positive`` (documented mapping, recorded in provenance).
    Unknown labels, empty text, and unknown split values raise an
    :class:`AdapterError` listing every offending file line; duplicates and
    contradictory labels are reported in provenance, never silently dropped.
    Without a ``split`` column, only ``train``/``validation`` are produced
    via a seeded grouped stratified split at ``validation_fraction``.
    """

    name = "five_level_csv"

    def __init__(
        self,
        validation_fraction: float = 0.2,
        seed: int = 42,
        filename: str = CSV_FILENAME,
    ) -> None:
        _validate_fraction(validation_fraction)
        self.validation_fraction = validation_fraction
        self.seed = seed
        self.filename = filename

    def load(self, root: Path) -> AdapterResult:
        root = Path(root)
        path = root / self.filename
        rows = _read_csv_rows(path, FIVE_LEVEL_COLUMNS, FIVE_LEVEL_OPTIONAL_COLUMNS)
        has_split_column = "split" in rows[0][1] if rows else False
        bad_labels: list[str] = []
        empty_text_lines: list[int] = []
        bad_splits: list[str] = []
        records: list[Record] = []
        declared: list[str] | None = [] if has_split_column else None
        for line_no, row in rows:
            text = normalize_text(row["text"])
            if not text:
                empty_text_lines.append(line_no)
            raw_label = row["label"].strip()
            canonical = FIVE_LEVEL_TO_CANONICAL.get(raw_label)
            if canonical is None:
                bad_labels.append(f"line {line_no} (label {raw_label!r})")
            split_value: str | None = None
            if has_split_column:
                split_value = row["split"].strip()
                if split_value not in SPLITS:
                    bad_splits.append(f"line {line_no} (split {split_value!r})")
                assert declared is not None
                declared.append(split_value)
            # Rows with problems produce placeholder records only; they are
            # discarded when the collected problems raise below.
            records.append(Record(text, canonical or raw_label, "train"))
        problems = []
        if bad_labels:
            problems.append(
                f"unknown labels (expected one of {sorted(FIVE_LEVEL_TO_CANONICAL)}): "
                + ", ".join(bad_labels)
            )
        if empty_text_lines:
            problems.append(f"empty text at line(s): {_line_list(empty_text_lines)}")
        if bad_splits:
            problems.append(
                f"unknown split values (expected one of {SPLITS}): {', '.join(bad_splits)}"
            )
        if problems:
            raise AdapterError(f"{path}: " + "; ".join(problems))
        records_by_split = _assign_splits(
            records,
            declared=declared,
            validation_fraction=self.validation_fraction,
            seed=self.seed,
        )
        provenance = {
            "dataset_id": "five_level_csv",
            "adapter": {"name": self.name, "version": ADAPTERS_VERSION},
            "source": [str(path)],
            "sha256": _sha256_by_split(records_by_split),
            "label_mapping": FIVE_LEVEL_LABEL_MAPPING,
            "splits_created_by": "split_column" if has_split_column else "validation_fraction",
            "integrity": _integrity_summary(records_by_split),
        }
        return AdapterResult(records_by_split, provenance)


class NeuEscAdapter:
    """Adapter for the NEU-ESC-style 4-class CSV export.

    Expects a UTF-8(-sig) CSV at ``<root>/data.csv`` (override with
    ``filename``) with columns ``text,label`` and labels exactly
    ``negative``/``neutral``/``positive``/``toxic`` (see
    :data:`NEU_ESC_LABELS`, per the NEU-ESC paper's 4-class schema). An
    optional ``split`` column assigns splits; otherwise a seeded grouped
    stratified validation split is used.

    ``toxic`` is a DISTINCT class - it is never mapped into the 3-class
    canonical space (in particular, never silently mapped to ``negative``).
    The ``toxic_policy`` decides what happens to toxic records:

    - ``"error"`` (default): :meth:`load` refuses when any toxic record is
      present, explaining the choice; a file without toxic records loads.
    - ``"exclude"``: toxic records are dropped before splitting; their count
      and file line numbers are reported in provenance.
    - ``"keep"``: toxic records stay in the result labeled ``toxic`` (excluded
      from the 3-class training labels; a 3-class consumer must filter them)
      and are reported separately in provenance.

    ``"exclude"`` and ``"keep"`` require a non-empty ``justification``, which
    is recorded verbatim in provenance. Without a ``split`` column, keeping
    toxic records means they participate in the grouped stratified split, so
    the toxicity stratum needs at least two distinct-text groups.
    """

    name = "neu_esc"

    def __init__(
        self,
        toxic_policy: str = TOXIC_POLICY_ERROR,
        justification: str | None = None,
        validation_fraction: float = 0.2,
        seed: int = 42,
        filename: str = CSV_FILENAME,
    ) -> None:
        if toxic_policy not in TOXIC_POLICIES:
            raise ValueError(f"toxic_policy must be one of {TOXIC_POLICIES}; got {toxic_policy!r}")
        if toxic_policy in (TOXIC_POLICY_EXCLUDE, TOXIC_POLICY_KEEP) and not (
            isinstance(justification, str) and justification.strip()
        ):
            raise ValueError(
                f"toxic_policy={toxic_policy!r} requires a non-empty justification "
                "explaining why excluding (or keeping) toxic records is acceptable"
            )
        _validate_fraction(validation_fraction)
        self.toxic_policy = toxic_policy
        self.justification = justification
        self.validation_fraction = validation_fraction
        self.seed = seed
        self.filename = filename

    def load(self, root: Path) -> AdapterResult:
        root = Path(root)
        path = root / self.filename
        rows = _read_csv_rows(path, NEU_ESC_COLUMNS, FIVE_LEVEL_OPTIONAL_COLUMNS)
        has_split_column = "split" in rows[0][1] if rows else False
        bad_labels: list[str] = []
        empty_text_lines: list[int] = []
        bad_splits: list[str] = []
        records: list[Record] = []
        toxic_line_numbers: list[int] = []
        declared: list[str] | None = [] if has_split_column else None
        for line_no, row in rows:
            text = normalize_text(row["text"])
            if not text:
                empty_text_lines.append(line_no)
            raw_label = row["label"].strip()
            if raw_label not in NEU_ESC_LABELS:
                bad_labels.append(f"line {line_no} (label {raw_label!r})")
            split_value = None
            if has_split_column:
                split_value = row["split"].strip()
                if split_value not in SPLITS:
                    bad_splits.append(f"line {line_no} (split {split_value!r})")
            if raw_label == TOXIC_LABEL:
                toxic_line_numbers.append(line_no)
                if self.toxic_policy == TOXIC_POLICY_EXCLUDE:
                    continue  # dropped before splitting; count stays in provenance
            if has_split_column:
                assert declared is not None
                declared.append(split_value)
            records.append(Record(text, raw_label, "train"))
        problems = []
        if bad_labels:
            problems.append(
                f"unknown labels (expected one of {list(NEU_ESC_LABELS)}): {', '.join(bad_labels)}"
            )
        if empty_text_lines:
            problems.append(f"empty text at line(s): {_line_list(empty_text_lines)}")
        if bad_splits:
            problems.append(
                f"unknown split values (expected one of {SPLITS}): {', '.join(bad_splits)}"
            )
        if problems:
            raise AdapterError(f"{path}: " + "; ".join(problems))
        if toxic_line_numbers and self.toxic_policy == TOXIC_POLICY_ERROR:
            raise AdapterError(
                f"{path}: found {len(toxic_line_numbers)} toxic record(s) on line "
                f"{_line_list(toxic_line_numbers)} of the file. Toxic is a distinct NEU-ESC "
                "class and is never mapped into the 3-class canonical space; pass "
                "toxic_policy='exclude' (drop toxic records) or toxic_policy='keep' "
                "(retain them labeled 'toxic'), each with a justification recorded in "
                "provenance"
            )
        records_by_split = _assign_splits(
            records,
            declared=declared,
            validation_fraction=self.validation_fraction,
            seed=self.seed,
        )
        provenance = {
            "dataset_id": "neu_esc",
            "adapter": {"name": self.name, "version": ADAPTERS_VERSION},
            "source": [str(path)],
            "sha256": _sha256_by_split(records_by_split),
            "label_mapping": NEU_ESC_LABEL_MAPPING,
            "toxic_policy": self.toxic_policy,
            "toxic_count": len(toxic_line_numbers),
            "toxic_line_numbers": toxic_line_numbers,
            "integrity": _integrity_summary(records_by_split),
        }
        if self.toxic_policy in (TOXIC_POLICY_EXCLUDE, TOXIC_POLICY_KEEP):
            provenance["justification"] = self.justification
        return AdapterResult(records_by_split, provenance)


class CustomJsonlAdapter:
    """Adapter for the custom-dataset JSONL export (Workstream 2 schema).

    Expects a UTF-8(-sig) JSON Lines file at ``<root>/data.jsonl`` (override
    with ``filename``); each line is one JSON object with exactly the fields
    in :data:`CUSTOM_DATASET_FIELDS` plus the optional ``split`` field. The
    ``CUSTOM_DATASET_*``, ``ANNOTATION_*``, and ``SOURCE_TYPE_*`` constants
    above are the schema's single source of truth; the dataset writer
    imports them from this module.

    Strict validation: unknown or missing fields, invalid enum values, empty
    required strings, empty text, and duplicate ``record_id`` values raise an
    :class:`AdapterError` naming the offending lines. Only records with
    ``annotation_status == "approved"`` become loadable records;
    ``draft``/``annotated`` lines are reported in provenance as not-approved
    counts. Split assignment uses the ``split`` field when every approved
    record declares one (mixing declared and undeclared splits is an error),
    else a seeded grouped stratified split at ``validation_fraction``.
    """

    name = "custom_jsonl"

    def __init__(
        self,
        validation_fraction: float = 0.2,
        seed: int = 42,
        filename: str = JSONL_FILENAME,
    ) -> None:
        _validate_fraction(validation_fraction)
        self.validation_fraction = validation_fraction
        self.seed = seed
        self.filename = filename

    def load(self, root: Path) -> AdapterResult:
        root = Path(root)
        path = root / self.filename
        if not path.exists():
            raise AdapterError(f"Expected dataset file is missing: {path}")
        allowed_fields = set(CUSTOM_DATASET_FIELDS + CUSTOM_DATASET_OPTIONAL_FIELDS)
        id_lines: dict[str, int] = {}
        records: list[Record] = []
        declared: list[str] = []  # aligned with records; "" = no split declared
        status_counts: dict[str, int] = dict.fromkeys(ANNOTATION_STATUSES, 0)
        problems: list[str] = []

        def problem(message: str) -> None:
            problems.append(message)

        with path.open(encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    problem(f"line {line_no}: blank line")
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    problem(f"line {line_no}: invalid JSON ({exc.msg})")
                    continue
                if not isinstance(payload, dict):
                    problem(f"line {line_no}: expected a JSON object, got {type(payload).__name__}")
                    continue
                unknown = sorted(set(payload) - allowed_fields)
                missing = [field for field in CUSTOM_DATASET_FIELDS if field not in payload]
                if unknown:
                    problem(
                        f"line {line_no}: unknown field(s) {unknown}; "
                        f"allowed: {sorted(allowed_fields)}"
                    )
                if missing:
                    problem(f"line {line_no}: missing field(s) {missing}")
                if unknown or missing:
                    continue
                for field, allowed in (
                    ("source_type", SOURCE_TYPES),
                    ("annotation_method", ANNOTATION_METHODS),
                    ("annotation_status", ANNOTATION_STATUSES),
                ):
                    if payload[field] not in allowed:
                        problem(
                            f"line {line_no}: {field} {payload[field]!r}; "
                            f"expected one of {list(allowed)}"
                        )
                for field in ("record_id", "provenance_reference", "dataset_version"):
                    value = payload[field]
                    if not isinstance(value, str) or not value.strip():
                        problem(f"line {line_no}: {field} must be a non-empty string")
                text = normalize_text(payload["text"]) if isinstance(payload["text"], str) else ""
                if not text:
                    problem(f"line {line_no}: text must be a non-empty string")
                label = payload["sentiment_label"]
                if label not in CANONICAL_LABELS:
                    problem(
                        f"line {line_no}: sentiment_label {label!r}; "
                        f"expected one of {list(CANONICAL_LABELS)}"
                    )
                split_value = payload.get("split")
                if split_value is not None and split_value not in SPLITS:
                    problem(f"line {line_no}: split {split_value!r}; expected one of {SPLITS}")
                record_id = payload["record_id"]
                if isinstance(record_id, str) and record_id.strip():
                    if record_id in id_lines:
                        problem(
                            f"line {line_no}: duplicate record_id {record_id!r} "
                            f"(first seen on line {id_lines[record_id]})"
                        )
                    else:
                        id_lines[record_id] = line_no
                status = payload["annotation_status"]
                if isinstance(status, str) and status in status_counts:
                    status_counts[status] += 1
                if status == ANNOTATION_STATUS_APPROVED:
                    records.append(Record(text, label, "train"))
                    declared.append(split_value if split_value is not None else "")
        if problems:
            raise AdapterError(f"{path}: " + "; ".join(problems))
        has_any_declared = any(declared)
        has_all_declared = all(declared)
        if has_any_declared and not has_all_declared:
            raise AdapterError(
                f"{path}: either every approved record must declare a split or none may; "
                "mixing declared and undeclared splits is ambiguous"
            )
        clean_declared = declared if has_all_declared else None
        records_by_split = _assign_splits(
            records,
            declared=clean_declared,
            validation_fraction=self.validation_fraction,
            seed=self.seed,
        )
        not_approved = {
            status: count
            for status, count in status_counts.items()
            if status != ANNOTATION_STATUS_APPROVED
        }
        provenance = {
            "dataset_id": "custom_jsonl",
            "adapter": {"name": self.name, "version": ADAPTERS_VERSION},
            "source": [str(path)],
            "sha256": _sha256_by_split(records_by_split),
            "label_mapping": "sentiment_label values are canonical labels used as-is",
            "annotation_status_counts": status_counts,
            "not_approved_counts": not_approved,
            "not_approved_total": sum(not_approved.values()),
            "approved_count": status_counts[ANNOTATION_STATUS_APPROVED],
            "splits_created_by": "split_field" if has_all_declared else "validation_fraction",
            "integrity": _integrity_summary(records_by_split),
        }
        return AdapterResult(records_by_split, provenance)


class UitVsfcAdapter:
    """Adapter for the line-aligned UIT-VSFC layout, gated as always.

    Wraps :func:`vfi.data.loading.load_dataset` (``<root>/<split>/`` with
    ``sentences.txt``/``sentiments.txt``) and records the provenance of the
    :class:`RawLabelMapping` used: the raw integer encoding is an unresolved
    question, so provenance carries ``label_mapping_source`` plus the
    evidence string. The default mapping stays the unverified SEACrowd
    reading; pass an explicit ``label_mapping`` once the encoding is
    confirmed. This adapter reads local files only - it never downloads
    anything.
    """

    name = "uit_vsfc"

    def __init__(self, label_mapping: RawLabelMapping | None = None) -> None:
        self.label_mapping = label_mapping

    def load(self, root: Path) -> AdapterResult:
        mapping = self.label_mapping or DEFAULT_RAW_LABEL_MAPPING
        try:
            records_by_split = load_dataset(root, label_mapping=mapping)
        except DataLoadingError as exc:
            raise AdapterError(f"UIT-VSFC load failed: {exc}") from exc
        provenance = {
            "dataset_id": UIT_VSFC_DATASET_ID,
            "adapter": {"name": self.name, "version": ADAPTERS_VERSION},
            "source": [str(Path(root))],
            "sha256": _sha256_by_split(records_by_split),
            "label_mapping": {
                "raw_to_canonical": {
                    str(raw): canonical
                    for raw, canonical in sorted(mapping.raw_to_canonical.items())
                },
                "evidence": mapping.evidence,
            },
            "label_mapping_source": mapping.source,
            "integrity": _integrity_summary(records_by_split),
        }
        return AdapterResult(records_by_split, provenance)
