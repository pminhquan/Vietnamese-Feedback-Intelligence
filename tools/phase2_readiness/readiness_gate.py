"""Offline Phase 2 readiness gate for VFI: an operational checklist tool.

The gate reads a JSON "evidence package" the OWNER prepares by hand and decides
whether a planned Phase 2 (fine-tuning) experiment satisfies the program's
scientific and operational prerequisites. It encodes gates G1-G5 of
``docs/phase2/00-executive-decision.md``, the M1 pilot contract of
``docs/phase2/pilot-experiment-plan.md``, and the label-audit mandate of
``docs/phase2/label-semantics-evidence.md`` section 7, plus risks R-01
(untrusted pickle checkpoints), R-06 (near-duplicate leakage) and R-08
(accidental weight commits) from ``docs/phase2/risk-register.md``.

WHAT THIS TOOL IS
- A completeness and internal-consistency checker over self-reported evidence.
  It turns "did we forget something?" into a mechanical, repeatable,
  fail-closed checklist with one verdict per readiness category.

LIMITATIONS (read before trusting a PASS)
- It does NOT validate legal rights. A license statement or access-condition
  claim in the input is taken at face value; only the owner can establish what
  a license or agreement actually permits.
- It does NOT verify claims against the real world. No URL, file, signature,
  SHA, date, or measurement in the input is checked against anything.
- It CANNOT detect untrue evidence. A field can be complete, well-formed and
  simply false. A PASS means the package is complete and self-consistent, not
  that the experiment is safe or that any claim is true.
- Self-declared checkboxes are rejected by design (e.g. ``license_status:
  "ok"`` is not license evidence); concrete evidence fields are required
  instead, and placeholder statements ("unknown", "ok") are not accepted.
- Present-but-incomplete evidence (a required field missing, empty, or a
  ``TODO:`` placeholder) yields UNKNOWN, never PASS.

USAGE
- ``readiness_gate.py --schema``
  Print an INCOMPLETE onboarding template (valid JSON; completed-action
  claims are ``TODO:`` placeholders or not-done flags, so feeding it back
  through ``--input`` does NOT pass) to stdout, then exit 0. Takes
  precedence over ``--input``.
- ``readiness_gate.py --input PATH [--json]``
  Evaluate the evidence file. Human-readable checklist on stdout by default;
  ``--json`` prints the machine-readable summary (schema_version,
  generated_at_utc, per-category verdicts, overall, next_actions) instead.
- Exit codes: 0 = overall PASS; 1 = overall BLOCKED or UNKNOWN (anything not
  green); 2 = usage/schema error (no input, unreadable file, invalid JSON,
  unknown top-level key). Every parse/validation failure fails CLOSED: the
  verdict is BLOCKED with the reason, never a traceback.

SCHEMA (v2)
Top level: required ``schema_version`` (must be 2) plus the 11 required
category objects listed below. Legacy ``schema_version`` 1 inputs are
rejected with exit code 2 and a migration hint. Unknown top-level keys are
rejected, and so are unknown fields inside a category. Wrong types,
out-of-range/non-finite numbers, invalid 40-hex/64-hex SHAs, non-ISO-8601
dates, synthetic fixture markers, and mismatched split/holdout fingerprints
are BLOCKED; a required field that is absent, empty, or a ``TODO:`` placeholder
yields UNKNOWN (evidence insufficient). Placeholder matching is
case-insensitive ("todo", "TODO: x", "todo x"). Values inside free-form lists
and extra pinned-version entries are not placeholder-scanned; load-bearing
slots (required packages, split fingerprint maps) are validated
independently.

1. dataset_provenance: dataset_id, dataset_version, dataset_adapter
   (custom_jsonl | five_level_csv | neu_esc | uit_vsfc), synthetic_fixture
   (bool; true is BLOCKED - synthetic fixtures never pass real readiness),
   access_condition_class (owner_collected_human_approved | licensed_public |
   gated_click_accept | author_agreement_signed), source_url (required for
   external classes, optional for owner_collected_human_approved),
   license_statement (concrete terms), access_evidence_reference (route-
   appropriate reference; bool rights_confirmed or checkbox strings never
   pass). Only uit_vsfc requires author_agreement_signed (and is BLOCKED under
   any other access class).
2. label_mapping_verification: method matching the declared adapter:
   - custom_jsonl -> human_approved_canonical (+ verification_reference)
   - five_level_csv -> ordinal_five_to_three (+ verification_reference)
   - neu_esc -> neu_esc_named_labels (+ verification_reference,
     toxic_policy='exclude', toxic_justification, toxic_excluded_count >= 0;
     toxic_policy='keep' or 'error' is BLOCKED for training readiness)
   - uit_vsfc -> blind_audit (audit_sample_size >= 100, agreement_rate >= 0.9,
     resulting_mapping 0/1/2 permutation, verification_reference) or
     author_confirmation (resulting_mapping, quote_reference)
3. split_definition: split_policy (physical | grouped_validation),
   train_source, validation_source, test_source (strings), train_size,
   validation_size, test_size (positive ints), fingerprint_algorithm
   (string; warning when not sha256), train_fingerprint,
   validation_fingerprint, test_fingerprint (distinct 64-hex sha256 strings).
4. leakage_preflight: exact_match_planned (bool, must be true),
   exact_match_executed (bool), exact_match_report_reference (text when
   executed), preflight_passed (bool), preflight_report_sha256 (64-hex when
   executed), preflight_synthetic_fixture (bool; true is BLOCKED),
   preflight_contradictions_count (int, must be 0),
   preflight_split_fingerprints (map with train/validation/test 64-hex hashes
   matching split_definition; preflight access to test is allowed and
   required), near_duplicate_method, near_duplicate_threshold (0-1),
   near_duplicate_executed (bool), near_duplicate_report_reference (text when
   executed), near_duplicate_split_fingerprints (all three effective hashes),
   preflight_created_at (timestamp from the Phase 1 receipt), and
   unresolved_conflicts (int, must be 0).
5. phase1_integrity: phase1_head_sha (40-hex), baseline_evidence_reference
   (string), baseline_evidence_sha256 (64-hex), baseline_result_scope
   (train_validation | declared_holdout | synthetic_fixture; synthetic_fixture
   is BLOCKED), baseline_split_fingerprints (train + validation 64-hex hashes
   matching split_definition; 'test' is forbidden here),
   baseline_holdout_fingerprint (64-hex matching split_definition.test_fingerprint
   when baseline_result_scope is declared_holdout), and
   baseline_preflight_report_sha256 (same frozen receipt as leakage_preflight),
   baseline_training_started_at (timestamp from training_report.json), and
   test_excluded_from_fit_and_selection (bool, must be true).
6. model_checkpoint: checkpoint_id (string), revision_sha (40-hex),
   license_name (string), weights_format (safetensors passes; pickle_bin is
   BLOCKED per R-01), checksum_recorded (bool, must be true) plus
   checksum_sha256 (64-hex; required when checksum_recorded is true).
7. tokenization_contract: segmenter_name, segmenter_version,
   determinism_evidence_reference (strings), max_length (positive int;
   warning when it exceeds PhoBERT's documented 256), truncation_side
   (left | right; warning when left, the pilot contract fixes right).
8. dependency_plan: pinned_versions (object with semver-looking torch /
   transformers / segmenter entries), lockfile_reference (string).
9. compute_feasibility: device (cpu | cuda), vram_gb (positive number,
   required when device is cuda), oom_recovery_plan (string).
10. artifact_privacy_controls: gitignore_patterns (list of strings; must
    contain ``*.bin`` and ``*.pt``), reports_pii_policy (string), and
    verified_against_repository (bool; a declared pattern list that has not
    been checked against the actual repository .gitignore is UNKNOWN, not
    PASS).
11. owner_approval: approver (string), approval_date (ISO-8601, must not be
    in the future), scope_statement (string). A missing or non-passing owner
    approval forces the overall verdict to BLOCKED regardless of everything
    else.

PRIVACY / SAFETY
- Pure stdlib, no network, no file writes, no subprocesses: a read-only
  checker. The input is declared non-sensitive by protocol, but the tool
  never prints any input-derived value longer than 120 characters (a fixed
  truncation marker is appended), and its reasons quote only field names,
  enum values, and numbers.

EXTENDING
- Add the category to CATEGORY_ORDER, a field table to FIELD_SPECS, a check
  function registered in CHECKS, and an honest, INCOMPLETE entry to
  build_template() (``TODO:`` placeholders / not-done flags only; passing
  fixtures live only in the test suite).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA_VERSION = 2
OUTPUT_SCHEMA_VERSION = 1

PASS = "pass"
BLOCKED = "blocked"
UNKNOWN = "unknown"

EXIT_PASS = 0
EXIT_BLOCKED = 1  # also used for overall UNKNOWN (anything not green)
EXIT_USAGE = 2

MAX_PRINTED_VALUE = 120
TRUNCATION_MARKER = "...[truncated]"

CANONICAL_CLASSES = ("negative", "neutral", "positive")
DATASET_ADAPTERS = ("custom_jsonl", "five_level_csv", "neu_esc", "uit_vsfc")
ACCESS_CONDITION_CLASSES = (
    "owner_collected_human_approved",
    "licensed_public",
    "gated_click_accept",
    "author_agreement_signed",
)
EXTERNAL_ACCESS_CLASSES = frozenset(
    {"licensed_public", "gated_click_accept", "author_agreement_signed"}
)
ADAPTER_ALLOWED_ACCESS: dict[str, frozenset[str]] = {
    "custom_jsonl": frozenset(ACCESS_CONDITION_CLASSES),
    "five_level_csv": EXTERNAL_ACCESS_CLASSES,
    "neu_esc": frozenset({"gated_click_accept"}),
    "uit_vsfc": frozenset({"author_agreement_signed"}),
}
LABEL_METHODS = (
    "human_approved_canonical",
    "ordinal_five_to_three",
    "neu_esc_named_labels",
    "author_confirmation",
    "blind_audit",
)
ADAPTER_ALLOWED_LABEL_METHODS: dict[str, tuple[str, ...]] = {
    "custom_jsonl": ("human_approved_canonical",),
    "five_level_csv": ("ordinal_five_to_three",),
    "neu_esc": ("neu_esc_named_labels",),
    "uit_vsfc": ("author_confirmation", "blind_audit"),
}
SPLIT_POLICIES = ("physical", "grouped_validation")
BASELINE_RESULT_SCOPES = ("train_validation", "declared_holdout", "synthetic_fixture")
ALL_SPLITS = ("train", "validation", "test")
FIT_SPLITS = ("train", "validation")

LABEL_AGREEMENT_THRESHOLD = 0.9
MIN_BLIND_AUDIT_SAMPLE = 100
PHOBERT_MAX_LENGTH = 256
REQUIRED_GITIGNORE_PATTERNS = ("*.bin", "*.pt")
REQUIRED_PINNED_PACKAGES = ("torch", "transformers", "segmenter")
PLACEHOLDER_LICENSE_STATEMENTS = frozenset(
    {"unknown", "ok", "yes", "true", "none", "n/a", "na", "tbd", "todo", "tobedone"}
)
NON_EVIDENCE_STATEMENTS = PLACEHOLDER_LICENSE_STATEMENTS | frozenset(
    {"confirmed", "approved", "signed", "accepted", "rights_confirmed"}
)
#: Values matching this prefix (case-insensitive, with or without a colon)
#: are placeholders, not evidence: any text field set to "TODO: ..." /
#: "todo ..." is treated as missing (UNKNOWN), so onboarding templates
#: cannot accidentally pass as completed evidence packages.
PLACEHOLDER_PREFIX = "TODO:"


def _is_placeholder(value: str) -> bool:
    """Case-insensitive placeholder match: 'TODO: x', 'todo x', 'TODO'."""
    normalized = value.strip().lower()
    return normalized == "todo" or normalized.startswith("todo:") or normalized.startswith("todo ")


def _is_non_evidence_statement(value: str) -> bool:
    """Match bare status words ('ok', 'confirmed', 'rights_confirmed', ...)."""
    return value.strip().lower() in NON_EVIDENCE_STATEMENTS


HEX40_RE = re.compile(r"[0-9a-fA-F]{40}")
HEX64_RE = re.compile(r"[0-9a-fA-F]{64}")
IDENTIFIER_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}")
SEMVER_RE = re.compile(r"\d+\.\d+(\.\d+)?([+\-][0-9A-Za-z.\-+]+)?")

# Field kinds understood by the structural validator.
KIND_TEXT = "text"
KIND_INT = "int"
KIND_NUMBER = "number"
KIND_BOOL = "bool"
KIND_ENUM = "enum"
KIND_TEXT_LIST = "text_list"
KIND_STRING_MAP = "string_map"
KIND_CLASS_MAPPING = "class_mapping"

CATEGORY_ORDER: tuple[str, ...] = (
    "dataset_provenance",
    "label_mapping_verification",
    "split_definition",
    "leakage_preflight",
    "phase1_integrity",
    "model_checkpoint",
    "tokenization_contract",
    "dependency_plan",
    "compute_feasibility",
    "artifact_privacy_controls",
    "owner_approval",
)


@dataclass(frozen=True)
class FieldSpec:
    """Structural contract for one evidence field inside a category."""

    kind: str
    required: bool = True
    choices: tuple[str, ...] = ()


FIELD_SPECS: dict[str, dict[str, FieldSpec]] = {
    "dataset_provenance": {
        "dataset_id": FieldSpec(KIND_TEXT),
        "dataset_version": FieldSpec(KIND_TEXT),
        "dataset_adapter": FieldSpec(KIND_ENUM, choices=DATASET_ADAPTERS),
        "synthetic_fixture": FieldSpec(KIND_BOOL),
        "access_condition_class": FieldSpec(KIND_ENUM, choices=ACCESS_CONDITION_CLASSES),
        "source_url": FieldSpec(KIND_TEXT, required=False),
        "license_statement": FieldSpec(KIND_TEXT),
        "access_evidence_reference": FieldSpec(KIND_TEXT),
    },
    "label_mapping_verification": {
        "method": FieldSpec(KIND_ENUM, choices=LABEL_METHODS),
        "verification_reference": FieldSpec(KIND_TEXT, required=False),
        "audit_sample_size": FieldSpec(KIND_INT, required=False),
        "agreement_rate": FieldSpec(KIND_NUMBER, required=False),
        "resulting_mapping": FieldSpec(KIND_CLASS_MAPPING, required=False),
        "quote_reference": FieldSpec(KIND_TEXT, required=False),
        "toxic_policy": FieldSpec(
            KIND_ENUM,
            required=False,
            choices=("error", "exclude", "keep"),
        ),
        "toxic_justification": FieldSpec(KIND_TEXT, required=False),
        "toxic_excluded_count": FieldSpec(KIND_INT, required=False),
    },
    "split_definition": {
        "split_policy": FieldSpec(KIND_ENUM, choices=SPLIT_POLICIES),
        "train_source": FieldSpec(KIND_TEXT),
        "validation_source": FieldSpec(KIND_TEXT),
        "test_source": FieldSpec(KIND_TEXT),
        "train_size": FieldSpec(KIND_INT),
        "validation_size": FieldSpec(KIND_INT),
        "test_size": FieldSpec(KIND_INT),
        "fingerprint_algorithm": FieldSpec(KIND_TEXT),
        "train_fingerprint": FieldSpec(KIND_TEXT),
        "validation_fingerprint": FieldSpec(KIND_TEXT),
        "test_fingerprint": FieldSpec(KIND_TEXT),
    },
    "leakage_preflight": {
        "exact_match_planned": FieldSpec(KIND_BOOL),
        "exact_match_executed": FieldSpec(KIND_BOOL),
        "exact_match_report_reference": FieldSpec(KIND_TEXT, required=False),
        "preflight_passed": FieldSpec(KIND_BOOL),
        "preflight_report_sha256": FieldSpec(KIND_TEXT, required=False),
        "preflight_created_at": FieldSpec(KIND_TEXT),
        "preflight_synthetic_fixture": FieldSpec(KIND_BOOL),
        "preflight_contradictions_count": FieldSpec(KIND_INT),
        "preflight_split_fingerprints": FieldSpec(KIND_STRING_MAP),
        "near_duplicate_method": FieldSpec(KIND_TEXT),
        "near_duplicate_threshold": FieldSpec(KIND_NUMBER),
        "near_duplicate_executed": FieldSpec(KIND_BOOL),
        "near_duplicate_report_reference": FieldSpec(KIND_TEXT, required=False),
        "near_duplicate_split_fingerprints": FieldSpec(KIND_STRING_MAP),
        "unresolved_conflicts": FieldSpec(KIND_INT),
    },
    "phase1_integrity": {
        "phase1_head_sha": FieldSpec(KIND_TEXT),
        "baseline_evidence_reference": FieldSpec(KIND_TEXT),
        "baseline_evidence_sha256": FieldSpec(KIND_TEXT, required=False),
        "baseline_preflight_report_sha256": FieldSpec(KIND_TEXT),
        "baseline_training_started_at": FieldSpec(KIND_TEXT),
        "baseline_result_scope": FieldSpec(KIND_ENUM, choices=BASELINE_RESULT_SCOPES),
        "baseline_split_fingerprints": FieldSpec(KIND_STRING_MAP),
        "baseline_holdout_fingerprint": FieldSpec(KIND_TEXT, required=False),
        "test_excluded_from_fit_and_selection": FieldSpec(KIND_BOOL),
    },
    "model_checkpoint": {
        "checkpoint_id": FieldSpec(KIND_TEXT),
        "revision_sha": FieldSpec(KIND_TEXT),
        "license_name": FieldSpec(KIND_TEXT),
        "weights_format": FieldSpec(KIND_ENUM, choices=("safetensors", "pickle_bin")),
        "checksum_recorded": FieldSpec(KIND_BOOL),
        "checksum_sha256": FieldSpec(KIND_TEXT, required=False),
    },
    "tokenization_contract": {
        "segmenter_name": FieldSpec(KIND_TEXT),
        "segmenter_version": FieldSpec(KIND_TEXT),
        "determinism_evidence_reference": FieldSpec(KIND_TEXT),
        "max_length": FieldSpec(KIND_INT),
        "truncation_side": FieldSpec(KIND_ENUM, choices=("left", "right")),
    },
    "dependency_plan": {
        "pinned_versions": FieldSpec(KIND_STRING_MAP),
        "lockfile_reference": FieldSpec(KIND_TEXT),
    },
    "compute_feasibility": {
        "device": FieldSpec(KIND_ENUM, choices=("cpu", "cuda")),
        "vram_gb": FieldSpec(KIND_NUMBER, required=False),
        "oom_recovery_plan": FieldSpec(KIND_TEXT),
    },
    "artifact_privacy_controls": {
        "gitignore_patterns": FieldSpec(KIND_TEXT_LIST),
        "reports_pii_policy": FieldSpec(KIND_TEXT),
        "verified_against_repository": FieldSpec(KIND_BOOL),
    },
    "owner_approval": {
        "approver": FieldSpec(KIND_TEXT),
        "approval_date": FieldSpec(KIND_TEXT),
        "scope_statement": FieldSpec(KIND_TEXT),
    },
}


@dataclass(frozen=True)
class CheckOutcome:
    """Result of one category check: verdict, one-line reason, and fix hint."""

    verdict: str
    reason: str
    action: str = ""
    warning: str = ""


@dataclass(frozen=True)
class CategoryResult:
    category: str
    verdict: str
    reason: str
    action: str = ""
    warning: str = ""


@dataclass(frozen=True)
class GateReport:
    categories: tuple[CategoryResult, ...]
    overall: str
    next_actions: tuple[str, ...]
    input_error: str = ""


CheckFn = Any  # Callable[[dict[str, Any], dict[str, Any]], CheckOutcome]

CHECKS: dict[str, CheckFn] = {}  # populated by the @check decorator below


def check(category: str) -> Any:
    """Register a category check function in the data-driven CHECKS mapping."""

    def register(fn: CheckFn) -> CheckFn:
        CHECKS[category] = fn
        return fn

    return register


def _clip(value: Any) -> str:
    """Render a value for output; never emit more than MAX_PRINTED_VALUE chars."""
    text = value if isinstance(value, str) else str(value)
    if len(text) <= MAX_PRINTED_VALUE:
        return text
    keep = MAX_PRINTED_VALUE - len(TRUNCATION_MARKER)
    return text[:keep] + TRUNCATION_MARKER


def _pass(reason: str, warning: str = "") -> CheckOutcome:
    return CheckOutcome(PASS, _clip(reason), warning=_clip(warning))


def _blocked(reason: str, action: str) -> CheckOutcome:
    return CheckOutcome(BLOCKED, _clip(reason), action=_clip(action))


def _unknown(reason: str, action: str) -> CheckOutcome:
    return CheckOutcome(UNKNOWN, _clip(reason), action=_clip(action))


def _type_problem(name: str, value: Any, spec: FieldSpec) -> str | None:
    """Return a hard-violation message, or None when the value is well formed."""
    if spec.kind in (KIND_TEXT, KIND_ENUM):
        if not isinstance(value, str):
            return f"field '{name}' must be a string"
        if spec.kind == KIND_ENUM and value not in spec.choices:
            expected = ", ".join(spec.choices)
            return f"field '{name}' must be one of: {expected} (got '{_clip(value)}')"
        return None
    if spec.kind == KIND_INT:
        if isinstance(value, bool) or not isinstance(value, int):
            return f"field '{name}' must be an integer"
        return None
    if spec.kind == KIND_NUMBER:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"field '{name}' must be a number"
        if isinstance(value, float) and not math.isfinite(value):
            return f"field '{name}' must be a finite number"
        return None
    if spec.kind == KIND_BOOL:
        if not isinstance(value, bool):
            return f"field '{name}' must be a boolean"
        return None
    if spec.kind == KIND_TEXT_LIST:
        if not isinstance(value, list):
            return f"field '{name}' must be a list of strings"
        if any(not isinstance(item, str) or not item.strip() for item in value):
            return f"field '{name}' must contain only non-empty strings"
        return None
    if spec.kind == KIND_STRING_MAP:
        if not isinstance(value, dict):
            return f"field '{name}' must be an object mapping names to strings"
        if any(
            not isinstance(key, str) or not isinstance(item, str) or not item.strip()
            for key, item in value.items()
        ):
            return f"field '{name}' must map names to non-empty strings"
        return None
    if spec.kind == KIND_CLASS_MAPPING:
        well_formed = (
            isinstance(value, list)
            and len(value) == 3
            and all(isinstance(item, str) and item in CANONICAL_CLASSES for item in value)
            and len(set(value)) == 3
        )
        if not well_formed:
            expected = "/".join(CANONICAL_CLASSES)
            return (
                f"field '{name}' must list the three canonical classes "
                f"({expected}) exactly once, in raw 0/1/2 order"
            )
        return None
    return f"field '{name}' has an unsupported field spec kind '{spec.kind}'"


def _collect_field_issues(
    fields: dict[str, Any], specs: dict[str, FieldSpec]
) -> tuple[list[str], list[str]]:
    """Validate one category against its field specs.

    Returns (hard violations, missing evidence). Hard violations (unknown
    field, wrong type, bad enum) fail closed; missing or empty required
    fields are treated as insufficient evidence, not as contradictions.
    """
    blocked = [f"unknown field '{_clip(name)}'" for name in sorted(fields) if name not in specs]
    missing: list[str] = []
    for name, spec in specs.items():
        if name not in fields:
            if spec.required:
                missing.append(name)
            continue
        value = fields[name]
        if spec.kind == KIND_TEXT and isinstance(value, str) and _is_placeholder(value):
            missing.append(f"{name} (placeholder '{PLACEHOLDER_PREFIX}...' value is not evidence)")
            continue
        if spec.kind in (KIND_TEXT, KIND_ENUM) and isinstance(value, str) and not value.strip():
            if spec.required:
                missing.append(f"{name} (empty)")
            continue
        problem = _type_problem(name, value, spec)
        if problem is not None:
            blocked.append(problem)
    return blocked, missing


# Per-category actions used when structural validation fails, so that even a
# schema error points the owner at the concrete evidence to provide.
STRUCTURAL_ACTIONS: dict[str, str] = {
    "dataset_provenance": (
        "dataset_provenance: self-declared checkboxes (e.g. rights_confirmed, "
        "license_status) are not evidence - record dataset_id, dataset_version, "
        "dataset_adapter, synthetic_fixture, access_condition_class, "
        "license_statement and access_evidence_reference"
    ),
    "label_mapping_verification": (
        "label_mapping_verification: record an adapter-appropriate verification "
        "method and concrete mapping evidence"
    ),
    "split_definition": (
        "split_definition: record split_policy, train/validation/test sources, "
        "positive sizes, fingerprint_algorithm and 64-hex split fingerprints"
    ),
    "leakage_preflight": (
        "leakage_preflight: record completed non-synthetic preflight evidence "
        "matching evaluation_preflight.json (sha256 and 3 split fingerprints)"
    ),
    "phase1_integrity": (
        "phase1_integrity: record phase1_head_sha, real baseline artifact scope "
        "and fingerprints, and test_excluded_from_fit_and_selection"
    ),
}


def _evaluate_fields(fields: dict[str, Any], category: str) -> CheckOutcome | None:
    """Run the structural field validation; None means structurally valid."""
    blocked, missing = _collect_field_issues(fields, FIELD_SPECS[category])
    if blocked:
        action = STRUCTURAL_ACTIONS.get(
            category, f"{category}: correct the invalid field values listed in the reason"
        )
        return _blocked("; ".join(blocked), action)
    if missing:
        return _unknown(
            f"evidence missing: {', '.join(missing)}",
            f"{category}: supply the missing evidence fields",
        )
    return None


def _looks_synthetic_dataset_id(dataset_id: str) -> bool:
    """Return True when a structured dataset_id identifies a synthetic fixture."""
    normalized = dataset_id.strip().lower()
    tokens = set(re.split(r"[._\-]+", normalized))
    return bool(tokens & {"synthetic", "fixture", "mock", "dummy"}) or "synthetic" in normalized


def _is_valid_source_url(url: str) -> bool:
    """Return True when source_url is an explicit http(s) URL with a network location."""
    parsed = urlsplit(url.strip())
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _get_declared_adapter(evidence: dict[str, Any]) -> str | None:
    """Return dataset_provenance.dataset_adapter if declared and valid, else None."""
    provenance = evidence.get("dataset_provenance")
    if not isinstance(provenance, dict):
        return None
    adapter = provenance.get("dataset_adapter")
    return adapter if isinstance(adapter, str) and adapter in DATASET_ADAPTERS else None


def _check_reference_text(
    value: Any, *, field_name: str, missing_msg: str, action: str
) -> CheckOutcome | None:
    """Validate that an evidence reference string is non-empty and not a bare status."""
    if not isinstance(value, str) or not value.strip():
        return _unknown(missing_msg, action)
    if _is_non_evidence_statement(value):
        return _unknown(
            f"'{field_name}' ('{_clip(value.strip())}') is a self-declared status, "
            "not an evidence reference",
            action,
        )
    return None


def _extract_split_fingerprints(evidence: dict[str, Any]) -> dict[str, str] | None:
    """Extract lowercase 64-hex split fingerprints from split_definition if complete."""
    split_section = evidence.get("split_definition")
    if not isinstance(split_section, dict):
        return None
    extracted: dict[str, str] = {}
    for split_name in ALL_SPLITS:
        raw = split_section.get(f"{split_name}_fingerprint")
        if not isinstance(raw, str) or not HEX64_RE.fullmatch(raw.strip()):
            return None
        extracted[split_name] = raw.strip().lower()
    if len(set(extracted.values())) != len(ALL_SPLITS):
        return None
    return extracted


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        timestamp = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return timestamp if timestamp.tzinfo is not None else None


def _validate_split_fingerprint_map(
    mapping: dict[str, str],
    *,
    expected: dict[str, str] | None,
    required_splits: tuple[str, ...],
    forbidden_splits: tuple[str, ...],
    field_name: str,
    action: str,
) -> CheckOutcome | None:
    """Validate a split -> 64-hex sha256 fingerprint dictionary and cross-match splits."""
    unknown_keys = sorted(key for key in mapping if key not in ALL_SPLITS)
    if unknown_keys:
        clipped = ", ".join(_clip(key) for key in unknown_keys)
        return _blocked(f"'{field_name}' contains unknown split(s): {clipped}", action)
    for split_name in forbidden_splits:
        if split_name in mapping:
            return _blocked(
                f"'{field_name}' must not contain '{split_name}' "
                "(fit and selection must exclude the test split)",
                action,
            )
    absent = [split_name for split_name in required_splits if split_name not in mapping]
    if absent:
        return _unknown(
            f"'{field_name}' lacks fingerprint(s) for {', '.join(absent)}",
            action,
        )
    for split_name in required_splits:
        fp = mapping[split_name].strip()
        if not HEX64_RE.fullmatch(fp):
            return _blocked(
                f"'{field_name}.{split_name}' must be a 64-character hexadecimal "
                "sha256 fingerprint",
                action,
            )
        if expected is not None and fp.lower() != expected[split_name]:
            return _blocked(
                f"'{field_name}.{split_name}' does not match "
                f"split_definition.{split_name}_fingerprint",
                action,
            )
    return None


@check("dataset_provenance")
def _check_dataset_provenance(fields: dict[str, Any], _evidence: dict[str, Any]) -> CheckOutcome:
    category = "dataset_provenance"
    early = _evaluate_fields(fields, category)
    if early is not None:
        return early
    if fields["synthetic_fixture"] is True:
        return _blocked(
            "'synthetic_fixture' is true - synthetic fixtures never pass Phase 2 readiness",
            f"{category}: use real approved data and set 'synthetic_fixture' to false",
        )
    dataset_id = fields["dataset_id"].strip()
    if not IDENTIFIER_RE.fullmatch(dataset_id):
        return _blocked(
            f"'dataset_id' ('{_clip(dataset_id)}') contains disallowed characters",
            f"{category}: use an alphanumeric identifier matching {IDENTIFIER_RE.pattern}",
        )
    if _looks_synthetic_dataset_id(dataset_id):
        return _blocked(
            f"'dataset_id' ('{_clip(dataset_id)}') identifies a synthetic/fixture dataset",
            f"{category}: replace the synthetic fixture with an approved real dataset",
        )
    if _is_non_evidence_statement(fields["dataset_version"]):
        return _unknown(
            f"'dataset_version' ('{_clip(fields['dataset_version'].strip())}') is not a version",
            f"{category}: record the concrete dataset version or snapshot identifier",
        )
    adapter = fields["dataset_adapter"]
    access = fields["access_condition_class"]
    allowed_access = ADAPTER_ALLOWED_ACCESS[adapter]
    if access not in allowed_access:
        allowed_str = ", ".join(sorted(allowed_access))
        return _blocked(
            f"adapter '{adapter}' is incompatible with access_condition_class "
            f"'{access}' (allowed: {allowed_str})",
            f"{category}: set 'access_condition_class' to one of {allowed_str} for '{adapter}'",
        )
    source_url = fields.get("source_url", "")
    if access in EXTERNAL_ACCESS_CLASSES:
        if not isinstance(source_url, str) or not source_url.strip():
            return _unknown(
                f"access condition '{access}' requires 'source_url'",
                f"{category}: record the official dataset http(s) URL in 'source_url'",
            )
        if not _is_valid_source_url(source_url):
            return _blocked(
                f"'source_url' ('{_clip(source_url.strip())}') must be an http(s) URL",
                f"{category}: record a valid http(s) URL for the authoritative dataset source",
            )
    elif isinstance(source_url, str) and source_url.strip():
        if _is_non_evidence_statement(source_url):
            return _unknown(
                f"'source_url' ('{_clip(source_url.strip())}') is a bare status word",
                f"{category}: record an actual URL/locator or leave 'source_url' empty",
            )
    statement = fields["license_statement"].strip()
    if _is_non_evidence_statement(statement):
        return _unknown(
            f"'license_statement' ('{_clip(statement)}') does not name a license or terms - "
            "a self-declared status is not evidence",
            f"{category}: record the concrete license or ownership terms from the source",
        )
    access_ref_err = _check_reference_text(
        fields["access_evidence_reference"],
        field_name="access_evidence_reference",
        missing_msg=f"access condition '{access}' requires 'access_evidence_reference'",
        action=f"{category}: record the route-specific evidence reference for '{access}'",
    )
    if access_ref_err is not None:
        return access_ref_err
    return _pass(f"provenance complete (self-reported, unverified): {adapter} via {access}")


@check("label_mapping_verification")
def _check_label_mapping_verification(
    fields: dict[str, Any], evidence: dict[str, Any]
) -> CheckOutcome:
    category = "label_mapping_verification"
    early = _evaluate_fields(fields, category)
    if early is not None:
        return early
    adapter = _get_declared_adapter(evidence)
    method = fields["method"]
    if adapter is not None and method not in ADAPTER_ALLOWED_LABEL_METHODS[adapter]:
        expected = ", ".join(ADAPTER_ALLOWED_LABEL_METHODS[adapter])
        return _blocked(
            f"label mapping method '{method}' is incompatible with dataset_adapter "
            f"'{adapter}' (expected: {expected})",
            f"{category}: use an applicable verification method for '{adapter}' ({expected})",
        )
    has_toxic_fields = (
        bool(str(fields.get("toxic_policy", "")).strip())
        or bool(str(fields.get("toxic_justification", "")).strip())
        or fields.get("toxic_excluded_count") is not None
    )
    if method != "neu_esc_named_labels" and has_toxic_fields:
        return _blocked(
            f"toxic_* fields apply only to 'neu_esc_named_labels' (got method '{method}')",
            f"{category}: remove toxic_* fields when using method '{method}'",
        )
    if method == "human_approved_canonical":
        ref_err = _check_reference_text(
            fields.get("verification_reference"),
            field_name="verification_reference",
            missing_msg="method 'human_approved_canonical' requires 'verification_reference'",
            action=f"{category}: reference the human review/approval log for canonical labels",
        )
        if ref_err is not None:
            return ref_err
        return _pass("human-approved canonical labels verified (self-reported)")
    if method == "ordinal_five_to_three":
        ref_err = _check_reference_text(
            fields.get("verification_reference"),
            field_name="verification_reference",
            missing_msg="method 'ordinal_five_to_three' requires 'verification_reference'",
            action=f"{category}: reference the 5-level ordinal collapse specification",
        )
        if ref_err is not None:
            return ref_err
        return _pass("ordinal 5-level to 3-class mapping verified (self-reported)")
    if method == "neu_esc_named_labels":
        ref_err = _check_reference_text(
            fields.get("verification_reference"),
            field_name="verification_reference",
            missing_msg="method 'neu_esc_named_labels' requires 'verification_reference'",
            action=f"{category}: reference the NEU-ESC label verification record",
        )
        if ref_err is not None:
            return ref_err
        toxic_policy = str(fields.get("toxic_policy", "")).strip()
        if not toxic_policy:
            return _unknown(
                "method 'neu_esc_named_labels' requires 'toxic_policy' ('exclude' for training)",
                f"{category}: set toxic_policy='exclude' with justification and excluded count",
            )
        if toxic_policy != "exclude":
            return _blocked(
                f"NEU-ESC 'toxic_policy' '{toxic_policy}' cannot pass Phase 2 readiness - "
                "'exclude' with justification and counts is required",
                f"{category}: audit and exclude Toxic rows (toxic_policy='exclude') "
                "before training",
            )
        just_err = _check_reference_text(
            fields.get("toxic_justification"),
            field_name="toxic_justification",
            missing_msg="toxic_policy 'exclude' requires 'toxic_justification'",
            action=f"{category}: record written justification for excluding Toxic rows",
        )
        if just_err is not None:
            return just_err
        excluded_count = fields.get("toxic_excluded_count")
        if excluded_count is None:
            return _unknown(
                "toxic_policy 'exclude' requires integer 'toxic_excluded_count' (>= 0)",
                f"{category}: record the audited count of excluded Toxic rows",
            )
        if excluded_count < 0:
            return _blocked(
                f"'toxic_excluded_count' must be non-negative (got {excluded_count})",
                f"{category}: record a non-negative integer count of excluded Toxic rows",
            )
        return _pass(f"NEU-ESC named labels (self-reported); toxic excluded count={excluded_count}")
    # method in ("blind_audit", "author_confirmation"): raw 0/1/2 numeric mapping
    mapping = fields.get("resulting_mapping")
    if mapping is None:
        return _unknown(
            f"method '{method}' requires 'resulting_mapping' in raw 0/1/2 order",
            f"{category}: record the canonical 3-class permutation for raw 0/1/2",
        )
    pinned = "raw order pinned as 0={}, 1={}, 2={}".format(*mapping)
    sample = fields.get("audit_sample_size")
    rate = fields.get("agreement_rate")
    if sample is not None and sample <= 0:
        return _blocked(
            f"'audit_sample_size' {sample} must be a positive integer",
            f"{category}: record a positive audit sample size",
        )
    if rate is not None and not 0.0 <= rate <= 1.0:
        return _blocked(
            f"'agreement_rate' {_clip(rate)} is outside the 0-1 range",
            f"{category}: record the real agreement rate as a number between 0 and 1",
        )
    if method == "blind_audit":
        ref_err = _check_reference_text(
            fields.get("verification_reference"),
            field_name="verification_reference",
            missing_msg="method 'blind_audit' requires 'verification_reference'",
            action=f"{category}: reference the completed blind-audit file or log",
        )
        if ref_err is not None:
            return ref_err
        if sample is None or rate is None:
            return _unknown(
                "method 'blind_audit' requires 'audit_sample_size' and 'agreement_rate'",
                f"{category}: record both audit_sample_size and agreement_rate",
            )
        if sample < MIN_BLIND_AUDIT_SAMPLE:
            return _blocked(
                f"blind audit sample size {sample} is below the required "
                f"{MIN_BLIND_AUDIT_SAMPLE} lines",
                f"{category}: audit at least {MIN_BLIND_AUDIT_SAMPLE} lines before "
                "trusting the mapping",
            )
        if rate < LABEL_AGREEMENT_THRESHOLD:
            return _blocked(
                f"blind audit agreement rate {rate} is below the required "
                f"{LABEL_AGREEMENT_THRESHOLD}",
                f"{category}: extend or improve the audit until agreement >= "
                f"{LABEL_AGREEMENT_THRESHOLD}, or obtain author confirmation with a quote",
            )
        return _pass(
            f"blind audit ({sample} lines, rate {rate} >= {LABEL_AGREEMENT_THRESHOLD}; "
            f"self-reported): {pinned}"
        )
    # method == "author_confirmation": a quote reference is mandatory.
    quote = fields.get("quote_reference", "")
    if not isinstance(quote, str) or not quote.strip() or _is_non_evidence_statement(quote):
        return _blocked(
            "method 'author_confirmation' requires 'quote_reference' (the author's "
            "quoted statement and where it was given)",
            f"{category}: record the author's quoted label-mapping statement with a reference",
        )
    return _pass(f"author confirmation with quote reference; {pinned} (self-reported)")


@check("split_definition")
def _check_split_definition(fields: dict[str, Any], _evidence: dict[str, Any]) -> CheckOutcome:
    category = "split_definition"
    early = _evaluate_fields(fields, category)
    if early is not None:
        return early
    for name in ("train_size", "validation_size", "test_size"):
        if fields[name] <= 0:
            return _blocked(
                f"'{name}' must be a positive integer (got {fields[name]})",
                f"{category}: record the expected size of each split as a positive integer",
            )
    normalized_fps: dict[str, str] = {}
    for split_name in ALL_SPLITS:
        field_name = f"{split_name}_fingerprint"
        raw_fp = fields[field_name].strip()
        if not HEX64_RE.fullmatch(raw_fp):
            return _blocked(
                f"'{field_name}' must be a 64-character hexadecimal sha256 fingerprint",
                f"{category}: record the 64-hex sha256 fingerprint for '{split_name}'",
            )
        normalized_fps[split_name] = raw_fp.lower()
    if len(set(normalized_fps.values())) != len(ALL_SPLITS):
        return _blocked(
            "split fingerprints must be pairwise distinct across train, validation and test",
            f"{category}: verify splits do not collapse or duplicate across train/val/test",
        )
    warning = ""
    algorithm = fields["fingerprint_algorithm"].strip()
    if algorithm.lower() != "sha256":
        warning = (
            f"fingerprint algorithm '{_clip(algorithm)}' differs from the project standard sha256"
        )
    policy = fields["split_policy"]
    return _pass(
        f"splits ({policy}), sizes and distinct fingerprints recorded (self-reported)",
        warning,
    )


@check("leakage_preflight")
def _check_leakage_preflight(fields: dict[str, Any], evidence: dict[str, Any]) -> CheckOutcome:
    category = "leakage_preflight"
    early = _evaluate_fields(fields, category)
    if early is not None:
        return early
    if fields["exact_match_planned"] is False:
        return _blocked(
            "'exact_match_planned' is false - the exact-match preflight is mandatory (risk R-06)",
            f"{category}: plan the exact-match cross-split check before training",
        )
    threshold = fields["near_duplicate_threshold"]
    if not 0.0 <= threshold <= 1.0:
        return _blocked(
            f"'near_duplicate_threshold' {_clip(threshold)} is outside the 0-1 range",
            f"{category}: record a near-duplicate similarity threshold between 0 and 1",
        )
    if fields["preflight_synthetic_fixture"] is True:
        return _blocked(
            "'preflight_synthetic_fixture' is true - synthetic preflight reports cannot pass",
            f"{category}: run 'vfi evaluate --preflight-only' on the real dataset splits",
        )
    if fields["exact_match_executed"] and fields["preflight_passed"] is False:
        return _blocked(
            "'preflight_passed' is false - leakage preflight failed",
            f"{category}: resolve all leakage/contradiction findings until preflight passes",
        )
    # Planning is not readiness: each check family must have been EXECUTED
    # with a recorded report reference before training can start.
    for family in ("exact_match", "near_duplicate"):
        label = family.replace("_", "-")
        if fields[f"{family}_executed"] is False:
            return _unknown(
                f"the {label} preflight is planned but not executed - "
                "intentions are not execution evidence",
                f"{category}: run the {label} preflight and record its "
                "report reference before declaring training readiness",
            )
        ref_err = _check_reference_text(
            fields.get(f"{family}_report_reference"),
            field_name=f"{family}_report_reference",
            missing_msg=f"the {label} preflight is marked executed but its "
            "'report reference' is missing",
            action=f"{category}: record where the executed {label} report lives",
        )
        if ref_err is not None:
            return ref_err
    contradictions = fields["preflight_contradictions_count"]
    if contradictions < 0:
        return _blocked(
            f"'preflight_contradictions_count' must not be negative (got {contradictions})",
            f"{category}: record the non-negative count of cross-split contradictions",
        )
    if contradictions > 0:
        return _blocked(
            f"{contradictions} cross-split label contradiction(s) reported by preflight",
            f"{category}: resolve every cross-split label contradiction before training",
        )
    unresolved = fields["unresolved_conflicts"]
    if unresolved < 0:
        return _blocked(
            f"'unresolved_conflicts' must not be negative (got {unresolved})",
            f"{category}: record the number of unresolved cross-split conflicts (0 or more)",
        )
    if unresolved > 0:
        return _blocked(
            f"{unresolved} unresolved cross-split conflict(s) recorded - leakage conflicts "
            "must be resolved before training (risk R-06)",
            f"{category}: resolve or annotate every cross-split conflict, then set "
            "'unresolved_conflicts' to 0",
        )
    preflight_sha = fields.get("preflight_report_sha256", "")
    if not isinstance(preflight_sha, str) or not preflight_sha.strip():
        return _unknown(
            "executed preflight requires 'preflight_report_sha256'",
            f"{category}: record the 64-hex sha256 hash of evaluation_preflight.json",
        )
    if not HEX64_RE.fullmatch(preflight_sha.strip()):
        return _blocked(
            "'preflight_report_sha256' must be a 64-character hexadecimal sha256 hash",
            f"{category}: fix 'preflight_report_sha256' (64 hex characters)",
        )
    if _parse_timestamp(fields["preflight_created_at"]) is None:
        return _blocked(
            "'preflight_created_at' must be an ISO-8601 timestamp with a UTC offset",
            f"{category}: copy created_at from evaluation_preflight.json",
        )
    expected_splits = _extract_split_fingerprints(evidence)
    for field_name in ("preflight_split_fingerprints", "near_duplicate_split_fingerprints"):
        fp_err = _validate_split_fingerprint_map(
            fields[field_name],
            expected=expected_splits,
            required_splits=ALL_SPLITS,
            forbidden_splits=(),
            field_name=field_name,
            action=f"{category}: match all three audit fingerprints to split_definition",
        )
        if fp_err is not None:
            return fp_err
    method = _clip(fields["near_duplicate_method"])
    return _pass(
        f"leakage preflight passed (self-reported): '{method}', threshold {threshold}, 0 conflicts"
    )


@check("phase1_integrity")
def _check_phase1_integrity(fields: dict[str, Any], evidence: dict[str, Any]) -> CheckOutcome:
    category = "phase1_integrity"
    early = _evaluate_fields(fields, category)
    if early is not None:
        return early
    if not HEX40_RE.fullmatch(fields["phase1_head_sha"]):
        return _blocked(
            "'phase1_head_sha' must be a 40-character hexadecimal git SHA",
            f"{category}: fix 'phase1_head_sha' - record the exact Phase 1 head "
            "commit SHA (40 hex characters)",
        )
    if fields["test_excluded_from_fit_and_selection"] is False:
        return _blocked(
            "'test_excluded_from_fit_and_selection' is false - "
            "fit and model selection must exclude test",
            f"{category}: investigate and enforce strict test exclusion from fit/selection",
        )
    scope = fields["baseline_result_scope"]
    if scope == "synthetic_fixture":
        return _blocked(
            "'baseline_result_scope' is 'synthetic_fixture' - "
            "baseline runs on synthetic fixtures cannot satisfy readiness",
            f"{category}: run the Phase 1 baseline on the real splits and record its scope",
        )
    ref_err = _check_reference_text(
        fields["baseline_evidence_reference"],
        field_name="baseline_evidence_reference",
        missing_msg="baseline evidence reference is missing",
        action=f"{category}: record the path to the baseline report artifact",
    )
    if ref_err is not None:
        return ref_err
    baseline_hash = fields.get("baseline_evidence_sha256", "")
    if not isinstance(baseline_hash, str) or not baseline_hash.strip():
        return _unknown(
            "baseline evidence referenced but 'baseline_evidence_sha256' absent - "
            "the reference is unconfirmed",
            f"{category}: record the sha256 hash of the referenced baseline metrics artifact",
        )
    if not HEX64_RE.fullmatch(baseline_hash.strip()):
        return _blocked(
            "'baseline_evidence_sha256' must be a 64-character hexadecimal sha256 hash",
            f"{category}: fix 'baseline_evidence_sha256' - record the artifact's sha256 "
            "(64 hex characters)",
        )
    trained_at = _parse_timestamp(fields["baseline_training_started_at"])
    if trained_at is None:
        return _blocked(
            "'baseline_training_started_at' must be an ISO-8601 timestamp with a UTC offset",
            f"{category}: copy training_started_at from training_report.json",
        )
    baseline_preflight_sha = fields["baseline_preflight_report_sha256"].strip()
    if not HEX64_RE.fullmatch(baseline_preflight_sha):
        return _blocked(
            "'baseline_preflight_report_sha256' must be a 64-character hexadecimal sha256 hash",
            f"{category}: record the digest of the pre-training evaluation_preflight.json",
        )
    preflight = evidence.get("leakage_preflight")
    preflight = preflight if isinstance(preflight, dict) else {}
    preflight_sha = preflight.get("preflight_report_sha256")
    checked_at = _parse_timestamp(preflight.get("preflight_created_at"))
    if (
        not isinstance(preflight_sha, str)
        or not HEX64_RE.fullmatch(preflight_sha.strip())
        or (checked_at is None)
    ):
        return _unknown(
            "a complete frozen preflight receipt is required to verify the baseline link",
            f"{category}: complete the leakage_preflight evidence",
        )
    if baseline_preflight_sha.lower() != preflight_sha.strip().lower() or checked_at > trained_at:
        return _blocked(
            "the baseline must reference the same frozen preflight from before training",
            f"{category}: compare the receipt digest and both timestamps against the saved reports",
        )
    expected_splits = _extract_split_fingerprints(evidence)
    fp_err = _validate_split_fingerprint_map(
        fields["baseline_split_fingerprints"],
        expected=expected_splits,
        required_splits=FIT_SPLITS,
        forbidden_splits=("test",),
        field_name="baseline_split_fingerprints",
        action=f"{category}: record matching train/validation fingerprints (never test)",
    )
    if fp_err is not None:
        return fp_err
    holdout_fp = fields.get("baseline_holdout_fingerprint", "")
    has_holdout = isinstance(holdout_fp, str) and bool(holdout_fp.strip())
    if scope == "declared_holdout":
        if not has_holdout:
            return _unknown(
                "'baseline_result_scope' is 'declared_holdout' but "
                "'baseline_holdout_fingerprint' is absent",
                f"{category}: record the 64-hex test fingerprint evaluated by vfi evaluate",
            )
        if not HEX64_RE.fullmatch(holdout_fp.strip()):
            return _blocked(
                "'baseline_holdout_fingerprint' must be a 64-character hexadecimal sha256 hash",
                f"{category}: fix 'baseline_holdout_fingerprint' (64 hex characters)",
            )
        if expected_splits is not None and holdout_fp.strip().lower() != expected_splits["test"]:
            return _blocked(
                "'baseline_holdout_fingerprint' does not match split_definition.test_fingerprint",
                f"{category}: re-run baseline evaluation on the declared holdout split",
            )
    elif has_holdout:
        return _blocked(
            "'baseline_holdout_fingerprint' must be empty when "
            "'baseline_result_scope' is 'train_validation'",
            f"{category}: either clear 'baseline_holdout_fingerprint' or use 'declared_holdout'",
        )
    return _pass(f"Phase 1 head SHA valid; {scope} baseline hashed; test excluded (self-reported)")


@check("model_checkpoint")
def _check_model_checkpoint(fields: dict[str, Any], _evidence: dict[str, Any]) -> CheckOutcome:
    early = _evaluate_fields(fields, "model_checkpoint")
    if early is not None:
        return early
    category = "model_checkpoint"
    if not HEX40_RE.fullmatch(fields["revision_sha"]):
        return _blocked(
            "'revision_sha' must be a 40-character hexadecimal revision SHA",
            f"{category}: fix 'revision_sha' - pin the exact checkpoint revision "
            "(40 hex characters)",
        )
    if fields["weights_format"] == "pickle_bin":
        return _blocked(
            "weights_format 'pickle_bin' is forbidden - untrusted pickle execution risk (R-01)",
            f"{category}: use a safetensors distribution of the checkpoint",
        )
    if fields["checksum_recorded"] is False:
        return _blocked(
            "'checksum_recorded' is false - a checksum must be recorded before any "
            "checkpoint is loaded",
            f"{category}: record the checkpoint checksum (algorithm plus hash) at download time",
        )
    checksum = fields.get("checksum_sha256", "")
    if not isinstance(checksum, str) or not checksum.strip():
        return _unknown(
            "'checksum_recorded' is true but 'checksum_sha256' is absent",
            f"{category}: record the checksum hash string itself",
        )
    if not re.fullmatch(r"[0-9a-fA-F]{64}", checksum.strip()):
        return _blocked(
            "'checksum_sha256' must be a 64-character hexadecimal hash",
            f"{category}: record the checkpoint artifact's actual sha256 (64 hex characters)",
        )
    checkpoint_id = _clip(fields["checkpoint_id"])
    return _pass(
        f"checkpoint (self-reported; hub not verified): '{checkpoint_id}' pinned, checksum recorded"
    )


@check("tokenization_contract")
def _check_tokenization_contract(fields: dict[str, Any], evidence: dict[str, Any]) -> CheckOutcome:
    early = _evaluate_fields(fields, "tokenization_contract")
    if early is not None:
        return early
    category = "tokenization_contract"
    max_length = fields["max_length"]
    if max_length <= 0:
        return _blocked(
            f"'max_length' must be a positive integer (got {max_length})",
            f"{category}: record the sequence-length policy as a positive integer",
        )
    warnings: list[str] = []
    checkpoint_evidence = evidence.get("model_checkpoint")
    checkpoint_id = ""
    if isinstance(checkpoint_evidence, dict):
        raw_id = checkpoint_evidence.get("checkpoint_id", "")
        if isinstance(raw_id, str):
            checkpoint_id = raw_id.strip().lower()
    # Warn when the length exceeds PhoBERT's maximum; without a checkpoint id
    # the plan defaults to PhoBERT, so the warning still applies.
    if max_length > PHOBERT_MAX_LENGTH and ("phobert" in checkpoint_id or not checkpoint_id):
        warnings.append(
            f"max_length {max_length} exceeds PhoBERT's documented maximum of {PHOBERT_MAX_LENGTH}"
        )
    if fields["truncation_side"] == "left":
        warnings.append("the M1 pilot contract specifies right-side truncation")
    segmenter = _clip(fields["segmenter_name"])
    version = _clip(fields["segmenter_version"])
    reason = (
        f"segmenter '{segmenter}' {version} with determinism evidence; "
        f"max_length {max_length}; {fields['truncation_side']} truncation"
    )
    return _pass(reason, warning="; ".join(warnings))


@check("dependency_plan")
def _check_dependency_plan(fields: dict[str, Any], _evidence: dict[str, Any]) -> CheckOutcome:
    early = _evaluate_fields(fields, "dependency_plan")
    if early is not None:
        return early
    category = "dependency_plan"
    pinned = fields["pinned_versions"]
    absent = [name for name in REQUIRED_PINNED_PACKAGES if name not in pinned]
    if absent:
        return _unknown(
            f"pinned_versions lacks entries for {', '.join(absent)}",
            f"{category}: pin exact versions for torch, transformers and the segmenter",
        )
    invalid = [
        f"{name}='{_clip(str(pinned[name]))}'"
        for name in REQUIRED_PINNED_PACKAGES
        if not SEMVER_RE.fullmatch(str(pinned[name]))
    ]
    if invalid:
        return _blocked(
            f"pinned_versions entries are not semver-looking: {', '.join(invalid)}",
            f"{category}: pin exact version strings such as '2.5.1' or '2.5.1+cu121'",
        )
    return _pass("torch/transformers/segmenter versions pinned and a lockfile reference recorded")


@check("compute_feasibility")
def _check_compute_feasibility(fields: dict[str, Any], _evidence: dict[str, Any]) -> CheckOutcome:
    early = _evaluate_fields(fields, "compute_feasibility")
    if early is not None:
        return early
    category = "compute_feasibility"
    device = fields["device"]
    if device == "cuda":
        vram = fields.get("vram_gb")
        if vram is None:
            return _unknown(
                "device 'cuda' requires 'vram_gb'",
                f"{category}: record the available VRAM in GB",
            )
        if vram <= 0:
            return _blocked(
                f"'vram_gb' must be a positive number (got {_clip(vram)})",
                f"{category}: record the real VRAM budget",
            )
        return _pass(f"device 'cuda' with {_clip(vram)} GB VRAM and an OOM recovery plan")
    return _pass("device 'cpu' with an OOM recovery plan")


@check("artifact_privacy_controls")
def _check_artifact_privacy_controls(
    fields: dict[str, Any], _evidence: dict[str, Any]
) -> CheckOutcome:
    early = _evaluate_fields(fields, "artifact_privacy_controls")
    if early is not None:
        return early
    category = "artifact_privacy_controls"
    patterns = fields["gitignore_patterns"]
    missing = [pattern for pattern in REQUIRED_GITIGNORE_PATTERNS if pattern not in patterns]
    if missing:
        return _blocked(
            f"gitignore_patterns missing required entries: {', '.join(missing)} "
            "(weight-commit risk R-08)",
            f"{category}: add the missing ignore patterns before the first checkpoint download",
        )
    if fields["verified_against_repository"] is False:
        return _unknown(
            "declared patterns are not verified against the repository "
            "('verified_against_repository' is false)",
            f"{category}: verify the declared patterns against the repository "
            "('git check-ignore ...') and set the flag true",
        )
    return _pass(
        "gitignore covers *.bin/*.pt; PII policy stated; patterns verified against "
        "the repository (self-reported)"
    )


@check("owner_approval")
def _check_owner_approval(fields: dict[str, Any], _evidence: dict[str, Any]) -> CheckOutcome:
    early = _evaluate_fields(fields, "owner_approval")
    if early is not None:
        return early
    category = "owner_approval"
    raw_date = fields["approval_date"].strip()
    try:
        approval_date = datetime.fromisoformat(raw_date)
    except ValueError:
        return _blocked(
            f"'approval_date' ('{_clip(raw_date)}') is not an ISO-8601 date/time",
            f"{category}: record the approval date in ISO-8601 format (e.g. 2026-11-07)",
        )
    if approval_date.date() > datetime.now(UTC).date():
        return _blocked(
            "'approval_date' is in the future",
            f"{category}: record the real approval date",
        )
    return _pass("owner approval recorded: approver, ISO-8601 date and scope statement")


def _overall_verdict(results: list[CategoryResult]) -> str:
    """PASS only if every category passes; owner approval can force BLOCKED."""
    by_category = {result.category: result.verdict for result in results}
    if by_category.get("owner_approval", BLOCKED) != PASS:
        return BLOCKED
    verdicts = {result.verdict for result in results}
    if BLOCKED in verdicts:
        return BLOCKED
    if UNKNOWN in verdicts:
        return UNKNOWN
    return PASS


def evaluate_evidence(payload: dict[str, Any]) -> GateReport:
    """Evaluate a schema-valid evidence payload into a GateReport."""
    results: list[CategoryResult] = []
    for category in CATEGORY_ORDER:
        if category not in payload:
            results.append(
                CategoryResult(
                    category=category,
                    verdict=BLOCKED,
                    reason="category missing from the evidence file",
                    action=f"{category}: add the missing category section "
                    "(see --schema for the template)",
                )
            )
            continue
        value = payload[category]
        if not isinstance(value, dict):
            results.append(
                CategoryResult(
                    category=category,
                    verdict=BLOCKED,
                    reason=f"category must be a JSON object (got {_json_type_name(value)})",
                    action=f"{category}: replace the value with an object of evidence fields",
                )
            )
            continue
        outcome = CHECKS[category](value, payload)
        results.append(
            CategoryResult(
                category=category,
                verdict=outcome.verdict,
                reason=outcome.reason,
                action=outcome.action,
                warning=outcome.warning,
            )
        )
    overall = _overall_verdict(results)
    actions = tuple(result.action for result in results if result.verdict != PASS and result.action)
    return GateReport(categories=tuple(results), overall=overall, next_actions=actions)


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _input_error_report(reason: str) -> GateReport:
    """Fail-closed report for input files that cannot be evaluated at all."""
    categories = tuple(
        CategoryResult(category=cat, verdict=BLOCKED, reason="not evaluated - input error")
        for cat in CATEGORY_ORDER
    )
    return GateReport(
        categories=categories,
        overall=BLOCKED,
        next_actions=(f"fix the evidence file: {reason}",),
        input_error=reason,
    )


def _report_to_dict(report: GateReport) -> dict[str, Any]:
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall": report.overall,
        "input_error": report.input_error or None,
        "categories": {
            result.category: {
                "verdict": result.verdict,
                "reason": result.reason,
                "warning": result.warning,
            }
            for result in report.categories
        },
        "next_actions": list(report.next_actions),
    }


def _render_human(report: GateReport) -> str:
    lines = ["VFI Phase 2 Readiness Gate (offline operational checklist)"]
    if report.input_error:
        lines.append(f"INPUT ERROR: {report.input_error}")
    lines.append("")
    for result in report.categories:
        row = f"  [{result.verdict.upper():<7}] {result.category}: {result.reason}"
        if result.warning:
            row += f" (warning: {result.warning})"
        lines.append(row)
    pass_count = sum(1 for result in report.categories if result.verdict == PASS)
    unknown_count = sum(1 for result in report.categories if result.verdict == UNKNOWN)
    blocked_count = sum(1 for result in report.categories if result.verdict == BLOCKED)
    lines.append("")
    lines.append(
        f"Overall: {report.overall.upper()} "
        f"({pass_count} pass, {unknown_count} unknown, {blocked_count} blocked)"
    )
    if report.next_actions:
        lines.append("Next actions:")
        lines.extend(
            f"  {index}. {action}" for index, action in enumerate(report.next_actions, start=1)
        )
    lines.append("")
    lines.append(
        "Reminder: operational checklist only - it does not validate legal rights, "
        "does not verify claims against the real world, and cannot detect untrue "
        "evidence statements."
    )
    return "\n".join(lines)


def _emit(report: GateReport, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(_report_to_dict(report), indent=2))
    else:
        print(_render_human(report))


def _fail_closed(reason: str, *, as_json: bool) -> int:
    _emit(_input_error_report(reason), as_json=as_json)
    return EXIT_USAGE


def build_template() -> dict[str, Any]:
    """An honest, INCOMPLETE onboarding template for the owner.

    This is a schema example, NOT an evidence package and NOT a passing
    example: every field that would claim a completed real-world action
    (a signed agreement, an executed audit, an owner approval, a pinned
    revision) is a ``TODO:`` placeholder or an explicit not-done flag, so
    feeding this template back through ``--input`` yields UNKNOWN/BLOCKED
    verdicts and a nonzero exit code. Passing synthetic fixtures exist only
    inside the test suite, clearly marked as fabricated.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_provenance": {
            "dataset_id": "TODO: record the dataset identifier (e.g. feedback_v1)",
            "dataset_version": "TODO: record the dataset version or snapshot identifier",
            "dataset_adapter": "custom_jsonl",
            "synthetic_fixture": False,
            "access_condition_class": "owner_collected_human_approved",
            "source_url": "",
            "license_statement": "TODO: quote the concrete license or ownership terms "
            "from the authoritative source (do not declare a bare status)",
            "access_evidence_reference": "TODO: record the route-specific evidence "
            "reference (consent/approval log, license file, click-accept date, or "
            "signed agreement)",
        },
        "label_mapping_verification": {
            "method": "human_approved_canonical",
            "verification_reference": "TODO: reference the mapping verification log or document",
        },
        "split_definition": {
            "split_policy": "physical",
            "train_source": "TODO: describe the train split source",
            "validation_source": "TODO: describe the validation split source",
            "test_source": "TODO: describe the test split source",
            "train_size": 0,
            "validation_size": 0,
            "test_size": 0,
            "fingerprint_algorithm": "sha256",
            "train_fingerprint": "TODO: record the 64-hex sha256 fingerprint of train",
            "validation_fingerprint": "TODO: record the 64-hex sha256 fingerprint of validation",
            "test_fingerprint": "TODO: record the 64-hex sha256 fingerprint of test",
        },
        "leakage_preflight": {
            "exact_match_planned": True,
            "exact_match_executed": False,
            "exact_match_report_reference": "",
            "preflight_passed": False,
            "preflight_report_sha256": "",
            "preflight_created_at": "",
            "preflight_synthetic_fixture": False,
            "preflight_contradictions_count": 0,
            "preflight_split_fingerprints": {},
            "near_duplicate_method": "TODO: choose and record the near-duplicate method",
            "near_duplicate_threshold": 0.9,
            "near_duplicate_executed": False,
            "near_duplicate_report_reference": "",
            "near_duplicate_split_fingerprints": {},
            "unresolved_conflicts": 0,
        },
        "phase1_integrity": {
            "phase1_head_sha": "TODO: record the exact 40-hex Phase 1 head commit SHA",
            "baseline_evidence_reference": "TODO: reference the produced baseline metrics "
            "artifact (only meaningful after a real run)",
            "baseline_evidence_sha256": "",
            "baseline_preflight_report_sha256": "",
            "baseline_training_started_at": "",
            "baseline_result_scope": "train_validation",
            "baseline_split_fingerprints": {},
            "baseline_holdout_fingerprint": "",
            "test_excluded_from_fit_and_selection": True,
        },
        "model_checkpoint": {
            "checkpoint_id": "TODO: record the checkpoint id you will actually use",
            "revision_sha": "TODO: pin the exact 40-hex revision SHA from the hub",
            "license_name": "TODO: quote the license from the model card",
            "weights_format": "safetensors",
            "checksum_recorded": False,
            "checksum_sha256": "",
        },
        "tokenization_contract": {
            "segmenter_name": "TODO: record the segmenter (e.g. underthesea) and version",
            "segmenter_version": "TODO: record the pinned segmenter version",
            "determinism_evidence_reference": "TODO: reference the segmentation determinism "
            "check (only meaningful once it has been run)",
            "max_length": 256,
            "truncation_side": "right",
        },
        "dependency_plan": {
            "pinned_versions": {
                "torch": "TODO",
                "transformers": "TODO",
                "segmenter": "TODO",
            },
            "lockfile_reference": "TODO: reference the lockfile once created",
        },
        "compute_feasibility": {
            "device": "cuda",
            "vram_gb": 0,
            "oom_recovery_plan": "TODO: record the pre-declared OOM recovery ladder",
        },
        "artifact_privacy_controls": {
            "gitignore_patterns": [
                "*.bin",
                "*.pt",
                "*.ckpt",
                "*.safetensors",
                "data/raw/",
                "logs/",
                ".env",
            ],
            "reports_pii_policy": "TODO: state the PII policy for committed reports "
            "(aggregates and line indices only; no raw feedback text)",
            "verified_against_repository": False,
        },
        "owner_approval": {
            "approver": "TODO: the owner fills this in personally at approval time",
            "approval_date": "TODO: ISO-8601 date of the real approval",
            "scope_statement": "TODO: state the approved scope (e.g. the M1 pilot)",
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="readiness_gate",
        description=(
            "Offline Phase 2 readiness gate: checks a self-reported evidence "
            "package for completeness and internal consistency. See the module "
            "docstring for the schema and its limitations."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        metavar="PATH",
        help="path to the owner-prepared evidence JSON file",
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        help=(
            "print an INCOMPLETE onboarding template (valid JSON) and exit: TODO "
            "placeholders yield UNKNOWN/BLOCKED verdicts - it is an example of the "
            "schema, NOT an example of passing evidence"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the machine-readable JSON summary instead of the human checklist",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the exit code (0 pass / 1 blocked / 2 usage).

    ``argv`` defaults to an empty argument list so the module can be driven
    in-process; the ``__main__`` wrapper passes ``sys.argv[1:]``.
    """
    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else [])
    if args.schema:
        print(json.dumps(build_template(), indent=2))
        return EXIT_PASS
    if args.input is None:
        return _fail_closed("no input given: pass --input PATH (or --schema)", as_json=args.json)
    try:
        # utf-8-sig tolerates a BOM, common in hand-edited Windows JSON files.
        raw = args.input.read_text(encoding="utf-8-sig")
    except OSError:
        return _fail_closed("cannot read the --input file", as_json=args.json)
    except UnicodeDecodeError:
        return _fail_closed("the --input file is not valid UTF-8 text", as_json=args.json)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _fail_closed(
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            as_json=args.json,
        )
    if not isinstance(payload, dict):
        return _fail_closed("the top-level JSON value must be an object", as_json=args.json)
    unknown_keys = sorted(
        str(key) for key in payload if key not in CATEGORY_ORDER and key != "schema_version"
    )
    if unknown_keys:
        return _fail_closed(
            f"unknown top-level key(s): {', '.join(_clip(key) for key in unknown_keys)}",
            as_json=args.json,
        )
    if "schema_version" not in payload:
        return _fail_closed(
            f"missing 'schema_version' (this gate requires schema {SCHEMA_VERSION})",
            as_json=args.json,
        )
    version = payload["schema_version"]
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        if not isinstance(version, bool) and version == 1:
            return _fail_closed(
                "legacy schema_version 1 is no longer supported - migrate dataset provenance, "
                "label mapping, split fingerprints, preflight, and phase1 fields to schema "
                f"{SCHEMA_VERSION} (see --schema)",
                as_json=args.json,
            )
        return _fail_closed(
            f"unsupported 'schema_version' (this gate implements schema {SCHEMA_VERSION})",
            as_json=args.json,
        )
    try:
        report = evaluate_evidence(payload)
    except Exception as exc:  # fail-closed belt: never crash with a traceback
        return _fail_closed(
            f"internal evaluation error ({type(exc).__name__}); treat the evidence as blocked",
            as_json=args.json,
        )
    _emit(report, as_json=args.json)
    if report.overall == PASS:
        return EXIT_PASS
    return EXIT_BLOCKED


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
