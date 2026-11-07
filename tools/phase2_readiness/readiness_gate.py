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
- Present-but-incomplete evidence (a required field missing or empty) yields
  UNKNOWN, never PASS.

USAGE
- ``readiness_gate.py --schema``
  Print a filled template evidence file (valid JSON; it doubles as a minimal
  PASSing example) to stdout, then exit 0. Takes precedence over ``--input``.
- ``readiness_gate.py --input PATH [--json]``
  Evaluate the evidence file. Human-readable checklist on stdout by default;
  ``--json`` prints the machine-readable summary (schema_version,
  generated_at_utc, per-category verdicts, overall, next_actions) instead.
- Exit codes: 0 = overall PASS; 1 = overall BLOCKED or UNKNOWN (anything not
  green); 2 = usage/schema error (no input, unreadable file, invalid JSON,
  unknown top-level key). Every parse/validation failure fails CLOSED: the
  verdict is BLOCKED with the reason, never a traceback.

SCHEMA (v1)
Top level: optional ``schema_version`` (must be 1) plus the 11 required
category objects listed below. Unknown top-level keys are rejected, and so are
unknown fields inside a category. Wrong types, out-of-range numbers, invalid
40-hex SHAs and non-ISO-8601 dates are BLOCKED; a required field that is
absent or empty yields UNKNOWN (evidence insufficient).

1. dataset_provenance: source_url (string), license_statement (string),
   access_condition_class (author_agreement_signed | gated_click_accept |
   open), agreement_reference (required when the class is
   author_agreement_signed).
2. label_mapping_verification: method (author_confirmation | blind_audit),
   audit_sample_size (positive int; >= 100 for blind_audit),
   agreement_rate (number in 0-1; blind_audit needs >= 0.9 to PASS),
   resulting_mapping (three distinct canonical class names drawn from
   negative/neutral/positive, in raw 0/1/2 order; either audit hypothesis is
   accepted - the audit decides), quote_reference (required when the method
   is author_confirmation; missing quote = BLOCKED).
3. split_definition: train_source, validation_source, test_source (strings),
   train_size, validation_size, test_size (positive ints),
   fingerprint_algorithm (string; warning when it is not sha256).
4. leakage_preflight: exact_match_planned (bool, must be true) and
   near_duplicate_method (string) + near_duplicate_threshold (number in 0-1).
   Both checks are required: exact match alone misses R-06 near-duplicates.
5. phase1_integrity: phase1_head_sha (40-hex), baseline_evidence_reference
   (string), test_set_untouched (bool, must be true).
6. model_checkpoint: checkpoint_id (string), revision_sha (40-hex),
   license_name (string), weights_format (safetensors passes; pickle_bin is
   BLOCKED per R-01), checksum_recorded (bool, must be true) plus
   checksum_sha256 (string; required when checksum_recorded is true).
7. tokenization_contract: segmenter_name, segmenter_version,
   determinism_evidence_reference (strings), max_length (positive int;
   warning when it exceeds PhoBERT's documented 256), truncation_side
   (left | right; warning when left, the pilot contract fixes right).
8. dependency_plan: pinned_versions (object with semver-looking torch /
   transformers / segmenter entries), lockfile_reference (string).
9. compute_feasibility: device (cpu | cuda), vram_gb (positive number,
   required when device is cuda), oom_recovery_plan (string).
10. artifact_privacy_controls: gitignore_patterns (list of strings; must
    contain ``*.bin`` and ``*.pt`` to PASS), reports_pii_policy (string).
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
  function registered in CHECKS, and a filled example to build_template().
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
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
LABEL_AGREEMENT_THRESHOLD = 0.9
MIN_BLIND_AUDIT_SAMPLE = 100
PHOBERT_MAX_LENGTH = 256
REQUIRED_GITIGNORE_PATTERNS = ("*.bin", "*.pt")
REQUIRED_PINNED_PACKAGES = ("torch", "transformers", "segmenter")
PLACEHOLDER_LICENSE_STATEMENTS = frozenset(
    {"unknown", "ok", "yes", "true", "none", "n/a", "na", "tbd"}
)

HEX40_RE = re.compile(r"[0-9a-fA-F]{40}")
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
        "source_url": FieldSpec(KIND_TEXT),
        "license_statement": FieldSpec(KIND_TEXT),
        "access_condition_class": FieldSpec(
            KIND_ENUM,
            choices=("author_agreement_signed", "gated_click_accept", "open"),
        ),
        "agreement_reference": FieldSpec(KIND_TEXT, required=False),
    },
    "label_mapping_verification": {
        "method": FieldSpec(KIND_ENUM, choices=("author_confirmation", "blind_audit")),
        "audit_sample_size": FieldSpec(KIND_INT),
        "agreement_rate": FieldSpec(KIND_NUMBER),
        "resulting_mapping": FieldSpec(KIND_CLASS_MAPPING),
        "quote_reference": FieldSpec(KIND_TEXT, required=False),
    },
    "split_definition": {
        "train_source": FieldSpec(KIND_TEXT),
        "validation_source": FieldSpec(KIND_TEXT),
        "test_source": FieldSpec(KIND_TEXT),
        "train_size": FieldSpec(KIND_INT),
        "validation_size": FieldSpec(KIND_INT),
        "test_size": FieldSpec(KIND_INT),
        "fingerprint_algorithm": FieldSpec(KIND_TEXT),
    },
    "leakage_preflight": {
        "exact_match_planned": FieldSpec(KIND_BOOL),
        "near_duplicate_method": FieldSpec(KIND_TEXT),
        "near_duplicate_threshold": FieldSpec(KIND_NUMBER),
    },
    "phase1_integrity": {
        "phase1_head_sha": FieldSpec(KIND_TEXT),
        "baseline_evidence_reference": FieldSpec(KIND_TEXT),
        "test_set_untouched": FieldSpec(KIND_BOOL),
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
        if spec.kind in (KIND_TEXT, KIND_ENUM) and isinstance(value, str) and not value.strip():
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
        "dataset_provenance: self-declared checkboxes are not evidence - record "
        "source_url, license_statement, access_condition_class and, for signed "
        "access, agreement_reference"
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


@check("dataset_provenance")
def _check_dataset_provenance(fields: dict[str, Any], _evidence: dict[str, Any]) -> CheckOutcome:
    early = _evaluate_fields(fields, "dataset_provenance")
    if early is not None:
        return early
    access = fields["access_condition_class"]
    statement = fields["license_statement"].strip()
    if statement.lower() in PLACEHOLDER_LICENSE_STATEMENTS:
        return _unknown(
            f"'license_statement' ('{_clip(statement)}') does not name a license - "
            "a self-declared status is not evidence",
            "dataset_provenance: record the concrete license name and terms from "
            "the authoritative source",
        )
    if access == "author_agreement_signed":
        reference = fields.get("agreement_reference", "")
        if not isinstance(reference, str) or not reference.strip():
            return _unknown(
                "access condition 'author_agreement_signed' requires "
                "'agreement_reference' (signed file or agreement email date)",
                "dataset_provenance: record the agreement reference (file path or "
                "the agreement email date)",
            )
    return _pass(
        f"provenance complete: source, license statement and access condition '{access}' recorded"
    )


@check("label_mapping_verification")
def _check_label_mapping_verification(
    fields: dict[str, Any], _evidence: dict[str, Any]
) -> CheckOutcome:
    early = _evaluate_fields(fields, "label_mapping_verification")
    if early is not None:
        return early
    category = "label_mapping_verification"
    method = fields["method"]
    sample = fields["audit_sample_size"]
    rate = fields["agreement_rate"]
    mapping = fields["resulting_mapping"]
    pinned = "raw order pinned as 0={}, 1={}, 2={}".format(*mapping)
    if not 0.0 <= rate <= 1.0:
        return _blocked(
            f"'agreement_rate' {_clip(rate)} is outside the 0-1 range",
            f"{category}: record the real agreement rate as a number between 0 and 1",
        )
    if sample <= 0:
        return _blocked(
            f"'audit_sample_size' {sample} must be a positive integer",
            f"{category}: record a positive audit sample size",
        )
    if method == "blind_audit":
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
            f"blind audit of {sample} lines with agreement rate {rate} "
            f"(>= {LABEL_AGREEMENT_THRESHOLD}); {pinned}"
        )
    # method == "author_confirmation": a quote reference is mandatory.
    quote = fields.get("quote_reference", "")
    if not isinstance(quote, str) or not quote.strip():
        return _blocked(
            "method 'author_confirmation' requires 'quote_reference' (the author's "
            "quoted statement and where it was given)",
            f"{category}: record the author's quoted label-mapping statement with a reference",
        )
    return _pass(f"author confirmation with quote reference; {pinned}")


@check("split_definition")
def _check_split_definition(fields: dict[str, Any], _evidence: dict[str, Any]) -> CheckOutcome:
    early = _evaluate_fields(fields, "split_definition")
    if early is not None:
        return early
    category = "split_definition"
    for name in ("train_size", "validation_size", "test_size"):
        if fields[name] <= 0:
            return _blocked(
                f"'{name}' must be a positive integer (got {fields[name]})",
                f"{category}: record the expected size of each split as a positive integer",
            )
    warning = ""
    algorithm = fields["fingerprint_algorithm"].strip()
    if algorithm.lower() != "sha256":
        warning = (
            f"fingerprint algorithm '{_clip(algorithm)}' differs from the project standard sha256"
        )
    return _pass(
        "train/validation/test descriptors, sizes and fingerprint algorithm recorded", warning
    )


@check("leakage_preflight")
def _check_leakage_preflight(fields: dict[str, Any], _evidence: dict[str, Any]) -> CheckOutcome:
    early = _evaluate_fields(fields, "leakage_preflight")
    if early is not None:
        return early
    category = "leakage_preflight"
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
    method = _clip(fields["near_duplicate_method"])
    return _pass(
        f"exact-match plus near-duplicate ('{method}', threshold {threshold}) preflights planned"
    )


@check("phase1_integrity")
def _check_phase1_integrity(fields: dict[str, Any], _evidence: dict[str, Any]) -> CheckOutcome:
    early = _evaluate_fields(fields, "phase1_integrity")
    if early is not None:
        return early
    category = "phase1_integrity"
    if not HEX40_RE.fullmatch(fields["phase1_head_sha"]):
        return _blocked(
            "'phase1_head_sha' must be a 40-character hexadecimal git SHA",
            f"{category}: fix 'phase1_head_sha' - record the exact Phase 1 head "
            "commit SHA (40 hex characters)",
        )
    if fields["test_set_untouched"] is False:
        return _blocked(
            "'test_set_untouched' is false - the Phase 1 test split must remain untouched",
            f"{category}: investigate the test-split violation before any Phase 2 work",
        )
    return _pass(
        "Phase 1 head SHA valid, baseline evidence referenced, test set declared untouched"
    )


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
    checkpoint_id = _clip(fields["checkpoint_id"])
    return _pass(
        f"safetensors checkpoint '{checkpoint_id}' pinned to a revision with a recorded checksum"
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
    return _pass("gitignore covers *.bin and *.pt and a reports PII policy is stated")


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
    """A filled, schema-valid evidence file that doubles as a PASS example."""
    return {
        "schema_version": 1,
        "dataset_provenance": {
            "source_url": "https://github.com/kietnv/VietnameseDatasets (UIT-VSFC, "
            "author delivery after signing the corpus user agreement)",
            "license_statement": "UIT-VSFC corpus user agreement signed; "
            "non-commercial research use permitted by the authors",
            "access_condition_class": "author_agreement_signed",
            "agreement_reference": "signed agreement stored locally; request email "
            "sent 2026-11-01 to kietnv@uit.edu.vn",
        },
        "label_mapping_verification": {
            "method": "blind_audit",
            "audit_sample_size": 135,
            "agreement_rate": 0.92,
            "resulting_mapping": ["negative", "neutral", "positive"],
        },
        "split_definition": {
            "train_source": "data/raw/uit_vsfc/train (author-delivered archive, "
            "line-paired sentences.txt/sentiments.txt)",
            "validation_source": "data/raw/uit_vsfc/validation (same delivery)",
            "test_source": "data/raw/uit_vsfc/test (same delivery)",
            "train_size": 11426,
            "validation_size": 1583,
            "test_size": 3166,
            "fingerprint_algorithm": "sha256",
        },
        "leakage_preflight": {
            "exact_match_planned": True,
            "near_duplicate_method": "tfidf_cosine",
            "near_duplicate_threshold": 0.9,
        },
        "phase1_integrity": {
            "phase1_head_sha": "95f7687735c785367fa0a6f9b1efbf0e284aac7f",
            "baseline_evidence_reference": "reports/metrics/phase1_baseline.json "
            "(sha256 recorded in the artifact index)",
            "test_set_untouched": True,
        },
        "model_checkpoint": {
            "checkpoint_id": "vinai/phobert-base",
            "revision_sha": "9a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b",
            "license_name": "MIT",
            "weights_format": "safetensors",
            "checksum_recorded": True,
            "checksum_sha256": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        },
        "tokenization_contract": {
            "segmenter_name": "py_vncorenlp",
            "segmenter_version": "0.1.0",
            "determinism_evidence_reference": "logs/tokenization/determinism_report.md "
            "(double-segmentation fingerprints equal)",
            "max_length": 256,
            "truncation_side": "right",
        },
        "dependency_plan": {
            "pinned_versions": {
                "torch": "2.5.1+cu121",
                "transformers": "4.46.2",
                "segmenter": "0.1.0",
            },
            "lockfile_reference": "requirements-phase2.txt (sha256 recorded in the Phase 2 branch)",
        },
        "compute_feasibility": {
            "device": "cuda",
            "vram_gb": 6,
            "oom_recovery_plan": "pre-declared ladder: halve micro-batch and double "
            "accumulation (8x4 -> 4x8 -> 2x16 -> 1x32)",
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
            "reports_pii_policy": "no raw student text in committed reports; "
            "aggregates and line indices only",
        },
        "owner_approval": {
            "approver": "VFI project owner",
            "approval_date": "2026-11-01",
            "scope_statement": "approved for the M1 pilot as scoped in "
            "docs/phase2/pilot-experiment-plan.md",
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
        help="print the filled template evidence file (valid JSON) and exit",
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
    version = payload.get("schema_version", SCHEMA_VERSION)
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        return _fail_closed(
            "unsupported 'schema_version' (this gate implements schema 1)", as_json=args.json
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
