"""Tests for the offline Phase 2 readiness gate (synthetic evidence only).

Two distinct synthetic artifacts are used here and they must never be
confused:

- ``build_template()`` (from the tool) is the user-facing ``--schema``
  onboarding example. It is deliberately INCOMPLETE: every field that would
  claim a completed real-world action (signed agreement, executed audit,
  owner approval...) is a ``TODO:`` placeholder, so feeding the template
  back through ``--input`` must NOT pass.
- ``_passing_fixture()`` (below) is a FABRICATED full-PASS evidence package
  that exists ONLY inside this test file to exercise the verdict logic. It
  is not an example for real use and must never move into the tool or any
  user-facing documentation.

Nothing here touches real datasets, weights, or the network. The tool must
stay a pure read-only checker, so one test also asserts that a run creates
no files.
"""

import getpass
import importlib.util
import io
import json
import math
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "phase2_readiness" / "readiness_gate.py"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("vfi_readiness_gate", TOOL_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load readiness gate module from {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Required before exec_module: the dataclasses machinery resolves string
    # annotations through sys.modules under the module's own name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


readiness_gate = _load_tool()

CATEGORY_NAMES = readiness_gate.CATEGORY_ORDER
TRUNCATION_MARKER = readiness_gate.TRUNCATION_MARKER
TRAIN_FP = "1" * 64
VAL_FP = "2" * 64
TEST_FP = "3" * 64
PREFLIGHT_SHA = "4" * 64


def _passing_fixture() -> dict[str, Any]:
    """FABRICATED full-PASS evidence package for verdict-logic tests only.

    Every "executed" flag and reference below is invented test input; no
    agreement, audit, or approval actually exists. This fixture must stay
    in this file (see the module docstring).
    """
    return {
        "schema_version": 2,
        "dataset_provenance": {
            "dataset_id": "uit_vsfc_v1",
            "dataset_version": "1.0.0",
            "dataset_adapter": "uit_vsfc",
            "synthetic_fixture": False,
            "access_condition_class": "author_agreement_signed",
            "source_url": "https://example.org/uit-vsfc",
            "license_statement": "fabricated academic usage agreement terms",
            "access_evidence_reference": "fabricated agreement email date 2026-11-01",
        },
        "label_mapping_verification": {
            "method": "blind_audit",
            "verification_reference": "fabricated reports/label_blind_audit.json",
            "audit_sample_size": 135,
            "agreement_rate": 0.92,
            "resulting_mapping": ["negative", "neutral", "positive"],
        },
        "split_definition": {
            "split_policy": "physical",
            "train_source": "fabricated train split descriptor",
            "validation_source": "fabricated validation split descriptor",
            "test_source": "fabricated test split descriptor",
            "train_size": 11426,
            "validation_size": 1583,
            "test_size": 3166,
            "fingerprint_algorithm": "sha256",
            "train_fingerprint": TRAIN_FP,
            "validation_fingerprint": VAL_FP,
            "test_fingerprint": TEST_FP,
        },
        "leakage_preflight": {
            "exact_match_planned": True,
            "exact_match_executed": True,
            "exact_match_report_reference": "fabricated reports/evaluation_preflight.json",
            "preflight_passed": True,
            "preflight_report_sha256": PREFLIGHT_SHA,
            "preflight_created_at": "2026-11-02T08:00:00+00:00",
            "preflight_synthetic_fixture": False,
            "preflight_contradictions_count": 0,
            "preflight_split_fingerprints": {
                "train": TRAIN_FP,
                "validation": VAL_FP,
                "test": TEST_FP,
            },
            "near_duplicate_method": "tfidf_cosine",
            "near_duplicate_threshold": 0.9,
            "near_duplicate_executed": True,
            "near_duplicate_report_reference": "fabricated reports/near_dup_preflight.json",
            "near_duplicate_split_fingerprints": {
                "train": TRAIN_FP,
                "validation": VAL_FP,
                "test": TEST_FP,
            },
            "unresolved_conflicts": 0,
        },
        "phase1_integrity": {
            "phase1_head_sha": "9" * 40,
            "baseline_evidence_reference": "fabricated reports/training_report.json",
            "baseline_evidence_sha256": "b" * 64,
            "baseline_preflight_report_sha256": PREFLIGHT_SHA,
            "baseline_training_started_at": "2026-11-02T08:30:00+00:00",
            "baseline_result_scope": "train_validation",
            "baseline_split_fingerprints": {
                "train": TRAIN_FP,
                "validation": VAL_FP,
            },
            "baseline_holdout_fingerprint": "",
            "test_excluded_from_fit_and_selection": True,
        },
        "model_checkpoint": {
            "checkpoint_id": "fixture-org/phobert-placeholder",
            "revision_sha": "a" * 40,
            "license_name": "MIT",
            "weights_format": "safetensors",
            "checksum_recorded": True,
            "checksum_sha256": "c" * 64,
        },
        "tokenization_contract": {
            "segmenter_name": "fabricated segmenter",
            "segmenter_version": "0.0.0",
            "determinism_evidence_reference": "fabricated determinism report",
            "max_length": 256,
            "truncation_side": "right",
        },
        "dependency_plan": {
            "pinned_versions": {"torch": "0.0.0", "transformers": "0.0.0", "segmenter": "0.0.0"},
            "lockfile_reference": "fabricated lockfile reference",
        },
        "compute_feasibility": {
            "device": "cuda",
            "vram_gb": 6,
            "oom_recovery_plan": "fabricated OOM recovery plan",
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
            "reports_pii_policy": "fabricated PII policy statement",
            "verified_against_repository": True,
        },
        "owner_approval": {
            "approver": "fabricated approver",
            "approval_date": "2026-01-01",
            "scope_statement": "fabricated scope statement",
        },
    }


def _owner_collected_fixture() -> dict[str, Any]:
    payload = _passing_fixture()
    payload["dataset_provenance"] = {
        "dataset_id": "owner_feedback_v1",
        "dataset_version": "2026-11-01",
        "dataset_adapter": "custom_jsonl",
        "synthetic_fixture": False,
        "access_condition_class": "owner_collected_human_approved",
        "source_url": "",
        "license_statement": "owner-collected course evaluations; IRB-exempt internal release",
        "access_evidence_reference": "data/manifests/owner_collection_log_2026-11-01.json",
    }
    payload["label_mapping_verification"] = {
        "method": "human_approved_canonical",
        "verification_reference": "data/manifests/human_review_signoff_v1.json",
    }
    return payload


def _licensed_public_fixture() -> dict[str, Any]:
    payload = _passing_fixture()
    payload["dataset_provenance"] = {
        "dataset_id": "five_level_public_v1",
        "dataset_version": "2.1.0",
        "dataset_adapter": "five_level_csv",
        "synthetic_fixture": False,
        "access_condition_class": "licensed_public",
        "source_url": "https://example.org/datasets/five-level-public",
        "license_statement": "CC-BY-4.0 permissive public attribution license",
        "access_evidence_reference": "licenses/five_level_public_CC-BY-4.0.txt",
    }
    payload["label_mapping_verification"] = {
        "method": "ordinal_five_to_three",
        "verification_reference": "docs/phase2/label-semantics-evidence.md#five-level-ordinal",
    }
    payload["split_definition"]["split_policy"] = "grouped_validation"
    payload["phase1_integrity"]["baseline_result_scope"] = "declared_holdout"
    payload["phase1_integrity"]["baseline_evidence_reference"] = "reports/eval_holdout.json"
    payload["phase1_integrity"]["baseline_holdout_fingerprint"] = TEST_FP
    return payload


def _neu_esc_gated_fixture() -> dict[str, Any]:
    payload = _passing_fixture()
    payload["dataset_provenance"] = {
        "dataset_id": "neu_esc_v2",
        "dataset_version": "2024-09",
        "dataset_adapter": "neu_esc",
        "synthetic_fixture": False,
        "access_condition_class": "gated_click_accept",
        "source_url": "https://huggingface.co/datasets/example/neu-esc",
        "license_statement": "Hugging Face gated click-accept research license terms",
        "access_evidence_reference": "gated hub terms accepted 2026-11-10 under owner account",
    }
    payload["label_mapping_verification"] = {
        "method": "neu_esc_named_labels",
        "verification_reference": "docs/phase2/label-semantics-evidence.md#neu-esc",
        "toxic_policy": "exclude",
        "toxic_justification": "Toxic is orthogonal moderation label; out of 3-class sentiment",
        "toxic_excluded_count": 42,
    }
    return payload


def _with(category: str, **overrides: Any) -> dict[str, Any]:
    payload = _passing_fixture()
    payload[category].update(overrides)
    return payload


def _without_category(category: str) -> dict[str, Any]:
    payload = _passing_fixture()
    del payload[category]
    return payload


def _without_field(category: str, field_name: str) -> dict[str, Any]:
    payload = _passing_fixture()
    del payload[category][field_name]
    return payload


def _run_cli(*args: str) -> tuple[int, str]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = readiness_gate.main(list(args))
    return code, buffer.getvalue()


def _run(tmp_path: Path, payload: Any, *flags: str) -> tuple[int, str]:
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return _run_cli("--input", str(path), *flags)


def _summary(out: str) -> dict[str, Any]:
    return json.loads(out)


# ---------------------------------------------------------------------------
# Full pass, output shape, and template round trip
# ---------------------------------------------------------------------------


def test_full_pass_fixture_exits_zero_with_complete_json(tmp_path: Path) -> None:
    code, out = _run(tmp_path, _passing_fixture(), "--json")
    assert code == 0
    summary = _summary(out)
    assert set(summary) == {
        "schema_version",
        "generated_at_utc",
        "overall",
        "input_error",
        "categories",
        "next_actions",
    }
    assert summary["schema_version"] == 1
    assert summary["overall"] == "pass"
    assert summary["input_error"] is None
    assert summary["next_actions"] == []
    assert set(summary["categories"]) == set(CATEGORY_NAMES)
    assert all(entry["verdict"] == "pass" for entry in summary["categories"].values())
    assert all(
        set(entry) == {"verdict", "reason", "warning"} for entry in summary["categories"].values()
    )
    # generated_at_utc is an ISO-8601 UTC timestamp.
    datetime.strptime(summary["generated_at_utc"], "%Y-%m-%dT%H:%M:%SZ")


def test_human_output_lists_every_category_and_limitations(tmp_path: Path) -> None:
    code, out = _run(tmp_path, _passing_fixture())
    assert code == 0
    for name in CATEGORY_NAMES:
        assert name in out
    assert "Overall: PASS" in out
    assert "does not validate legal rights" in out
    assert "cannot detect untrue evidence" in out


def test_schema_template_round_trips_to_non_pass(tmp_path: Path) -> None:
    """P0 regression: the --schema onboarding example must NOT pass the gate.

    The template is an honest, incomplete evidence package. Feeding it back
    through --input must exit nonzero with an overall verdict that is not
    PASS; otherwise the template would imply completed actions (signed
    agreement, executed audit, owner approval) that never happened.
    """
    code, out = _run_cli("--schema")
    assert code == 0  # the --schema command itself succeeds
    template = json.loads(out)  # must parse as JSON
    assert template["schema_version"] == 2
    assert set(CATEGORY_NAMES).issubset(template)
    code, out = _run(tmp_path, template, "--json")  # round trip
    assert code != 0
    assert _summary(out)["overall"] != "pass"


def test_template_contains_no_fictional_executed_claims() -> None:
    """The template must not imply signing, auditing, or approving happened."""
    template = json.loads(json.dumps(readiness_gate.build_template()))
    rendered = json.dumps(template)
    for fictional_phrase in (
        "user agreement signed",
        "agreement signed",
        "request email sent",
        "9a1b2c3d",  # the removed invented revision SHA
        "9f86d081",  # the removed invented checksum
        '"checksum_recorded": true',
        '"verified_against_repository": true',
        '"preflight_passed": true',
    ):
        assert fictional_phrase not in rendered, fictional_phrase
    assert template["model_checkpoint"]["checksum_recorded"] is False
    assert template["artifact_privacy_controls"]["verified_against_repository"] is False
    assert template["leakage_preflight"]["preflight_passed"] is False
    # The Phase 1 head and split fingerprints must be placeholders, not real hashes.
    assert not readiness_gate.HEX40_RE.fullmatch(template["phase1_integrity"]["phase1_head_sha"])
    assert not readiness_gate.HEX64_RE.fullmatch(template["split_definition"]["train_fingerprint"])
    # Owner approval fields must be placeholders, not a fabricated approval.
    approver = template["owner_approval"]["approver"]
    assert readiness_gate.PLACEHOLDER_PREFIX in approver or not approver.strip()


def test_template_contains_no_local_identifiers() -> None:
    """No username, home path, or hostname may leak into the template."""
    rendered = json.dumps(readiness_gate.build_template())
    try:
        username = getpass.getuser()
    except Exception:  # pragma: no cover - exotic environments
        username = ""
    if username and len(username) >= 3:
        assert username not in rendered
    home = str(Path.home())
    if len(home) >= 3:
        assert home not in rendered


def test_todo_placeholder_values_are_unknown(tmp_path: Path) -> None:
    code, out = _run(tmp_path, _passing_fixture(), "--json")
    assert code == 0
    payload = _passing_fixture()
    payload["tokenization_contract"]["segmenter_name"] = "TODO: install and pin the segmenter"
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    summary = _summary(out)
    entry = summary["categories"]["tokenization_contract"]
    assert entry["verdict"] == "unknown"
    assert "placeholder" in entry["reason"]


def test_tool_writes_no_files(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(_passing_fixture()), encoding="utf-8")
    code, _ = _run_cli("--input", str(path))
    assert code == 0
    assert {entry.name for entry in tmp_path.iterdir()} == {"evidence.json"}


# ---------------------------------------------------------------------------
# Schema validation failures
# ---------------------------------------------------------------------------


def test_missing_category_is_blocked_with_next_action(tmp_path: Path) -> None:
    code, out = _run(tmp_path, _without_category("split_definition"), "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["overall"] == "blocked"
    assert summary["categories"]["split_definition"]["verdict"] == "blocked"
    assert any("split_definition" in action for action in summary["next_actions"])


def test_unknown_top_level_key_exits_two(tmp_path: Path) -> None:
    payload = _passing_fixture()
    payload["extra_note"] = "not allowed"
    code, out = _run(tmp_path, payload, "--json")
    assert code == 2
    summary = _summary(out)
    assert summary["overall"] == "blocked"  # fail closed
    assert summary["input_error"] is not None
    assert "extra_note" in summary["input_error"]


def test_non_object_top_level_exits_two(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text("[]", encoding="utf-8")
    code, out = _run_cli("--input", str(path), "--json")
    assert code == 2
    assert "object" in _summary(out)["input_error"]


def test_missing_input_file_exits_two(tmp_path: Path) -> None:
    code, out = _run_cli("--input", str(tmp_path / "absent.json"), "--json")
    assert code == 2
    assert _summary(out)["overall"] == "blocked"


def test_invalid_json_exits_two_with_reason(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text("{not json", encoding="utf-8")
    code, out = _run_cli("--input", str(path), "--json")
    assert code == 2
    assert "invalid JSON" in _summary(out)["input_error"]


def test_bad_schema_version_exits_two(tmp_path: Path) -> None:
    payload = _passing_fixture()
    payload["schema_version"] = 99
    code, out = _run(tmp_path, payload, "--json")
    assert code == 2
    assert "schema_version" in _summary(out)["input_error"]


def test_no_arguments_is_a_usage_error() -> None:
    code, out = _run_cli()
    assert code == 2
    assert "--input" in out


def test_subprocess_never_prints_a_traceback(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text("{not json", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-I", str(TOOL_PATH), "--input", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "Traceback" not in proc.stdout


def test_wrong_field_type_is_blocked(tmp_path: Path) -> None:
    code, out = _run(tmp_path, _with("label_mapping_verification", agreement_rate="high"), "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["overall"] == "blocked"
    assert summary["categories"]["label_mapping_verification"]["verdict"] == "blocked"


def test_null_category_value_is_blocked(tmp_path: Path) -> None:
    payload = _passing_fixture()
    payload["compute_feasibility"] = None
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["categories"]["compute_feasibility"]["verdict"] == "blocked"
    assert "JSON object" in summary["categories"]["compute_feasibility"]["reason"]


@pytest.mark.parametrize(
    ("category", "field_name"),
    [("phase1_integrity", "phase1_head_sha"), ("model_checkpoint", "revision_sha")],
)
def test_bad_hex_sha_is_blocked(tmp_path: Path, category: str, field_name: str) -> None:
    code, out = _run(tmp_path, _with(category, **{field_name: "nothex"}), "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["categories"][category]["verdict"] == "blocked"
    assert any(field_name in action for action in summary["next_actions"])


def test_bad_iso_approval_date_is_blocked(tmp_path: Path) -> None:
    code, out = _run(tmp_path, _with("owner_approval", approval_date="07/11/2026"), "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["overall"] == "blocked"
    assert summary["categories"]["owner_approval"]["verdict"] == "blocked"


def test_future_approval_date_is_blocked(tmp_path: Path) -> None:
    code, out = _run(tmp_path, _with("owner_approval", approval_date="2099-01-01"), "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["categories"]["owner_approval"]["verdict"] == "blocked"
    assert "future" in summary["categories"]["owner_approval"]["reason"]


@pytest.mark.parametrize(
    ("category", "field_name", "value"),
    [
        ("label_mapping_verification", "agreement_rate", 1.2),
        ("label_mapping_verification", "agreement_rate", -0.1),
        ("leakage_preflight", "near_duplicate_threshold", 1.5),
    ],
)
def test_out_of_range_float_is_blocked(
    tmp_path: Path, category: str, field_name: str, value: float
) -> None:
    code, out = _run(tmp_path, _with(category, **{field_name: value}), "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["categories"][category]["verdict"] == "blocked"
    assert "outside the 0-1 range" in summary["categories"][category]["reason"]


def test_non_positive_split_size_is_blocked(tmp_path: Path) -> None:
    code, out = _run(tmp_path, _with("split_definition", test_size=0), "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["categories"]["split_definition"]["verdict"] == "blocked"


def test_unknown_field_inside_category_is_blocked(tmp_path: Path) -> None:
    payload = _with("split_definition", mystery_field="x")
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["categories"]["split_definition"]["verdict"] == "blocked"
    assert "mystery_field" in summary["categories"]["split_definition"]["reason"]


# ---------------------------------------------------------------------------
# Verdict logic: UNKNOWN, owner approval override, thresholds
# ---------------------------------------------------------------------------


def test_insufficient_evidence_is_unknown(tmp_path: Path) -> None:
    code, out = _run(tmp_path, _without_field("dataset_provenance", "license_statement"), "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["overall"] == "unknown"
    assert summary["categories"]["dataset_provenance"]["verdict"] == "unknown"
    assert "license_statement" in summary["categories"]["dataset_provenance"]["reason"]


def test_missing_owner_approval_blocks_overall(tmp_path: Path) -> None:
    code, out = _run(tmp_path, _without_category("owner_approval"), "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["overall"] == "blocked"
    assert any("owner_approval" in action for action in summary["next_actions"])


def test_incomplete_owner_approval_still_blocks_overall(tmp_path: Path) -> None:
    code, out = _run(tmp_path, _without_field("owner_approval", "approver"), "--json")
    assert code == 1
    assert _summary(out)["overall"] == "blocked"


@pytest.mark.parametrize(
    ("rate", "expected_verdict"),
    [(0.85, "blocked"), (0.9, "pass"), (0.92, "pass")],
)
def test_label_agreement_threshold(tmp_path: Path, rate: float, expected_verdict: str) -> None:
    payload = _with(
        "label_mapping_verification",
        method="blind_audit",
        audit_sample_size=135,
        agreement_rate=rate,
    )
    code, out = _run(tmp_path, payload, "--json")
    summary = _summary(out)
    assert summary["categories"]["label_mapping_verification"]["verdict"] == expected_verdict
    assert summary["overall"] == expected_verdict
    assert code == (0 if expected_verdict == "pass" else 1)


def test_blind_audit_sample_below_100_is_blocked(tmp_path: Path) -> None:
    payload = _with("label_mapping_verification", audit_sample_size=45)
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["categories"]["label_mapping_verification"]["verdict"] == "blocked"
    assert "100" in summary["categories"]["label_mapping_verification"]["reason"]


def test_author_confirmation_without_quote_is_blocked(tmp_path: Path) -> None:
    payload = _with("label_mapping_verification", method="author_confirmation")
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["categories"]["label_mapping_verification"]["verdict"] == "blocked"
    assert "quote_reference" in summary["categories"]["label_mapping_verification"]["reason"]


def test_author_confirmation_with_quote_passes(tmp_path: Path) -> None:
    payload = _with(
        "label_mapping_verification",
        method="author_confirmation",
        quote_reference="author email 2026-11-05, quoted in docs/phase2/label-audit.md",
    )
    code, out = _run(tmp_path, payload, "--json")
    assert code == 0
    assert _summary(out)["overall"] == "pass"


@pytest.mark.parametrize(
    "mapping",
    [["positive", "positive", "positive"], ["negative", "neutral"], ["negative", "neutral", "odd"]],
)
def test_resulting_mapping_must_be_canonical_permutation(tmp_path: Path, mapping: list) -> None:
    payload = _with("label_mapping_verification", resulting_mapping=mapping)
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["categories"]["label_mapping_verification"]["verdict"] == "blocked"


# ---------------------------------------------------------------------------
# Category-specific hard rules
# ---------------------------------------------------------------------------


def test_self_declared_license_checkbox_does_not_pass(tmp_path: Path) -> None:
    payload = _passing_fixture()
    payload["dataset_provenance"] = {"license_status": "ok"}
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["overall"] == "blocked"
    entry = summary["categories"]["dataset_provenance"]
    assert entry["verdict"] == "blocked"
    assert "license_status" in entry["reason"]


def test_placeholder_variants_case_and_colon_insensitive(tmp_path: Path) -> None:
    """Regression: 'todo: x', 'TODO x' and 'Todo: x' must not pass anywhere."""
    placeholders = (
        "todo: sign the agreement first",
        "TODO sign the agreement first",
        "Todo: sign",
    )
    for placeholder in placeholders:
        payload = _with("dataset_provenance", access_evidence_reference=placeholder)
        code, out = _run(tmp_path, payload, "--json")
        summary = _summary(out)
        entry = summary["categories"]["dataset_provenance"]
        assert entry["verdict"] == "unknown", (placeholder, entry)
        assert summary["overall"] != "pass"
        assert "placeholder" in entry["reason"]


def test_checksum_sha256_must_be_hex64(tmp_path: Path) -> None:
    """Regression: a recorded checksum must look like a sha256, not free text."""
    for junk in ("tbd", "junk-not-a-hash", "deadbeef"):
        payload = _with("model_checkpoint", checksum_sha256=junk)
        code, out = _run(tmp_path, payload, "--json")
        summary = _summary(out)
        entry = summary["categories"]["model_checkpoint"]
        assert entry["verdict"] == "blocked", (junk, entry)
        assert "64-character hexadecimal" in entry["reason"]
    # A bare "todo" is caught earlier as a placeholder (UNKNOWN, not BLOCKED).
    payload = _with("model_checkpoint", checksum_sha256="todo")
    code, out = _run(tmp_path, payload, "--json")
    summary = _summary(out)
    assert summary["categories"]["model_checkpoint"]["verdict"] == "unknown"


def test_pass_reasons_carry_self_reported_qualifier(tmp_path: Path) -> None:
    """Regression: honest qualifiers must survive the 120-char output clip."""
    code, out = _run(tmp_path, _passing_fixture(), "--json")
    assert code == 0
    summary = _summary(out)
    for category in (
        "dataset_provenance",
        "label_mapping_verification",
        "split_definition",
        "leakage_preflight",
        "model_checkpoint",
        "artifact_privacy_controls",
        "phase1_integrity",
    ):
        reason = summary["categories"][category]["reason"]
        assert "self-reported" in reason, (category, reason)
        assert TRUNCATION_MARKER not in reason, (category, reason)


def test_placeholder_license_statement_is_not_evidence(tmp_path: Path) -> None:
    code, out = _run(tmp_path, _with("dataset_provenance", license_statement="Unknown"), "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["overall"] == "unknown"
    assert summary["categories"]["dataset_provenance"]["verdict"] == "unknown"


def test_signed_access_without_access_evidence_reference_is_unknown(tmp_path: Path) -> None:
    payload = _without_field("dataset_provenance", "access_evidence_reference")
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["categories"]["dataset_provenance"]["verdict"] == "unknown"


def test_pickle_bin_checkpoint_is_blocked(tmp_path: Path) -> None:
    code, out = _run(tmp_path, _with("model_checkpoint", weights_format="pickle_bin"), "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["overall"] == "blocked"
    assert "R-01" in summary["categories"]["model_checkpoint"]["reason"]
    assert any("safetensors" in action for action in summary["next_actions"])


@pytest.mark.parametrize("checksum_recorded", [False])
def test_unrecorded_checkpoint_checksum_is_blocked(tmp_path: Path, checksum_recorded: bool) -> None:
    payload = _with("model_checkpoint", checksum_recorded=checksum_recorded)
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    assert _summary(out)["categories"]["model_checkpoint"]["verdict"] == "blocked"


def test_checksum_true_without_hash_is_unknown(tmp_path: Path) -> None:
    code, out = _run(tmp_path, _without_field("model_checkpoint", "checksum_sha256"), "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["overall"] == "unknown"
    assert summary["categories"]["model_checkpoint"]["verdict"] == "unknown"


def test_phase1_baseline_without_integrity_hash_is_unknown(tmp_path: Path) -> None:
    """A referenced baseline artifact without a recorded hash is not evidence."""
    payload = _without_field("phase1_integrity", "baseline_evidence_sha256")
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    summary = _summary(out)
    entry = summary["categories"]["phase1_integrity"]
    assert entry["verdict"] == "unknown"
    assert "baseline_evidence_sha256" in entry["reason"]


def test_phase1_baseline_with_sha256_passes(tmp_path: Path) -> None:
    code, out = _run(tmp_path, _passing_fixture(), "--json")
    assert code == 0
    assert _summary(out)["categories"]["phase1_integrity"]["verdict"] == "pass"


def test_phase1_test_set_touched_is_blocked(tmp_path: Path) -> None:
    code, out = _run(
        tmp_path, _with("phase1_integrity", test_excluded_from_fit_and_selection=False), "--json"
    )
    assert code == 1
    summary = _summary(out)
    assert summary["categories"]["phase1_integrity"]["verdict"] == "blocked"


@pytest.mark.parametrize("pattern", ["*.bin", "*.pt"])
def test_gitignore_missing_required_pattern_is_blocked(tmp_path: Path, pattern: str) -> None:
    patterns = [p for p in _passing_fixture()["artifact_privacy_controls"]["gitignore_patterns"]]
    patterns.remove(pattern)
    payload = _with("artifact_privacy_controls", gitignore_patterns=patterns)
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["overall"] == "blocked"
    assert "R-08" in summary["categories"]["artifact_privacy_controls"]["reason"]
    assert pattern in summary["categories"]["artifact_privacy_controls"]["reason"]


def test_artifact_patterns_declared_but_unverified_is_unknown(tmp_path: Path) -> None:
    """Claimed gitignore patterns without repository verification stay UNKNOWN."""
    payload = _with("artifact_privacy_controls", verified_against_repository=False)
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    summary = _summary(out)
    entry = summary["categories"]["artifact_privacy_controls"]
    assert entry["verdict"] == "unknown"
    assert "not verified" in entry["reason"]


def test_exact_match_alone_is_not_pass(tmp_path: Path) -> None:
    payload = _passing_fixture()
    del payload["leakage_preflight"]["near_duplicate_method"]
    del payload["leakage_preflight"]["near_duplicate_threshold"]
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["overall"] == "unknown"
    assert summary["categories"]["leakage_preflight"]["verdict"] == "unknown"


def test_exact_match_disabled_is_blocked(tmp_path: Path) -> None:
    code, out = _run(tmp_path, _with("leakage_preflight", exact_match_planned=False), "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["categories"]["leakage_preflight"]["verdict"] == "blocked"
    assert "R-06" in summary["categories"]["leakage_preflight"]["reason"]


def test_leakage_planned_but_not_executed_is_unknown(tmp_path: Path) -> None:
    """P1 regression: an intended check is not an executed check."""
    payload = _with(
        "leakage_preflight",
        exact_match_executed=False,
        near_duplicate_executed=False,
    )
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    summary = _summary(out)
    entry = summary["categories"]["leakage_preflight"]
    assert entry["verdict"] == "unknown"
    assert "planned" in entry["reason"] and "not executed" in entry["reason"]


def test_leakage_executed_without_report_reference_is_unknown(tmp_path: Path) -> None:
    payload = _without_field("leakage_preflight", "exact_match_report_reference")
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    summary = _summary(out)
    entry = summary["categories"]["leakage_preflight"]
    assert entry["verdict"] == "unknown"
    assert "report reference" in entry["reason"]


def test_leakage_unresolved_conflicts_block_readiness(tmp_path: Path) -> None:
    payload = _with("leakage_preflight", unresolved_conflicts=2)
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    summary = _summary(out)
    entry = summary["categories"]["leakage_preflight"]
    assert entry["verdict"] == "blocked"
    assert "unresolved" in entry["reason"]


def test_leakage_requires_execution_evidence_not_intentions(tmp_path: Path) -> None:
    """Only executed preflights with recorded references can PASS."""
    planned_only = _with(
        "leakage_preflight",
        exact_match_executed=False,
        near_duplicate_executed=False,
    )
    executed = _passing_fixture()
    code, out = _run(tmp_path, planned_only, "--json")
    assert _summary(out)["categories"]["leakage_preflight"]["verdict"] != "pass"
    code, out = _run(tmp_path, executed, "--json")
    assert code == 0
    assert _summary(out)["categories"]["leakage_preflight"]["verdict"] == "pass"


def test_cuda_without_vram_is_unknown(tmp_path: Path) -> None:
    code, out = _run(tmp_path, _without_field("compute_feasibility", "vram_gb"), "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["overall"] == "unknown"
    assert summary["categories"]["compute_feasibility"]["verdict"] == "unknown"


@pytest.mark.parametrize("vram", [0, -1])
def test_non_positive_vram_is_blocked(tmp_path: Path, vram: float) -> None:
    code, out = _run(tmp_path, _with("compute_feasibility", vram_gb=vram), "--json")
    assert code == 1
    assert _summary(out)["categories"]["compute_feasibility"]["verdict"] == "blocked"


def test_missing_pinned_version_entry_is_unknown(tmp_path: Path) -> None:
    payload = _passing_fixture()
    del payload["dependency_plan"]["pinned_versions"]["segmenter"]
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["overall"] == "unknown"
    assert summary["categories"]["dependency_plan"]["verdict"] == "unknown"


def test_non_semver_pin_is_blocked(tmp_path: Path) -> None:
    payload = _with(
        "dependency_plan",
        pinned_versions={"torch": "latest", "transformers": "4.46.2", "segmenter": "0.1.0"},
    )
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["categories"]["dependency_plan"]["verdict"] == "blocked"
    assert "torch" in summary["categories"]["dependency_plan"]["reason"]


# ---------------------------------------------------------------------------
# Warnings, sanitization, and exit codes
# ---------------------------------------------------------------------------


def test_oversized_max_length_warns_without_blocking(tmp_path: Path) -> None:
    # The warning is PhoBERT-specific: the evidence must name a phobert checkpoint.
    payload = _passing_fixture()
    payload["model_checkpoint"]["checkpoint_id"] = "vinai/phobert-base"
    payload["tokenization_contract"]["max_length"] = 512
    code, out = _run(tmp_path, payload, "--json")
    assert code == 0
    summary = _summary(out)
    assert summary["overall"] == "pass"
    entry = summary["categories"]["tokenization_contract"]
    assert entry["verdict"] == "pass"
    assert "256" in entry["warning"]


def test_sanitization_truncates_long_input_values(tmp_path: Path) -> None:
    long_value = "x" * 200
    payload = _with("compute_feasibility", device=long_value)
    code, human_out = _run(tmp_path, payload)
    assert code == 1
    assert TRUNCATION_MARKER in human_out
    assert long_value not in human_out
    code, json_out = _run(tmp_path, payload, "--json")
    assert code == 1
    assert TRUNCATION_MARKER in json_out
    assert long_value not in json_out


def test_exit_code_is_one_for_unknown_overall(tmp_path: Path) -> None:
    code, out = _run(tmp_path, _without_field("compute_feasibility", "oom_recovery_plan"), "--json")
    assert code == 1
    assert _summary(out)["overall"] == "unknown"


def test_exit_code_is_two_survives_unknown_flag_usage(tmp_path: Path) -> None:
    # argparse rejects unknown flags with exit code 2 (no traceback).
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(_passing_fixture()), encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        readiness_gate.main(["--input", str(path), "--definitely-not-a-flag"])
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# Schema v2 dataset routes, adapter/mapping rules, and fingerprint checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [_owner_collected_fixture, _licensed_public_fixture, _neu_esc_gated_fixture],
)
def test_non_uit_vsfc_dataset_routes_pass(tmp_path: Path, factory: Any) -> None:
    payload = factory()
    code, out = _run(tmp_path, payload, "--json")
    assert code == 0, out
    summary = _summary(out)
    assert summary["overall"] == "pass"
    for category in ("dataset_provenance", "label_mapping_verification", "split_definition"):
        reason = summary["categories"][category]["reason"]
        assert "self-reported" in reason
        assert TRUNCATION_MARKER not in reason


def test_missing_and_v1_schema_versions_exit_two_with_migration_hint(tmp_path: Path) -> None:
    missing = _passing_fixture()
    del missing["schema_version"]
    code, out = _run(tmp_path, missing, "--json")
    assert code == 2
    assert "schema_version" in _summary(out)["input_error"]

    legacy = _passing_fixture()
    legacy["schema_version"] = 1
    code, out = _run(tmp_path, legacy, "--json")
    assert code == 2
    err = _summary(out)["input_error"]
    assert "schema_version 1" in err and "schema 2" in err


def test_synthetic_fixture_markers_are_blocked(tmp_path: Path) -> None:
    code, out = _run(tmp_path, _with("dataset_provenance", synthetic_fixture=True), "--json")
    assert code == 1
    assert _summary(out)["categories"]["dataset_provenance"]["verdict"] == "blocked"

    code, out = _run(
        tmp_path, _with("dataset_provenance", dataset_id="synthetic_feedback_v1"), "--json"
    )
    assert code == 1
    assert _summary(out)["categories"]["dataset_provenance"]["verdict"] == "blocked"

    code, out = _run(
        tmp_path, _with("leakage_preflight", preflight_synthetic_fixture=True), "--json"
    )
    assert code == 1
    assert _summary(out)["categories"]["leakage_preflight"]["verdict"] == "blocked"

    code, out = _run(
        tmp_path, _with("phase1_integrity", baseline_result_scope="synthetic_fixture"), "--json"
    )
    assert code == 1
    assert _summary(out)["categories"]["phase1_integrity"]["verdict"] == "blocked"


def test_rights_confirmed_bool_and_bare_status_words_rejected(tmp_path: Path) -> None:
    payload = _with("dataset_provenance", rights_confirmed=True)
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    entry = _summary(out)["categories"]["dataset_provenance"]
    assert entry["verdict"] == "blocked"
    assert "rights_confirmed" in entry["reason"]

    for word in ("confirmed", "rights_confirmed", "signed", "ok"):
        payload = _with("dataset_provenance", access_evidence_reference=word)
        code, out = _run(tmp_path, payload, "--json")
        assert code == 1
        assert _summary(out)["categories"]["dataset_provenance"]["verdict"] == "unknown"


@pytest.mark.parametrize(
    ("adapter", "access_class"),
    [
        ("uit_vsfc", "licensed_public"),
        ("uit_vsfc", "gated_click_accept"),
        ("uit_vsfc", "owner_collected_human_approved"),
        ("five_level_csv", "owner_collected_human_approved"),
        ("neu_esc", "owner_collected_human_approved"),
        ("neu_esc", "licensed_public"),
    ],
)
def test_adapter_access_incompatibilities_are_blocked(
    tmp_path: Path, adapter: str, access_class: str
) -> None:
    payload = _with(
        "dataset_provenance", dataset_adapter=adapter, access_condition_class=access_class
    )
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    assert _summary(out)["categories"]["dataset_provenance"]["verdict"] == "blocked"


def test_external_access_requires_valid_source_url(tmp_path: Path) -> None:
    payload = _licensed_public_fixture()
    payload["dataset_provenance"]["source_url"] = ""
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    assert _summary(out)["categories"]["dataset_provenance"]["verdict"] == "unknown"

    payload["dataset_provenance"]["source_url"] = "not-a-url"
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    assert _summary(out)["categories"]["dataset_provenance"]["verdict"] == "blocked"


def test_author_confirmation_works_without_blind_audit_numbers(tmp_path: Path) -> None:
    payload = _passing_fixture()
    payload["label_mapping_verification"] = {
        "method": "author_confirmation",
        "resulting_mapping": ["negative", "neutral", "positive"],
        "quote_reference": "email 2026-11-05 from author confirming 0/1/2",
    }
    code, out = _run(tmp_path, payload, "--json")
    assert code == 0, out
    assert _summary(out)["overall"] == "pass"


@pytest.mark.parametrize(
    ("payload_factory", "bad_method"),
    [
        (_passing_fixture, "human_approved_canonical"),
        (_owner_collected_fixture, "blind_audit"),
        (_licensed_public_fixture, "neu_esc_named_labels"),
        (_neu_esc_gated_fixture, "blind_audit"),
    ],
)
def test_adapter_and_label_method_mismatch_is_blocked(
    tmp_path: Path, payload_factory: Any, bad_method: str
) -> None:
    payload = payload_factory()
    payload["label_mapping_verification"]["method"] = bad_method
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    assert _summary(out)["categories"]["label_mapping_verification"]["verdict"] == "blocked"


@pytest.mark.parametrize("bad_policy", ["keep", "error"])
def test_neu_esc_non_exclude_toxic_policy_is_blocked(tmp_path: Path, bad_policy: str) -> None:
    payload = _neu_esc_gated_fixture()
    payload["label_mapping_verification"]["toxic_policy"] = bad_policy
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    assert _summary(out)["categories"]["label_mapping_verification"]["verdict"] == "blocked"


def test_neu_esc_missing_justification_or_negative_count(tmp_path: Path) -> None:
    payload = _neu_esc_gated_fixture()
    payload["label_mapping_verification"]["toxic_justification"] = ""
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    assert _summary(out)["categories"]["label_mapping_verification"]["verdict"] == "unknown"

    payload = _neu_esc_gated_fixture()
    payload["label_mapping_verification"]["toxic_excluded_count"] = -1
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    assert _summary(out)["categories"]["label_mapping_verification"]["verdict"] == "blocked"

    payload = _neu_esc_gated_fixture()
    payload["label_mapping_verification"]["toxic_excluded_count"] = 0
    code, out = _run(tmp_path, payload, "--json")
    assert code == 0


def test_toxic_fields_rejected_on_non_neu_esc_method(tmp_path: Path) -> None:
    payload = _owner_collected_fixture()
    payload["label_mapping_verification"]["toxic_policy"] = "exclude"
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    assert _summary(out)["categories"]["label_mapping_verification"]["verdict"] == "blocked"


@pytest.mark.parametrize("bad_float", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_fail_closed(tmp_path: Path, bad_float: float) -> None:
    payload = _with("label_mapping_verification", agreement_rate=bad_float)
    report = readiness_gate.evaluate_evidence(payload)
    assert report.overall == "blocked"
    cat = {c.category: c for c in report.categories}["label_mapping_verification"]
    assert cat.verdict == "blocked"

    # Also test JSON text containing NaN/Infinity parses fail-closed via CLI.
    path = tmp_path / "nan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    code, out = _run_cli("--input", str(path), "--json")
    assert code == 1
    assert _summary(out)["overall"] == "blocked"


def test_split_fingerprints_must_be_valid_and_distinct(tmp_path: Path) -> None:
    payload = _with("split_definition", validation_fingerprint=TRAIN_FP)
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    assert _summary(out)["categories"]["split_definition"]["verdict"] == "blocked"

    payload = _with("split_definition", test_fingerprint="not-64-hex")
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    assert _summary(out)["categories"]["split_definition"]["verdict"] == "blocked"


def test_preflight_requirements_and_fingerprint_linkage(tmp_path: Path) -> None:
    # Missing 'test' in preflight_split_fingerprints yields UNKNOWN (incomplete cross-split check).
    payload = _passing_fixture()
    del payload["leakage_preflight"]["preflight_split_fingerprints"]["test"]
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    assert _summary(out)["categories"]["leakage_preflight"]["verdict"] == "unknown"

    # Preflight mismatching split_definition yields BLOCKED.
    payload = _passing_fixture()
    payload["leakage_preflight"]["preflight_split_fingerprints"]["validation"] = "f" * 64
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    assert _summary(out)["categories"]["leakage_preflight"]["verdict"] == "blocked"

    # Contradictions or failed preflight_passed or missing preflight sha fail closed / unknown.
    code, out = _run(
        tmp_path, _with("leakage_preflight", preflight_contradictions_count=1), "--json"
    )
    assert code == 1
    assert _summary(out)["categories"]["leakage_preflight"]["verdict"] == "blocked"

    code, out = _run(tmp_path, _with("leakage_preflight", preflight_passed=False), "--json")
    assert code == 1
    assert _summary(out)["categories"]["leakage_preflight"]["verdict"] == "blocked"

    code, out = _run(
        tmp_path, _without_field("leakage_preflight", "preflight_report_sha256"), "--json"
    )
    assert code == 1
    assert _summary(out)["categories"]["leakage_preflight"]["verdict"] == "unknown"


def test_planned_exact_match_with_no_passing_receipt_is_unknown(tmp_path: Path) -> None:
    payload = _with("leakage_preflight", exact_match_executed=False, preflight_passed=False)
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    assert _summary(out)["categories"]["leakage_preflight"]["verdict"] == "unknown"


def test_phase1_split_and_holdout_fingerprint_rules(tmp_path: Path) -> None:
    # 'test' must NEVER appear in fit/selection baseline_split_fingerprints.
    payload = _passing_fixture()
    payload["phase1_integrity"]["baseline_split_fingerprints"]["test"] = TEST_FP
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    assert _summary(out)["categories"]["phase1_integrity"]["verdict"] == "blocked"

    payload = _passing_fixture()
    payload["phase1_integrity"]["baseline_split_fingerprints"]["train"] = "e" * 64
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    assert _summary(out)["categories"]["phase1_integrity"]["verdict"] == "blocked"

    payload = _with("phase1_integrity", baseline_result_scope="declared_holdout")
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    assert _summary(out)["categories"]["phase1_integrity"]["verdict"] == "unknown"

    payload = _with(
        "phase1_integrity",
        baseline_result_scope="declared_holdout",
        baseline_holdout_fingerprint="e" * 64,
    )
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    assert _summary(out)["categories"]["phase1_integrity"]["verdict"] == "blocked"

    payload = _with("phase1_integrity", baseline_holdout_fingerprint=TEST_FP)
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    assert _summary(out)["categories"]["phase1_integrity"]["verdict"] == "blocked"


@pytest.mark.parametrize(
    ("test_fingerprint", "verdict"), [(None, "unknown"), ("f" * 64, "blocked")]
)
def test_near_duplicate_audit_must_cover_declared_holdout(
    tmp_path: Path, test_fingerprint: str | None, verdict: str
) -> None:
    payload = _passing_fixture()
    fingerprints = payload["leakage_preflight"]["near_duplicate_split_fingerprints"]
    if test_fingerprint is None:
        del fingerprints["test"]
    else:
        fingerprints["test"] = test_fingerprint
    code, out = _run(tmp_path, payload, "--json")
    assert code == 1
    assert _summary(out)["categories"]["leakage_preflight"]["verdict"] == verdict


@pytest.mark.parametrize(
    "overrides",
    [
        {"baseline_preflight_report_sha256": "f" * 64},
        {"baseline_training_started_at": "2026-11-02T08:30:00+01:00"},
        {"baseline_training_started_at": "2026-11-02T08:30:00"},
    ],
)
def test_baseline_requires_matching_pretraining_receipt(
    tmp_path: Path, overrides: dict[str, str]
) -> None:
    code, out = _run(tmp_path, _with("phase1_integrity", **overrides), "--json")
    assert code == 1
    assert _summary(out)["categories"]["phase1_integrity"]["verdict"] == "blocked"
