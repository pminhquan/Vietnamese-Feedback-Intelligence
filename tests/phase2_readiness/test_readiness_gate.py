"""Tests for the offline Phase 2 readiness gate (synthetic evidence only).

Every fixture is a synthetic evidence package built in memory from the tool's
own template; nothing here touches real datasets, weights, or the network.
The tool must stay a pure read-only checker, so one test also asserts that a
run creates no files.
"""

import copy
import importlib.util
import io
import json
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


def _passing_fixture() -> dict[str, Any]:
    return copy.deepcopy(readiness_gate.build_template())


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


def test_schema_output_is_valid_json_and_round_trips_to_pass(tmp_path: Path) -> None:
    code, out = _run_cli("--schema")
    assert code == 0
    template = json.loads(out)  # must parse as JSON
    assert set(CATEGORY_NAMES).issubset(template)
    code, out = _run(tmp_path, template, "--json")  # round trip
    assert code == 0
    assert _summary(out)["overall"] == "pass"


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


def test_placeholder_license_statement_is_not_evidence(tmp_path: Path) -> None:
    code, out = _run(tmp_path, _with("dataset_provenance", license_statement="Unknown"), "--json")
    assert code == 1
    summary = _summary(out)
    assert summary["overall"] == "unknown"
    assert summary["categories"]["dataset_provenance"]["verdict"] == "unknown"


def test_signed_access_without_agreement_reference_is_unknown(tmp_path: Path) -> None:
    payload = _without_field("dataset_provenance", "agreement_reference")
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


def test_phase1_test_set_touched_is_blocked(tmp_path: Path) -> None:
    code, out = _run(tmp_path, _with("phase1_integrity", test_set_untouched=False), "--json")
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
    code, out = _run(tmp_path, _with("tokenization_contract", max_length=512), "--json")
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
