"""Red-team tests for the hardware probe and Phase 2 readiness gate.

Hardware probe cases verify path-canonicalization, env-spoof, and argparse
privacy guarantees. The appended readiness gate section reproduces historical
false accepts (synthetic fixtures, checkbox flags, unlinked fingerprints,
leaked test splits, stale schema omission) and historical false rejects
(legitimate non-UIT-VSFC dataset routes and declared holdout/preflight test
access).
"""

import getpass
import importlib.util
import io
import json
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest

TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "phase2_readiness" / "hardware_probe.py"
GATE_TOOL_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "phase2_readiness" / "readiness_gate.py"
)
FAKE_SMI_CSV = "NVIDIA GeForce RTX 3060 Laptop GPU, 6144, 617.42, 8.6\n"


def _load_tool(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hardware_probe = _load_tool(TOOL_PATH, "vfi_hardware_probe_redteam")
readiness_gate = _load_tool(GATE_TOOL_PATH, "vfi_readiness_gate_redteam")


def _fake_which(_binary: str) -> str:
    return "C:/fake-tools/nvidia-smi.exe"


def _fake_runner(_cmd: list[str], _timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(_cmd, 0, stdout=FAKE_SMI_CSV, stderr="")


def _make_fixture_repo(tmp_path: Path) -> Path:
    """Copy the probe into a temporary repository layout (module + .git)."""
    tool_dir = tmp_path / "fixture-repo" / "tools" / "phase2_readiness"
    tool_dir.mkdir(parents=True)
    shutil.copy2(TOOL_PATH, tool_dir / "hardware_probe.py")
    (tmp_path / "fixture-repo" / ".git").mkdir()
    return tmp_path / "fixture-repo"


def test_output_with_extended_prefix_bypass_writes_inside_fixture_repo(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Reproduced: ``\\\\?\\``-prefixed ``--output`` writes inside the repo.

    On Python 3.12 ``Path.resolve()`` reduces ``\\\\?\\C:\\a\\b\\leak.json`` to
    the DRIVE-RELATIVE string ``C:a\\b\\leak.json``; the refusal check sees a
    path that is not under the repo root, then ``mkdir``/``write_text`` resolve
    the drive-relative string against the per-drive working directory of the
    probe process. With the CWD inside the repository (the documented normal
    usage), the detailed results land inside it. Observed end-to-end on
    2026-11-07: exit code 0, file created at
    ``<repo>\\project-personal\\VFI-Phase2-Research\\redteam_leak_a.json``.

    Desired behavior asserted here: the crafted spelling must be refused like
    the plain path (exit 1, nothing written).
    """
    fixture_repo = _make_fixture_repo(tmp_path)
    probe = _load_tool(
        fixture_repo / "tools" / "phase2_readiness" / "hardware_probe.py",
        f"redteam_probe_ext_{abs(hash(str(fixture_repo)))}",
    )
    monkeypatch.chdir(fixture_repo)
    target = fixture_repo / "leak.json"
    assert probe._repo_internal_refusal(probe._resolve_output(target)) is not None
    extended = "\\\\?\\" + str(target)  # e.g. \\?\C:\...\fixture-repo\leak.json
    exit_code = probe.main(["--output", extended], which=_fake_which, runner=_fake_runner)
    captured = capsys.readouterr()
    assert exit_code == 1, "crafted extended-prefix path must be refused like the plain path"
    assert "refusing" in captured.err
    assert not list(fixture_repo.rglob("leak.json")), "details file must not be written"


def test_output_trailing_space_component_bypass_writes_inside_fixture_repo(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Reproduced: trailing-space component defeats the parents comparison.

    ``D:\\...\\repo\\docs \\leak.json`` (space before the backslash) resolves to
    ``\\\\?\\D:\\...\\repo\\docs\\leak.json``: the component itself is
    canonicalized, but the result carries the Windows extended-length prefix,
    and ``root in dest.parents`` compares plain ``D:\\...`` anchors against
    ``\\\\?\\D:\\...`` anchors, so the refusal is missed. Observed end-to-end on
    2026-11-07: exit code 0 and the file landed in the REAL ``docs/`` directory
    of the clone.

    Desired behavior asserted here: the crafted spelling must be refused like
    the plain path (exit 1, nothing written).
    """
    fixture_repo = _make_fixture_repo(tmp_path)
    probe = _load_tool(
        fixture_repo / "tools" / "phase2_readiness" / "hardware_probe.py",
        f"redteam_probe_space_{abs(hash(str(fixture_repo)))}",
    )
    (fixture_repo / "subdir").mkdir()
    monkeypatch.chdir(fixture_repo)
    crafted = str(fixture_repo / "subdir") + " \\leak.json"  # 'subdir ' with trailing space
    exit_code = probe.main(["--output", crafted], which=_fake_which, runner=_fake_runner)
    captured = capsys.readouterr()
    assert exit_code == 1, "trailing-space crafted path must be refused like the plain path"
    assert "refusing" in captured.err
    assert not (fixture_repo / "subdir" / "leak.json").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows drive-letter paths")
def test_unc_admin_share_spelling_is_refused(tmp_path: Path) -> None:
    """Reproduced (API level): UNC spelling of a repo path is not refused.

    ``\\\\localhost\\<drive>$\\...`` falls back under ``realpath`` to a literal
    drive-rooted path (``D:\\localhost\\D$\\...``), which is not under the repo
    root, so the refusal returns None. Manually verified end-to-end on
    2026-11-07: the write then lands in a new ``D:\\localhost\\...`` tree,
    i.e. at a different location than the string the guard checked. This test
    only checks the guard decision, so nothing is written anywhere.

    Desired behavior asserted here: a UNC spelling must be handled explicitly
    (refused or canonicalized), never silently accepted as outside the repo.
    """
    drive = tmp_path.drive  # e.g. 'C:' — same spelling trick as the manual run
    unc = f"\\\\localhost\\{drive[0]}$\\some\\repo\\leak.json"
    resolved = hardware_probe._resolve_output(Path(unc))
    assert str(resolved).lower() != unc.lower(), "resolve() must canonicalize UNC spellings"
    assert hardware_probe._repo_internal_refusal(resolved) is None or (
        hardware_probe._find_repo_root(TOOL_PATH) in resolved.parents
    )


def test_env_spoofing_does_not_disable_guard_on_output_echo(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Reproduced: spoofed env vars make the guard miss the real username.

    ``getpass.getuser()`` prefers LOGNAME/USER/LNAME over USERNAME, and
    ``Path.home()`` follows USERPROFILE. With those spoofed, the guard checks
    the fake values; a ``--output`` path that embeds the REAL username is then
    echoed to stderr unredacted (observed: ``written to C:\\Users\\<real>\\...``),
    while the same invocation under the real environment is redacted. The
    destination below embeds the real username but stays inside tmp_path.

    Desired behavior asserted here: the echo must stay redacted even when the
    environment the guard reads has been manipulated.
    """
    real_user = getpass.getuser()
    for var in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        monkeypatch.setenv(var, "innocent_user1")
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "fakehome"))
    destination = tmp_path / f"u={real_user}" / "probe-out.json"
    exit_code = hardware_probe.main(
        ["--output", str(destination)], which=_fake_which, runner=_fake_runner
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert destination.exists()
    assert real_user not in captured.err, "spoofed env must not disable redaction of the echo"
    assert "[redacted:" in captured.err


def test_argparse_unrecognized_arguments_are_not_echoed_unredacted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reproduced: argv is echoed verbatim before any guard runs.

    ``hardware_probe --bogusflag leak <path-with-username>`` exits 2 and the
    argparse message repeats the caller-supplied value on stderr without the
    redaction the module docstring promises for error strings. No *new*
    information leaks (the caller typed it), but the documented guarantee does
    not hold on this path, and the echo lands unredacted in captured logs.

    Desired behavior asserted here: the argparse echo must respect the same
    redaction contract as every other error message.
    """
    real_user = getpass.getuser()
    with pytest.raises(SystemExit) as excinfo:
        hardware_probe.main(["--bogusflag", f"leak C:/Users/{real_user}/secret.json"])
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "unrecognized arguments" in captured.err
    assert real_user not in captured.err, "argparse echo must be redacted like other errors"
    assert "[redacted:" in captured.err


def test_computername_spoof_does_not_change_hostname_guard(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Control (verified covered): COMPUTERNAME is spoofable but platform.node()
    ignores it, so hostname redaction cannot be disabled this way."""
    import platform as _platform

    monkeypatch.setenv("COMPUTERNAME", "FakeHost")
    assert _platform.node() != "FakeHost"
    exit_code = hardware_probe.main(["--json"], which=_fake_which, runner=_fake_runner)
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "FakeHost" not in captured.out


def test_covered_path_crafts_are_still_refused() -> None:
    """Pin the surface verified COVERED so it cannot regress silently.

    Case variants, forward slashes, ``..`` traversal, ``.`` segments and
    trailing dot/space on the FILE name all canonicalize (or compare
    case-folded) to a path under the repository root and are refused.
    """
    root = TOOL_PATH.resolve().parents[2]
    plain = root / "docs" / "probe-out.json"
    assert hardware_probe._repo_internal_refusal(hardware_probe._resolve_output(plain)) is not None
    for crafted in (
        str(plain).replace(plain.drive, plain.drive.lower()),
        str(plain).replace("\\", "/"),
        str(root / "docs" / "sub" / ".." / "probe-out.json"),
        str(root / "docs" / "." / "probe-out.json"),
        str(plain) + ".",
        str(plain) + " ",
    ):
        resolved = hardware_probe._resolve_output(Path(crafted))
        refusal = hardware_probe._repo_internal_refusal(resolved)
        assert refusal is not None, f"crafted path was NOT refused: {crafted!r} -> {resolved!r}"


# ---------------------------------------------------------------------------
# Phase 2 readiness gate red-team regressions (false accepts & false rejects)
# ---------------------------------------------------------------------------

_RT_TRAIN_FP = "a1" * 32
_RT_VAL_FP = "b2" * 32
_RT_TEST_FP = "c3" * 32


def _gate_base_uit_vsfc_payload() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "dataset_provenance": {
            "dataset_id": "uit_vsfc_redteam",
            "dataset_version": "1.0.0",
            "dataset_adapter": "uit_vsfc",
            "synthetic_fixture": False,
            "access_condition_class": "author_agreement_signed",
            "source_url": "https://example.org/uit-vsfc",
            "license_statement": "signed academic license terms from UIT NLP group",
            "access_evidence_reference": "signed_agreement_2026-11-10.pdf",
        },
        "label_mapping_verification": {
            "method": "blind_audit",
            "verification_reference": "reports/uit_blind_audit_120.json",
            "audit_sample_size": 120,
            "agreement_rate": 0.95,
            "resulting_mapping": ["negative", "neutral", "positive"],
        },
        "split_definition": {
            "split_policy": "physical",
            "train_source": "uit-vsfc/train.txt",
            "validation_source": "uit-vsfc/dev.txt",
            "test_source": "uit-vsfc/test.txt",
            "train_size": 11426,
            "validation_size": 1583,
            "test_size": 3166,
            "fingerprint_algorithm": "sha256",
            "train_fingerprint": _RT_TRAIN_FP,
            "validation_fingerprint": _RT_VAL_FP,
            "test_fingerprint": _RT_TEST_FP,
        },
        "leakage_preflight": {
            "exact_match_planned": True,
            "exact_match_executed": True,
            "exact_match_report_reference": "reports/evaluation_preflight.json",
            "preflight_passed": True,
            "preflight_report_sha256": "d4" * 32,
            "preflight_created_at": "2026-11-02T08:00:00+00:00",
            "preflight_synthetic_fixture": False,
            "preflight_contradictions_count": 0,
            "preflight_split_fingerprints": {
                "train": _RT_TRAIN_FP,
                "validation": _RT_VAL_FP,
                "test": _RT_TEST_FP,
            },
            "near_duplicate_method": "char_ngram_jaccard",
            "near_duplicate_threshold": 0.9,
            "near_duplicate_executed": True,
            "near_duplicate_report_reference": "reports/near_duplicate_audit.json",
            "near_duplicate_split_fingerprints": {
                "train": _RT_TRAIN_FP,
                "validation": _RT_VAL_FP,
                "test": _RT_TEST_FP,
            },
            "unresolved_conflicts": 0,
        },
        "phase1_integrity": {
            "phase1_head_sha": "db9b9210" * 5,
            "baseline_evidence_reference": "reports/training_report.json",
            "baseline_evidence_sha256": "e5" * 32,
            "baseline_preflight_report_sha256": "d4" * 32,
            "baseline_training_started_at": "2026-11-02T08:30:00+00:00",
            "baseline_result_scope": "train_validation",
            "baseline_split_fingerprints": {
                "train": _RT_TRAIN_FP,
                "validation": _RT_VAL_FP,
            },
            "baseline_holdout_fingerprint": "",
            "test_excluded_from_fit_and_selection": True,
        },
        "model_checkpoint": {
            "checkpoint_id": "vinai/phobert-base-v2",
            "revision_sha": "12" * 20,
            "license_name": "MIT",
            "weights_format": "safetensors",
            "checksum_recorded": True,
            "checksum_sha256": "f6" * 32,
        },
        "tokenization_contract": {
            "segmenter_name": "underthesea",
            "segmenter_version": "6.8.4",
            "determinism_evidence_reference": "reports/seg_check.json",
            "max_length": 256,
            "truncation_side": "right",
        },
        "dependency_plan": {
            "pinned_versions": {
                "torch": "2.5.1",
                "transformers": "4.46.2",
                "segmenter": "6.8.4",
            },
            "lockfile_reference": "locks/phase2.lock",
        },
        "compute_feasibility": {
            "device": "cuda",
            "vram_gb": 6,
            "oom_recovery_plan": "batch 16 -> 8 -> grad accum 2 -> cpu fallback",
        },
        "artifact_privacy_controls": {
            "gitignore_patterns": ["*.bin", "*.pt", "*.safetensors", "data/raw/"],
            "reports_pii_policy": "aggregate metrics and anonymized indices only",
            "verified_against_repository": True,
        },
        "owner_approval": {
            "approver": "project-owner",
            "approval_date": "2026-11-10",
            "scope_statement": "Phase 2 M1 readiness verification",
        },
    }


def _gate_run(tmp_path: Path, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    path = tmp_path / "gate_evidence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = readiness_gate.main(["--input", str(path), "--json"])
    return code, json.loads(buf.getvalue())


def test_redteam_false_accept_synthetic_smoke_payload_is_blocked(tmp_path: Path) -> None:
    """Smoke-test fixtures or synthetic dataset IDs must NEVER pass readiness."""
    for mutator in (
        lambda p: p["dataset_provenance"].update({"synthetic_fixture": True}),
        lambda p: p["dataset_provenance"].update({"dataset_id": "synthetic_smoke_fixture"}),
        lambda p: p["leakage_preflight"].update({"preflight_synthetic_fixture": True}),
        lambda p: p["phase1_integrity"].update({"baseline_result_scope": "synthetic_fixture"}),
    ):
        payload = _gate_base_uit_vsfc_payload()
        mutator(payload)
        code, summary = _gate_run(tmp_path, payload)
        assert code == 1
        assert summary["overall"] == "blocked"


def test_redteam_false_accept_rights_confirmed_checkbox_is_rejected(tmp_path: Path) -> None:
    """Bool rights_confirmed checkbox or bare status word must never pass provenance."""
    payload = _gate_base_uit_vsfc_payload()
    payload["dataset_provenance"]["rights_confirmed"] = True
    code, summary = _gate_run(tmp_path, payload)
    assert code == 1
    assert summary["categories"]["dataset_provenance"]["verdict"] == "blocked"

    payload = _gate_base_uit_vsfc_payload()
    payload["dataset_provenance"]["access_evidence_reference"] = "rights_confirmed"
    code, summary = _gate_run(tmp_path, payload)
    assert code == 1
    assert summary["categories"]["dataset_provenance"]["verdict"] == "unknown"


def test_redteam_false_accept_uit_vsfc_bypass_routes_are_blocked(tmp_path: Path) -> None:
    """UIT-VSFC cannot claim licensed_public/gated access or non-0/1/2 mapping."""
    payload = _gate_base_uit_vsfc_payload()
    payload["dataset_provenance"]["access_condition_class"] = "licensed_public"
    code, summary = _gate_run(tmp_path, payload)
    assert code == 1
    assert summary["categories"]["dataset_provenance"]["verdict"] == "blocked"

    payload = _gate_base_uit_vsfc_payload()
    payload["label_mapping_verification"] = {
        "method": "human_approved_canonical",
        "verification_reference": "skipped 0/1/2 check",
    }
    code, summary = _gate_run(tmp_path, payload)
    assert code == 1
    assert summary["categories"]["label_mapping_verification"]["verdict"] == "blocked"


def test_redteam_false_accept_neu_esc_toxic_keep_or_unjustified_fails(tmp_path: Path) -> None:
    """NEU-ESC with toxic_policy='keep' or missing exclusion justification cannot pass."""
    payload = _gate_base_uit_vsfc_payload()
    payload["dataset_provenance"].update(
        {
            "dataset_id": "neu_esc_corpus",
            "dataset_adapter": "neu_esc",
            "access_condition_class": "gated_click_accept",
            "access_evidence_reference": "HF gated click-accept 2026-11-10",
        }
    )
    payload["label_mapping_verification"] = {
        "method": "neu_esc_named_labels",
        "verification_reference": "docs/phase2/label-semantics-evidence.md#neu-esc",
        "toxic_policy": "keep",
        "toxic_justification": "trying to keep toxic class during 3-class sentiment fit",
        "toxic_excluded_count": 0,
    }
    code, summary = _gate_run(tmp_path, payload)
    assert code == 1
    assert summary["categories"]["label_mapping_verification"]["verdict"] == "blocked"


def test_redteam_false_accept_legacy_preflight_without_test_or_mismatch_fails(
    tmp_path: Path,
) -> None:
    """Legacy data_report-style preflight omitting 'test' or using stale splits must not pass."""
    payload = _gate_base_uit_vsfc_payload()
    del payload["leakage_preflight"]["preflight_split_fingerprints"]["test"]
    code, summary = _gate_run(tmp_path, payload)
    assert code == 1
    assert summary["categories"]["leakage_preflight"]["verdict"] == "unknown"

    payload = _gate_base_uit_vsfc_payload()
    payload["leakage_preflight"]["preflight_split_fingerprints"]["test"] = "99" * 32
    code, summary = _gate_run(tmp_path, payload)
    assert code == 1
    assert summary["categories"]["leakage_preflight"]["verdict"] == "blocked"


def test_redteam_false_accept_leaked_test_or_mismatched_holdout_is_blocked(tmp_path: Path) -> None:
    """Putting test in fit fingerprints or mismatched holdout fingerprint fails closed."""
    payload = _gate_base_uit_vsfc_payload()
    payload["phase1_integrity"]["baseline_split_fingerprints"]["test"] = _RT_TEST_FP
    code, summary = _gate_run(tmp_path, payload)
    assert code == 1
    assert summary["categories"]["phase1_integrity"]["verdict"] == "blocked"

    payload = _gate_base_uit_vsfc_payload()
    payload["phase1_integrity"]["baseline_result_scope"] = "declared_holdout"
    payload["phase1_integrity"]["baseline_holdout_fingerprint"] = "00" * 32
    code, summary = _gate_run(tmp_path, payload)
    assert code == 1
    assert summary["categories"]["phase1_integrity"]["verdict"] == "blocked"


def test_redteam_false_accept_missing_or_v1_schema_version_exits_two(tmp_path: Path) -> None:
    """Schema version omission must never silently default to schema v2."""
    payload = _gate_base_uit_vsfc_payload()
    del payload["schema_version"]
    code, summary = _gate_run(tmp_path, payload)
    assert code == 2
    assert summary["overall"] == "blocked"

    payload = _gate_base_uit_vsfc_payload()
    payload["schema_version"] = 1
    code, summary = _gate_run(tmp_path, payload)
    assert code == 2
    assert "schema_version 1" in summary["input_error"]


def test_redteam_false_reject_custom_jsonl_owner_collected_passes(tmp_path: Path) -> None:
    """Owner-collected custom_jsonl without source_url or UIT-VSFC agreement must PASS."""
    payload = _gate_base_uit_vsfc_payload()
    payload["dataset_provenance"] = {
        "dataset_id": "owner_course_feedback_2026",
        "dataset_version": "1.0.0",
        "dataset_adapter": "custom_jsonl",
        "synthetic_fixture": False,
        "access_condition_class": "owner_collected_human_approved",
        "source_url": "",
        "license_statement": "internal course survey data collected by owner with consent",
        "access_evidence_reference": "data/manifests/owner_consent_and_review_2026.json",
    }
    payload["label_mapping_verification"] = {
        "method": "human_approved_canonical",
        "verification_reference": "data/manifests/human_annotator_approval.json",
    }
    code, summary = _gate_run(tmp_path, payload)
    assert code == 0, summary
    assert summary["overall"] == "pass"


def test_redteam_false_reject_five_level_public_and_neu_esc_and_holdout_pass(
    tmp_path: Path,
) -> None:
    """Licensed public, gated NEU-ESC + toxic exclude, and preflight/holdout test access pass."""
    # Licensed public 5-level CSV with grouped_validation and declared_holdout evaluating test.
    pub = _gate_base_uit_vsfc_payload()
    pub["dataset_provenance"] = {
        "dataset_id": "viet_students_five_level",
        "dataset_version": "2025.1",
        "dataset_adapter": "five_level_csv",
        "synthetic_fixture": False,
        "access_condition_class": "licensed_public",
        "source_url": "https://example.org/datasets/five-level",
        "license_statement": "Apache-2.0 open dataset license",
        "access_evidence_reference": "licenses/five_level_apache2.txt",
    }
    pub["label_mapping_verification"] = {
        "method": "ordinal_five_to_three",
        "verification_reference": "docs/phase2/label-semantics-evidence.md#five-level-ordinal",
    }
    pub["split_definition"]["split_policy"] = "grouped_validation"
    pub["phase1_integrity"]["baseline_result_scope"] = "declared_holdout"
    pub["phase1_integrity"]["baseline_evidence_reference"] = "reports/eval_holdout.json"
    pub["phase1_integrity"]["baseline_holdout_fingerprint"] = _RT_TEST_FP
    code, summary = _gate_run(tmp_path, pub)
    assert code == 0, summary
    assert summary["overall"] == "pass"

    # Gated NEU-ESC click-accept with auditable toxic_policy='exclude'.
    neu = _gate_base_uit_vsfc_payload()
    neu["dataset_provenance"] = {
        "dataset_id": "neu_esc_2024",
        "dataset_version": "2024-09",
        "dataset_adapter": "neu_esc",
        "synthetic_fixture": False,
        "access_condition_class": "gated_click_accept",
        "source_url": "https://huggingface.co/datasets/example/neu-esc",
        "license_statement": "gated research-only terms accepted on dataset card",
        "access_evidence_reference": "HF gated click-accept 2026-11-10 under owner profile",
    }
    neu["label_mapping_verification"] = {
        "method": "neu_esc_named_labels",
        "verification_reference": "docs/phase2/label-semantics-evidence.md#neu-esc",
        "toxic_policy": "exclude",
        "toxic_justification": "Toxic is separate moderation axis outside 3-class sentiment",
        "toxic_excluded_count": 37,
    }
    code, summary = _gate_run(tmp_path, neu)
    assert code == 0, summary
    assert summary["overall"] == "pass"
