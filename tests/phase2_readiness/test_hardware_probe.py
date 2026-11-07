"""Tests for the offline hardware probe.

All nvidia-smi interaction is mocked (fake CSV output, missing binary, failing
execution); no real subprocess is launched and no network access is needed.
"""

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "phase2_readiness" / "hardware_probe.py"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("vfi_hardware_probe", TOOL_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load hardware probe module from {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hardware_probe = _load_tool()

FAKE_SMI_CSV = (
    "NVIDIA GeForce RTX 3060 Laptop GPU, 6144, 617.42, 8.6\n"
    "NVIDIA GeForce RTX 4090, 24564, 550.54, 8.9\n"
)


def _fake_which(_binary: str) -> str:
    return "C:/fake-tools/nvidia-smi.exe"


def _fake_missing_which(_binary: str) -> str | None:
    return None


def _fake_runner(_cmd: list[str], _timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(_cmd, 0, stdout=FAKE_SMI_CSV, stderr="")


def _failing_runner(_cmd: list[str], _timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(_cmd, 1, stdout="", stderr="driver exploded")


def _timeout_runner(_cmd: list[str], _timeout: int) -> subprocess.CompletedProcess[str]:
    raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=_timeout)


def _garbage_runner(_cmd: list[str], _timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(_cmd, 0, stdout="this is not csv at all\n", stderr="")


def test_parse_gpu_line_valid() -> None:
    gpu = hardware_probe._parse_gpu_line("NVIDIA GeForce RTX 3060 Laptop GPU, 6144, 617.42, 8.6")
    assert gpu == {
        "name": "NVIDIA GeForce RTX 3060 Laptop GPU",
        "vram_mib": 6144,
        "driver_version": "617.42",
        "compute_capability": "8.6",
    }


def test_parse_gpu_line_garbage_returns_none() -> None:
    assert hardware_probe._parse_gpu_line("not csv") is None
    assert hardware_probe._parse_gpu_line("Name, notanumber, 1.0, 8.6") is None
    assert hardware_probe._parse_gpu_line("") is None


def test_query_nvidia_smi_missing_binary_is_not_an_error() -> None:
    result = hardware_probe.query_nvidia_smi(which=_fake_missing_which)
    assert result["nvidia_smi_available"] is False
    assert result["gpus"] == []
    assert result["query_error"] is None
    assert result["reason"] is not None
    assert "not found" in result["reason"]


def test_query_nvidia_smi_parses_fake_csv() -> None:
    result = hardware_probe.query_nvidia_smi(which=_fake_which, runner=_fake_runner)
    assert result["nvidia_smi_available"] is True
    assert result["query_error"] is None
    assert [gpu["name"] for gpu in result["gpus"]] == [
        "NVIDIA GeForce RTX 3060 Laptop GPU",
        "NVIDIA GeForce RTX 4090",
    ]
    assert result["gpus"][0]["vram_mib"] == 6144
    assert result["raw_output"] == FAKE_SMI_CSV


def test_query_nvidia_smi_nonzero_exit_records_error() -> None:
    result = hardware_probe.query_nvidia_smi(which=_fake_which, runner=_failing_runner)
    assert result["nvidia_smi_available"] is True
    assert result["gpus"] == []
    assert result["query_error"] is not None
    assert "driver exploded" in result["query_error"]


def test_query_nvidia_smi_timeout_records_error() -> None:
    result = hardware_probe.query_nvidia_smi(which=_fake_which, runner=_timeout_runner)
    assert result["nvidia_smi_available"] is True
    assert result["gpus"] == []
    assert result["query_error"] is not None
    assert "failed" in result["query_error"]


def test_query_nvidia_smi_garbage_output_yields_empty_gpus() -> None:
    result = hardware_probe.query_nvidia_smi(which=_fake_which, runner=_garbage_runner)
    assert result["nvidia_smi_available"] is True
    assert result["gpus"] == []
    assert result["query_error"] is None


def test_summary_is_valid_json_and_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("getpass.getuser", lambda: "secret_user")
    monkeypatch.setattr("platform.node", lambda: "secret-host")
    summary, _details = hardware_probe.collect_report(which=_fake_which, runner=_fake_runner)
    text = json.dumps(summary)
    assert "secret_user" not in text
    assert "secret-host" not in text
    assert hardware_probe.find_sensitive_matches(text) == []
    parsed = json.loads(text)
    assert parsed["cpu"]["cores_logical"] >= 1
    assert "host" not in parsed


def test_assert_sanitized_raises_on_username(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("getpass.getuser", lambda: "leaky_user_42")
    with pytest.raises(hardware_probe.SanitizationError):
        hardware_probe.assert_sanitized("report for leaky_user_42 with hardware facts")


def test_assert_sanitized_raises_on_home_path() -> None:
    home = Path.home()
    with pytest.raises(hardware_probe.SanitizationError):
        hardware_probe.assert_sanitized(f"details saved under {home}\\reports")


def test_main_json_stdout_is_valid_and_sanitized(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("getpass.getuser", lambda: "secret_user")
    monkeypatch.setattr("platform.node", lambda: "secret-host")
    exit_code = hardware_probe.main(["--json"], which=_fake_which, runner=_fake_runner)
    captured = capsys.readouterr()
    assert exit_code == 0
    parsed = json.loads(captured.out)
    assert parsed["gpu"]["gpus"][0]["name"] == "NVIDIA GeForce RTX 3060 Laptop GPU"
    assert "secret_user" not in captured.out
    assert "secret-host" not in captured.out


def test_main_text_stdout_contains_facts_and_no_secrets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("getpass.getuser", lambda: "secret_user")
    monkeypatch.setattr("platform.node", lambda: "secret-host")
    exit_code = hardware_probe.main([], which=_fake_which, runner=_fake_runner)
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "RTX 3060 Laptop GPU" in captured.out
    assert "physical_cores" in captured.out
    assert "secret_user" not in captured.out
    assert "secret-host" not in captured.out


def test_main_missing_gpu_is_exit_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    exit_code = hardware_probe.main([], which=_fake_missing_which)
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "not available" in captured.out
    assert "not found" in captured.out


def test_main_output_inside_cwd_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    (tmp_path / "repo").mkdir()
    monkeypatch.chdir(tmp_path / "repo")
    destination = tmp_path / "repo" / "details.json"
    exit_code = hardware_probe.main(
        ["--output", str(destination)], which=_fake_which, runner=_fake_runner
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert not destination.exists()
    assert "refusing" in captured.err


def test_main_output_outside_cwd_writes_details(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    (tmp_path / "repo").mkdir()
    (tmp_path / "outside").mkdir()
    monkeypatch.chdir(tmp_path / "repo")
    destination = tmp_path / "outside" / "details.json"
    exit_code = hardware_probe.main(
        ["--output", str(destination)], which=_fake_which, runner=_fake_runner
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert destination.exists()
    details = json.loads(destination.read_text(encoding="utf-8"))
    assert "host" in details
    assert details["host"]["username"]
    assert details["gpu"]["raw_output"] == FAKE_SMI_CSV
    assert "wrote" in captured.err or "written" in captured.err


def test_main_without_output_writes_temp_details(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    exit_code = hardware_probe.main([], which=_fake_which, runner=_fake_runner)
    captured = capsys.readouterr()
    assert exit_code == 0
    marker = "temp file: "
    assert marker in captured.err
    temp_path = Path(captured.err.split(marker, 1)[1].strip())
    try:
        assert temp_path.exists()
        details = json.loads(temp_path.read_text(encoding="utf-8"))
        assert "host" in details
    finally:
        temp_path.unlink(missing_ok=True)


def test_collect_disks_reports_fixed_drives() -> None:
    disks = hardware_probe.collect_disks()
    assert len(disks) >= 1
    for disk in disks:
        assert disk["total_gb"] > 0
        assert disk["free_gb"] >= 0
        assert disk["root"].endswith("\\") or disk["root"] == "/"


def test_physical_core_count_never_exceeds_logical() -> None:
    physical = hardware_probe._windows_physical_cores()
    if physical is not None:
        logical = hardware_probe.collect_cpu()["cores_logical"]
        assert 1 <= physical <= logical
