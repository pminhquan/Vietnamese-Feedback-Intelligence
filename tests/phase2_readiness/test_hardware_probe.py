"""Tests for the offline hardware probe.

All nvidia-smi interaction is mocked (fake CSV output, missing binary, failing
execution); no real subprocess is launched and no network access is needed.
Default execution must create no files anywhere; detailed results exist only
through an explicit ``--output`` pointing outside the repository.
"""

import getpass
import importlib.util
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest

TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "phase2_readiness" / "hardware_probe.py"
REPO_ROOT = TOOL_PATH.parents[2]


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


def _oserror_runner(_cmd: list[str], _timeout: int) -> subprocess.CompletedProcess[str]:
    raise OSError(13, "Permission denied: C:/Users/leaky_user_42/restricted/nvidia-smi.exe")


def _fail_if_mkstemp_called(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("tempfile.mkstemp was called; default runs must create no files")


def _probe_temp_markers() -> set[Path]:
    """Existing vfi-hardware-probe-* files in the OS temp dir (old-version residue)."""
    return set(Path(tempfile.gettempdir()).glob("vfi-hardware-probe-*"))


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


def test_query_nvidia_smi_oserror_is_rendered_without_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("getpass.getuser", lambda: "leaky_user_42")
    result = hardware_probe.query_nvidia_smi(which=_fake_which, runner=_oserror_runner)
    assert result["nvidia_smi_available"] is True
    assert result["gpus"] == []
    assert result["query_error"] is not None
    # OSError(13, ...) resolves to the PermissionError subclass on all platforms.
    assert "PermissionError" in result["query_error"]
    assert "errno 13" in result["query_error"]
    assert "leaky_user_42" not in result["query_error"]
    assert "C:/Users" not in result["query_error"]
    assert "Permission denied" not in result["query_error"]


def test_query_nvidia_smi_stderr_with_sensitive_value_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("getpass.getuser", lambda: "leaky_user_42")

    def _sensitive_stderr_runner(
        _cmd: list[str], _timeout: int
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(_cmd, 1, stdout="", stderr="called by leaky_user_42")

    result = hardware_probe.query_nvidia_smi(which=_fake_which, runner=_sensitive_stderr_runner)
    assert result["query_error"] is not None
    assert "[redacted:" in result["query_error"]
    assert "leaky_user_42" not in result["query_error"]
    assert "username" in result["query_error"]


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


def test_main_missing_gpu_is_exit_zero_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    before = _probe_temp_markers()
    monkeypatch.setattr(tempfile, "mkstemp", _fail_if_mkstemp_called)
    exit_code = hardware_probe.main([], which=_fake_missing_which)
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "not available" in captured.out
    assert "not found" in captured.out
    assert _probe_temp_markers() == before
    assert captured.err == ""


def test_main_garbage_gpu_output_is_exit_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    exit_code = hardware_probe.main([], which=_fake_which, runner=_garbage_runner)
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "no GPU parsed" in captured.out


def test_main_without_output_writes_no_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("getpass.getuser", lambda: "secret_user")
    monkeypatch.setattr("platform.node", lambda: "secret-host")
    before = _probe_temp_markers()
    monkeypatch.setattr(tempfile, "mkstemp", _fail_if_mkstemp_called)
    exit_code = hardware_probe.main([], which=_fake_which, runner=_fake_runner)
    captured = capsys.readouterr()
    assert exit_code == 0
    assert _probe_temp_markers() == before
    assert "secret_user" not in captured.out + captured.err
    assert "temp file" not in captured.err


def test_find_repo_root_walks_up_to_git_entry(tmp_path: Path) -> None:
    git_dir_repo = tmp_path / "repo-dir"
    (git_dir_repo / "tools" / "phase2_readiness").mkdir(parents=True)
    (git_dir_repo / ".git").mkdir()
    module_file = git_dir_repo / "tools" / "phase2_readiness" / "hardware_probe.py"
    assert hardware_probe._find_repo_root(module_file) == git_dir_repo

    git_file_repo = tmp_path / "repo-file"
    (git_file_repo / "tools" / "phase2_readiness").mkdir(parents=True)
    (git_file_repo / ".git").write_text("gitdir: ../ elsewhere\n", encoding="utf-8")
    module_file = git_file_repo / "tools" / "phase2_readiness" / "hardware_probe.py"
    assert hardware_probe._find_repo_root(module_file) == git_file_repo


def test_find_repo_root_falls_back_without_git(tmp_path: Path) -> None:
    start = tmp_path / "a" / "b" / "module.py"
    if any((parent / ".git").exists() for parent in (start, *start.parents)):
        pytest.skip("a parent directory contains .git; fallback path not reachable here")
    assert hardware_probe._find_repo_root(start) == start.parents[2]


def test_find_repo_root_detects_actual_clone() -> None:
    assert hardware_probe._find_repo_root(TOOL_PATH) == REPO_ROOT


def test_resolve_output_strips_extended_prefix(tmp_path: Path) -> None:
    r"""A ``\\?\``-prefixed argument must canonicalize to the same plain path.

    ``Path.resolve()`` keeps the extended prefix, so it is stripped before and
    after resolving; otherwise the refusal comparison sees ``\\?\D:\`` anchors
    where the repository root is plain ``D:\``.
    """
    plain = tmp_path / "out.json"
    extended = Path("\\\\?\\" + str(plain))
    assert hardware_probe._resolve_output(extended) == plain.resolve()


def test_output_extended_prefix_inside_repo_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    r"""Red team hole 1: a ``\\?\``-prefixed repo path is refused like the plain one."""
    monkeypatch.chdir(tmp_path)  # a CWD outside the repository
    destination = REPO_ROOT / "docs" / "probe-details-must-not-exist.json"
    exit_code = hardware_probe.main(
        ["--output", "\\\\?\\" + str(destination)], which=_fake_which, runner=_fake_runner
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert not destination.exists()
    assert "refusing" in captured.err


def test_output_trailing_space_component_inside_repo_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    r"""Red team hole 2: a trailing-space component is refused like the plain one.

    Windows canonicalizes ``docs \leak.json`` to a ``\\?\``-prefixed path
    during resolve(), so the refusal must compare the prefix-stripped spelling.
    """
    monkeypatch.chdir(tmp_path)  # a CWD outside the repository
    crafted = str(REPO_ROOT / "docs") + " \\probe-details-must-not-exist.json"
    exit_code = hardware_probe.main(["--output", crafted], which=_fake_which, runner=_fake_runner)
    captured = capsys.readouterr()
    assert exit_code == 1
    assert not (REPO_ROOT / "docs" / "probe-details-must-not-exist.json").exists()
    assert "refusing" in captured.err


def test_output_unc_repo_spelling_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    r"""Red team hole 3: a UNC spelling of a repo path is refused.

    ``Path.resolve()`` leaves UNC spellings unchanged on this platform, so the
    probe maps the local admin-share spelling to its drive-letter path first;
    the refusal then fires on the true (repo-internal) location.
    """
    monkeypatch.chdir(tmp_path)  # a CWD outside the repository
    tail = str(REPO_ROOT)[3:]  # repo path below the drive root
    crafted = (
        f"\\\\localhost\\{REPO_ROOT.drive[0]}$\\{tail}\\docs\\probe-details-must-not-exist.json"
    )
    exit_code = hardware_probe.main(["--output", crafted], which=_fake_which, runner=_fake_runner)
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "refusing" in captured.err
    assert not (REPO_ROOT / "docs" / "probe-details-must-not-exist.json").exists()


def test_output_foreign_unc_path_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    r"""Red team hole 3: UNC paths on servers other than this machine are refused outright."""
    for crafted in (
        r"\\some-host\some-share\details.json",
        r"\\?\UNC\some-host\some-share\details.json",
    ):
        exit_code = hardware_probe.main(
            ["--output", crafted], which=_fake_which, runner=_fake_runner
        )
        captured = capsys.readouterr()
        assert exit_code == 1, crafted
        assert "refusing" in captured.err, crafted
        assert "UNC" in captured.err, crafted


def test_resolve_output_maps_local_admin_share_spelling(tmp_path: Path) -> None:
    r"""A local admin-share spelling resolves to its drive-letter path.

    ``\\localhost\C$\tail`` names the local ``C:\tail``; canonicalizing it
    lets the repo refusal see the true location instead of a stray tree.
    """
    unc = f"\\\\localhost\\{tmp_path.drive[0]}$\\some\\repo\\leak.json"
    resolved = hardware_probe._resolve_output(Path(unc))
    assert str(resolved).lower() != unc.lower(), "UNC spelling must be canonicalized"
    assert resolved == Path(f"{tmp_path.drive}\\some\\repo\\leak.json").resolve()
    # Outside the repository, so the refusal does not fire for it.
    assert hardware_probe._repo_internal_refusal(resolved) is None


def test_resolve_output_maps_repo_internal_admin_share_to_refusal() -> None:
    r"""A UNC spelling of a repo path maps to the repo path and is refused."""
    tail = str(REPO_ROOT)[3:]  # repo path below the drive root
    unc = f"\\\\localhost\\{REPO_ROOT.drive[0]}$\\{tail}\\docs\\leak.json"
    resolved = hardware_probe._resolve_output(Path(unc))
    assert resolved == Path(REPO_ROOT / "docs" / "leak.json")
    assert hardware_probe._repo_internal_refusal(resolved) is not None


def test_repo_internal_refusal_rejects_drive_relative_paths() -> None:
    r"""A drive-relative destination (``C:relative\out.json``) is refused outright.

    Its location depends on the per-drive working directory, so no parents
    comparison against the repository root could be trusted.
    """
    refusal = hardware_probe._repo_internal_refusal(Path(r"C:relative-dir\out.json"))
    assert refusal is not None
    assert "drive-relative" in refusal


def test_main_output_inside_repo_is_refused_from_other_cwd(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)  # a CWD outside the repository
    destination = REPO_ROOT / "docs" / "probe-details-must-not-exist.json"
    exit_code = hardware_probe.main(
        ["--output", str(destination)], which=_fake_which, runner=_fake_runner
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert not destination.exists()
    assert "refusing" in captured.err


def test_main_output_inside_repo_is_refused_when_cwd_is_repo(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    destination = REPO_ROOT / "tools" / "probe-details-must-not-exist.json"
    exit_code = hardware_probe.main(
        ["--output", str(destination)], which=_fake_which, runner=_fake_runner
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert not destination.exists()
    assert "refusing" in captured.err


def test_main_output_outside_repo_writes_details(
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
    assert "written" in captured.err
    assert "RTX 3060 Laptop GPU" in captured.out


def test_main_output_creates_missing_parent_directories(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / "new" / "deep" / "details.json"
    exit_code = hardware_probe.main(
        ["--output", str(destination)], which=_fake_which, runner=_fake_runner
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert destination.exists()
    assert "written" in captured.err


def test_main_output_relative_path_resolves_against_cwd(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    exit_code = hardware_probe.main(
        ["--output", "probe-rel-out.json"], which=_fake_which, runner=_fake_runner
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert (tmp_path / "probe-rel-out.json").exists()
    assert "written" in captured.err


def test_output_path_with_username_is_written_but_not_echoed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr("getpass.getuser", lambda: "leaky_user_42")
    destination = tmp_path / "details-leaky_user_42.json"
    exit_code = hardware_probe.main(
        ["--output", str(destination)], which=_fake_which, runner=_fake_runner
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert destination.exists()
    details = json.loads(destination.read_text(encoding="utf-8"))
    assert details["host"]["username"] == "leaky_user_42"  # opt-in file holds details
    assert "leaky_user_42" not in captured.out + captured.err
    assert "[redacted:" in captured.err


def test_sensitive_values_gather_environment_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Identity env vars are redaction needles even when they carry forged values."""
    monkeypatch.setenv("USERNAME", "env_spoof_user_7")
    monkeypatch.setenv("USERPROFILE", "C:/Users/env_spoof_home_user_9")
    categories = dict(hardware_probe._sensitive_values())
    assert categories["env_username"] == "env_spoof_user_7"
    # A USERPROFILE that looks like a path also contributes its basename.
    assert categories["env_userprofile_basename"] == "env_spoof_home_user_9"


def test_sensitive_values_survive_environment_spoofing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spoofing every identity env var must not blind the guard to the real login.

    getpass.getuser() prefers LOGNAME/USER/LNAME/USERNAME and Path.home()
    follows USERPROFILE/HOME, so both are forgeable through the environment;
    os.getlogin() asks the OS for the logon-session name and ignores the
    environment on Windows.
    """
    real_login = os.getlogin()
    for var in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        monkeypatch.setenv(var, "innocent_user_1")
    monkeypatch.setenv("USERPROFILE", "C:/Users/innocent_home")
    values = [value for _category, value in hardware_probe._sensitive_values()]
    assert "innocent_user_1" in values, "forged env values become redaction needles too"
    assert real_login in values, "OS-reported login name must survive env spoofing"


def test_redact_sensitive_catches_spoofed_environment_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("USERPROFILE", "C:/Users/spoofed_home_user_9")
    redacted = hardware_probe._redact_sensitive("saved under spoofed_home_user_9")
    assert "[redacted:" in redacted
    assert "spoofed_home_user_9" not in redacted


def test_unwritable_output_reports_error_without_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr("getpass.getuser", lambda: "leaky_user_42")
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory", encoding="utf-8")
    destination = blocker / "details.json"
    exit_code = hardware_probe.main(
        ["--output", str(destination)], which=_fake_which, runner=_fake_runner
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "cannot create parent directory" in captured.err
    assert str(tmp_path) not in captured.err
    assert "leaky_user_42" not in captured.out + captured.err


def test_symlink_into_repo_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    link = tmp_path / "link-into-repo.json"
    target = REPO_ROOT / "docs" / "probe-details-symlink-test.json"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform/account")
    try:
        exit_code = hardware_probe.main(
            ["--output", str(link)], which=_fake_which, runner=_fake_runner
        )
        captured = capsys.readouterr()
        assert exit_code == 1
        assert not target.exists()
        assert "refusing" in captured.err
    finally:
        link.unlink(missing_ok=True)
        target.unlink(missing_ok=True)


def test_repo_internal_symlink_resolving_outside_is_allowed(tmp_path: Path) -> None:
    fake_repo = tmp_path / "fake-repo"
    fake_repo.mkdir()
    (fake_repo / ".git").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = fake_repo / "link.json"
    try:
        link.symlink_to(outside / "details.json")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform/account")
    try:
        exit_code = hardware_probe._write_details_to_output({"host": {"k": "v"}}, link)
        assert exit_code == 0
        assert (outside / "details.json").exists()
    finally:
        link.unlink(missing_ok=True)


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


def test_argument_parser_redacts_usage_error_values(capsys: pytest.CaptureFixture[str]) -> None:
    r"""Red team hole 5: argparse usage errors redact caller-supplied argv values.

    The fixed template (``unrecognized arguments: ...``) stays recognizable;
    only sensitive values inside it are replaced, and argparse's exit code 2
    is preserved.
    """
    real_user = getpass.getuser()
    parser = hardware_probe._SanitizingArgumentParser(prog="hardware_probe")
    parser.add_argument("--json", action="store_true")
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--bogusflag", f"leak C:/Users/{real_user}/secret.json"])
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "unrecognized arguments" in captured.err
    assert real_user not in captured.err
    assert "[redacted:" in captured.err
    assert "usage:" in captured.err
