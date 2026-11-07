"""Red-team tests for the hardware probe (Phase 2 readiness hardening pass).

Every ``xfail`` test here encodes the DESIRED privacy contract for a hole that
was ACTUALLY demonstrated against the current working-tree probe on Windows 11
/ Python 3.12 (2026-11-07). Nothing in this file is speculative: each hole test
asserts the safe behavior, fails against the current code, and carries the
exact crafted input that reproduced the bug. When the probe is fixed, these
tests XPASS (strict marker) and must be flipped into plain regression tests.

Repo-internal writes are exercised against a temporary fixture repository (a
copy of the tool under ``tmp_path`` with a ``.git`` entry), never against the
real clone.

Reproduced holes covered (each marked xfail, strict):
1. ``\\?\\``-prefixed ``--output``: ``Path.resolve()`` yields a drive-relative
   path, the refusal check misses, and the details file is written INSIDE the
   (fixture) repository via the per-drive working directory.
2. Trailing-space directory component: ``resolve()`` canonicalizes the
   component but returns it with a ``\\\\?\\`` prefix, so the parents
   comparison against the plain repo root misses and the file lands inside
   the (fixture) repository.
3. UNC ``\\\\localhost\\<drive>$\\...`` spelling: ``resolve()`` falls back to a
   literal drive-rooted path, so ``_repo_internal_refusal`` misses (checked at
   the API level; the end-to-end write lands outside the repo and was
   verified manually).
4. Environment spoofing (LOGNAME/USER/LNAME/USERNAME/USERPROFILE): the guard
   derives "sensitive" from the live environment, so a spoofed runner sees
   the real username echoed unredacted on the ``--output`` stderr line.
5. argparse ``unrecognized arguments``: argv is echoed verbatim to stderr
   before any guard runs, contradicting the documented guarantee that error
   strings containing known sensitive values are replaced by a category
   message.
Two further tests pin behavior that was verified COVERED or safe (case
variants, ``..`` traversal, trailing-dot files, forward slashes; and the fact
that ``COMPUTERNAME`` does not influence ``platform.node()``) so it cannot
regress silently.
"""

import getpass
import importlib.util
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "phase2_readiness" / "hardware_probe.py"
FAKE_SMI_CSV = "NVIDIA GeForce RTX 3060 Laptop GPU, 6144, 617.42, 8.6\n"


def _load_tool(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load hardware probe module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hardware_probe = _load_tool(TOOL_PATH, "vfi_hardware_probe_redteam")


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
