r"""Offline, stdlib-only hardware probe for VFI Phase 2 readiness.

Collects OS, CPU, RAM, GPU, disk, Python and library facts about the current
machine without network access, third-party dependencies, or model weights.

Privacy contract:
- Default execution creates NO files of any kind; stdout receives a SANITIZED
  summary only (no username, hostname, serials, MACs, tokens, or personal
  paths), enforced by a fail-closed guard.
- Detailed results (including hostname, username, Python executable path and
  raw nvidia-smi output) are written ONLY when an explicit ``--output PATH``
  is passed. Destinations that resolve (symlinks followed) inside this
  repository are refused with exit code 1 and nothing is written. The
  refusal compares a canonicalized spelling: Windows extended-length
  prefixes (``\\?\``) are stripped before and after resolving, which also
  covers spellings that Windows canonicalizes while re-adding the prefix (a
  ``\\?\``-prefixed argument, or a directory component with a trailing
  space); a UNC spelling of a local admin share (``\\localhost\<drive>$\...``,
  also via 127.0.0.1 or this machine's hostname) is mapped to its true
  drive-letter path so the refusal applies to the real location, and every
  other UNC/network device path is refused outright. The repository root is
  located by walking up from this module's file to the enclosing ``.git``
  entry, independent of the current working directory.
- Error messages never contain OS exception text (Windows OSErrors embed
  absolute paths); they carry only the exception class, the errno, and, when
  it passes the guard below, the user-supplied destination path. Any error
  string that would contain a known sensitive value is replaced by a generic
  category message. Argparse usage errors redact caller-supplied values in
  place (keeping the fixed message template) and keep argparse's exit
  code 2.
- Limits of the sanitization guard: it matches identity candidates gathered
  from the OS (``getpass.getuser()``, ``os.getlogin()``, ``platform.node()``,
  ``Path.home()``) and from the process environment (``USERNAME``,
  ``LOGNAME``, ``USER``, ``LNAME``, ``USERPROFILE``, ``HOME``, plus their
  path basenames), so casually spoofing those variables cannot blind it to
  the real login name. It cannot detect arbitrary personal strings (other
  accounts' names, other machines' hostnames, tokens, serial numbers),
  values shorter than 3 characters, or an identity changed below the
  environment. It is a best-effort seatbelt for this tool's own output, not
  a general-purpose secret scanner.

Exit codes: 0 = probe succeeded (a missing GPU is NOT an error),
1 = probe error (collection failure, sanitization violation, unsafe output
path, or unwritable destination).
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import getpass
import importlib.util
import json
import os
import platform
import shutil
import string
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

NVIDIA_SMI_BINARY = "nvidia-smi"
NVIDIA_SMI_QUERY_FIELDS = "name,memory.total,driver_version,compute_cap"
NVIDIA_SMI_TIMEOUT_SECONDS = 15
GIB = 1024**3
SUMMARY_SCHEMA_VERSION = 1
INTERESTING_LIBRARIES = ("numpy", "sklearn", "torch", "transformers")
# Identity-bearing environment variables the sanitization guard also matches
# on (see _sensitive_values): spoofing them must not blind the guard.
ENV_IDENTITY_VARS = ("USERNAME", "LOGNAME", "USER", "LNAME", "USERPROFILE", "HOME")
# Windows extended-length path prefixes (device namespace).
EXTENDED_PATH_PREFIX = "\\\\?\\"
EXTENDED_UNC_PATH_PREFIX = "\\\\?\\UNC\\"

WhichFn = Callable[[str], str | None]
RunnerFn = Callable[..., subprocess.CompletedProcess[str]]


class SanitizationError(Exception):
    """Raised when probe output would leak a sensitive value."""


def _default_runner(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def _parse_gpu_line(line: str) -> dict[str, Any] | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 4 or not parts[0]:
        return None
    try:
        vram_mib = int(float(parts[1]))
    except ValueError:
        return None
    return {
        "name": parts[0],
        "vram_mib": vram_mib,
        "driver_version": parts[2],
        "compute_capability": parts[3],
    }


def query_nvidia_smi(
    which: WhichFn | None = None,
    runner: RunnerFn | None = None,
) -> dict[str, Any]:
    """Query NVIDIA GPUs via nvidia-smi. Absence of the binary is not an error.

    Returns a dict with keys: nvidia_smi_available, gpus, query_error, reason,
    raw_output. ``which`` and ``runner`` are injectable for tests.
    """
    which = which or shutil.which
    runner = runner or _default_runner
    resolved = which(NVIDIA_SMI_BINARY)
    if resolved is None:
        return {
            "nvidia_smi_available": False,
            "gpus": [],
            "query_error": None,
            "reason": f"{NVIDIA_SMI_BINARY} not found on PATH; reporting no NVIDIA GPU",
            "raw_output": None,
        }
    cmd = [resolved, f"--query-gpu={NVIDIA_SMI_QUERY_FIELDS}", "--format=csv,noheader,nounits"]
    try:
        completed = runner(cmd, NVIDIA_SMI_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "nvidia_smi_available": True,
            "gpus": [],
            "query_error": (
                f"nvidia-smi execution failed: {_redact_sensitive(_subprocess_error_text(exc))}"
            ),
            "reason": None,
            "raw_output": None,
        }
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        detail = detail or f"exit {completed.returncode}"
        return {
            "nvidia_smi_available": True,
            "gpus": [],
            "query_error": f"nvidia-smi failed: {_redact_sensitive(detail)}",
            "reason": None,
            "raw_output": None,
        }
    raw = completed.stdout or ""
    gpus = [gpu for line in raw.splitlines() if (gpu := _parse_gpu_line(line)) is not None]
    return {
        "nvidia_smi_available": True,
        "gpus": gpus,
        "query_error": None,
        "reason": None,
        "raw_output": raw,
    }


def _registry_cpu_name() -> str | None:
    if sys.platform != "win32":
        return None
    try:
        import winreg

        key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        return str(value).strip() or None
    except OSError:
        return None


def _windows_physical_cores() -> int | None:
    if sys.platform != "win32":
        return None
    try:
        kernel32 = ctypes.windll.kernel32

        class _CacheDescriptor(ctypes.Structure):
            _fields_ = [
                ("Level", ctypes.c_ubyte),
                ("Associativity", ctypes.c_ubyte),
                ("LineSize", ctypes.c_ushort),
                ("Size", ctypes.c_uint),
                ("Type", ctypes.c_int),
            ]

        class _InfoUnion(ctypes.Union):
            _fields_ = [
                ("Cache", _CacheDescriptor),
                ("ProcessorCore", ctypes.c_ulonglong),
                ("NumaNode", ctypes.c_uint),
            ]

        class _LogicalProcessorInfo(ctypes.Structure):
            _fields_ = [
                ("ProcessorMask", ctypes.c_size_t),
                ("Relationship", ctypes.c_int),
                ("Info", _InfoUnion),
            ]

        needed = ctypes.c_ulong(0)
        ok = kernel32.GetLogicalProcessorInformation(None, ctypes.byref(needed))
        if not ok and needed.value == 0:
            return None
        count = needed.value // ctypes.sizeof(_LogicalProcessorInfo)
        if count <= 0:
            return None
        buffer = (_LogicalProcessorInfo * count)()
        if not kernel32.GetLogicalProcessorInformation(buffer, ctypes.byref(needed)):
            return None
        cores = 0
        for entry in buffer:
            # RelationProcessorCore: exactly one entry per physical core.
            if entry.Relationship == 0:
                cores += 1
        return cores or None
    except Exception:
        return None


def collect_cpu() -> dict[str, Any]:
    logical = os.cpu_count() or 1
    physical = _windows_physical_cores()
    model = _registry_cpu_name() or platform.processor() or "unknown"
    return {"model": model, "cores_physical": physical, "cores_logical": logical}


def _installed_ram_bytes() -> int | None:
    if sys.platform == "win32":
        try:
            kilobytes = ctypes.c_ulonglong(0)
            if ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(kilobytes)):
                return int(kilobytes.value) * 1024
        except Exception:
            return None
        return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return int(page_size) * int(page_count)
    except (AttributeError, OSError, ValueError):
        return None


def collect_ram() -> dict[str, Any]:
    installed = _installed_ram_bytes()
    return {"installed_ram_gb": round(installed / GIB, 2) if installed else None}


def collect_os() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
    }


def collect_python() -> dict[str, Any]:
    libraries: dict[str, bool] = {}
    for name in INTERESTING_LIBRARIES:
        try:
            libraries[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            libraries[name] = False
    return {
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "libraries_detected": libraries,
    }


def _windows_drive_type(root: str) -> int | None:
    if sys.platform != "win32":
        return None
    try:
        return int(ctypes.windll.kernel32.GetDriveTypeW(ctypes.c_wchar_p(root)))
    except Exception:
        return None


def collect_disks() -> list[dict[str, Any]]:
    """List fixed local drives (letters only; no volume labels or serials)."""
    if sys.platform == "win32":
        candidates = [f"{letter}:\\" for letter in string.ascii_uppercase[2:]]
    else:
        candidates = [os.sep]
    disks: list[dict[str, Any]] = []
    for root in candidates:
        if not os.path.exists(root):
            continue
        drive_type = _windows_drive_type(root)
        if drive_type is not None and drive_type != 3:  # 3 = DRIVE_FIXED
            continue
        try:
            usage = shutil.disk_usage(root)
        except OSError:
            continue
        if usage.total <= 0:
            continue
        disks.append(
            {
                "root": root,
                "total_gb": round(usage.total / GIB, 1),
                "free_gb": round(usage.free / GIB, 1),
            }
        )
    return disks


def collect_report(
    which: WhichFn | None = None,
    runner: RunnerFn | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collect (sanitized_summary, detailed_results) with a single GPU query."""
    gpu_probe = query_nvidia_smi(which=which, runner=runner)
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "os": collect_os(),
        "python": collect_python(),
        "cpu": collect_cpu(),
        "ram": collect_ram(),
        "gpu": {
            "nvidia_smi_available": gpu_probe["nvidia_smi_available"],
            "query_error": gpu_probe["query_error"],
            "reason": gpu_probe["reason"],
            "gpus": [
                {
                    "name": gpu["name"],
                    "vram_mib": gpu["vram_mib"],
                    "driver_version": gpu["driver_version"],
                    "compute_capability": gpu["compute_capability"],
                }
                for gpu in gpu_probe["gpus"]
            ],
        },
        "disks": collect_disks(),
    }
    details = copy.deepcopy(summary)
    details["gpu"]["raw_output"] = gpu_probe["raw_output"]
    details["cpu"]["platform_processor"] = platform.processor()
    details["host"] = {
        "hostname": platform.node(),
        "username": getpass.getuser(),
        "python_executable": sys.executable,
    }
    return summary, details


def _path_basename(text: str) -> str:
    """Return the last path component of *text* ("" when it has none)."""
    return text.replace("\\", "/").rsplit("/", 1)[-1]


def _sensitive_values() -> list[tuple[str, str]]:
    r"""Gather candidate (category, value) pairs for the sanitization guard.

    Sources:
    - ``getpass.getuser()``: the documented username source. It prefers
      LOGNAME/USER/LNAME/USERNAME from the environment, so a manipulated
      environment can forge it.
    - ``os.getlogin()``: the OS-reported logon-session name; on Windows it
      ignores the environment, so it survives casual env spoofing.
    - ``platform.node()``: hostname.
    - ``Path.home()``: follows USERPROFILE/HOME, so it can be forged too.
    - every identity-bearing environment variable in
      :data:`ENV_IDENTITY_VARS`, plus its path basename when it looks like a
      path: forged values become redaction needles too, and when only some
      variables are forged the remaining genuine ones still match.

    Fail closed: extra candidates only widen redaction; a spurious match
    replaces output with a generic category message, never with the value.
    This defends against casual environment manipulation and against this
    tool echoing its own inputs; it is NOT a general-purpose secret scanner.
    """
    values: list[tuple[str, str]] = []
    for category, source in (("username", getpass.getuser), ("login_name", os.getlogin)):
        try:
            value = source().strip()
        except Exception:
            continue
        if value:
            values.append((category, value))
    node = platform.node()
    if node:
        values.append(("hostname", node))
    home = Path.home()
    values.append(("home_path", str(home)))
    if home.name:
        values.append(("home_dir_name", home.name))
    for var in ENV_IDENTITY_VARS:
        raw = (os.environ.get(var) or "").strip()
        if not raw:
            continue
        values.append((f"env_{var.lower()}", raw))
        base = _path_basename(raw)
        if base and base != raw:
            values.append((f"env_{var.lower()}_basename", base))
    return values


def find_sensitive_matches(payload: str) -> list[str]:
    lowered = payload.lower()
    matches = []
    for category, value in _sensitive_values():
        needle = value.strip().lower()
        if len(needle) >= 3 and needle in lowered:
            matches.append(category)
    return matches


def assert_sanitized(payload: str) -> None:
    matches = find_sensitive_matches(payload)
    if matches:
        msg = (
            "sanitization guard: output would contain sensitive value(s) "
            f"(categories: {', '.join(sorted(set(matches)))}); refusing to print"
        )
        raise SanitizationError(msg)


def _redact_sensitive(text: str) -> str:
    """Return *text*, or a generic category message when it holds sensitive values.

    Best-effort: only the known identity candidates (OS-reported username and
    login name, hostname, home path/name, and the identity environment
    variables) are detected; arbitrary personal strings are not (see the
    module docstring for the exact guarantee and its limits).
    """
    matches = find_sensitive_matches(text)
    if not matches:
        return text
    categories = ", ".join(sorted(set(matches)))
    return f"[redacted: potential sensitive value(s): {categories}]"


def _redact_sensitive_in_place(text: str) -> str:
    """Replace each sensitive value inside *text* with a category marker.

    Unlike :func:`_redact_sensitive` (which replaces the whole string), this
    keeps the surrounding text readable. It exists for messages built from a
    fixed template that must stay recognizable, such as argparse usage errors
    ("unrecognized arguments: ..."). Longer values are replaced first so a
    value that contains another one is fully covered.

    Fail closed: if any sensitive value still matches after the replacements
    (e.g. an echoed value whose case differs from the candidate), the whole
    text falls back to the generic category message of
    :func:`_redact_sensitive`.
    """
    redacted = text
    for category, value in sorted(_sensitive_values(), key=lambda pair: len(pair[1]), reverse=True):
        needle = value.strip()
        if len(needle) >= 3 and needle in redacted:
            redacted = redacted.replace(needle, f"[redacted:{category}]")
    if find_sensitive_matches(redacted):
        return _redact_sensitive(text)
    return redacted


def _subprocess_error_text(exc: BaseException) -> str:
    """Render a subprocess/OSError for summaries without exception text.

    Only the exception class and errno (when present) survive; the message is
    dropped because OS errors can embed absolute paths containing personal
    values. Callers pass the result through :func:`_redact_sensitive` as a
    second layer.
    """
    errno = getattr(exc, "errno", None)
    if errno is not None:
        return f"{type(exc).__name__} errno {errno}"
    return type(exc).__name__


def _errno_text(exc: OSError) -> str:
    """Render an OSError without its message, which may embed absolute paths."""
    if exc.errno is not None:
        return f"{type(exc).__name__} errno {exc.errno}"
    return type(exc).__name__


def _find_repo_root(start: Path) -> Path:
    """Return the repository root containing the module file *start*.

    Walks up from *start* looking for a ``.git`` entry (a file or a directory,
    so linked worktrees count too), independent of the current working
    directory. Falls back to ``start.parents[2]`` — the repository root of a
    normal checkout (``<root>/tools/phase2_readiness/<module>.py``) — when no
    ``.git`` entry is found above *start*.
    """
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    try:
        return start.parents[2]
    except IndexError:  # pragma: no cover - the module file always has 3+ parents
        return start


def _strip_extended_prefix(text: str) -> str:
    r"""Remove Windows extended-length prefixes from *text*.

    ``\\?\UNC\server\share\...`` becomes ``\\server\share\...`` and
    ``\\?\X:\...`` becomes ``X:\...``; the prefix match is case-insensitive
    and the remainder keeps its case. Anything else is returned unchanged.
    """
    lowered = text.lower()
    if lowered.startswith(EXTENDED_UNC_PATH_PREFIX.lower()):
        return "\\\\" + text[len(EXTENDED_UNC_PATH_PREFIX) :]
    if lowered.startswith(EXTENDED_PATH_PREFIX.lower()):
        return text[len(EXTENDED_PATH_PREFIX) :]
    return text


def _local_admin_share_path(text: str) -> str | None:
    r"""Map a local admin-share UNC spelling to its drive-letter path.

    ``\\localhost\C$\tail`` (also via ``127.0.0.1`` or this machine's own
    hostname, any drive letter, case-insensitive) names the local
    ``C:\tail``. Returning the drive-letter spelling lets the repo refusal
    see the true location instead of a stray tree. Every other UNC device
    path returns None and is refused outright by :func:`_repo_internal_refusal`.
    """
    server, sep, rest = text[2:].partition("\\")
    if not sep:
        return None
    share, sep, tail = rest.partition("\\")
    if not sep:
        tail = ""
    local_servers = {"localhost", "127.0.0.1"}
    node = platform.node()
    if node:
        local_servers.add(node.lower())
    if server.lower() not in local_servers:
        return None
    if len(share) == 2 and share[0] in string.ascii_letters and share[1] == "$":
        drive = f"{share[0].upper()}:\\"
        return drive + tail if tail else drive
    return None


def _resolve_output(output: Path) -> Path:
    r"""Expand ``~`` and canonicalize *output* for the refusal comparison.

    ``Path.resolve()`` alone is not enough on Windows (each behavior below was
    reproduced against Python 3.12):

    - ``resolve()`` keeps the ``\\?\`` extended prefix, both for arguments the
      caller already wrote that way and for components Windows canonicalizes
      (e.g. a trailing-space directory). Anchors like ``\\?\D:\`` never match
      the plain repository root, so the prefix is stripped before resolving
      and again from the result.
    - ``resolve()`` passes UNC spellings through unchanged (and its fallback
      for unreachable shares is not trustworthy), so UNC inputs are
      canonicalized explicitly: a local admin-share spelling is mapped to its
      drive-letter path and then resolved normally (so the repo refusal
      applies to the true location), and every other UNC device path becomes
      the ``\\?\UNC\`` device form, which the refusal rejects outright.
    """
    text = _strip_extended_prefix(str(output.expanduser()))
    if text.startswith("\\\\"):
        local = _local_admin_share_path(text)
        if local is None:
            return Path(EXTENDED_UNC_PATH_PREFIX + text[2:])
        text = local
    resolved = Path(text).resolve()
    return Path(_strip_extended_prefix(str(resolved)))


def _repo_internal_refusal(dest: Path) -> str | None:
    r"""Return a refusal message when *dest* is unsafe as an output location.

    The check runs on the canonicalized spelling (extended prefixes stripped,
    so plain drive-letter anchors are always compared) and refuses outright:
    UNC/network device paths other than local admin-share spellings, which
    :func:`_resolve_output` has already mapped to their drive-letter path (a
    remaining UNC share cannot be verified to be free of repository aliases),
    and drive-relative paths (their location depends on the per-drive working
    directory, so no parents comparison could be trusted). Otherwise it
    refuses paths inside the repository.
    """
    dest = Path(_strip_extended_prefix(str(dest)))
    if str(dest).startswith("\\\\"):
        return (
            "hardware_probe: refusing UNC/network device paths for --output; "
            "choose a fully qualified local drive-letter path outside the repository."
        )
    if dest.drive and not dest.root:
        return (
            "hardware_probe: refusing a drive-relative --output path whose location "
            "depends on the working directory; choose a fully qualified local "
            "drive-letter path outside the repository."
        )
    root = _find_repo_root(Path(__file__).resolve())
    if dest == root or root in dest.parents:
        return (
            "hardware_probe: refusing --output inside the VFI repository; "
            "detailed results may contain personal values. Choose a path outside it."
        )
    return None


def render_text_summary(summary: dict[str, Any]) -> str:
    lines = ["VFI Phase 2 hardware probe (sanitized summary)"]
    lines.append(f"generated_at_utc: {summary['generated_at_utc']}")
    os_info = summary["os"]
    os_text = (
        f"{os_info['system']} {os_info['release']} ({os_info['os_version']}, {os_info['machine']})"
    )
    lines.append(f"os: {os_text}")
    py_info = summary["python"]
    lines.append(f"python: {py_info['implementation']} {py_info['python_version']}")
    libs = py_info["libraries_detected"]
    lib_text = " ".join(
        f"{name}={'yes' if found else 'no'}" for name, found in sorted(libs.items())
    )
    lines.append(f"python_libraries: {lib_text}")
    cpu = summary["cpu"]
    physical = cpu["cores_physical"]
    physical_text = str(physical) if physical is not None else "unknown"
    lines.append(
        f"cpu: {cpu['model']} | physical_cores: {physical_text} "
        f"| logical_cores: {cpu['cores_logical']}"
    )
    ram = summary["ram"]
    installed_gb = ram["installed_ram_gb"]
    ram_text = f"{installed_gb:.2f} GiB installed" if installed_gb else "unknown"
    lines.append(f"ram: {ram_text}")
    gpu = summary["gpu"]
    if not gpu["nvidia_smi_available"]:
        lines.append(f"gpu: nvidia-smi not available ({gpu['reason']})")
    elif gpu["gpus"]:
        lines.append(f"gpu: nvidia-smi available, {len(gpu['gpus'])} device(s)")
        for device in gpu["gpus"]:
            device_text = (
                f"{device['name']} | vram_mib: {device['vram_mib']} "
                f"| driver: {device['driver_version']} "
                f"| compute_capability: {device['compute_capability']}"
            )
            lines.append(f"  - {device_text}")
    else:
        lines.append(f"gpu: nvidia-smi available but no GPU parsed (error: {gpu['query_error']})")
    lines.append("disks:")
    for disk in summary["disks"]:
        lines.append(f"  - {disk['root']} total {disk['total_gb']} GiB, free {disk['free_gb']} GiB")
    return "\n".join(lines)


def _write_details_to_output(details: dict[str, Any], output: Path) -> int:
    dest = _resolve_output(output)
    refusal = _repo_internal_refusal(dest)
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return 1
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(
            f"hardware_probe: cannot create parent directory for detailed results "
            f"({_errno_text(exc)})",
            file=sys.stderr,
        )
        return 1
    try:
        dest.write_text(json.dumps(details, indent=2), encoding="utf-8")
    except OSError as exc:
        print(
            f"hardware_probe: cannot write detailed results to "
            f"{_redact_sensitive(str(dest))} ({_errno_text(exc)})",
            file=sys.stderr,
        )
        return 1
    print(
        f"hardware_probe: detailed results written to {_redact_sensitive(str(dest))}",
        file=sys.stderr,
    )
    return 0


class _SanitizingArgumentParser(argparse.ArgumentParser):
    r"""ArgumentParser that redacts caller-supplied values in usage errors.

    argparse echoes argv verbatim in usage-error messages (e.g.
    ``unrecognized arguments: leak C:\Users\<user>\secret.json``), which would
    bypass the redaction contract every other error path follows. The message
    is passed through :func:`_redact_sensitive_in_place` before printing, so
    the fixed template stays recognizable while the caller-supplied values are
    redacted. The usage line and argparse's exit code 2 are unchanged.
    """

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {_redact_sensitive_in_place(message)}\n")


def main(
    argv: list[str] | None = None,
    which: WhichFn | None = None,
    runner: RunnerFn | None = None,
) -> int:
    parser = _SanitizingArgumentParser(
        prog="hardware_probe",
        description="Offline stdlib-only hardware probe for VFI Phase 2 readiness.",
    )
    parser.add_argument("--json", action="store_true", help="print the sanitized summary as JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write detailed (unsanitized) results to this JSON path, outside the repository",
    )
    args = parser.parse_args(argv)
    if args.output is not None:
        refusal = _repo_internal_refusal(_resolve_output(args.output))
        if refusal is not None:
            print(refusal, file=sys.stderr)
            return 1
    try:
        summary, details = collect_report(which=which, runner=runner)
        text = json.dumps(summary, indent=2) if args.json else render_text_summary(summary)
        assert_sanitized(text)
    except SanitizationError as exc:
        print(f"hardware_probe: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"hardware_probe: probe error: {_redact_sensitive(repr(exc))}", file=sys.stderr)
        return 1
    print(text)
    if args.output is not None:
        return _write_details_to_output(details, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
