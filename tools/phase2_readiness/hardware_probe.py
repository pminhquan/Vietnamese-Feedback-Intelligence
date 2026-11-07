"""Offline, stdlib-only hardware probe for VFI Phase 2 readiness.

Collects OS, CPU, RAM, GPU, disk, Python and library facts about the current
machine without network access, third-party dependencies, or model weights.

Privacy contract:
- stdout receives a SANITIZED summary only (no username, hostname, serials,
  MACs, tokens, or personal paths), enforced by a fail-closed guard.
- Detailed results (including hostname, username, Python executable path and
  raw nvidia-smi output) are written ONLY to an explicit ``--output PATH``
  that lies outside the current directory tree, or to an OS temp file.

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
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

NVIDIA_SMI_BINARY = "nvidia-smi"
NVIDIA_SMI_QUERY_FIELDS = "name,memory.total,driver_version,compute_cap"
NVIDIA_SMI_TIMEOUT_SECONDS = 15
GIB = 1024**3
SUMMARY_SCHEMA_VERSION = 1
INTERESTING_LIBRARIES = ("numpy", "sklearn", "torch", "transformers")

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
            "query_error": f"nvidia-smi execution failed: {exc}",
            "reason": None,
            "raw_output": None,
        }
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        detail = detail or f"exit {completed.returncode}"
        return {
            "nvidia_smi_available": True,
            "gpus": [],
            "query_error": f"nvidia-smi failed: {detail}",
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


def _sensitive_values() -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    try:
        values.append(("username", getpass.getuser()))
    except Exception:
        pass
    node = platform.node()
    if node:
        values.append(("hostname", node))
    home = Path.home()
    values.append(("home_path", str(home)))
    if home.name:
        values.append(("home_dir_name", home.name))
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
    dest = output.expanduser().resolve()
    cwd = Path.cwd().resolve()
    if dest == cwd or cwd in dest.parents:
        print(
            "hardware_probe: refusing --output inside the current directory tree; "
            "detailed results may contain personal values. Choose a path outside it.",
            file=sys.stderr,
        )
        return 1
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(details, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"hardware_probe: cannot write detailed results: {exc}", file=sys.stderr)
        return 1
    print(f"hardware_probe: detailed results written to {dest}", file=sys.stderr)
    return 0


def _write_details_to_temp(details: dict[str, Any]) -> int:
    try:
        handle, temp_path = tempfile.mkstemp(prefix="vfi-hardware-probe-", suffix=".json")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(details, stream, indent=2)
    except OSError as exc:
        print(f"hardware_probe: cannot write detailed results to temp file: {exc}", file=sys.stderr)
        return 1
    print(f"hardware_probe: detailed results written to temp file: {temp_path}", file=sys.stderr)
    return 0


def main(
    argv: list[str] | None = None,
    which: WhichFn | None = None,
    runner: RunnerFn | None = None,
) -> int:
    parser = argparse.ArgumentParser(
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
        dest = args.output.expanduser().resolve()
        cwd = Path.cwd().resolve()
        if dest == cwd or cwd in dest.parents:
            print(
                "hardware_probe: refusing --output inside the current directory tree; "
                "detailed results may contain personal values. Choose a path outside it.",
                file=sys.stderr,
            )
            return 1
    try:
        summary, details = collect_report(which=which, runner=runner)
        text = json.dumps(summary, indent=2) if args.json else render_text_summary(summary)
        assert_sanitized(text)
    except SanitizationError as exc:
        print(f"hardware_probe: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"hardware_probe: probe error: {exc!r}", file=sys.stderr)
        return 1
    print(text)
    if args.output is not None:
        return _write_details_to_output(details, args.output)
    return _write_details_to_temp(details)


if __name__ == "__main__":
    raise SystemExit(main())
