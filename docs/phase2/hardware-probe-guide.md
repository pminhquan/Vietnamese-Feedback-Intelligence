# Hardware Probe Guide

How to run, read, and share the output of `tools/phase2_readiness/hardware_probe.py` — the offline, stdlib-only hardware probe for VFI Phase 2 readiness.

## What it does

- Collects OS, CPU, RAM, GPU, disk, Python, and ML-library facts about the current machine.
- Runs entirely offline, uses only the Python standard library, downloads nothing, installs nothing, and never touches model weights.
- Queries `nvidia-smi` via subprocess only if the binary is present; a machine without NVIDIA tooling is reported cleanly, not as an error.
- Prints a **sanitized** summary to stdout and writes **detailed** (unsanitized) results only where you explicitly allow it (see Privacy below).
- Exit codes: `0` = probe succeeded (a missing GPU is not an error); `1` = probe error (collection failure, sanitization guard triggered, unsafe `--output` path, or unwritable destination).

## How to run

From any clone of the repository (Python 3.12+; no venv or dependencies required):

```bash
# 1. Human-readable sanitized summary on stdout
py -3.12 tools/phase2_readiness/hardware_probe.py

# 2. Machine-readable sanitized summary on stdout
py -3.12 tools/phase2_readiness/hardware_probe.py --json

# 3. Sanitized summary plus a detailed JSON file written OUTSIDE the repo
py -3.12 tools/phase2_readiness/hardware_probe.py --output "D:/temp/vfi-probe-details.json"
```

If `--output` is omitted, detailed results are written to an OS temp file and its path is printed to **stderr** (never to stdout, so the sanitized stdout stays paste-safe).

## Reading the output

Example of the sanitized stdout summary (fields are `key: value`, one fact per line; GPU devices are indented):

```
VFI Phase 2 hardware probe (sanitized summary)
generated_at_utc: 2026-11-07T03:12:53+00:00
os: Windows 11 (10.0.26300, AMD64)
python: CPython 3.12.10
python_libraries: numpy=no sklearn=no torch=no transformers=no
cpu: Intel(R) Core(TM) i5-10500H CPU @ 2.50GHz | physical_cores: 6 | logical_cores: 12
ram: 32.00 GiB installed
gpu: nvidia-smi available, 1 device(s)
  - NVIDIA GeForce RTX 3060 Laptop GPU | vram_mib: 6144 | driver: 617.42 | compute_capability: 8.6
disks:
  - C:\ total 475.9 GiB, free 391.5 GiB
  - D:\ total 931.5 GiB, free 521.5 GiB
```

Field notes:

- `python_libraries` shows whether `numpy`, `sklearn`, `torch`, `transformers` are importable **without importing them** (module existence check only — heavy packages are not executed).
- `cpu` physical cores come from the Windows API (`GetLogicalProcessorInformation`); on non-Windows systems the field is `unknown` and logical cores come from `os.cpu_count()`.
- `ram` is physically installed memory from `GetPhysicallyInstalledSystemMemory` on Windows.
- GPU `vram_mib` comes from `nvidia-smi` and is authoritative; Windows WMI `AdapterRAM` is not (it is a 32-bit field and truncates GPUs above 4 GiB).
- `disks` lists fixed local drives only (drive letters, totals/free in GiB; no volume labels, serials, or paths).

## Privacy model (read before sharing)

- **stdout is sanitized by construction**: it is built from a whitelist of hardware facts. It never includes your username, hostname, MAC address, serial numbers, tokens, or personal paths (only bare drive letters like `C:\`).
- A fail-closed guard runs before anything is printed: if a sensitive value (username, hostname, home directory path/name) would appear in stdout anyway, the probe exits `1` with an error naming the *category* (never the value). If that ever triggers, do not disable it — report it as a bug.
- **Detailed results are unsanitized** and additionally contain: hostname, username, the Python executable path, the resolved `nvidia-smi` path, and raw `nvidia-smi` output. They are written only to:
  - an explicit `--output PATH`, **or**
  - an OS temp file (path on stderr).
- `--output` paths that resolve **inside the current directory tree** are refused (exit `1`). The probe assumes you run it from the repository root, so this prevents detailed results ever landing in the repo.
- Temp files live in your user temp directory (a path that contains your username) and are never deleted automatically — delete them after use if you want no local trace.

### Sharing a sanitized summary

Safe: run without `--output` and copy the stdout text (or `--json` output) into an issue, document, or chat.

Not safe: pasting the temp file or a `--output` file (contains username/hostname), or pasting stderr if you do not want the temp path visible.

```bash
# Share this file's contents:
py -3.12 tools/phase2_readiness/hardware_probe.py --json > "D:/temp/vfi-probe-summary.json"
# Do NOT share: the temp/detailed file, or the stderr text.
```

## Tests

The probe is covered by unit tests that mock all subprocess interaction (fake `nvidia-smi` CSV, missing binary, failing execution), assert JSON validity, and assert that no username/hostname/home-path leaks into sanitized output:

```bash
# from the repository root, in the dev venv (pytest + ruff in [dev] extras)
python -m pytest tests/phase2_readiness
python -m ruff check tools/phase2_readiness tests/phase2_readiness
```

## Limitations

- Windows-first: physical core count and installed RAM use Windows APIs; on other OSes those fields fall back to portable sources or `unknown`.
- GPU facts require NVIDIA drivers (`nvidia-smi`) to be installed; AMD/Intel discrete GPUs are not queried (only the integrated controller name may appear via OS APIs, and `Win32_VideoController.AdapterRAM` is intentionally not used for VRAM because it truncates above 4 GiB).
- The probe reports library *presence*, not versions — importing torch/transformers would be slow and is deliberately avoided; versions belong in the actual Phase 2 environment once it exists.
