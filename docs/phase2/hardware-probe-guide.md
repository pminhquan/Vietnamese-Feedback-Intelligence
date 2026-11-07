# Hardware Probe Guide

How to run, read, and share the output of `tools/phase2_readiness/hardware_probe.py` — the offline, stdlib-only hardware probe for VFI Phase 2 readiness.

## What it does

- Collects OS, CPU, RAM, GPU, disk, Python, and ML-library facts about the current machine.
- Runs entirely offline, uses only the Python standard library, downloads nothing, installs nothing, and never touches model weights.
- Queries `nvidia-smi` via subprocess only if the binary is present; a machine without NVIDIA tooling is reported cleanly, not as an error.
- Prints a **sanitized** summary to stdout and creates **no files**; detailed (unsanitized) results are written only when you explicitly pass `--output PATH` (see Privacy below).
- Exit codes: `0` = probe succeeded (a missing GPU is not an error); `1` = probe error (collection failure, sanitization guard triggered, unsafe `--output` path, or unwritable destination); `2` = usage error (bad command-line arguments).

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

If `--output` is omitted, the probe creates **no files anywhere** — stdout is the only output (stderr stays empty on success). Detailed results exist only when you explicitly pass `--output`.

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

- **stdout is sanitized by construction**: it is built from a whitelist of hardware facts. It never includes your username, hostname, MAC address, serial numbers, tokens, or personal paths (only bare drive letters like `C:\`). Two honest caveats: the guard never matches values shorter than 3 characters, and it recognizes only the identity candidates listed under "Limits of the sanitization guard" below — it is a seatbelt for this tool's own output, not a general secret scanner.
- A fail-closed guard runs before anything is printed: if any known identity candidate (the OS-reported login name and username, the hostname, the home directory path/name, or any of the `USERNAME`, `LOGNAME`, `USER`, `LNAME`, `USERPROFILE`, `HOME` environment variables and their path basenames) would appear in stdout anyway, the probe exits `1` with an error naming the *category* (never the value). If that ever triggers, do not disable it — report it as a bug.
- **Default runs create no files.** The old behavior of writing detailed results to an OS temp file has been removed.
- **Detailed results are unsanitized** and additionally contain: hostname, username, the Python executable path, the resolved `nvidia-smi` path, and raw `nvidia-smi` output. They are written **only** to the explicit `--output PATH` you pass.
- `--output` paths that resolve **inside the VFI repository** are refused (exit `1`, nothing written). The comparison runs on a canonicalized spelling: Windows extended-length prefixes (`\\?\`) are stripped before and after resolution, which also covers crafted spellings that `Path.resolve()` leaves prefixed — a `\\?\`-prefixed argument, or a directory component with a trailing space like `docs \leak.json` (a trailing-space *file* name survives resolution verbatim but is still refused, because its parent chain is under the repo root). A UNC spelling of a local admin share (`\\localhost\D$\...`, also via `127.0.0.1` or this machine's hostname) is mapped to its true drive-letter path first, so a UNC spelling of a repo path is refused like the plain one; every other UNC/network device path is refused outright. Drive-relative spellings (`D:leak.json`) are refused outright too, because their location depends on the per-drive working directory. The repository root is found by walking up from the probe module itself to the enclosing `.git` entry, so the check works no matter which directory you run the tool from (symlinks are resolved first: a link from outside that points into the repo is refused; a link inside the repo pointing outside is allowed).
- Error messages never repeat OS exception text (Windows OSErrors embed absolute paths); they carry only the error class, the errno, and — when it passes the guard below — the destination path you supplied. Error strings that would contain a known sensitive value are replaced by a generic category message. This includes argparse usage errors: an invalid flag or stray argument is echoed only after the same redaction (the fixed message template stays readable, the caller-supplied values do not), and usage errors exit with code `2`.

### Limits of the sanitization guard (be honest about them)

The guard matches a list of identity candidates gathered from two places: the OS (`getpass.getuser()`, `os.getlogin()`, `platform.node()`, `Path.home()`) and the process environment (`USERNAME`, `LOGNAME`, `USER`, `LNAME`, `USERPROFILE`, `HOME`, plus their path basenames). Gathering from the environment as well is deliberate: `getpass.getuser()` prefers `LOGNAME`/`USER`/`LNAME`/`USERNAME` and `Path.home()` follows `USERPROFILE`/`HOME`, so a caller that controls those variables could otherwise forge the guard's view of your identity (a red-team exercise demonstrated exactly that against the previous version). The defense is that `os.getlogin()` asks the OS for the logon-session name and ignores the environment on Windows, and that the forged values themselves become redaction needles. This defends against casual environment manipulation; it is not forensic-grade. It is a best-effort seatbelt for this tool's own output, **not** a general-purpose secret scanner. It cannot detect:

- arbitrary personal strings (your real name, email, project names, other people's data);
- sensitive values belonging to *other* accounts or machines;
- tokens, serial numbers, or MAC addresses that appear in unexpected places (e.g., inside a tool's stderr);
- values shorter than 3 characters (skipped to limit false positives);
- an identity changed below the environment (e.g., a process running under a different Windows account than the shell variables claim);
- anything inside the `--output` details file — that file is unsanitized **by design** and opt-in.

Treat any detailed file as personal data, and review stdout once before pasting it anywhere sensitive.

### Sharing a sanitized summary

Safe: run without `--output` and copy the stdout text (or `--json` output) into an issue, document, or chat.

Not safe: pasting a `--output` file (contains username/hostname), or pasting stderr if you passed an `--output` path containing personal directory names (the path is echoed back only when it passes the guard above; otherwise a redacted placeholder is printed instead).

```bash
# Share this file's contents:
py -3.12 tools/phase2_readiness/hardware_probe.py --json > "D:/temp/vfi-probe-summary.json"
# Do NOT share: the --output details file.
```

## Tests

The probe is covered by unit tests that mock all subprocess interaction (fake `nvidia-smi` CSV, missing binary, failing execution, garbage CSV, errors embedding a fake username), assert JSON validity, assert that no username/hostname/home-path leaks into sanitized output or error paths, assert that a default run creates no file (a `tempfile.mkstemp` tripwire plus temp-dir listing), and cover the repository-internal `--output` refusal from a foreign CWD and both symlink directions:

```bash
# from the repository root, in the dev venv (pytest + ruff in [dev] extras)
python -m pytest tests/phase2_readiness
python -m ruff check tools/phase2_readiness tests/phase2_readiness
```

## Limitations

- Windows-first: physical core count and installed RAM use Windows APIs; on other OSes those fields fall back to portable sources or `unknown`.
- GPU facts require NVIDIA drivers (`nvidia-smi`) to be installed; AMD/Intel discrete GPUs are not queried (only the integrated controller name may appear via OS APIs, and `Win32_VideoController.AdapterRAM` is intentionally not used for VRAM because it truncates above 4 GiB).
- The probe reports library *presence*, not versions — importing torch/transformers would be slow and is deliberately avoided; versions belong in the actual Phase 2 environment once it exists.
- The privacy guard only matches the identity candidates described under "Limits of the sanitization guard" above (OS-reported plus environment-provided; nothing shorter than 3 characters). The repository-internal refusal is only as good as its inputs: the `.git` walk-up (a module copied out of any repo falls back to its installation root) and the path canonicalization — which maps local admin-share UNC spellings to their drive-letter paths, refuses every other UNC device path and drive-relative spellings outright, and strips `\\?\` prefixes. A single-backslash drive-rooted spelling such as `\localhost\share\file` is resolved against the current drive's root, which lands outside the repository: an explicit, user-chosen location, but a surprising one.
