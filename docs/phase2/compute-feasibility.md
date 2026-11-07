# Phase 2 Compute & Hardware Feasibility

- Status: ready for Phase 2 decision review
- Date: 2026-11-07
- Scope: hardware measured on the owner's laptop (read-only probes); compute needs for Phase 2 workloads estimated from problem size, not benchmarked. Fine-tuning PhoBERT-base is discussed as planning input only; it is NOT authorized yet.
- Companion doc: [hardware-probe-guide.md](hardware-probe-guide.md) (how to re-run the probe yourself).

Evidence labels used below: `[MEASURED RESULT]` (actually ran and observed on 2026-11-07), `[ESTIMATE]` (projection with reasoning), `[ASSUMPTION]`, `[UNVERIFIED]`.

---

## 1. Measured environment

All commands were run read-only on the owner's machine. Windows paths and identifiers are omitted where irrelevant.

| Item | Value | Evidence |
|---|---|---|
| OS | Windows 11 Home, 64-bit, build 10.0.26300 | `[MEASURED RESULT]` `powershell Get-CimInstance Win32_OperatingSystem` |
| CPU | Intel Core i5-10500H @ 2.50 GHz, 6 physical cores / 12 logical | `[MEASURED RESULT]` `Win32_Processor`; independently confirmed by the probe tool |
| RAM | 32.00 GiB installed (2 x 16 GiB DDR4-3200 modules, configured 2933 MT/s) | `[MEASURED RESULT]` `Win32_PhysicalMemory` + kernel32 `GetPhysicallyInstalledSystemMemory` via probe |
| Discrete GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB VRAM, driver 617.42, compute capability 8.6 (Ampere) | `[MEASURED RESULT]` `nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv` |
| CUDA visibility | NVIDIA-SMI 617.42, CUDA UMD Version 13.4 | `[MEASURED RESULT]` `nvidia-smi` header |
| Integrated GPU | Intel UHD Graphics (driver 31.0.101.2140) | `[MEASURED RESULT]` `Win32_VideoController` |
| Disk C: | 475.9 GiB total, ~391.5 GiB free | `[MEASURED RESULT]` `py -3.12 -I -c "import shutil; print(shutil.disk_usage('C:/'))"` |
| Disk D: (project drive) | 931.5 GiB total, ~521.5 GiB free | `[MEASURED RESULT]` `py -3.12 -I -c "import shutil; print(shutil.disk_usage('D:/'))"` |
| Python | 3.12.10 (`py -3.12`) and 3.13 (default) installed | `[MEASURED RESULT]` `py -0p`, `py -3.12 --version` |
| PyTorch | NOT importable in either global Python (3.12 / 3.13): `ModuleNotFoundError: No module named 'torch'` | `[MEASURED RESULT]` `py -3.12 -c "import torch"` (importability check only; nothing installed) |
| numpy / scikit-learn in global 3.12 | NOT importable in the global interpreter | `[MEASURED RESULT]` `py -3.12 -I -c "import sklearn, numpy"` |

Notes:

- An earlier internal note suggested an "RTX 3060 6GB may exist". `[MEASURED RESULT]`: confirmed by my own `nvidia-smi` query — RTX 3060 Laptop GPU, exactly 6144 MiB VRAM. Do not rely on `Win32_VideoController.AdapterRAM` (reported 4293918720 bytes ≈ 4.0 GiB) — that field is a 32-bit value and truncates above 4 GiB; `nvidia-smi` is authoritative.
- Where Phase 1's working environment (numpy/sklearn for the classical baselines) actually lives was not probed — the Phase 1 clone was intentionally off-limits for isolation. `[UNVERIFIED]` The global interpreters do not have them, so Phase 1 presumably runs in a project venv.
- Re-run everything above with one command using the probe tool (see [hardware-probe-guide.md](hardware-probe-guide.md)). `[MEASURED RESULT]` sanitized probe output from the real run on 2026-11-07:

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

## 2. Software readiness gaps (before any Phase 2 work)

- `[MEASURED RESULT]` PyTorch, transformers, and even numpy/scikit-learn are absent from the global Python interpreters. Phase 2 therefore needs a fresh project venv (the repo `pyproject.toml` already pins Python >= 3.12 and dev deps) plus a CUDA-enabled PyTorch wheel.
- Driver/CUDA compatibility: the installed driver reports CUDA UMD 13.4, so any CUDA runtime <= 13.4 is driver-compatible (NVIDIA drivers are backwards compatible with older CUDA runtimes). `[ASSUMPTION]` A current `torch` cu12x wheel will work on this driver; Ampere (compute capability 8.6) is a supported target of all recent torch releases. This was not verified by installation, per mission constraints.
- Disk for a CUDA torch install: `[ESTIMATE]` ~3-5 GiB for torch + bundled CUDA libraries, plus `[ESTIMATE]` < 1.5 GiB for PhoBERT-base weights/tokenizer later. Against ~521 GiB free on D:, disk is a non-constraint.
- RAM for training data: `[ESTIMATE]` the entire 11k-sentence tokenized dataset at max_len 256 is on the order of tens of MB in memory — negligible against 32 GiB.

## 3. Compute estimates

### 3.1 TF-IDF + LogReg on ~16k short texts (Phase 1 extension work)

Phase 1 recorded no measured training/inference timings, so everything here is a projection. `[ESTIMATE]` reasoning:

- Vectorization: 16k short Vietnamese texts (tens of tokens each), vocabulary likely 30k-60k features after tokenization. Sparse matrix ~16k x 50k with roughly 100-200 non-zeros per document -> ~2-3M non-zeros -> tens of MB. Fitting TF-IDF: seconds.
- LogisticRegression (liblinear/lbfgs on sparse input): seconds per fit; a 5-fold grid over a handful of C values: under ~5 minutes total on this CPU.
- Inference: at least thousands of documents per second per core; 16k documents scored in well under a minute.
- Memory: peak well under 1 GiB.

Conclusion: TF-IDF + LogReg workloads are trivially feasible on this laptop, CPU-only. `[ESTIMATE]` total experimentation time is dominated by human analysis, not compute (minutes per full grid search).

### 3.2 PhoBERT-base 3-class fine-tuning on ~11k train sentences

PhoBERT-base has ~135M parameters. `[ESTIMATE]` memory budget (mixed-precision AMP + AdamW, max_len 256):

- Weights/optimizer: fp16 compute copy ~0.27 GiB + fp32 master ~0.54 GiB + two fp32 Adam moments ~1.08 GiB + gradients ~0.27-0.54 GiB ≈ 2.2-2.4 GiB before activations.
- Activations: `[ESTIMATE]` roughly 0.6-1.0 GiB at batch 8, 1.2-2 GiB at batch 16, 2.5-4 GiB at batch 32 (all at max_len 256, without gradient checkpointing).
- Resulting fit on 6144 MiB VRAM: batch 8 comfortably; batch 16 tight but usually workable; batch 32 borderline/OOM — use gradient accumulation instead of raising batch size.

`[ESTIMATE]` throughput and wall-clock (RTX 3060 Laptop at seq len 256):

- Realistic training throughput: ~40-70 samples/s with AMP at batch 8-16 (bottom-up from typical BERT-base seq-256 step times on Ampere consumer GPUs, scaled down ~30-40% for laptop power/thermal limits).
- One epoch over 11k samples: ~3-5 minutes; a full 3-epoch run with evaluation and checkpointing: ~20-40 minutes.
- A small hyperparameter sweep (3-6 runs): ~1-4 GPU-hours.
- Batch inference over 16k texts (batch 32, fp16): ~150-400 texts/s -> ~1-2 minutes; add tokenization overhead, call it a few minutes end-to-end.
- Uncertainty: these are order-of-magnitude figures; laptop thermals and background load can shift them by 2x. Nothing above is a measurement.

`[ESTIMATE]` CPU-only fallback (no GPU used):

- BERT-base forward+backward at seq 256 on a 6-core mobile CPU: roughly 1-3 samples/s -> 1-3 hours per epoch -> ~4-10 hours for 3 epochs, high variance.
- 32 GiB RAM is sufficient (model + optimizer ~2.5-3 GiB resident).
- Verdict: technically feasible as an overnight batch job, impractical for iteration. Acceptable only as a one-off fallback if the GPU becomes unusable.

### 3.3 Feasibility verdict per workload

| Workload | This laptop | Verdict |
|---|---|---|
| TF-IDF + LogReg (~16k texts) | CPU-only, minutes, <1 GiB RAM | Trivially feasible `[ESTIMATE]` |
| PhoBERT-base fine-tune (~11k train, 3 classes, max_len 256) | RTX 3060 6 GiB, batch 8-16 + AMP + grad accumulation; ~0.5-1.5 GPU-h per run | Feasible `[ESTIMATE]` |
| PhoBERT-base on CPU only | ~4-10 h for 3 epochs | Feasible but painful; fallback only `[ESTIMATE]` |
| Batch inference 16k texts | ~1-2 min GPU, ~10-30 min CPU | Feasible `[ESTIMATE]` |

## 4. Environment options

| Option | Compute characteristics | Cost | Privacy | Fit for Phase 2 |
|---|---|---|---|---|
| (a) This laptop, as measured | i5-10500H 6C/12T, 32 GiB RAM, RTX 3060 6 GiB (CC 8.6), CUDA driver 13.4, ~521 GiB free on D: | Zero marginal cost | Data never leaves the machine | Sufficient for the stated Phase 2 scope `[ESTIMATE]`; 6 GiB VRAM caps batch size, laptop thermals cap long sweeps |
| (b) Free/cheap cloud notebooks | Typically single GPU in the 16 GiB class with session time limits and variable availability | Free tiers exist; specifics change often | Data must be uploaded — the feedback corpus leaves the laptop | Useful as a cross-check or when the local GPU is busy; do not plan around quotas `[UNVERIFIED]` |
| (c) Paid GPU VM | Larger VRAM (24+ GiB), reproducible env, long runs | Hourly billing; verify current pricing at decision time | Data must be uploaded | Only needed if scaling beyond PhoBERT-base at this dataset size — not indicated now `[ESTIMATE]` |

Deliberately excluded: current vendor prices, quotas, or GPU model guarantees for option (b)/(c) — those change frequently and must be verified when a decision is actually made. `[UNVERIFIED]`

Privacy note: the Vietnamese feedback corpus is user-generated content; uploading it to any third-party service (option b or c) should be treated as a licensing/consent question first and a convenience question second. Option (a) avoids the issue entirely.

## 5. Recommendation

1. Run Phase 2 on this laptop: the measured RTX 3060 Laptop 6 GiB + 32 GiB RAM + 6C/12T CPU comfortably covers TF-IDF work and is adequate `[ESTIMATE]` for PhoBERT-base 3-class fine-tuning at ~11k sentences with AMP, batch 8-16, and gradient accumulation to reach an effective batch of 32.
2. Defer cloud/VM options until a concrete trigger appears: a sweep that exceeds ~1 day of local GPU time, a model that cannot fit in 6 GiB even with checkpointing, or a requirement for guaranteed reproducibility beyond one machine.
3. Prerequisite before any training (not authorized yet): create a project venv, install a CUDA 12.x torch build plus transformers, and re-run the hardware probe to confirm torch sees the GPU (`torch.cuda.is_available()`), which this readiness pass deliberately did not install or verify.
4. Keep inference workloads on the local GPU too — batch inference for the full corpus is minutes, so there is no compute reason to move it.

## 6. Risks / open items

- `[ASSUMPTION]` torch cu12x wheels work on driver 617.42 / CUDA UMD 13.4 — verify on first install (see recommendation 3).
- `[ESTIMATE]` All runtime/VRAM figures in section 3 are projections; the first authorized training run should record real step times and VRAM (e.g., `torch.cuda.max_memory_allocated()`) to replace them.
- Laptop sustained-load behavior (thermals, power profile) may reduce GPU throughput below the estimated range on long sweeps.
- Phase 1's environment location was not inspected (isolation); numpy/sklearn availability for Phase 1-style work in a new venv is an install step, not a hardware constraint.
