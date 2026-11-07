# Handoff to Phase 1 Maintainer (Codex-anti) — Findings for the Phase 1 Branch

Prepared by: Red Team Agent E (Security & Privacy), Phase 2 readiness program, 2026-11-07.
Copy-ready handoff. Every claim is evidence-labeled; file:line references are against Phase 1
commit `95f7687` (`phase1/classical-ml-baselines`, PR #1 head
`95f7687735c785367fa0a6f9b1efbf0e284aac7f`) unless stated otherwise.

**Scope statement.** Phase 1 was reviewed READ-ONLY. No Phase 1 file, branch, or PR was
modified. All red-team reproduction ran against a disposable copy of the readiness tool inside
a temporary directory; the working clone was returned clean (verified: no residue files).

---

## (a) Raw-label mapping conflict — status: UNRESOLVED

The raw integer encoding of the author-distributed `{split}/sentiments.txt` files (UIT-VSFC)
is **not documented by any primary artifact** fetched to date (author GitHub README, KSE 2018
paper, either HF loader). Phase 1's `LABEL_TO_ID` (`loading.py:26`) inherits the SEACrowd
`ClassLabel` declaration (`0=positive, 1=negative, 2=neutral`), which conflicts with the
uitnlp card's "Data Fields" prose. A wrong mapping silently inverts every metric.

- Full evidence trail and decision requirements:
  [`label-semantics-evidence.md`](label-semantics-evidence.md) (verdict section, read first).
- Action for the maintainer: none in code. This is an owner verification gate at data receipt
  (recorded in `phase1-dependencies.md` §3 issue #9, risk R-07).

## (b) Confirmed Phase 1 issues (consolidated from phase1-dependencies.md §3)

Fourteen issues were confirmed during the read-only review; the table in
[`phase1-dependencies.md`](phase1-dependencies.md) §3 remains the authoritative list. Tight
summary (file:line | severity | one-line direction):

1. `baseline.py:222` (with `:178`) | High | `joblib.load` executes arbitrary pickle code behind
   advisory-only sidecar checks — record sha256 at save, verify at load, hard fail on mismatch.
2. `validation.py:43-59` | High | Exact-match-only leakage/duplicate detection misses
   near-duplicates — add an n-gram/MinHash audit report; keep exact-match behavior.
3. `baseline.py:212-213` | Medium | `config_hash` presence is checked, never equality —
   recompute and fail on mismatch.
4. `baseline.py:160-183` | Medium | `save_model` writes no artifact integrity hash — add
   sha256 + size to `meta.json`.
5. `cli.py:83-91` (written `:99`, printed `:101`) | Medium | `data_report.json` embeds the
   literal text of every cross-split duplicate — report counts + 12-hex text-hash prefixes
   instead (see also (c)2 below for the red-team confirmation of the full flow).
6. `.gitignore:32-37` | Medium | No `*.bin`/`*.pt`/`*.ckpt`/`*.h5`/`*.onnx` rules — add them
   before the first HF download (red-team `git check-ignore` evidence in (c)1).
7. `verify.py:56-60` | Medium | Structural gate covers only 3 paths + `*.joblib`/`*.pkl` —
   extend to `*.safetensors`, `*.bin`, `*.pt`, `*.ckpt`, `.env*`.
8. `splitting.py:41-46` | Medium | Fallback split does not group duplicates — split by
   normalized-text group, then expand.
9. `loading.py:26` | Medium | Label mapping sourced from an unverified declaration — owner
   gate at data receipt (see (a)).
10. `pyproject.toml:13-19` | Low (High if replicated in Phase 2) | Lower-bound-only pins —
    pin exact when adding the Phase 2 stack (`torch` ≥ 2.6 so `torch.load` defaults to
    `weights_only=True`, `transformers`, `accelerate`, `safetensors`, segmenter).
11. `baseline.py:214-221` | Low | sklearn version warning noisy/ignorable — align message with
    comparison; consider hard fail on major mismatch.
12. `cli.py:65-102` | Low | `prepare-data` always exits 0 even with leakage — add a gate flag
    or dedicated gate command.
13. `.github/workflows/ci.yml:9-32` | Low | No structural-check step, no `permissions:` block,
    tag-pinned actions — add checks + `permissions: contents: read`; optionally SHA-pin.
14. `cli.py:223-252` | Low | `predict` echoes full input text, no length cap — document;
    length cap belongs to Phase 3 input limits.

## (c) New Phase-2-relevant findings from the 2026-11-07 red-team pass

1. **Weight-format ignore gap, verified with `git check-ignore`.** [VERIFIED FACT] Against the
   Phase 1 rules at `95f7687` (`/models/` root-anchored, `*.joblib`, `*.pkl`, `*.safetensors`):
   `pytorch_model.bin` at the repository root, `checkpoints/epoch3.pt`,
   `artifacts/phobert/adapter_model.bin`, and `export/model.onnx` are all committable
   (NOT IGNORED); anything under `models/` is ignored regardless of extension. The readiness
   clone's `.gitignore:33` uses the unanchored `models/` variant, which is the direction to
   copy: extension rules + unanchored directory rules, before the first HF download (risk
   R-08, gate G3).
2. **`data_report.json` PII flow, end-to-end.** [VERIFIED FACT] `vfi prepare-data` embeds raw
   student feedback text via `find_leakage` (`validation.py:43-61` returns a mapping keyed by
   the normalized text) into `reports/artifacts/data_report.json` (`cli.py:83-91`, `:99`) and
   stdout (`:101`). The only protection is the single ignore rule for `reports/artifacts/`;
   `config.py:141` (`reports_dir` override) can point the report anywhere, including an
   unignored directory — the report would then be committable with raw feedback inside.
   Direction: counts + hash prefixes (issue #5), and treat a non-default `reports_dir` as a
   Phase 2 reporting risk (R-05).
3. **Hardware probe (`tools/phase2_readiness/hardware_probe.py`) — three reproduced
   repo-refusal bypasses** (details in `risk-register.md`, hardening pass section): crafted
   `--output` paths (`\\?\`-prefixed, `\?\`-rooted, trailing-space components, UNC
   `\\localhost\D$\...` spellings) bypass the repository-internal refusal; the `\\?\` and
   trailing-space forms were verified to write the detailed (username/hostname-bearing) file
   INSIDE the repository tree, where it is committable. Fix direction: canonicalize (strip
   `\\?\`/`\\?\UNC\` prefixes, expand drive-relative forms) or re-verify the destination with
   `Path.resolve(strict=True)` after the parent directory exists, and re-run the refusal check
   at write time. Red-team tests live in `tests/phase2_readiness/test_red_team.py` (strict
   xfail: they pass when the holes are fixed). RESOLUTION UPDATE 2026-11-07: all five
   reproduced holes (including the environment-spoofing and argparse-echo findings below)
   have since been fixed in `hardware_probe.py` with the xfail markers converted to passing
   regression tests — see `00-executive-decision.md` (hardening pass) and
   `hardware-probe-guide.md` for the corrected guarantees and limits.
4. **True CI state, verified via `gh` on 2026-11-07.** [VERIFIED FACT] PR #1
   (`phase1/classical-ml-baselines`): OPEN, Draft, MERGEABLE, CI green (run 42324940437,
   success, 2026-11-07T01:36:08Z; an earlier failed run 42324583725 at 01:32Z was superseded).
   PR #2 (`research/phase2-readiness`, head `fd91f4701d86380b99a4555a8ca26ef5aadea522`):
   OPEN, no status checks at all — the branch carries no CI workflow and `gh run list` shows
   zero runs for it. CI exists only on `pull_request`/`push` to `main` (`ci.yml:9-32`).
5. **Sanitization-guard limit worth knowing in Phase 2 reporting.** [VERIFIED FACT] The probe's
   guard derives its sensitive values from the live environment (`getpass.getuser()` prefers
   `LOGNAME`/`USER`/`LNAME` over `USERNAME`; `Path.home()` follows `USERPROFILE`); spoofed
   environment variables disable redaction of the `--output` echo. The same env-derived-value
   pattern should be avoided in any Phase 2 log/report sanitizer.

## (d) Review scope

Phase 1 (branch `phase1/classical-ml-baselines`, PR #1 head
`95f7687735c785367fa0a6f9b1efbf0e284aac7f`) was reviewed READ-ONLY: no Phase 1 file was
edited, no commit/push/merge was made, no dataset was downloaded, and the probe tool was never
run with a real `--output` into the Phase 1 repository. The read-only review that produced
§3 above was likewise non-mutating (git history scan + `git show`).
