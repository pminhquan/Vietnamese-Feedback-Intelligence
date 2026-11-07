# Phase 2 Risk Register — Security, Privacy & Failure-Mode Review

Prepared by: Research Agent E (Security, Privacy & Failure-Mode Review), Phase 2 readiness program.
Date: 2026-11-07.
Scope: threats introduced or amplified by **Phase 2** (fine-tuning a Vietnamese pretrained
transformer, e.g. PhoBERT, on UIT-VSFC student feedback), on top of the Phase 1 codebase.

Reviewed Phase 1 ref: `95f7687735c785367fa0a6f9b1efbf0e284aac7f` (head of
`phase1/classical-ml-baselines`, Draft PR #1: OPEN, MERGEABLE, CI green — verified via
`gh pr view 1`). File:line citations below refer to that commit unless stated otherwise.

## Evidence legend

- **[VERIFIED FACT]** — confirmed by reading the reviewed ref (file:line cited), by running a
  read-only command (stated), or by a primary upstream source (linked).
- **[ASSUMPTION]** — reasonable inference from verified facts; stated reasoning given.
- **[UNVERIFIED]** — plausible but not confirmed; needs owner/agent action to settle.

## Severity model

| Likelihood \ Impact | Low | Medium | High |
| ------------------- | ------- | ------ | ---- |
| **High**            | Medium  | High   | Critical |
| **Medium**          | Low     | Medium | High |
| **Low**             | Low     | Low    | Medium |

Responsibility legend: **Owner** = repository owner (human decision/data actions);
**Agent** = AI coding agents (Phase 2 implementation agents, including Codex-anti).

## Risk summary

| ID | Risk | L | I | Severity |
| ---- | ------------------------------------------------------------ | --- | --- | -------- |
| R-01 | Untrusted pretrained checkpoints (pickle `.bin` deserialization) | M | H | High |
| R-02 | Remote code execution via `trust_remote_code=True` (HF datasets/models) | L | H | Medium |
| R-03 | Unsafe local joblib artifacts (Phase 1 save/load pattern carried into Phase 2) | L | H | Medium |
| R-04 | Dataset provenance and license enforcement gap (UIT-VSFC user agreement) | M | H | High |
| R-05 | Student-feedback privacy: PII in reports, logs, PR diffs | M | M | Medium |
| R-06 | Hidden cross-split near-duplicate leakage inflating test metrics | M | H | High |
| R-07 | Ambiguous raw-vs-canonical label mapping (silent label inversion) | M | H | High |
| R-08 | Accidentally committed weights / raw data / `.env` (gitignore + gate gaps) | M | H | High |
| R-09 | Repeated test-set tuning (no mechanical enforcement of test lockout) | M | H | High |
| R-10 | Dependency/version drift (transformers/torch unpinned, no lockfile) | H | M | High |
| R-11 | Misleading accuracy claims (unit tests or synthetic runs read as real results) | L | H | Medium |
| R-12 | Vietnamese segmentation toolchain supply chain (VnCoreNLP/underthesea) | M | M | Medium |
| R-13 | Phase 2 training code invisible to CI (CPU-only CI, no structural gate in CI) | H | M | High |

---

## R-01 — Untrusted pretrained checkpoints (pickle deserialization)

**Description.** Phase 2 will download third-party model checkpoints. Many PyTorch checkpoints
(including the original `vinai/phobert-base`) are distributed as `pytorch_model.bin`, which is a
pickle payload loaded via `torch.load`; a malicious or tampered `.bin` (or any `*.pt`/`*.pth`
artifact) executes arbitrary code at load time. The safetensors format is not executable and is
the safe default.

**Affected assets.** Developer machine and any credentials in the environment (e.g. `.env`,
HF token, git credentials), the repository, trained artifacts derived from a poisoned base model.

**Likelihood: Medium.** The canonical `vinai/phobert-base` repo is long-lived and widely used,
but Phase 2 will also be tempted to compare against third-party *fine-tuned* checkpoints, which
are the realistic attack surface. [VERIFIED FACT, external] `vinai/phobert-base` ships
`pytorch_model.bin`; a safetensors variant was auto-converted for `vinai/phobert-base-v2`
(HF discussion: https://huggingface.co/vinai/phobert-base-v2/discussions/3), and HF provides a
conversion space (https://huggingface.co/spaces/safetensors/convert). [VERIFIED FACT, external]
PyTorch 2.6 (Jan 2025) flipped the `torch.load` default to `weights_only=True` as a security
measure (https://pytorch.org/blog/pytorch2-6/); older torch, or an explicit
`weights_only=False`, re-opens arbitrary code execution.

**Impact: High.** Full workstation compromise (code execution outside any sandbox).

**Severity: High.**

**Mitigation.**
1. Prefer checkpoints with a native `model.safetensors`; otherwise convert the chosen checkpoint
   once with the official safetensors converter and commit only the *decision + hash*, not the weights.
2. Pin every checkpoint by repo id **and** revision SHA (`from_pretrained(..., revision="<sha>")`),
   and record the SHA256 of each downloaded file in `meta.json`-style provenance (reusing the
   Phase 1 sidecar convention, `baseline.py:226-243`).
3. Pin `torch>=2.6` so `weights_only=True` is the default; never pass `weights_only=False`;
   optionally set `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1` in local run scripts for defense in depth.
4. Only canonical, org-published checkpoints for the primary experiment; treat any third-party
   fine-tuned checkpoint as untrusted input requiring explicit owner approval.

**Responsibility.** *Agent:* implement pinned-revision + checksum recording, safe loading calls.
*Owner:* approve which checkpoints are used; no agent should silently swap in a new checkpoint.

**Evidence.** [VERIFIED FACT] Phase 1 contains no HF/torch code yet (`pyproject.toml:13-19`);
the risk is entirely Phase 2-new. External facts verified from the linked primary sources.

---

## R-02 — Remote code execution via `trust_remote_code=True`

**Description.** `trust_remote_code=True` on `from_pretrained`/`load_dataset` executes
repository-supplied Python on the local machine. The SEACrowd UIT-VSFC mirror is a
loading-script dataset (`docs/dataset-uit-vsfc.md:35` [VERIFIED FACT]), i.e. exactly the kind of
resource that historically required `trust_remote_code=True`.

**Affected assets.** Developer machine; project data pipeline integrity.

**Likelihood: Low.** [VERIFIED FACT, external] Hugging Face `datasets` 3.0 **removed** dataset
script support entirely — script datasets raise `RuntimeError: Dataset scripts are no longer
supported` and `trust_remote_code` no longer has any effect
(https://github.com/huggingface/datasets/releases/tag/3.0.1,
https://huggingface.co/docs/datasets/v3.4.1/en/dataset_script). Also, Phase 1 already reads the
corpus as local line-aligned text files (`loading.py:69-106`), which is the correct pattern to
keep. Residual risk: habitually passing `trust_remote_code=True` to `AutoModel.from_pretrained`
for some exotic checkpoint.

**Impact: High.** Arbitrary code execution from a model repo.

**Severity: Medium.**

**Mitigation.**
1. Project rule: **never** pass `trust_remote_code=True` anywhere in `src/vfi/`; load UIT-VSFC
   from local files (existing Phase 1 path), not via `load_dataset("SEACrowd/uit_vsfc")`.
2. Use `AutoTokenizer`/`AutoModel` only with checkpoints that load with pure config+weights
   (PhoBERT does; it has a native `BertModel`-based implementation in transformers).
3. If a dataset ever seems to require it, stop and record a provenance note instead
   (same discipline as `docs/dataset-uit-vsfc.md`).

**Responsibility.** *Agent:* code review check — grep for `trust_remote_code` before any PR.
*Owner:* final say on any new data source.

**Evidence.** [VERIFIED FACT] `docs/dataset-uit-vsfc.md:35` (mirror is loading-script only);
[VERIFIED FACT, external] datasets 3.0 removal notes; [VERIFIED FACT] `loading.py:69-106`
local-file loading already in place.

---

## R-03 — Unsafe local joblib artifacts (Phase 1 save/load pattern)

**Description.** Phase 1 persists models with `joblib.dump` (`baseline.py:178`) and loads them
with `joblib.load` (`baseline.py:222`) — pickle deserialization. `load_model` applies
"trusted-artifact" checks first (`baseline.py:199-221`), but they are **advisory**: the
`config_hash` is checked for *presence* only (`baseline.py:212-213`), the sklearn version
mismatch only *warns* (`baseline.py:214-221`), and there is no integrity hash of
`model.joblib` itself, so `meta.json` can be re-stamped by whatever wrote the pickle. Any code
(or person) with write access to `models/` gains code execution at load time.

**Affected assets.** Workstation (code execution), integrity of reported results
(a swapped artifact would silently change predictions).

**Likelihood: Low.** Single-user, local-only project; exploitation requires local write access
to `models/`. [ASSUMPTION] No multi-user deployment exists in scope (Phase 3 serving would
change this).

**Impact: High.** Arbitrary code execution; silent model substitution.

**Severity: Medium.**

**Mitigation.**
1. Keep joblib only for sklearn pipelines (Phase 1 scope). For Phase 2, save transformer
   weights via `safetensors` + tokenizer files, never a pickled wrapper object.
2. Record `sha256(model.joblib)` in `meta.json` at save time; verify at load time
   (direction for Codex-anti, see `phase1-dependencies.md` issue #3).
3. Compare the recorded `config_hash` against the hash recomputed from the current config
   at load time, or fail loudly on mismatch.
4. If Phase 3 ever serves artifacts from a shared location, add an explicit signature/manifest
   step instead of relying on the sidecar.

**Responsibility.** *Agent (Codex-anti):* implement hash + hash-verification in
`save_model`/`load_model` (Phase 1 code change — not mine to make). *Agent (Phase 2):* do not
reuse joblib for transformer checkpoints. *Owner:* accept the residual local-trust model.

**Evidence.** [VERIFIED FACT] `baseline.py:178, 222` (pickle dump/load); `baseline.py:212-213`
(presence-only hash check); `baseline.py:214-221` (warn-only version check); no artifact digest
anywhere in `save_model` (`baseline.py:160-183`).

---

## R-04 — Dataset provenance and license enforcement

**Description.** UIT-VSFC's license is **unknown** and distribution requires a signed corpus
user agreement (`docs/dataset-uit-vsfc.md:8-18` [VERIFIED FACT]). The HF mirror
(`SEACrowd/uit_vsfc`) pulls the same Google Drive files without conveying that agreement.
Phase 2 pressure ("I need real data to fine-tune") makes bypassing the agreement — grabbing the
mirror or a random re-upload — the single most tempting shortcut.

**Affected assets.** Legal/ethical standing of the project (it doubles as an internship
portfolio, `AGENTS.md:6`), all trained models and reported numbers (built on unlicensed data
would be unusable), public repo reputation.

**Likelihood: Medium.** The blocker is loud and documented, but Phase 2 cannot meaningfully
start without data, and the mirror is one command away.

**Impact: High.** Any results from unlicensed data are unpublishable and must be retracted;
violating a signed-agreement dataset is worse than never starting.

**Severity: High.**

**Mitigation.**
1. Hard gate: no Phase 2 training run on real data until the owner records that the signed
   agreement was returned (extend `AGENTS.md` rule 5 to Phase 2 explicitly).
2. Record provenance in every run's metadata: source channel ("author email, agreement signed
   on YYYY-MM-DD"), not just file fingerprints. Phase 1's metadata convention
   (`baseline.py:226-243`) is the place to add a `dataset_provenance` field.
3. `vfi prepare-data` already fingerprints each split (`cli.py:81`); keep that as the
   tamper-evidence mechanism and store the "expected" fingerprints alongside the provenance note.
4. Agents must not add download scripts that bypass the agreement (`AGENTS.md:20-23`).

**Responsibility.** *Owner:* sign and return the agreement; state when permission exists.
*Agent:* refuse/flag any data acquisition path that bypasses it; add provenance fields.

**Evidence.** [VERIFIED FACT] `docs/dataset-uit-vsfc.md:6-18` (verdict BLOCKED, license
unknown, agreement required), `:53-57` (author's repo instruction to sign the agreement);
`AGENTS.md:20-23` (rule 5); `HANDOFF.md:10-12` (dataset blocker). [VERIFIED FACT] full git
history across all refs contains no dataset files (scan `git log --all --name-only` for
data/artifact extensions returned nothing).

---

## R-05 — Student-feedback privacy (PII in reports, logs, PR diffs)

**Description.** The corpus is student opinions naming lecturers, courses, facilities
(`docs/dataset-uit-vsfc.md:61-70` [VERIFIED FACT]). Even without classic name/ID PII in the raw
files, sentiment-bearing sentences about *named* people are personal data. Phase 2 amplifies the
surface: error-analysis reports for a neural model naturally quote many more misclassified
examples; training logs may echo dataset samples; PR descriptions may quote fragments; and one
Phase 1 report already embeds literal texts (below).

**Affected assets.** Raw `data/raw/` (gitignored), `reports/artifacts/*.json` (gitignored),
committed milestone reports in `reports/`, PR text, console logs, any future CI logs.

**Likelihood: Medium.** [VERIFIED FACT] `cli.py:83-91` writes the **literal text of every
cross-split duplicate** into `reports/artifacts/data_report.json` (`:99`) and prints it to
stdout (`:101`) — the pattern of "texts inside artifacts" already exists. Committed reports are
covered by explicit rules (`reports/README.md:11`, `dataset-uit-vsfc.md:66-69`), which lowers
but does not eliminate the risk once error analysis becomes per-example.

**Impact: Medium.** Privacy harm to students/lecturers quoted out of context; reputational cost.

**Severity: Medium.**

**Mitigation.**
1. Change `cmd_prepare_data` to report duplicate/leakage **counts and text hashes** (e.g. first
   12 hex of sha256) instead of raw strings (direction for Codex-anti, issue #5 in
   `phase1-dependencies.md`).
2. Phase 2 error-analysis artifacts: keep per-example details in `reports/artifacts/`
   (gitignored); commit only aggregates plus at most short truncated/paraphrased fragments
   (existing rule, `dataset-uit-vsfc.md:68-69`).
3. Never paste raw feedback into PR descriptions, issues, or commit messages; the gitignore
   cannot protect those.
4. If training logs echo samples, route them to a gitignored log dir; scrub before sharing.

**Responsibility.** *Agent (Codex-anti):* fix the data-report text embedding. *Agent
(Phase 2):* design error-analysis outputs aggregate-first. *Owner:* reviews every committed
milestone report before merge (PR rule, `HANDOFF.md:89`).

**Evidence.** [VERIFIED FACT] `cli.py:83-91, 99, 101`; `docs/dataset-uit-vsfc.md:61-70`;
`reports/README.md:11`; [VERIFIED FACT] `.gitignore:33-34` excludes `/reports/artifacts/`.

---

## R-06 — Hidden cross-split near-duplicate leakage

**Description.** Phase 1 leakage/duplicate detection is **exact-match after NFC + whitespace
normalization** (`validation.py:43-59`; `normalize_text` at `loading.py:48-59`), and the code
itself states near-duplicate leakage is out of scope (`validation.py:52-53`;
`project-spec.md:85-86`). Vietnamese feedback abounds in near-duplicates that survive exact
matching: diacritic variants ("không" vs "khong"), teencode/abbreviation, punctuation and word
order edits, plus legitimate verbatim repeats which are *deliberately not removed*
(`validation.py:32-40`, `cli.py:92-95`). A fine-tuned transformer memorizes near-duplicates far
more strongly than TF-IDF + LogReg, so train→test near-duplicate leakage inflates the headline
test macro-F1 precisely where Phase 2 claims its improvement.

**Affected assets.** All Phase 2 test metrics and the honest Phase-1-vs-Phase-2 comparison
(`docs/roadmap.md:22-23`); model selection decisions.

**Likelihood: Medium.** [ASSUMPTION] The real corpus's near-duplicate rate is unknown (no real
data has been run yet — all results are from synthetic fixtures, `HANDOFF.md:10-12`), but short
survey feedback is a genre where repeats are common; the fallback split path additionally never
dedupes before splitting (`splitting.py:41-46`), so exact duplicates can already straddle
train/validation today.

**Impact: High.** Invalid benchmark claims; silently invalid model selection; the flagship
Phase 2 deliverable would be wrong in a way that is invisible without an audit.

**Severity: High.**

**Mitigation.**
1. Before the first Phase 2 test evaluation, run a near-duplicate audit across all split pairs:
   character/word n-gram Jaccard or MinHash-LSH at a documented threshold, plus the existing
   exact-match check. Report the near-duplicate rate per split pair in `reports/`.
2. If leakage is found: prefer group-aware splitting (assign near-duplicate clusters to one
   split) — but note the official test split is fixed by the corpus, so the audit result must be
   *disclosed alongside* the metrics rather than silently re-splitting.
3. Extend `find_leakage` behind a separate function so Phase 1 exact-match behavior (and its
   tests) stays intact (direction for Codex-anti).
4. The fallback `use_official_splits: false` path should deduplicate (group by normalized text)
   before `stratified_split` (`splitting.py:41-46`).

**Responsibility.** *Agent:* implement the near-dup audit tooling and the dedup-aware fallback
split; report rates. *Owner:* decide, on evidence, how the official splits' near-duplicate rate
is disclosed in the milestone report.

**Evidence.** [VERIFIED FACT] `validation.py:43-59` (exact-match only, docstring admits scope
limit), `loading.py:48-59` (normalization = NFC + whitespace), `validation.py:32-40` (duplicates
reported, not removed), `cli.py:92-95` (note in report), `project-spec.md:85-86` (known
limitation), `splitting.py:41-46` (no dedup before fallback split).

---

## R-07 — Ambiguous raw-vs-canonical label mapping

**Description.** Raw UIT-VSFC integers are mapped `0=positive, 1=negative, 2=neutral`
(`loading.py:26`) into the canonical alphabetical space `negative=0, neutral=1, positive=2`
(`loading.py:29-32`). The two spaces are **inverted orders of each other**, so any Phase 2 code
that bypasses `RAW_SENTIMENT_TO_LABEL` (e.g. reading `sentiments.txt` directly for a HF
`Dataset`, or interpreting logits' argmax as a raw id) silently flips positive/negative. The
whole mapping chain is also **sourced from the SEACrowd loader's `ClassLabel` declaration, not
the corpus author's documentation** — flagged as unverified in
`docs/dataset-uit-vsfc.md:37-41` and `data/README.md:29-30` [VERIFIED FACT].

**Affected assets.** Every label in training, validation, test; all metrics; the confusion
matrix; any Phase 2 dataset conversion step.

**Likelihood: Medium.** The mapping is explicit and centralized (good), but it has never been
checked against the author's README (no real data accessed yet), and Phase 2 adds a second
consumer (tokenizer/trainer pipeline) that might bypass it.

**Impact: High.** A silent inversion produces confidently wrong models and metrics; because all
three classes shift, accuracy may even look plausible, making it hard to notice.

**Severity: High.**

**Mitigation.**
1. Owner verifies sample lines against the author's README immediately upon receiving the
   corpus (already the documented plan: `data/README.md:29-30`,
   `docs/dataset-uit-vsfc.md:79`), *before* any training run.
2. Keep the conversion in exactly one place (`RAW_SENTIMENT_TO_LABEL`, `loading.py:26`) and make
   Phase 2's dataset builder consume `Record` objects — never re-parse `sentiments.txt`.
3. Add a smoke assertion to `prepare-data` output that per-class counts differ across classes
   and match author-documented distribution ranges, when available [UNVERIFIED: author README
   contents not yet accessible].
4. Any Phase 2 id↔label table must be derived from `LABEL_TO_ID`/`CANONICAL_LABELS`, not
   re-declared.

**Responsibility.** *Owner:* verify against author docs; state the outcome in writing.
*Agent:* single-source the mapping; forbid re-parsing in Phase 2 code.

**Evidence.** [VERIFIED FACT] `loading.py:8-13, 26-32`; `docs/dataset-uit-vsfc.md:33, 37-41`;
`data/README.md:15, 29-30`; `project-spec.md:52-54`.

---

## R-08 — Accidentally committed weights, raw data, or secrets

**Description.** Phase 2 downloads gigabyte-scale checkpoints and produces many artifact files
in working directories. Phase 1's `.gitignore` covers `.env*`/`*.key` (`:17-19`), local data
(`:22-30`), and `/models/`, `/reports/artifacts/`, `*.joblib`, `*.pkl`, `*.safetensors`
(`:33-37`) [VERIFIED FACT] — but has **no rule for `*.bin`, `*.pt`, `*.ckpt`, `*.h5`,
`*.onnx`** (e.g. HF `pytorch_model.bin`, PEFT `adapter_model.bin`), so the classic Phase 2
artifact is committable if it lands outside `/models/` (which is also root-anchored — a diff
caddb4a→95f7687 changed `models/` to `/models/`). The verify gate checks only three sample
paths and tracked-patterns `*.joblib`/`*.pkl` (`verify.py:56-60`), and CI runs no structural
check at all (`ci.yml`).

**Affected assets.** Public repository (license violation + PII if data leaks; repo bloat if
weights leak); git history (removal then requires history rewrite, which `AGENTS.md:18-19`
forbids doing casually).

**Likelihood: Medium.** Git history is currently clean across all refs [VERIFIED FACT: scanned
`git log --all --name-only` for artifact/data extensions — none found], but Phase 2 changes the
default artifact formats to exactly the uncovered ones.

**Impact: High.** Unlicensed data or raw student text published publicly; force-push pressure
conflicts with rule 4; multi-hundred-MB blobs degrade the repo permanently.

**Severity: High.**

**Mitigation.**
1. Before the first HF download: extend `.gitignore` with `*.bin`, `*.pt`, `*.ckpt`, `*.h5`,
   `*.onnx` (and negations for any intentional small binary fixtures), and extend
   `verify.py:check_no_tracked` to the same list plus `*.safetensors` and `.env`
   (direction for Codex-anti, issues #9-10 in `phase1-dependencies.md`).
2. Add the structural checks to CI so the gate cannot be skipped locally
   (issue #11).
3. Configure local HF cache outside the repo (default `~/.cache/huggingface`) and, when a
   checkpoint must live in-tree, put it under an explicitly gitignored path.
4. Pre-commit habit (owner): `git status --short` review before every commit; the gate enforces
   the rest.

**Responsibility.** *Agent (Codex-anti):* gitignore/verify/CI hardening (small, contained
Phase 1 diffs). *Agent (Phase 2):* never `git add -A`; stage deliberately. *Owner:* review
before merge.

**Evidence.** [VERIFIED FACT] `.gitignore:1-43` (full contents; gaps listed),
`verify.py:56-60`, `ci.yml:9-32` (no structural step, no `permissions:` block, actions pinned
by tag `@v4`/`@v5` not SHA); `git diff caddb4a 95f7687 -- .gitignore` (root-anchoring change);
[VERIFIED FACT] history scan clean.

---

## R-09 — Repeated test-set tuning

**Description.** The protocol forbids touching the test split until selection is finalized
(`docs/experiment-protocol.md:9-11, 73-79` [VERIFIED FACT]) and `vfi evaluate` is deliberately
separate (`cli.py:189-220`). But nothing *mechanically* enforces it: `vfi evaluate` is cheap,
repeatable, exits 0, and keeps no ledger. Phase 2 raises the stakes — many checkpoints/epochs
and a large transformer gap to Phase 1 baselines make "just check it on test this once"
recursively tempting (each "final" run begets another).

**Affected assets.** Validity of every published metric; the honest Phase 1 vs Phase 2
comparison; model selection decisions.

**Likelihood: Medium.** Human/process risk, not a code bug; the honesty rules exist but are
only words without a counter.

**Impact: High.** Overfit-by-selection test numbers; the exact failure the protocol exists to
prevent.

**Severity: High.**

**Mitigation.**
1. Test-set evaluation ledger: every `vfi evaluate` run appends an entry (timestamp, config
   hash, test fingerprint, metrics) to an append-only `reports/artifacts/test_ledger.jsonl`;
   the committed milestone report must cite its ledger line. >1 test evaluation per selected
   configuration requires a written justification in the report.
2. Keep the Phase 1 structural separation (train/validate → save → evaluate) unchanged in
   Phase 2; new training code must not import or touch test records (`cli.py:116` records
   `test_untouched` — preserve that field).
3. Model selection dashboards/summaries read validation metrics only.

**Responsibility.** *Owner:* discipline; approve the single final test evaluation per
milestone. *Agent:* implement the ledger and preserve the "test untouched" reporting; flag any
change that makes test data reachable from training code.

**Evidence.** [VERIFIED FACT] `docs/experiment-protocol.md:9-11, 73-79`; `cli.py:116, 125-126`
(comment "deliberately NOT touched"), `cli.py:189-220` (unrestricted evaluate); [ASSUMPTION]
no mechanical counter exists anywhere in Phase 1 (verified by reading cli.py/verify.py).

---

## R-10 — Dependency and version drift

**Description.** Phase 1 dependencies are lower-bound only (`pyproject.toml:13-19`:
`numpy>=1.26`, `pandas>=2.1`, `scikit-learn>=1.4`, `joblib>=1.3`, `pyyaml>=6.0`; only
`ruff==0.17.4` is exact, `:25`). Phase 2 adds `torch`, `transformers`, `accelerate`,
`segmentation` libs — the fastest-moving, most mutually-coupled versions in the Python ML
ecosystem (torch↔CUDA, transformers↔tokenizers, etc.). Drift changes predictions (sklearn
documented as able to change predictions across versions — hence the load-time warning,
`baseline.py:214-221`) and breaks reproducibility of a trained checkpoint (a re-run with
newer transformers may not reproduce a reported metric at all).

**Affected assets.** Reproducibility record (`meta.json` currently stores sklearn + Python
versions, `baseline.py:237-238`); CI vs local parity; historical experiment re-runs.

**Likelihood: High.** New unpinned ML stacks drift within weeks; there is no lockfile in the
repo.

**Impact: Medium.** Non-reproducible metrics, CI/local divergence, occasional breakage; costs
time and credibility but not security.

**Severity: High.**

**Mitigation.**
1. Pin exact versions for the Phase 2 stack in `pyproject.toml` (`torch==x.y.z`,
   `transformers==x.y.z`, `accelerate==...`, segmentation lib pinned) and commit a lockfile
   (`uv lock`/`pip-tools`) if the owner adopts one; record all of them in training metadata
   (extend `build_training_metadata`, `baseline.py:226-243`).
2. `load_model`-style version checks for transformers (warn on any mismatch, mirroring the
   sklearn warning) for any loaded Phase 2 artifact.
3. Add a `meta.json` field `library_versions` capturing `torch.__version__`,
   `transformers.__version__`, segmentation tool version at train time.

**Responsibility.** *Agent:* pin versions, extend metadata, add the warn-on-mismatch check.
*Owner:* approve version bumps (no silent upgrades mid-milestone).

**Evidence.** [VERIFIED FACT] `pyproject.toml:13-19, 25`; `baseline.py:237-238`;
`load_model` warning behavior at `baseline.py:214-221`.

---

## R-11 — Misleading accuracy claims

**Description.** Risk that synthetic-fixture unit test results or an accidental misconfiguration
get reported as real model performance. Phase 1 has strong countermeasures: honesty rules
(`docs/experiment-protocol.md:73-79`), AGENTS rule 1 ("unit tests passing is not model
quality", `AGENTS.md:10-12`), no numbers claimed anywhere yet (`HANDOFF.md:10-12`,
`project-spec.md:73-74`), test suite explicitly synthetic (`tests/conftest.py:1-7`). Phase 2
adds new claim surfaces: "loss went down" ≠ accuracy; a model card or README accidentally
quoting a validation number as a test number; comparing against the Phase 1 baseline under a
different protocol.

**Affected assets.** Public credibility of the portfolio project; all reported metrics.

**Likelihood: Low.** The rules are explicit, current, and repeated; no violation so far.

**Impact: High.** A single fabricated/mislabeled number undermines the entire portfolio premise.

**Severity: Medium.**

**Mitigation.**
1. Every committed metric table must state: split, dataset fingerprint, config hash, command —
   mirroring the Phase 1 evaluation JSON fields (`cli.py:206-213`).
2. Validation vs test numbers must be visually distinct in reports (Phase 1 already separates
   `validation_report` from `eval_*.json`); keep that separation in Phase 2.
3. Same-metric, same-protocol comparison rule: Phase 2 numbers only comparable to Phase 1 when
   splits, fingerprints, and macro-F1 definition match (`docs/roadmap.md:22-23`).
4. Never report training loss as a quality metric in committed docs.

**Responsibility.** *Owner:* final review of every committed report. *Agent:* produce
aggregate-first, provenance-complete reports; never invent numbers (`AGENTS.md:10-12`).

**Evidence.** [VERIFIED FACT] `docs/experiment-protocol.md:73-79`; `AGENTS.md:10-12`;
`tests/conftest.py:1-7`; `cli.py:206-213`; `project-spec.md:73-74, 83-84`.

---

## R-12 — Vietnamese segmentation toolchain supply chain

**Description.** Phase 2 plans word segmentation (VnCoreNLP/underthesea — `docs/roadmap.md:20`
[VERIFIED FACT]). VnCoreNLP wraps a Java tool and downloads model `.jar` archives from external
hosts at setup time; underthesea ships models via its own hosting. Both add a non-pip supply
chain (Java version, jar integrity) that Phase 1's pip-only process does not cover, and both
versions materially affect tokenization and therefore results.

**Affected assets.** Reproducibility (segmentation version changes token sequences); workstation
(external downloads); CI feasibility (Java on runners).

**Likelihood: Medium.** Segmentation is on the Phase 2 plan; the tools are mainstream but the
artifact delivery is informal (direct downloads).

**Impact: Medium.** Silent tokenization drift between experiments; unverified jar download.

**Severity: Medium.**

**Mitigation.**
1. Prefer `underthesea` (pip-delivered, pinned) unless VnCoreNLP is shown to be required;
   record the exact segmentation tool + version in training metadata.
2. If VnCoreNLP is used: pin the jar by SHA256, download once into a gitignored local dir,
   document the Java version; never download at train time.
3. Note [UNVERIFIED]: whether PhoBERT's recommended usage actually needs segmentation for the
   chosen checkpoint/tokenizer — check the model card at Phase 2 start; if not needed, drop the
   dependency entirely.

**Responsibility.** *Agent:* evaluate the options, pin and record. *Owner:* approve the added
dependency (rule 7 discipline, `AGENTS.md:27-28`).

**Evidence.** [VERIFIED FACT] `docs/roadmap.md:20` (VnCoreNLP/underthesea named as the plan);
`docs/learning-notes.md:130-133` (WordPiece + segmentation expectations). Tool delivery
mechanics [ASSUMPTION based on the tools' public documentation].

---

## R-13 — Phase 2 training code invisible to CI

**Description.** CI is CPU-only lint+pytest (`ci.yml:31-32`; `HANDOFF.md:87-88` states "no data
downloads, CPU-only"). GPU training can never run in CI, so training-loop code paths (data
collation, trainer wiring, checkpoint saving/loading, label pipeline into the model) get zero
automated coverage unless deliberately designed for it. Additionally, the structural
artifact-exclusion gate exists only in the local `verify.py`, not CI (see R-08). [VERIFIED
FACTS]

**Affected assets.** Correctness of the training pipeline; the protection against committing
artifacts (R-08) is weaker in CI where it matters most (unknown local states don't run CI;
merges do).

**Likelihood: High.** Certain by construction — CI cannot execute GPU training.

**Impact: Medium.** Regressions discovered late (owner's machine, mid-experiment); a broken
artifact gate in the exact phase that produces heavy artifacts.

**Severity: High.**

**Mitigation.**
1. Architect Phase 2 for testability: pure functions for dataset building, tokenization,
   label alignment, metrics — unit-tested on tiny synthetic fixtures (extend the Phase 1
   conftest pattern, `tests/conftest.py`); mock the tokenizer where possible.
2. Add a CPU-smoke test that builds the model head (or a tiny config) and runs one forward
   pass if dependency weight allows; otherwise keep a documented manual smoke checklist.
3. Port `verify.py` structural checks into the CI workflow (see R-08 mitigation 2).
4. Pin CI's Python to the same minor version as local dev (3.12 today — already aligned,
   `ci.yml:17` / `HANDOFF.md:16-22`).

**Responsibility.** *Agent:* testable design + CI additions. *Owner:* approve workflow
changes.

**Evidence.** [VERIFIED FACT] `ci.yml:9-32`; `HANDOFF.md:87-88`; `docs/roadmap.md:24-25`
(GPU feasibility is explicitly uncertain — "do not assume one exists").

---

## Top risks (by severity, then impact)

1. **R-01** Untrusted checkpoints (pickle `.bin` deserialization) — High.
2. **R-06** Hidden near-duplicate leakage inflating Phase 2 test metrics — High.
3. **R-08** Gitignore/verify/CI gaps let HF-era artifacts and data be committed — High.
4. **R-07** Raw-vs-canonical label mapping sourced from an unverified declaration — High.
5. **R-04** License/user-agreement enforcement under Phase 2 data pressure — High.

Close behind: R-09 (test-set tuning without a ledger), R-10 (dependency drift), R-13 (CI
blindness to training code).

---

## Hardening pass 2026-11-07 (Red Team Agent E — adversarial review; evidence-only updates)

An adversarial pass attacked the readiness package itself: the hardware probe (working-tree
version), the 17 dossier documents, and the planned Phase 2 workflow. No risk was re-scored
except where noted; history above is unchanged.

**R-05 (PII in reports) — evidence strengthened, likelihood unchanged (Medium).** [VERIFIED
FACT] The end-to-end flow was confirmed at commit `95f7687`: `find_leakage`
(`src/vfi/data/validation.py:43-61`) returns a mapping **keyed by the normalized student
text**, which `cmd_prepare_data` embeds verbatim into `reports/artifacts/data_report.json`
(`cli.py:83-91`, written `:99`, printed `:101`). The only commit protection is the
`reports/artifacts/` ignore rule; `config.py:141` (`reports_dir` override) can relocate the
report to any unignored path, after which raw feedback would be committable. New direction
added to mitigation 1: treat any non-default `reports_dir` as untrusted for Phase 2 reporting.

**R-08 (accidental weight commits) — evidence strengthened, likelihood unchanged (Medium).**
[VERIFIED FACT, `git check-ignore` in the readiness clone 2026-11-07] `models/torch_model.bin`,
`models/weights.pt`, `models/check.ckpt`, `models/model.safetensors`, `models/model.onnx` are
all ignored via the unanchored `models/` rule (`.gitignore:33`); `pytorch_model.bin` at the
root, `checkpoints/epoch3.pt`, `artifacts/phobert/adapter_model.bin`, and `export/model.onnx`
are NOT ignored — the gap is exactly the extensions outside `models/`, as claimed. Note: the
clone's `.gitignore:33` is the unanchored `models/` variant, which is more protective than
Phase 1's root-anchored `/models/` (`95f7687:.gitignore:33`); the extension additions remain
required before the first HF download.

**R-13 (Phase 2 code invisible to CI) — evidence strengthened, likelihood unchanged (High).**
[VERIFIED FACT, `gh` API 2026-11-07] PR #2 (`research/phase2-readiness`, head
`fd91f4701d86380b99a4555a8ca26ef5aadea522`) has **zero status checks**: the branch carries no
CI workflow and `gh run list` shows no runs for it. CI exists only on `pull_request`/`push`
to `main` (`ci.yml:9-32`). PR #1: OPEN, Draft, MERGEABLE, CI green (run 42324940437, success
2026-11-07T01:36:08Z, after a superseded failed run 42324583725).

**R-11 (misleading accuracy claims) — audit result: no violation found in the dossier.**
[VERIFIED FACT] No document claims real accuracy, completed training, or CI success on PR #2.
The only CI claim found — `00-executive-decision.md:11` and `phase1-dependencies.md:6`
("PR #1 … CI green") — is accurate per the `gh` evidence above (Draft/Open, MERGEABLE,
latest run success). Runtime figures in `compute-feasibility.md` carry `[ESTIMATE]` labels
(section 3 preface labels the whole section; only line 119 restates an estimate as fact and
could take an inline label).

**New tool-level finding: hardware probe repository-refusal bypass (feeds R-05/R-08; probe
fix belongs to the readiness package, not Phase 1).** [MEASURED, end-to-end reproduction
2026-11-07, Windows 11 / Python 3.12.10] Crafted `--output` paths defeat the
repository-internal refusal (`_repo_internal_refusal`, `hardware_probe.py:428`): the
`\\?\`-prefixed and `\?\`-rooted forms resolve to drive-relative strings
(`Path.resolve()` quirk), and a trailing-space component resolves to a `\\?\`-prefixed string;
in all three cases the parents comparison against the plain repo root misses, the probe exits
0, and the detailed (username/hostname-bearing) file is written INSIDE the repository tree —
observed at `<repo>\project-personal\VFI-Phase2-Research\redteam_leak_a.json` and in the real
`docs/` directory. Files landed there are committable (no ignore rule covers them). A UNC
spelling (`\\localhost\D$\...`) also bypasses the refusal but writes outside the repo (new
`D:\localhost\...` tree). Reproduced guard-side, lower severity: environment spoofing
(`LOGNAME`/`USER`/`LNAME`/`USERNAME`, `USERPROFILE`) blinds the sanitization guard to the real
username on the `--output` stderr echo (`getpass.getuser()` env preference +
`Path.home()`→`USERPROFILE`); `COMPUTERNAME` spoofing does NOT affect `platform.node()`
(hostname guard holds); argparse `unrecognized arguments` echoes argv verbatim, unredacted,
before any guard. Verified COVERED (no bypass found): case variants, forward slashes, `.`/`..`
traversal, trailing dot/space on the file name, 8.3 short names on this volume (no short name
allocated for the repo directory), symlink-into-repo, and stdout/stderr on all normal code
paths (byte-level audit with `-I` from 4 working directories × 6 flag combinations: zero
personal values emitted). Values not matching username/hostname/home reach the summary
unredacted (e.g. another account's path inside a failing tool's stderr) — a documented limit
(`hardware-probe-guide.md:66-73`), not a new hole. Fix direction for the probe: strip
`\\?\`/`\\?\UNC\` prefixes and expand drive-relative forms before comparison (or re-resolve
with `strict=True` after the parent exists and re-check at write time), and route the argparse
echo through the same redaction contract. Reproductions are pinned by strict-xfail tests in
`tests/phase2_readiness/test_red_team.py`; `pytest tests/phase2_readiness -q` passes
(33 passed, 5 xfailed), ruff clean. Doc fixes required in `hardware-probe-guide.md` (lines 59,
63, 64, 104) are queued for the QA agent.
