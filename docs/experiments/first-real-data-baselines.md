# First real-data baseline experiment

**Status: blocked on dataset access and provenance.** The first configuration
targets NEU-ESC because it covers education feedback and declares three
physical splits. No source has been selected for a real run. No real rows
were acquired, inspected, split, or scored. Data-contract tests use generated
fixtures. Their outputs carry `synthetic_fixture` scope.

## Source decisions

Metadata and publisher materials checked on 2026-11-15. Counts below are
publisher claims. The staged files, data quality, and class balance still
need to be checked against an authorized release.

| Candidate | Decision and evidence |
| --- | --- |
| **NEU-ESC** | **Conditional.** The [pinned Hugging Face metadata](https://huggingface.co/api/datasets/hung20gg/NEU-ESC/revision/daf543ad1992153cd2be9fec3cb59aa0fc714147) reports an automatic access gate, an Apache-2.0 tag, and three CSVs. The [original paper](https://arxiv.org/pdf/2506.23524v1) describes university Facebook groups and forums, including XamVN and Voz, and reports train/validation/test sizes of 23,048/3,305/6,613. Its label counts sum to 33,016, 50 above the split total. Neutral dominates, Toxic is a separate fourth label, and no independent annotation agreement or source permission was established from the materials reviewed. |
| **Multi-level sentiment Vietnamese** | **Conditional access, deferred for this benchmark.** The [pinned Hugging Face metadata](https://huggingface.co/api/datasets/sonlam1102/multilevel_sentiment_vietnamese/revision/e1e1395146124f84480be387ab618710de1673ac) reports an automatic gate, an MIT tag, and one CSV. The [original paper](https://journals.sagepub.com/doi/pdf/10.3233/FAIA210043) describes Tiki/Shopee reviews, two annotators with reported κ = 0.85, and expansion from 2,983 to 5,971 rows through more collection, round-trip translation and synonym substitution. The paper's 4,178/598/1,195 split counts are not verified in the hosted file. Levels 1–2/3/4–5 are the proposed negative/neutral/positive collapse after verification. An independent original-only holdout and grouping of derivative items with their originals are required. The [author's repository](https://github.com/sonlam1102/multi-level-sentiments) directs users to the gated release. A visible archive does not establish another permitted channel. |
| **UTS2017 Bank** | **Conditional.** The [publisher's pinned card](https://huggingface.co/datasets/undertheseanlp/UTS2017_Bank/blob/4d1319fa229720d392c1dc0e323d10b40eeacbd0/README.md) explicitly releases the dataset under Apache-2.0, and the [metadata](https://huggingface.co/api/datasets/undertheseanlp/UTS2017_Bank/revision/4d1319fa229720d392c1dc0e323d10b40eeacbd0) confirms it is ungated. The card describes public sources and banking communications and claims anonymization, without identifying original sources or the scope of contributor and bank permissions. Its sentiment JSONL has train/test, 1,977/494 examples, and three classes with sparse neutral support. The [published converter](https://huggingface.co/datasets/undertheseanlp/UTS2017_Bank/blob/4d1319fa229720d392c1dc0e323d10b40eeacbd0/preprocess.py) derives the overall label from the most frequent aspect label. The [publisher's technical report](https://huggingface.co/undertheseanlp/sen-1/blob/b5fd35d67636893618da05a895e3a2bbbc964ec9/TECHNICAL_REPORT.md#42-uts2017_bank-dataset) disagrees with the card on the positive/negative train counts. The task and domain differ from student feedback. |
| **FPT comments** | **Blocked.** The ungated release has a CSV layout and an education domain, but its publisher restricts it to FPT coursework. It includes scraped community comments and LLM-assisted labels. [Pinned publisher card](https://huggingface.co/datasets/lducc/fpt-comments-sentiment/blob/00a9491abe55175f4e8b176ed6470a691cbb36be/README.md) |
| **Vietnamese AI feedback OOD 3125** | **Blocked for fitting.** This ungated publisher release exposes raw and segmented variants of one 3,125-row test-only collection, with 0 negative, 1 neutral and 2 positive. Its CC BY 4.0 tag accompanies an anonymization claim, but the card does not specify collection permissions or provide a fit/selection split. Keep that test-only collection outside this experiment. [Pinned publisher card](https://huggingface.co/datasets/morex3/vietnamese-ai-feedback-ood-3125/blob/de7a66bdb1f9b38e5fd71d70953a300b8a6a3d45/README.md) |
| **ViDA and ViDia2Std** | **Blocked for this run.** ViDA annotates three sentiment values on functional segments of dialogue, using Babel Vietnamese transcripts. No permitted source of the annotated corpus was verified, and the original transcript pack has its own agreement. ViDia2Std is an ungated CC BY-NC 4.0 dialect translation corpus. Its three sentiment labels are generated for the test CSV only. [ViDA paper](https://aclanthology.org/L18-1630/), [Babel source terms](https://catalog.ldc.upenn.edu/LDC2017S01), [ViDia2Std card](https://huggingface.co/datasets/Biu3010/ViDia2Std/blob/5e2ab80218dad7eaa98bfced1d3362910fb6ef48/README.md), [ViDia2Std paper](https://arxiv.org/html/2603.10211v1) |
| **Owner-collected feedback** | **Conditional.** Use collection permission, documented origin, and records labeled and approved by a human. The [released JSONL workflow](../../README.md#build-an-independent-dataset) supports grouped splits and keeps unapproved records out. No local authorized source is available. |
| **UIT-VSFC** | **Inactive.** The [source agreement](../dataset-uit-vsfc.md) and verification of that corpus's encoding apply only to that route. |

For NEU-ESC, the owner would sign in on the
[official dataset page](https://huggingface.co/datasets/hung20gg/NEU-ESC),
read the visible terms, and submit access from their account if acceptable.
The [Hub access guide](https://huggingface.co/docs/hub/datasets-gated#access-gated-datasets-as-a-user)
describes the information shared on request. Keep a local reference to the
applicable terms and separately establish that the intended local use covers
the contributed feedback. A working download and a license tag establish
neither annotation quality nor the scope of the original-content rights.

## Stage and audit the authorized snapshot

1. Use exactly the revision named in the
   [NEU-ESC config](../../configs/experiments/neu-esc-first-baselines.yaml),
   or create a distinct run for a different revision. Save the raw files and
   source SHA-256 hashes under ignored `data/datasets/`. Retain the pinned
   URL, terms/access evidence reference, source date, expected files and
   deterministic conversion steps in an ignored manifest.
2. Confirm the actual release schema and meanings before conversion. The
   paper lists 0 Neutral, 1 Positive, 2 Negative and 3 Toxic. Verify this
   against the authorized release documentation and a local human review
   confined to train/validation. If a numeric mapping remains uncertain,
   stop. The adapter expects named labels. Stage UTF-8 `train.csv`,
   `validation.csv` and `test.csv` with headers `text,label` (optional
   `split` requires the canonical file's split name). Preserve the source
   partitions where viable. The adapter excludes Toxic and reports the
   source and exclusion counts per split. Keep the benchmark claim scoped
   to the resulting three-class subset.
3. Review split membership by respondent, course, source thread or other
   available reliable collection IDs. Keep each group in one split if
   partitioning is necessary. Record the group availability, decisions,
   deterministic rule and seed, and missing identifiers as a limitation.
   Review legitimate repeated feedback locally. Correct cross-split
   duplicates and contradictory labels before freezing. Train and
   validation need all three classes for an interpretable comparison.
4. Run a separate label-blind near-duplicate audit on all effective records
   for train/validation, train/test, and validation/test. Record the
   normalization, similarity threshold, numbers examined per split, each
   pair's findings and the treatment of unresolved groups, with the same
   effective fingerprints as preflight. A feasible method is exact Jaccard
   over sets of normalized character trigrams, using a postings index to
   compare pairs with a shared trigram and a threshold fixed before review.
   Handle very short text explicitly and review candidates locally. Reports
   contain only aggregates. `prepare-data --near-duplicates` examines at
   most the first 500 train/validation records and cannot establish this
   coverage. This full audit remains pending authorized data.

## Run and retain the validation experiment

Run from the repository root in the isolated Python 3.12 virtual environment.
For a fresh environment, install with `python -m pip install -e ".[dev]"`.
Copy the checked-in YAML into the ignored run directory. Enable
`rights_confirmed` only in that local copy after documenting the access
evidence. Leave `synthetic_fixture: false` for the real run.

```powershell
$run = 'reports/artifacts/neu-esc-daf543ad-baselines'
New-Item -ItemType Directory -Force $run | Out-Null
Copy-Item configs/experiments/neu-esc-first-baselines.yaml "$run/config.yaml"
```

Review and edit `config.yaml` locally after completing the access, schema,
and near-duplicate audit steps. Then use the reviewed copy for all commands.

```powershell
$py = '.\.venv\Scripts\python.exe'
$run = 'reports/artifacts/neu-esc-daf543ad-baselines'
$cfg = "$run/config.yaml"
git rev-parse HEAD | Set-Content -Encoding ascii "$run/git-sha.txt"
& $py -m pip freeze --all --exclude-editable | Set-Content -Encoding ascii "$run/environment.txt"
& $py -m vfi.cli prepare-data --config $cfg
if ($LASTEXITCODE -ne 0) { throw 'Data preparation failed' }
& $py -m vfi.cli evaluate --preflight-only --config $cfg
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed' }
$receipt = "$run/evaluation_preflight.json"
$preflight = Get-Content $receipt -Raw | ConvertFrom-Json
if ($preflight.passed -ne $true) { throw 'Preflight did not pass' }
if ($preflight.dataset_provenance.identity.synthetic_fixture -ne $false) {
    throw 'Expected real-data preflight evidence'
}
(Get-FileHash -Algorithm SHA256 $receipt).Hash.ToLowerInvariant() |
    Set-Content -Encoding ascii "$run/preflight-sha256.txt"
# Review the preflight and the full near-duplicate audit before fitting.
& $py -m vfi.cli train --config $cfg
if ($LASTEXITCODE -ne 0) { throw 'Baseline training failed' }
```

Run each snapshot in its own ignored directory. Keep the config, receipt,
audit, environment, and source manifest together. Before fitting, verify the
three effective SHA-256 fingerprints, zero exact cross-split conflicts and
zero contradictory labels in the receipt. Preflight uses the reserved test
only for integrity. Training reads train and validation only. It fits the
majority class and all TF-IDF candidates on train. Selection uses validation
macro-F1 and keeps test sealed for the common comparison.

Retain `training_report.json` and trusted local model metadata for the two
baselines. Verify the training timestamp follows preflight and the train
and validation fingerprints match. Report validation accuracy, macro-F1,
per-class metrics/support, class distribution and the confusion matrix.
The majority metadata has fit time, the trial table has candidate fit times,
and the report has total run time. The selected refit is not separately
timed by the released CLI. For validation latency, use the same samples from
the verified validation split and the existing `measure_latency` protocol
(batch 1, 10 warm-ups, 100 calls, seed 42, mean/median/p95). Record CPU and
software versions without machine identifiers. Review errors on validation
and publish aggregate categories and confusion counts with no raw feedback.

The handoff to [Phase 2](../phase2/evaluation-protocol.md) requires actual
validation results for both baselines, the frozen receipt and a matching
holdout for the eventual three-model comparison. This experiment does not
authorize final test scoring or PhoBERT fine-tuning.
