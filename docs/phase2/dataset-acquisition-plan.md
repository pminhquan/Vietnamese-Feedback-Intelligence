# Phase 2 Dataset Acquisition Plan

The dataset research was collected on 2026-11-07. The
[dataset landscape](dataset-landscape.md) and [license matrix](dataset-license-matrix.md)
record the source evidence and access conditions. No corpus was downloaded or approved
through this review.

## Select the Phase 1 benchmark first

Use independently collected Vietnamese feedback with human annotation and review, or a
Vietnamese sentiment dataset whose terms allow the intended use. Choose one dataset version
and freeze its train, validation, and test files before running the Phase 1 baselines.
Readiness depends on real benchmark evidence from that same version and holdout.

| Path | Evidence required before use |
|---|---|
| Owner-collected feedback | Document collection permission and provenance. Remove unnecessary personal details, label and approve records manually, then materialize a grouped split with the local [dataset workflow](../../README.md#build-an-independent-dataset). |
| Authorized five-level sentiment CSV | Confirm the release and permission terms. Verify that 1–2 are negative, 3 neutral, and 4–5 positive. Materialize separate split files and reserve the holdout. |
| NEU-ESC | The owner must accept the Hugging Face gate and assess the terms, including the uploader's rights over scraped content. Verify the delivered label encoding before exporting named labels. Record the explicit Toxic policy and exclusion count when benchmarking three classes. |
| UIT-VSFC | Optional adapter only. Signed corpus agreement and evidence for the original integer encoding remain prerequisites. The owner will not contact the authors or sign the agreement, so this path is inactive. |

Synthetic records can exercise the pipeline and remain marked synthetic after human review.
Synthetic runs do not qualify as real baseline or Phase 2 readiness evidence.

## Record an authorized data snapshot

1. Record the dataset identity, version, provenance and access evidence appropriate to the
   selected source. Keep private references and raw records local.
2. Use the documented [adapter layouts and label policies](../../data/README.md). Preserve
   separate files for fitting, selection, and test. Check for shared collection groups
   such as multiple responses from the same person or survey unit.
3. Run `vfi prepare-data` and `vfi evaluate --preflight-only` against the selected config.
   Resolve split overlap and contradictory labels before a clean benchmark. Preserve the
   passing receipt, source hashes and effective split fingerprints.
4. Run both Phase 1 baselines with `vfi train`. Keep selection on validation. Finalize the
   comparison plan before scoring with `vfi evaluate --model-id ...`. Preserve the
   evaluation artifact and the provenance required by the
   [experiment protocol](../experiment-protocol.md).

## NEU-ESC and comparability

The 2026-11-07 dataset review recorded a Hugging Face click-accept gate and an Apache-2.0
tag supplied by the uploader. The tag does not establish permission over the original
social-media text. The research also found a neutral-heavy distribution, different
collection channels from student surveys and no documented annotation protocol. See the
[NEU-ESC evidence](dataset-landscape.md#a2-neu-esc--vietnamese-educational-sentiment-classification).

Toxic is distinct from Negative. For a three-class benchmark with Toxic present, choose
`toxic_policy: exclude` with a written justification and report that the result is for
the non-Toxic subset. `keep` supports preparation audits only. Retain the same dataset,
mapping, exclusion policy and holdout when comparing a future model to the Phase 1
baselines. If those change, run both baselines again under the new protocol.

## Decision record

- [ ] Dataset, release/version, access and collection evidence recorded.
- [ ] Label semantics and any exclusions justified.
- [ ] Physical splits frozen, preflight passed and audit reports retained.
- [ ] Real Phase 1 baseline evidence from that dataset recorded.

The [UIT-VSFC mapping evidence](label-semantics-evidence.md) remains a historical
reference for the optional adapter. It does not block work on an independent dataset.
