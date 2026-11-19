# data/ — local datasets (NEVER committed)

Keep private datasets under an ignored directory such as `data/datasets/`
or `data/raw/`. Keep raw text, record IDs and annotation references out of
Git and reports. Use aggregate counts and whole-split fingerprints.

## Independent datasets

Point `data.raw_dir` at the directory of split files. Set the appropriate
`dataset_adapter`, a non-sensitive `dataset_id` and `dataset_version`, and
declare `rights_confirmed: true` only after checking permissible use.
No adapter verifies a license or downloads files.

| Adapter | Files | Label contract |
| --- | --- | --- |
| `custom_jsonl` | `train.jsonl`, `validation.jsonl`, `test.jsonl` | Export from `vfi dataset split`. Records retain their source and version and need human annotation and approval. |
| `five_level_csv` | `train.csv`, `validation.csv`, `test.csv` | UTF-8 with `text,label`. Use only an ordered sentiment scale where 1–2 means negative, 3 neutral, and 4–5 positive. Verify this interpretation against the source's label guide. |
| `neu_esc` | `train.csv`, `validation.csv`, `test.csv` | UTF-8 with `text,label`. Convert to named labels only after checking the authorized release's schema. Accepted names are Negative, Neutral, Positive, and Toxic. |

For NEU-ESC, `toxic_policy: error` is the default. Use `exclude` with a
`toxic_justification` to benchmark sentiment on the remaining subset, or `keep`
with a justification for a local preparation audit. Reports count exclusions.
The three-class models cannot fit or score `keep` mode. Toxic is a separate label.
Report the explicit exclusion policy when interpreting results.

Custom exports contain `record_id`, `text`, `sentiment_label`, `source_type`,
`provenance_reference`, `annotation_method`, `annotation_status`,
`dataset_version`, and `split`. They stay local. Synthetic exports require a
separate fixture run with `synthetic_fixture: true`; do not set this flag for
real records. Use versioned paths and preserve a frozen holdout.

Use a non-sensitive local provenance reference to link each real record to
its origin and consent or permission evidence. The CLI checks that this
reference is present. Review and confirm the underlying permission separately.
Dataset writes are serialized for each local store. If a record changes while
you review it, reopen it and confirm the current label.

## UIT-VSFC layout (after the owner obtains access)

```text
data/raw/uit_vsfc/
├── train/
│   ├── sentences.txt     # one feedback sentence per line (UTF-8)
│   ├── sentiments.txt    # one integer label per line, mapping verified by owner
│   └── topics.txt        # optional for Phase 1 sentiment work
├── validation/
│   └── ... (same three files)
└── test/
    └── ... (same three files)
```

Notes:

- Files must be line-aligned within each split (line N of `sentences.txt`
  corresponds to line N of `sentiments.txt`). `vfi prepare-data` validates
  train and validation files. `vfi evaluate --preflight-only` checks alignment
  and exact cross-split matches across all effective splits before training.
- When using `data.use_official_splits: false`, VFI derives validation from the
  train files. The raw `validation/` directory is optional in this mode.
- `utf-8-sig` is tolerated (a BOM written by Windows editors is stripped).
- The raw `0/1/2` mapping remains unresolved. Use the `uit_vsfc` adapter only
  after the corpus agreement and confirmation of the original file encoding.
  Configure `rights_confirmed: true`, `label_mapping_source: owner_confirmed`,
  `label_mapping_evidence`, and the explicit three-entry `label_mapping`.
  See `docs/dataset-uit-vsfc.md`.
- Do not rename files: `vfi.data.loading.load_split` expects exactly
  `sentences.txt` and `sentiments.txt` inside each split directory.
