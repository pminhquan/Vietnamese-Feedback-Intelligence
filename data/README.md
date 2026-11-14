# data/ — local datasets (NEVER committed)

This directory holds raw and processed datasets locally. **Raw data is never
committed to Git** — `.gitignore` excludes everything here except this
README. Only aggregate statistics (counts, fingerprints) and short,
truncated/paraphrased error-analysis fragments may appear in committed
reports.

## UIT-VSFC layout (after the owner obtains access)

```text
data/raw/uit_vsfc/
├── train/
│   ├── sentences.txt     # one feedback sentence per line (UTF-8)
│   ├── sentiments.txt    # one integer label per line: 0=positive, 1=negative, 2=neutral
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
- Before running experiments, sanity-check the `0/1/2` label encoding against
  the author's README — see `docs/dataset-uit-vsfc.md`.
- Do not rename files: `vfi.data.loading.load_split` expects exactly
  `sentences.txt` and `sentiments.txt` inside each split directory.
