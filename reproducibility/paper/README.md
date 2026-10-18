# Frozen SIGMETRICS evidence

This is a portable numerical reference for paper source commit
`ecd3c750168dea52aa80dd609f463b9b668689b0`. `reference.json` records the source
identity, file checksums and the historical stale-table-hash caveat. It does
not identify the exact submitted PDF.

From the checkout root:

```sh
python reproducibility/paper/replay.py
python -m unittest discover -s reproducibility/paper/tests -v
```

These commands require matplotlib and run without API, Vitis or live
experiment access. They preserve the original selected designs, cohort
exclusions, validation tiers and timeout policy. A passing replay verifies
archived evidence and calculations; it is not a new hardware or model rerun.

The copied `generated/`, `scripts/` and `tests/` files retain their original
bytes. Manuscript prose and PDFs are excluded; `reference.json` retains their source
identity and hashes for the external archive. The v6 prompt source under `prompts/` binds the
default runtime compatibility profile separately from the paper commit.
Render figures only from a scratch copy, so the frozen snapshot stays intact.
See [the paper contract](../../docs/paper-contract.md) for performance values,
prompt-version policy and full-rerun acceptance boundaries.
