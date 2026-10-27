# Changes

## v8 — historical branch rename, 2026-10-27

- Renamed `hlsmith-clean` to `v8` at the user's request; `v7` remains unassigned.
- Preserved the original tip `3b6081b02ae4daf9022e843cacb46ee850a61314` with the
  annotated tag `snapshot/20261027/v8` before this documentation update.
- Retained the full SIGMETRICS/Vitis research implementation, packaged runtime,
  paper prompt profile, numerical evidence replay, and controlled evaluation
  tooling. This rename makes no runtime changes.
- Kept reproduction commands and evidence qualifications. Historical results and
  fresh runs retain their separate provenance; unresolved or pending validation
  gates are not completed by renaming the branch.
- Identified [`HLSmiths-nano`](https://github.com/luozoupi/code_translation_c2hls/tree/HLSmiths-nano)
  as the minimal standalone successor, with no paper-reproduction claim.
  GitHub's default remains `v1`.

## HLSmith cleanup — historical consolidation

- Captured previously uncommitted SIGMETRICS sources as `v6`, preserving the existing commit ancestry.
- Organized runtime, agents, compiler execution, validation, optimization, provenance and research tools as an installable package, with historical entrypoints retained.
- Kept the paper's prompt templates intact and added a separately identified readable profile.
- Added portable historical fixtures, numerical evidence replay and captured-response compatibility checks.
- Prepared an interleaved DeepSeek V4.1-Flash-1017 comparison with explicit spending, compute and token caps.
- Removed generated reports and archived legacy translation files from the active tree; retained evidence in historical Git snapshots and the private archive.

Full experimental acceptance remains unestablished. See `docs/migration.md` and
`reproducibility/studies.json` for the preservation and evaluation requirements,
and the campaign records for fresh-run status.
