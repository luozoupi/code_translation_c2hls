# Cleanup, 27 October 2026

The active workbench keeps eight Python modules, one prompt and one CLI.
The runtime and prompt bytes are unchanged by this cleanup. The five focused
regression modules remain; they generate their own negative cases in temporary
directories.

Removed the three standalone scalar failure demonstrations and the redundant
second candidate file. Both saved responses remain embedded in `responses.json`.
The pre-cleanup snapshot preserves every removed file:
`snapshots/minimal-before-cleanup-20261027` (`951d6f1`).

The main README is shorter; detailed tool instructions live in
[backends.md](backends.md). Original validation receipts moved unchanged into
[evidence/](evidence/README.md). HLSFactory inputs, manifests, notices and licenses
retain their original paths and bytes.

## Local storage

The checkout occupies about 15 MB including its virtual environment and Git
history, down from about 14 GB. Optional tool installations and generated data
are outside the checkout. New runs should use `--out results/NAME`; this directory
is ignored by Git. Historical `runs/` is a compatibility link for old evidence.

| Original project path | Current storage |
| --- | --- |
| `runs/` | Compatibility link to `/mnt/data1/luo00466/hlsmith_minimal_archive_20261027/runs` |
| `.tools/` | Compatibility link to `/home/luo00466/.local/share/hlsmith-minimal/tools` |
| `.tools/bambu/bambu-2024.10.AppImage` | Link to the archive's `installers/bambu-2024.10.AppImage` |
| `dist/` | Archived under `build-artifacts/dist/`; resolve old wheel paths there |
| `build/` | Archived under `build-artifacts/build/` |

The archive root is `/mnt/data1/luo00466/hlsmith_minimal_archive_20261027`.
Its `manifest.json` maps original paths and hashes. All 5,757 regular files
(8,569,069,124 bytes), directories and symlinks were checked before retiring local
copies. A Git bundle was verified and cloned, its snapshot checked out, and
representative result/source/RTL files restored and hashed. `RESTORE.md` records
the recovery procedure. Recursive debug copies are retained as raw diagnostics;
they are not scientific results.

Build directories, old wheels, package metadata and Python/test caches were
removed from the active checkout. No active project jobs were found before
migration. The original `.tools/` link preserves embedded Conda prefixes.

## Verification

All 90 tests passed. Both relocated tool entrypoints still start. An installed
CLI run from `/tmp` passed the MVT native oracle and three Bambu/Verilator checks
with zero-ULP comparison: 4,626 cycles per call, 13,878 total. It used a saved
baseline and made no provider requests. Raw checks are outside the checkout at
`/home/luo00466/.local/share/hlsmith-minimal/checks/cleanup-20261027`.

[CLEANUP_VALIDATION.json](evidence/CLEANUP_VALIDATION.json) records archive
checksums, path mappings and the final checks. The separate research repository
and its historical experiments were unchanged.
