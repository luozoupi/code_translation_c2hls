# Archive HLS sol bulk artifacts (inode reclaim)

**Date:** 2026-07-25  
**Status:** Approved for implementation planning  
**Root:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp`  
**Motivation:** Scratch inode use ~97% (≈9.6M / 10M). ~8.2k `.autopilot` solution dirs under `c2hls_tmp`; each sol often has hundreds of files (`.autopilot` dominates).

## Goals

1. Reclaim **file/inode quota** by zipping bulky finished HLS solution artifacts **in place**.
2. After **successful verified** zip, delete only the archived bulk members.
3. Keep everything the pipeline still reads live (or for later harvest/export) on disk.
4. Never touch in-progress runs: **process check + ≥6h age gate**.

## Non-goals (v1)

- Reading archives from `hls_feedback.py` / eval (keep live files instead).
- Cron / systemd automation (CLI only).
- Archiving outside `c2hls_tmp` (optional `--root` allowed, default is `c2hls_tmp`).
- Deleting `syn/` or work-dir harvest logs.

## Approach

**Per-sol sidecar zip** (Approach 1): one `sol_bulk.zip` inside each finished solution directory. Restartable, local restore, incremental inode reclaim.

## Discovery

- Walk `--root` for directories named `.autopilot`.
- Unit of work = parent of `.autopilot` (typically `.../hls_proj/sol1`, also accept other `sol*`).
- Skip if `.sol_bulk_ok` already exists next to a valid `sol_bulk.zip` (idempotent).

## Finished gates (both required)

1. **Age:** solution directory mtime is **≥ 6 hours** ago (`--min-age-hours`, default 6).
2. **Process:** no `vitis_hls` / `vitis-run` / `xelab` / `xsim` (and close variants) with open files or cwd under the solution’s **top-level batch** tree under `--root`.

If either gate fails → skip (do not zip/delete).

## Keep on disk (never delete)

| Path | Reason |
|------|--------|
| Entire `syn/` | csynth reports / XML; pipeline + harvest |
| `.autopilot/db/*.xml` | `burst.xml`, `fe_messages.xml`, `be_messages.xml` (and any other db XMLs) |
| `sol*.log` | export/harvest (`sol1.log`) |
| Work-dir `logs/hls_run_tcl.log` | export/harvest (outside sol; not archived by this tool) |
| `sim/report/**` | cosim reports |
| `csim/report/**` | csim reports |
| `**/*.result.lat.rb` | cosim cycle harvest (`hls_eval._parse_lat_rpt_cycles`) |

Also leave in place after success: `sol_bulk.zip` and `.sol_bulk_ok`.

## Zip + delete after verify (if present)

Archive members are stored **relative to the sol directory**.

| Member | Notes |
|--------|--------|
| `.autopilot/**` except kept `db/*.xml` | Zip bulk; after verify delete non-kept autopilot files/dirs; leave `db/*.xml` |
| `.debug/` | whole tree |
| `impl/` | whole tree |
| `sim/` except `sim/report/` and `*.result.lat.rb` | whole remainder |
| `csim/` except `csim/report/` and `*.result.lat.rb` | whole remainder |
| Sol-root sidecars | `*.aps`, `*.directive`, `*.cfg`, `*_data.json`, `vitis-comp.json`, other sol-root `*.json` — **not** `sol*.log` |

`syn/` is **included in the zip for backup** but **never deleted**.

## Verify-then-delete protocol

1. Write `sol_bulk.zip.partial` (atomic create).
2. Add all intended members (including `syn/` for backup).
3. Integrity: CRC test (`zipfile.testzip` or `zip -T`).
4. Membership: every intended path present; sizes match.
5. Rename `.partial` → `sol_bulk.zip`.
6. Write `.sol_bulk_ok` (JSON: member count, uncompressed bytes, zip bytes, timestamp, tool version).
7. Delete **only** the delete-set members that passed verify.
8. On any failure: leave originals intact; remove partial zip; log and continue.

## CLI

`scripts/archive_hls_sol_bulk.py`

```text
--root PATH              default: <repo>/c2hls_tmp
--min-age-hours FLOAT    default: 6
--jobs N                 default: 4–8
--dry-run                discover + report only; no zip/delete
--limit N                process at most N sols (pilot)
--batch-prefix PREFIX    optional filter on top-level batch dir name
--strict                 non-zero exit if any sol hard-failed
```

## Scale execution

1. Dry-run over full root → JSONL candidates + projected file/inode savings.
2. Live pilot: one finished batch (`--batch-prefix` or `--limit`).
3. Full run with moderate `--jobs` (scratch I/O bound).

Concurrency: one worker per sol; never two workers on the same path. Resume via `.sol_bulk_ok`.

## Reporting

- Progress on stderr.
- JSONL report (default under cwd or `--report PATH`): one record per sol (`skipped` / `archived` / `failed`, reasons, member counts, bytes).
- Exit 0 unless `--strict` and ≥1 hard failure.

## Error handling

- Zip/verify failure → no deletes for that sol.
- Missing optional trees (no `sim/`, no `impl/`) → not an error.
- Empty delete-set after keep filtering → skip zip or write empty marker policy: **skip** (nothing to reclaim).

## Testing

- Unit: keep/delete classification on a synthetic sol fixture.
- Unit: verify refuses delete when a member is missing from zip.
- Integration (optional): tiny fixture under tmp → dry-run → live → assert keep-set remains, delete-set gone, zip tests clean.

## Success criteria

- Finished sols reclaim the bulk of `.autopilot` / `.debug` / `impl` / `sim`/`csim` (non-report) inodes.
- Pipeline can still harvest `.autopilot/db/*.xml`, `syn/report`, `sol*.log`, `logs/hls_run_tcl.log` without unzipping.
- Re-running the tool is a no-op on already-archived sols.
- Active (<6h or live process) trees are never modified.
