# Running migration validation

The canonical development checkout is `/home/luo00466/code_translation_c2hls`.
Its ignored `runs` link resolves to the private NAS directory
`/mnt/data1/luo00466/c2hls_runs_20261018`.

Both campaigns use immutable runtime snapshots under
`/mnt/data1/luo00466/c2hls_archive_20261018/runtime`: `v6` and
`hlsmith-clean-f9e94eb`. The latter is the exact committed clean implementation
`f9e94ebd945e7db726625da3ba9bed2c554d9f62`. Their member manifests bind every
committed file and link. The canonical checkout remains available for further
development; editing it does not silently change the registered evaluators.

The approved limits are $100 API spend, 500 DeepSeek worker-hours, 250 saved-design
worker-hours, and 1,000,000 aggregate input tokens per DeepSeek cell. GPU jobs
remain stopped. The private controller runs saved-design validation first and
the DeepSeek comparison second. Before API use it rechecks the official Flash
release and pricing; a changed release, changed rates or uncertain billing
stops the campaign for review. It never changes a branch default.

The durable operational files are under `runs/migration-validation-r2/`:

- `approved_limits.json` records the user's explicit limits.
- `controller_status.json` records the supervisor and active stage PIDs.
- `saved_designs.log` and `deepseek.log` retain controller output.
- `run_authorized_campaigns.py` and `recheck_deepseek_pricing.py` are private
  machine-specific launch helpers; their checksums are retained beside them.

On October 20 UTC (October 19 in Chicago), the user requested parallel Vitis
sessions. The saved-design controller supports `--workers`; this queue uses a
limit of four. Each worker keeps its registered inputs, evaluator snapshot,
Vitis 2023.2 binary, U280 target, 3.33 ns clock and per-phase timeouts. All scratch
files remain on NAS. The aggregate 250-hour allowance includes both the stopped
trial and all serial and parallel worker time.

The handoff first sends SIGTERM only to the old serial controller, whose handler
finishes its active cell before stopping dispatch. The replacement supervisor
waits for the old supervisor's lock, verifies that no cell remains active, and
copies the drained state and original launch receipt into a timestamped
`parallel-handoff-*` directory. It resumes only pending cells from the same
manifest. `parallel_transition.json` records the wait, handoff and active stage;
the new helper is `run_parallel_authorized_campaigns.py`. It retains the
original sequential DeepSeek runner and all API limits.

The new scheduling code is exported separately and recorded by path and hash.
It does not replace the registered v6 or `hlsmith-clean-f9e94eb` evaluators. New
execution epochs distinguish parallel checks from the preceding serial checks.
Wall-time censoring can change under concurrent load, so compare those segments
explicitly and retain the earlier LU pass/timeout asymmetry. Four worker slots
are an execution limit, not a promise of fourfold speedup or evidence of QoR
equivalence.

The key is read from the user's private local credential file. Its contents are
excluded from registration, run metadata and Git. Public reproduction commands
accept `DEEPSEEK_API_KEY` through the environment.

Inspect progress without changing it:

```bash
cat runs/migration-validation-r2/controller_status.json
cat runs/migration-validation-r2/parallel_transition.json
cat runs/saved-design-validation-20261018-r2/state.json
cat runs/cleanup-comparison-20261018-r2/state.json
```

To regenerate the saved-design report after the parallel handoff, use the
`controller_runtime` path recorded in `parallel_transition.json`. Running the
older scheduling module would omit execution-epoch fields from its report.
Keep the evaluator snapshot on `PYTHONPATH`:

```bash
export C2HLS_CONTROL_RUNTIME="/path/from/parallel_transition.json"
PYTHONPATH=/mnt/data1/luo00466/c2hls_archive_20261018/runtime/hlsmith-clean-f9e94eb \
python "$C2HLS_CONTROL_RUNTIME/hlsmith/saved_designs.py" report \
  --output /mnt/data1/luo00466/c2hls_runs_20261018/saved-design-validation-20261018-r2
```

The DeepSeek report still uses its original pinned implementation:

```bash
cd /mnt/data1/luo00466/c2hls_archive_20261018/runtime/hlsmith-clean-f9e94eb
python -m hlsmith.experiments report \
  --output /mnt/data1/luo00466/c2hls_runs_20261018/cleanup-comparison-20261018-r2
```

Scratch workspaces and fresh measurements stay on NAS. Reports retain failures,
timeouts, missing roles and unfinished cells. An interrupted active cell needs
evidence review before resuming. A fresh-generation campaign does not silently
resume or reroll. Changing source, inputs, prompts or skills requires a new
registration and preserves the old one as superseded.

The historical follow-up queue and its original shared Git owner remain at
their old local paths. Retire those last directories only after their live
dependencies are gone. Frozen checkpoint studies remain separate: their
available evidence and assets are listed in `reproducibility/studies.json`,
alongside the missing exact serving/evaluation bindings. No GPU job or training
run is part of this controller.


The initial hardware trial remains at `saved-design-validation-20261018` and
`migration-validation`. It found a clean-export omission and missing Vitis XML
clock metadata handling. The frozen candidate's native, CSim and RTL checks
passed, but the original qualification failure remains unchanged; the clean
worker failed during import. The corrected source and wheel were checked from
an independent Git export before creating `r2`. No design was repaired or
reselected. The original trial consumed 114.64893941092305 worker-seconds, so
`r2` receives only 249.96815307238586 remaining hours from the approved 250-hour
total. `saved-design-carry-forward.json` binds the original state and manifest
hashes. Both trial folders and every old measurement remain in place.
