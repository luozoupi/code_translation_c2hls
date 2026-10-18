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

The key is read from the user's private local credential file. Its contents are
excluded from registration, run metadata and Git. Public reproduction commands
accept `DEEPSEEK_API_KEY` through the environment.

Inspect progress without changing it:

```bash
cat runs/migration-validation-r2/controller_status.json
cat runs/saved-design-validation-20261018-r2/state.json
cat runs/cleanup-comparison-20261018-r2/state.json
```

To regenerate reports with the pinned implementation, run from its snapshot:

```bash
cd /mnt/data1/luo00466/c2hls_archive_20261018/runtime/hlsmith-clean-f9e94eb
python -m hlsmith.saved_designs report \
  --output /mnt/data1/luo00466/c2hls_runs_20261018/saved-design-validation-20261018-r2
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
