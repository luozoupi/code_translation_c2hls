# Running migration validation

The canonical development checkout is `/home/luo00466/code_translation_c2hls`.
Its ignored `runs` link resolves to the private NAS directory
`/mnt/data1/luo00466/c2hls_runs_20261018`.

Both campaigns use immutable runtime snapshots under
`/mnt/data1/luo00466/c2hls_archive_20261018/runtime`: `v6` and
`hlsmith-clean-79db16f`. The latter is the exact committed clean implementation
`79db16f28f4681394bdf2c3da50fc50bd6a47a45`. Their member manifests bind every
committed file and link. The canonical checkout remains available for further
development; editing it does not silently change the registered evaluators.

The approved limits are $100 API spend, 500 DeepSeek worker-hours, 250 saved-design
worker-hours, and 1,000,000 aggregate input tokens per DeepSeek cell. GPU jobs
remain stopped. The private controller runs saved-design validation first and
the DeepSeek comparison second. Before API use it rechecks the official Flash
release and pricing; a changed release, changed rates or uncertain billing
stops the campaign for review. It never changes a branch default.

The durable operational files are under `runs/migration-validation/`:

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
cat runs/migration-validation/controller_status.json
cat runs/saved-design-validation-20261018/state.json
cat runs/cleanup-comparison-20261018/state.json
```

To regenerate reports with the pinned implementation, run from its snapshot:

```bash
cd /mnt/data1/luo00466/c2hls_archive_20261018/runtime/hlsmith-clean-79db16f
python -m hlsmith.saved_designs report \
  --output /mnt/data1/luo00466/c2hls_runs_20261018/saved-design-validation-20261018
python -m hlsmith.experiments report \
  --output /mnt/data1/luo00466/c2hls_runs_20261018/cleanup-comparison-20261018
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
