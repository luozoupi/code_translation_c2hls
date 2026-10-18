# Registered coexistence of the backfill and main study

The historical backfill remains a four-worker queue. Every worker still executes
`artifacts/sigmetrics2027/registry_recovery_20260928/source/scripts/run_historical_cosim.py`
with its original frozen evaluator, manifests, and 3600-second timeout. The new
`scripts/resume_sigmetrics_cosim_coexist.py` changes only scheduling: it permits
Vitis processes descended from one authenticated main-study controller while
continuing to pause submissions for unrelated Vitis sessions.

The wrapper never sends a signal. It waits for the old controller to exit and
requires `state.json` to record `stopped` or `complete`, with no active workers,
before invoking the frozen scheduler. A handover lock prevents duplicate waiting
wrappers; the frozen scheduler additionally takes the original `queue.lock`.
SIGTERM or SIGINT during the wait cancels the handover without launching workers.
After resumption, the frozen scheduler's own signal handler drains active work.

## Preparation and review

First freeze the main study, including its `manifest.json` with schema
`c2hls.sigmetrics-recipe-main.v1` and
`source/scripts/run_sigmetrics_recipe_main.py`. Copy the coexistence wrapper into
a new scheduling directory before preparing the request. This copy must remain
unchanged: the request pins the wrapper's SHA-256.

Example commands (run from the SIGMETRICS worktree):

```bash
mkdir -p results_sweeps/historical_registry_recovery_20260928/scheduling_coexist_20260930
cp scripts/resume_sigmetrics_cosim_coexist.py results_sweeps/historical_registry_recovery_20260928/scheduling_coexist_20260930/
/home/luo00466/.conda/envs/py310_2/bin/python results_sweeps/historical_registry_recovery_20260928/scheduling_coexist_20260930/resume_sigmetrics_cosim_coexist.py --prepare --output results_sweeps/historical_registry_recovery_20260928 --change-request results_sweeps/historical_registry_recovery_20260928/scheduling_coexist_20260930/change_request.json --main-campaign results_sweeps/sigmetrics_recipe_main_20260930
/home/luo00466/.conda/envs/py310_2/bin/python results_sweeps/historical_registry_recovery_20260928/scheduling_coexist_20260930/resume_sigmetrics_cosim_coexist.py --check-only --output results_sweeps/historical_registry_recovery_20260928 --change-request results_sweeps/historical_registry_recovery_20260928/scheduling_coexist_20260930/change_request.json
```

Preparation records the old PID, `/proc` start ticks, command, controller source
hash, original manifest hash, all already completed measurement hashes, and the
main manifest/runner hashes. It saves a distinct before-handover snapshot of the
state, summary, scheduling status, and original launch receipt. It does not alter
campaign results or ask the old controller to stop.

The main controller writes `controller_identity.json` before launching tools.
The exemption requires its live PID and start ticks, matching frozen file hashes,
and the exact executable script and arguments:

```text
python -u <campaign>/source/scripts/run_sigmetrics_recipe_main.py run --output <absolute campaign>
```

The wrapper accepts Python execution with or without `-u`. An absent, terminated,
reused, or mismatched identity grants no exemption. Authentication is repeated
at each resource check; only descendants of that controller are exempted.

After reviewing the request, the operator may signal **only the recorded old
controller PID with SIGTERM after rechecking its start ticks**, allowing existing
workers to finish. Start the frozen wrapper with the same arguments as the check,
without `--check-only`; it waits up to 4800 seconds for a clean drain. Do not use
process-group termination or interrupt active validators. The main study may
begin during draining because its scratch directories and Vitis user directory
are separate. The new scheduler resumes remaining backfill work after the drain.

## Durable evidence and monitoring

- `coexist_scheduler_status.json`: new controller identity, scheduling source and
  request hashes, four-worker limit, fixed timeout, and transition status.
- `coexist_resource_gate.json`: authenticated main tool PIDs exempted from the
  external-session check, remaining external Vitis PIDs, and any authentication
  failure. Authentication failure never grants an exemption.
- The request directory holds distinct before-handover and after-drain snapshots;
  the latter includes hashes for every measurement completed during draining.
- Existing completed measurements are verified before resumption and after the
  frozen queue finishes. Neither failed measurements nor timeouts are relabeled
  or replayed by this scheduling change.

Focused verification: `tests/test_sigmetrics_cosim_coexist.py` covers exact
controller authentication, PID reuse, wrong commands/hashes, unrelated processes,
cycle-safe ancestry, controller exit during inspection, source/measurement drift,
non-overlapping handovers, and rejection of unclean drains. Combined with
`tests/test_historical_cosim_parallel.py`, 26 tests passed on 2026-09-30.
