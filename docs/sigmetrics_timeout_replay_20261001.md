# Separate longer-timeout RTL replay (October 1, 2026 UTC)

The registered 216-cell recipe study remains untouched and continues in
`c2hls_sigmetrics_recipe_main_20260930`. This sensitivity campaign replays only
outcomes that had a recorded 3,600-second RTL co-simulation timeout. It does not
change the main study's parent gate, registered cells, search policy, or result
denominator, and it never replaces the original backfill measurements.

The replay uses the frozen main-study parent builder for four unchanged public-C
parents and the frozen historical backfill worker for 102 distinct timed-out
validations (169 historical rows). It retains the original code, input suite,
toolchain, clock target, and validator. The only validation-policy change is the
RTL co-simulation timeout, from 3,600 to **10,800 seconds per RTL case**. Parent
CSim and synthesis are repeated by the frozen builder; their new measurements
are kept separately. Historical jobs receive new replay validation IDs linked to
the original IDs and timeout measurement hashes. No model API calls are made.

Two parent workers and two historical workers share a guarded controller. Its
maximum of four concurrent jobs plus the main study's two workers matches the
six-way Vitis concurrency previously used by the main study and backfill. New
jobs pause if free scratch falls below 100 GiB, available memory below 128 GiB,
one-minute load exceeds 70% of logical CPUs, the registered main controller
loses its process identity while marked running, or an unrelated Vitis process
appears. Every job uses distinct scratch and Vitis user-home directories.

The original four parents passed C simulation and synthesis; each timed out
after XSIM started. Their original synthesis estimates were:

| Parent | HLS estimated cycles | Original RTL outcome |
|---|---:|---|
| Jacobi-2D | 3,112,241 | 3,600-second timeout |
| 2mm | 25,296,077 | 3,600-second timeout |
| Seidel-2D | 132,556,483 | 3,600-second timeout |
| LU | 134,372,041 | 3,600-second timeout |

The preserved XSIM logs show advancing simulated time and repeated DSP OPMODE
warnings. That evidence is consistent with slow simulation; it does **not**
establish a functional deadlock or predict which jobs will pass at 10,800
seconds. The warning-heavy logs are large (roughly 4--167 MB per parent), so
replay storage is monitored by the resource gate. The original failure labels
remain timeouts until a separate replay records an independently checked result.

Launch: `c2hls_sigmetrics_timeout_replay_20261001` tmux session.
Controller: `scripts/run_sigmetrics_timeout_replay.py`.
Output: `results_sweeps/sigmetrics_timeout_replay_20261001/`.
Read `summary.json`, `state.json`, `launcher.log`, per-parent
`parents/<benchmark>/measurement.json`, and historical
`historical_timeout_replay_20261001/raw/measurements/<replay_id>.json`.
The frozen source and every original timeout measurement hash were verified
before launch. The historical worker's input checks passed offline for all 102
replay identities. A replay pass is sensitivity evidence, not a correction of
the frozen main study or a new agent-generated candidate.
