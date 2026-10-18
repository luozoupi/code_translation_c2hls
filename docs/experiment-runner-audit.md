# Prospective experiment runner audit

The runner is prepared for a paired DeepSeek comparison of frozen v6, cleaned
paper prompts, and cleaned researcher prompts: 26 public kernels × seven
requested configurations × three independent runs × three cohorts = 1,638
registered cells. This is a prospective development-corpus study. It is not a
completed full-paper reproduction, a held-out result, or an immutable-weight
hosted-model comparison.

## Safeguards checked and corrected

- Source files, the complete source file set, public inputs, skill catalog,
  shape registry, historical estimate inputs, fixed job order and state IDs
  are checked against registration. Every worker stage checks them again.
- Private workers require a running launch, positive API/compute/input caps,
  a matching API ledger, unchanged Vitis settings, remaining compute allowance
  and the active job identity. They reconstruct the controlled environment;
  an omitted API ledger, inherited role configuration, dotenv file or custom
  runner cannot silently change the study.
- Selection freezes the source and its exact synthesis-report hash before
  independent validation. A failed selected candidate is never replaced with
  another configuration after observing native or RTL outputs.
- The native comparator must pass against a nonempty CPU oracle. CSim and RTL
  require explicit independent comparison evidence against that same output
  hash. A bare successful tool exit is insufficient.
- Timing/resource feasibility and complete finite integer counts are required
  for the selected synthesis report. The fresh synthesis already performed
  inside co-simulation is read from its saved report and checked separately.
  Missing fresh evidence cannot become a qualified RTL pass. This adds no
  synthesis invocation.
- Measured cycles must be positive finite integers; booleans, estimates,
  nonpassing RTL and missing values cannot enter ratios. Reports retain every
  registered cell and show the paper/default and researcher-profile paired
  comparisons separately. A worker exiting without a terminal measurement
  stops dispatch.
- The validation CSim timeout is set before importing the evaluator, because
  that module captures timeout settings at import time. Native compile/run,
  CSim and complete fresh-synthesis/co-simulation limits stay distinct.

The seven-arm interpretation remains qualified: v6's single-shot entry point
does not inject recipes just because curated/all-positive flags are selected.
Those requested labels are not seven independently active treatments. The
runner preserves this behavior and reports it; changing it would be another
method change.

## Tool identity

Bounded version-only queries verified both installed runners as Vitis 2023.2:
`vitis-run` reports SW build 4026344, and classic `vitis_hls` reports SW build
4023990. The query commands, executable/settings hashes and captured output
are recorded in [validation-tool-versions.json](validation-tool-versions.json).
No synthesis or simulation job was executed. A paid launch repeats the version
check and pins both executable identities; a path containing `2023.2` alone
does not satisfy the check.

All comparison cohorts use the same verified `vitis-run` binary with
`--tcl --input_file`. Frozen v6 names that executable literally, so its verified
directory is prepended to PATH; the cleaned runtime receives the same absolute
binary and arguments through `HLS_TCL_RUNNER`. The launch receipt, selection
record and validation measurement retain this common-runner identity. Classic
`vitis_hls` remains a version-only portability check and is not a differing
compiler invocation in the comparison.

## Validation and remaining work

The focused experiment and API-budget tests passed without provider or HLS
jobs. They exercise state/source tampering, launch caps, owned-process cleanup,
registry checks, real archived synthesis reports, independent-output gates,
timing/resource failures, fresh-report requirements, finite cycle checks and
paired-report denominators.

No campaign has been launched or authorized by this audit. Final source
changes require a fresh registration and preflight; older source hashes are
intentionally invalidated. Explicit API, aggregate job-hour and per-cell input
limits, credentials and current-rate review remain launch prerequisites.

The offline paper replay and representative v6 control replay are separate
checks. Neither executes the archived hardware gate. Full archived source
revalidation, fresh model generation, original multi-model/open-model sweeps,
router and adapter studies remain distinct campaign work. The paper's GRPO
study still lacks the original trainer/checkpoint/raw evaluation package;
another training run cannot be labeled its exact reproduction.
