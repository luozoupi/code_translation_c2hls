# HPCA 2027 freeze-index normalizer

`scripts/normalize_hpca_freeze_index.py` is the deterministic boundary between
raw runner JSON and the paper artifact generator's result-manifest schema 2.
It does not scan result directories, rank duplicate runs, infer missing rows,
or read the active experiment summary. A human/evidence-freeze process must
name exactly one hash-pinned runner artifact for every preregistered
`(kernel, seed, method)` cell and exactly one baseline and expert source for
every kernel.

The intended publication flow is:

```text
explicit freeze-index v1
  -> hash/digest/identity validation
  -> normalize_hpca_freeze_index.py
  -> normalized result manifest v2
  -> hash-pinned evidence manifest v2
  -> generate_hpca_paper_artifacts.py
  -> one immutable artifact bundle
```

The normalizer writes nothing when any source is missing or inconsistent. An
existing byte-identical output is accepted; a non-identical output is never
overwritten.

## Freeze-index contract

The index schema is `c2hls.hpca-freeze-index.v1`. This example uses
descriptive placeholders and is not a publishable freeze:

```json
{
  "schema_version": "c2hls.hpca-freeze-index.v1",
  "target": {
    "vitis_version": "2023.2",
    "part": "xcu280-fsvh2892-2L-e",
    "clock_ns": "3.33"
  },
  "cohort": {
    "implementation_sha256": "<hash of fingerprint implementation object>",
    "prompts_sha256": "<hash of fingerprint prompts object>",
    "reference_isolation_sha256": "<hash of exact paper isolation overrides>",
    "decoding": {"temperature": "0.2", "top_p": "0.95", "max_completion_tokens": "8192"},
    "budgets": {"candidate_budget": "5", "llm_candidate_budget": "5"},
    "toolchain": {"flow_target": "vitis", "device_platform": "<U280 platform>"}
  },
  "methods": [
    {
      "id": "qwen_one_shot_best_of_five",
      "display_name": "Qwen best-of-five one-shot",
      "runner": "run_paper_baseline.py",
      "runner_method": "one_shot_best_of_five",
      "model": {"id": "<exact model ID>", "revision": "<resolved revision>"}
    }
  ],
  "expected_kernels": ["<kernel>"],
  "expected_cells": [
    {"kernel": "<kernel>", "seed": 0, "method": "qwen_one_shot_best_of_five"}
  ],
  "generated_rows": [
    {
      "kernel": "<kernel>",
      "seed": 0,
      "method": "qwen_one_shot_best_of_five",
      "runner": "run_paper_baseline.py",
      "run_id": "<opaque unique run ID>",
      "artifact": {"path": "<exact result JSON>", "sha256": "<64 hex>"},
      "json_pointer": "",
      "transcript": {
        "artifact": {"path": "<exact transcript JSON>", "sha256": "<64 hex>"},
        "json_pointer": ""
      },
      "reference_isolation_audit": {
        "artifact": {"path": "<exact audit JSON or result JSON>", "sha256": "<64 hex>"},
        "json_pointer": ""
      }
    }
  ],
  "frontiers": [
    {
      "kernel": "<kernel>",
      "baseline": {
        "source_kind": "reference_workflow_entry",
        "run_id": "<opaque unique baseline ID>",
        "artifact": {"path": "<fingerprinted result JSON>", "sha256": "<64 hex>"},
        "json_pointer": "/reference_validation/workflow/0"
      },
      "expert": {
        "source_kind": "reference_workflow_entry",
        "run_id": "<opaque unique expert ID>",
        "artifact": {"path": "<fingerprinted result JSON>", "sha256": "<64 hex>"},
        "json_pointer": "/reference_validation/workflow/4"
      }
    }
  ]
}
```

There must be exactly one `generated_rows` entry for every explicitly listed
`expected_cells` item, with no extras. The normalizer never fills a Cartesian
product. This is essential for the preregistered sparse replication campaign:
seed 0 contains all five methods, while seeds 1 and 2 contain only one-shot,
dynamic-no-skill, and dynamic-frozen-skill on three kernels. Consequently the
132-row matrix remains 132 method/model-qualified cells; no 24 unplanned
pragma-only or flash replicates are invented. A method ID is
the paper-facing model/workflow combination; `runner_method` is the exact
underlying controller method. This makes Qwen and Claude separate attested
cells rather than an unrecorded dimension. A generated row's JSON pointer
must be the empty string because runner identity and terminal evidence live at
the result root. Frontier pointers use RFC 6901 and select exact
`reference_validation.workflow` entries inside a fingerprinted result.
Relative artifact paths resolve from the index location.

Every source is read once; the exact bytes parsed are the bytes SHA-256
verified. Duplicate JSON keys are rejected. The normalizer then verifies the run fingerprint digest against its
canonical payload, `run.reproducibility.complete`, reference-blind mode,
benchmark, seed, model ID/revision, implementation/prompt cohort, decoding,
budget, probed Vitis version/settings, part, clock, runner schema, method contract,
strategy, and the full skill isolation/freeze state. Output provenance records only hashes,
pointers, adapters, and opaque IDs; it does not copy source filesystem paths.
It also recomputes `effective_llm_call_issues` over every recorded request;
the claimed reproducibility bit cannot hide an actual model, revision, prompt
hash, token cap, temperature, top-p, seed, or seed-support mismatch.
The separately pinned transcript must match
`reference_isolation_audit.transcript_sha256`; audit schema, counts, findings,
error state, and the embedded result copy must all agree.

Every record with passing synthesis must expose the five Vitis CSynth resource
counts (`bram`, `dsp`, `ff`, `lut`, and `uram`) as exact non-negative integers
and a positive finite `fmax_mhz`. The normalizer derives, rather than copies,
each utilization as `used / capacity`. For the fixed XCU280 target it uses the
part table (4032 BRAM18K, 9024 DSP, 2,607,360 FF, 1,303,680 LUT, and 960 URAM).
An optional `target.resource_capacities` object must contain all five values
and must agree with that table. Missing URAM, zero/invalid Fmax, fractional
resource counts, or a mismatched capacity is a hard refusal for both generated
and baseline/expert frontier records.

The normalized synthesis object includes the canonical SHA-256 of the exact
source report. For generated runs, that digest must equal the report digest on
the uniquely selected candidate event. The event code digest must also equal
both root `selected_code_sha256` and `cosim_target_code_sha256` (and the cosim
result's target digest when present). Thus a valid report from a different
candidate or a stale cosim result cannot supply the paper's resource/Fmax row.

## Accepted raw adapters

The baseline adapter accepts only `c2hls.paper-baseline.v1` results for
`one_shot_best_of_five` or `pragma_only`, and cross-checks the
`run_fingerprint.payload.paper_baseline` contract. It joins candidate, LLM,
and synthesis events by the producer's zero-based candidate index. A
pragma-only run may have placeholder candidates after a missing initial
translation; these consume neither an LLM call nor synthesis, but still need
an explicit candidate completion time.

The agentic adapter accepts only the matrix mappings below:

| Method | Fingerprinted strategy | Fingerprinted skills |
|---|---|---|
| `flash_c2hls` | `flash` | `skill_off` |
| `dynamic_no_skills` | `dynamic` | `skill_off` |
| `dynamic_frozen_skills` | `dynamic` | `skill_on`, `frozen: true` |

Agentic normalization requires one unified, contiguous candidate stream. That
stream must include candidates rejected before synthesis as well as synthesized
candidates; otherwise budget curves and failure denominators are not
reconstructible.

For both adapters, publishable cycles come only from a passing record with
`cosim.ran: true`. A predicted timeout/skip, a latency estimate, or a cycle
count attached to a nonpassing co-simulation is a hard error. Vitis-CSynth
latency is read only from integer `latency_cycles_worst` or `latency_cycles`;
the normalizer never converts nanoseconds or rounds a value.

Timeout, tool failure, and provider failure remain orthogonal typed status.
Correctness-stage timeouts contribute to the root timeout bit; correctness
tool failures and authenticated LLM-event errors contribute to the tool-failure
bit. `evaluation_status.provider_failure` must exactly equal the presence of
an error in the attributed LLM events (and any root copy must agree).

## Current producer gaps (2026-07-13)

The current `run_paper_baseline.py` / `PaperBaselineEngine` producer satisfies
the baseline adapter contract. Every candidate, including an early rejection
or pragma-only placeholder, records `cumulative_elapsed_seconds`; root counters
separately report selection synthesis evaluations, the selected-winner cosim
flow, and their exact `total_synthesis_calls`. The CLI-selected base seed also
overrides any inherited environment seed in the fingerprint and records the
requested/effective per-candidate seed schedule.

`reference_isolation.audit_history_file` now hashes the exact transcript bytes
it parses, emits `transcript_sha256`, and both runners persist that bound audit.
Historical artifacts missing any of these fields remain invalid; the
normalizer does not backfill them or infer zero-valued counters.

`run_agentic_sweep.py` / `C2HLSOrchestrator` currently lacks a complete event
stream with:

- stable candidate-evaluation joins to LLM calls;
- cumulative tokens, LLM calls, synthesis evaluations, and elapsed time;
- typed CSim and synthesis statuses;
- resource fit, timing, integer Vitis-CSynth latency, and failure class;
- the selected executed-cosim event marker; and
- root `total_synthesis_calls` and `selected_winner_cosim_count`.

The producer changes belong in
`C2HLSOrchestrator._record_llm_usage`, `_synth_and_test`,
`_optimization_step_attempt_single`, and `_run_selected_winner_cosim`. The
normalizer reports error code `agentic_candidate_telemetry_incomplete` with
the exact missing fields and these producer functions.

Baseline and expert frontier entries also remain failed outcomes unless each
has a passing, executed RTL co-simulation cycle count. A selected-only
reference co-simulation does not supply the measured plain-C baseline needed
for expert recovery; the freeze must point to independently revalidated
evidence for both roles.

Until the producers emit these fields and new runs are completed under a full
matching fingerprint, no normalized manifest should be attached to a paper
claim.

## Invocation

```sh
python scripts/normalize_hpca_freeze_index.py \
  --freeze-index paper_eval/freeze/freeze-index.json \
  --output paper_eval/freeze/results-v2.json
```

Refusals are emitted to standard error as structured JSON with `code`,
`location`, `missing_fields`, and `producer_functions`, and the command exits
with status 2. A successful command prints only the output path.
