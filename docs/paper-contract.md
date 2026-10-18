# Paper and prompt compatibility

The cleanup reference is the SIGMETRICS source at commit
`ecd3c750168dea52aa80dd609f463b9b668689b0` in the Overleaf repository
`6a3058ea0542c2fa6f168fe8`. Its entry point is `main-sigmetrics.tex`, titled
*HLSmith: Compiler-Guided Source Restructuring for FPGA Accelerators*.
The exact PDF submitted for review has not been identified. The local root PDF
predates later source changes, so it is not substituted for the submitted PDF.

## What is frozen

`reproducibility/paper/reference.json` binds 106 copied files by SHA-256. These
include the numerical exports and manifests, the portable Figure 8 source and
measurement evidence, five unmodified historical generators, both original
evidence test modules. Manuscript prose and PDFs stay in the external archive;
only their source commit and file hashes are included here. Historical paths
inside evidence remain provenance labels; offline replay does not open them.
The complete manuscript and large experiment workspaces belong in the archive.

The inherited `claims.yaml` and paper README describe earlier HPCA/pending
work. They are retained as historical records. The active SIGMETRICS source
and its numerical manifests define the values below.

| Cohort | Passing selected/comparator pairs | RTL recipe-disabled / selected geometric mean |
| --- | ---: | ---: |
| Sonnet 4.6 | 24 | 3.21× |
| Opus 4.8 | 25 | 3.01× |
| GPT-5.5 | 23 | 2.41× |

The commercial paper bundle records 153 passing roles out of 156. Its evidence
is reported Vitis C/RTL consistency; independent original-reference output
comparison is not recorded in that bundle. The same passing pairs define the
synthesis and RTL aggregates. Original partial-grid exclusions and five
asymmetric Grok rerun exclusions remain part of the calculation.

The open-model Figure 8 bundle records 84/108 selected and 57/108 comparator
passes within the fixed one-hour fresh synthesis plus co-simulation limit.
There are 52 passing pairs: 35 selected wins and 17 ties. Five comparator
identities remain unresolved. Successful three-hour retries are a separate
sensitivity cohort, with 94 selected and 65 paired passes. They do not enter
the one-hour bars. The 27-kernel cohort excludes `doitgen`.

All copied historical bytes remain unchanged. One historical commercial
manifest contains a stale output hash for
`generated/sigmetrics/commercial_validation_table.tex`; the table matches the
historical generator. `reference.json` records both the old declared hash and
the actual file hash. This explicit exception does not relax hash checking for
the copied snapshot.

## Offline verification

With the optional paper dependencies installed, run from the checkout root:

```sh
python reproducibility/paper/replay.py
python -m unittest discover -s reproducibility/paper/tests -v
python -m unittest discover -s tests -p test_prompt_profiles.py -v
```

Replay checks the copied bytes, verifies the portable evidence, and recomputes
the commercial and open-model numerical summaries. It reads no live experiment
state, writes no results, and makes no model or Vitis calls. The 26 copied paper
tests cover selection identity, metric/cohort consistency, incomplete evidence,
output checks, source/input hashes, and timeout separation. Figure rendering
is separate; run the original generators against a scratch copy rather than
overwriting this snapshot.

## Prompt profiles

`paper-sigmetrics-2026` is the default compatibility profile. It exports all 23
strings, the optimization map, and step lists from the frozen v6
`prompt_c2hls.py` without editing their bytes. The copied source in
`reproducibility/paper/prompts/` is the regression fixture. This preserves the
v6 interface; it does not establish that every historical experiment used the
same template bytes. Some historical controller and merged-library snapshots
were not recovered.

`researcher-v1` uses shorter, direct language about memory traffic, banking,
dependencies, compute width, and the target interface. It retains the format
fields, step names, testbench ABI requirements, header ownership, control
bundle, narrow/wide-bus boundary, buffer initialization, and tail handling.
It is a new experimental input. Selecting it changes model behavior and does
not inherit the paper's numerical performance claim.

`hlsmith.prompt_profiles.load_profile(name)` loads either profile lazily. Its
`exports` property returns isolated runtime values; `templates` provides the
named strings. Record `metadata` with every run: profile name, full exported
content SHA-256, source-file SHA-256, schema, and template count. Keep this
profile record separate from the skill catalog hash and model revision.

## Full reruns and acceptance

Use Vitis HLS 2023.2, U280 `xcu280-fsvh2892-2L-e`, and a 3.33-ns target where
those are the cohort's recorded settings. Freeze the benchmark, header,
testbench, metadata, comparator and tolerances; distinguish synthesis-cycle
estimates from observed RTL cycles. Keep LUT, FF, BRAM, URAM, DSP and timing
eligibility, selection budgets and timeout scope explicit.

Compare the cleaned runtime with its frozen predecessor using replayed model
responses and reports first. Compatibility requires the same prompt bytes,
source outputs, eligibility decisions, selected candidates, metrics, failure
statuses and source associations for the paper profile. Fresh model generation
is a separate paired experiment with matched inputs, budgets and deployment
settings. Report all registered failures and unresolved identities. Do not
replace a failed selected candidate after observing RTL outcomes.

Historical hosted aliases do not identify immutable weights. An unavailable
alias or original training package is a documented reproduction limitation,
not a reason to label a replacement experiment as historical reproduction.
The GRPO evidence is an external summary without imported trainer/checkpoint
identity or complete raw per-kernel records. Recover that package before
claiming a full GRPO retraining reproduction.

The later independent local commercial replay also differs from the paper's
supplied C/RTL bundle: 144 saved designs produce 121 passes, eight RTL
failures, eight timeouts, and seven failed independent C-output checks.
Preserve both evidence sets and their contracts. Code cleanup cannot turn
those failures into paper passes. An acceptance report must distinguish
preserved calculations, implementation compatibility, and fresh experimental
quality; numerical tolerances for fresh stochastic comparisons must be fixed
before their results are observed.
