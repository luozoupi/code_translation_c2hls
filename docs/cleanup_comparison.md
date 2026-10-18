# Paired DeepSeek cleanup comparison

This is a new prospective experiment comparing the frozen v6 implementation
with the cleaned implementation. It does not replace archived results, repair
historical ledgers, or claim exact reproduction of the commercial sweeps.

The matrix contains 26 public PolyBench kernels, seven configurations, three
fresh runs, and three cohorts: frozen implementation with paper prompts, clean
implementation with paper prompts, and clean implementation with researcher
prompts. All **1,638 cells** remain in reporting denominators. Cohorts are
interleaved within each kernel/configuration/repeat; arm and cohort order rotate
without consulting outcomes. Repeated hosted calls do not establish immutable
weights or seed-controlled repetitions.

The API identifier is **`deepseek-flash`**, endpoint `https://api.deepseek.com`.
`DeepSeek-V4.1-Flash-1017` is the declared provider release, not an invented API
identifier or immutable weights pin. Thinking is disabled. Requested and returned
model identities, request IDs, cache-hit/input/output usage, prompt hashes and
cost reservations are retained. The fresh ledger uses peak USD rates per million
tokens: input cache miss $0.30, input cache hit $0.006, output $1.20, checked
October 18, 2026 against [official pricing](https://api-docs.deepseek.com/quick_start/pricing/).
Recheck those rates before paid launch.

## Search and validation

The original commercial launcher defines these limits:

| Configuration | Runtime entrypoint | Turns | Quality turns | Generation limit |
| --- | --- | ---: | ---: | ---: |
| oneshot | single-shot `run_benchmark` | 1 | 0 | 2 h |
| flash_noskills / flash_curated / flash_allpos | single-shot `run_benchmark` | 5 | 4 | 4 h |
| ms_noskills / ms_curated / ms_allpos | `run_benchmark_multistep` | 5 | 4 | 4 h |

Historical “flash” means the single-shot translation/repair pipeline. It is not
the newer multistep `strategy=flash`. The historical launcher specifies no
universal five-request/five-synthesis cap. This study preserves per-arm turn and
quality limits, an 8,192-token response limit, and 2,400-second synthesis limit.
Launch additionally requires explicit campaign API/job-hour caps and a per-cell
aggregate input allowance. These new resource limits are reported if reached.

All cohorts receive the same staged public input and frozen skill library.
Generation is reference-blind and public-only; expert variants, reduced
co-simulation testbenches and overrides are excluded. Generation runs CSim and
synthesis with candidate co-simulation disabled. Selected source is frozen before
independent native, Vitis CSim and RTL checks using the unchanged full public
harness. No alternate winner is selected after a validation failure.

Native compilation/execution have separate 180-second limits for the plain
oracle and selected candidate; CSim has 300 seconds and synthesis-plus-RTL has
3,600 seconds. Their conservative combined allowance is 4,620 seconds per cell.
All jobs run sequentially. The controller reserves each cell's full remaining
allowance against the job-hour cap before dispatch and stops on unrelated Vitis
jobs, execution errors or uncertain billing. Observed owned descendants are
tracked by PID/start time and reaped on timeout, including detached sessions.

The old code's single-shot entrypoint does not necessarily inject recipes for
the curated/all-positive flags. The registration preserves those entrypoints;
the presence of a configuration label does not prove its guidance was active.
Inspect saved prompts before attributing effects to recipe policy. This is a
development corpus and does not establish held-out generalization.

## Register and inspect without spending

Run from the clean checkout. Supply the verified archive's frozen v6 runtime,
the public enhanced benchmark snapshot, the agreed frozen skill library, and
the imported commercial payload. These paths are explicit so registration
cannot quietly choose a different source tree or skill store.

```bash
python -m hlsmith.experiments register \
  --frozen-source /path/to/archive/runtime/v6 \
  --clean-source /path/to/clean/checkout \
  --benchmark-root /path/to/archived/enhanced/benchmarks \
  --skill-library /path/to/frozen/skills.json \
  --shape-registry /path/to/hlsfactory_output_shapes_fc27133.json \
  --historical-payload /path/to/import/payload/c2hls_newskills_commercial_sweep \
  --output /path/to/new/cleanup-comparison
python -m hlsmith.experiments estimate --output /path/to/new/cleanup-comparison
python -m hlsmith.experiments preflight --output /path/to/new/cleanup-comparison \
  --vitis-settings /path/to/Vitis/2023.2/settings64.sh
python -m hlsmith.experiments report --output /path/to/new/cleanup-comparison
```

Registration writes immutable identities, public input copies, the complete job
matrix, workload estimates and a pending state. It does not read keys, create a
billing ledger or call APIs/Vitis. Any changed source, prompt, skill, registry or
input blocks verification. Register after finalizing source changes.

The archive's 546 rows project approximately **111.39 M input and 17.99 M output
tokens**, repriced at **$55.01 without cache discounts**, and **140.14 sequential
generation job-hours** for the 1,638-cell study. This extrapolates other models'
workloads and excludes new final validation. Six archived failures lack usage
summaries. The combined timeout ceiling is **8,186.1 job-hours**; it is a ceiling,
not an expected runtime. Choose spending/compute caps after reviewing this
estimate; neither number authorizes a launch.

## Launch only after the caps are agreed

The CLI requires `--max-api-usd`, `--max-compute-hours`,
`--max-input-tokens-per-cell`, `--scratch-root`, `--vitis-settings`, and
`--rates-rechecked`. Set `DEEPSEEK_API_KEY` in the environment; its value is
never written into registration or state. No cap has a paid-work default.
The fresh `api_budget.json` uses reservations before requests and settles only
known provider usage. Unknown billing stops the campaign. Never reuse the
historical SIGMETRICS ledger.

An interrupted or failed campaign is preserved and does not silently resume or
reroll cells. Review its state, measurements and reservations before registering
an explicit continuation. Package extraction, source relocation and prompt
rewrites change identities and require new registrations.
