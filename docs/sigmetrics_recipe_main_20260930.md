# Main recipe representation study — September 30, 2026

The user authorized the 216-cell study to run alongside historical validation.
The new runner is `scripts/run_sigmetrics_recipe_main.py`. Its source snapshot
and `manifest.json` are immutable once prepared. This study is separate from
the six-cell engineering pilot, the historical mode comparison, and any future
step-goal enforcement experiment.

The denominator is twelve prespecified kernels × DeepSeek Flash/GPT-5.6 Luna ×
no additional recipes/flat same-content guidance/structured recipes × three
independent API runs. All 216 identities are registered before generation. The
M60,N40 Gram–Schmidt variant keeps its separate benchmark name. No kernel,
provider, arm, or run is replaced after an adverse result.

Each kernel starts from unchanged public C, validated on all five prespecified
inputs with five CSim runs, one parent CSynth run, and five RTL runs. Existing
ATAX construction is reused with source, input, toolchain, and measurement
hashes checked. Timing and resource feasibility are recorded separately:
a correct but infeasible parent can be an optimization starting point, but
cannot count as a feasible selected result. Parent construction uses no model
requests. A parent failure or timeout blocks its 18 registered cells. A
prospective router with no matching recipe also blocks all 18 cells, with no
manual replacement recipe.

For each passing parent, the frozen router selects at most three positive
recipes using only that parent's code and compiler report. Complete statements,
applicability conditions, and templates are identical between flat and structured
arms. The control retains the same generic HLS instructions and compiler
feedback but receives no added recipes. Exact first-call messages for both
providers and all three arms are captured offline, checked for equal content,
and frozen before any model outcome on the kernel. Subsequent feedback is
outcome-dependent; its sent prompt hashes remain in the usage ledger.

Each cell uses flash mode, one candidate sequence and up to five exhaustive
compiler-feedback attempts, with five provider requests and five selection
syntheses as separate upper bounds. Successful attempts do not end search early.
Malformed output or a prespecified token limit may end search earlier, and
all actual requests and tool invocations are reported. This is complete-design
search with feedback, not five sequential targeted optimization steps. The
QoR reviewer, deterministic QoR sweep, dynamic routing, skill learning, and
post-RTL reselection are disabled. The fastest eligible design or feasible
parent is selected before five-case RTL validation; a failed selected design
remains failed. Maximum measured RTL cycles across all five cases is the
performance observation. CSim, selection synthesis, RTL resynthesis, and parent
construction costs remain separate.

Two explicit controller corrections apply only to the frozen main snapshot:

- The common system instruction permits combined whole-design transformations,
  matching the flash user prompt, instead of demanding one transformation.
- Repair requests stop before a sixth unevaluated response. Earlier valid
  candidates remain eligible when the last attempt fails or a local token
  allowance prevents another request. Real provider/accounting failures halt
  new work and retain their evidence.

The existing shared API ledger retains its $300 cap and frozen conservative
rates. Each response has at most 16,384 tokens. The main allowance is 500,000
aggregate input tokens, five times the engineering pilot allowance. Each
request must also have a conservative input bound of at most 250,000 tokens,
checked before request accounting and below the OpenAI long-context pricing
threshold. Conservative input bounds use UTF-8 JSON bytes plus message framing;
actual provider token usage is reconciled. The registered 216-cell upper-bound
estimate is $48.233664 before the pilot's existing charges, with no cache discount
assumed. These are rate-based estimates, not invoices. Model aliases are not
immutable model snapshots, and the repetitions have no supported seed control.

The two-worker controller schedules one parent and one ready cell while parents
remain; afterwards it schedules up to two cells. Ready cells follow a fixed
order that rotates arm positions across repeats and reverses provider order on
alternate kernels. It never sorts by observed performance. Each job has its own
scratch and Vitis user directory under `/tmp/sigmetrics_recipe_main_20260930/`.
Root coordinates an authenticated allowlist with the historical scheduler before
launch. The main runner registers its PID, process start ticks, manifest hash,
and frozen runner hash in `controller_identity.json` before submitting work.

New submissions wait if free scratch space falls below 60 GiB, available memory
below 64 GiB, or one-minute load exceeds 0.8 per logical CPU. Waiting reasons and
observed resources appear in `state.json` and `summary.json`; active jobs drain
normally. SIGTERM/SIGINT requests also stop submissions and drain active jobs.
A controller is never restarted silently: prepared, completed, failed, and
interrupted identities are not rerolled. Individual tool timeouts are 300 s for
CSim, 1,200 s for CSynth, 3,600 s for each RTL case, and 300 s for the provider.
Parent and cell process groups have additional 24,000 s and 43,200 s bounds.

Prepare without API or Vitis calls:

```bash
/home/luo00466/.conda/envs/py310_2/bin/python scripts/run_sigmetrics_recipe_main.py prepare \
  --output /home/luo00466/code_translation-c2hls-sigmetrics2027/results_sweeps/sigmetrics_recipe_main_20260930
```

After the historical coexistence preflight, root launches the frozen copy:

```bash
/home/luo00466/.conda/envs/py310_2/bin/python -u \
  results_sweeps/sigmetrics_recipe_main_20260930/source/scripts/run_sigmetrics_recipe_main.py run \
  --output /home/luo00466/code_translation-c2hls-sigmetrics2027/results_sweeps/sigmetrics_recipe_main_20260930
```

`state.json`, `summary.json`, per-parent `measurement.json`, per-kernel
`qualification.json`, per-cell `result.json`, and the shared API ledger are the
canonical progress/evidence records. Accounting or infrastructure failure stops
new submissions; recorded correctness and timeout outcomes retain their
prespecified denominator. No efficacy claims follow merely from registration
or launch.
