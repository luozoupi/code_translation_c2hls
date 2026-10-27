# HLSmith — historical v8

**`v8` is the historical full SIGMETRICS/Vitis research implementation, formerly
named `hlsmith-clean`.** Its successor is
[`HLSmiths-nano`](https://github.com/luozoupi/code_translation_c2hls/tree/HLSmiths-nano),
a separate minimal standalone workflow. The nano branch does not reproduce the
paper's experiments; use this branch for the full research pipeline and its
preserved reproduction records.

HLSmith translates C/C++ kernels into FPGA implementations with compiler feedback. It checks functional behavior, synthesizes candidates with Vitis HLS, and searches for lower latency within the target device's resource and timing constraints.

This branch consolidates the implementation used for the SIGMETRICS submission.
The paper's results are preserved as frozen evidence; fresh evaluations are
recorded separately. The default prompt profile preserves the original model
inputs. The code before the 2026-10-27 rename documentation is retained at
`snapshot/20261027/v8` (`3b6081b02ae4daf9022e843cacb46ee850a61314`).

## Install

Use Python 3.10 or later. Vitis HLS and a C++ compiler are required to evaluate hardware candidates.

```bash
python -m pip install -e '.[anthropic,analysis,test]'
c2hls --help
```

Optional training dependencies are available through `.[training]`. Existing `python c2hls.py ...` commands and Python imports remain supported.

## Run a kernel

Configure your Vitis installation and model access in your shell or an untracked `.env` file:

```bash
export C2HLS_VITIS_SETTINGS=/path/to/Vitis_HLS/2023.2/settings64.sh
export HLS_TCL_RUNNER='vitis_hls -f'
export DEEPSEEK_API_KEY=your_key
export C2HLS_DEEPSEEK_THINKING=disabled
export C2HLS_REFERENCE_BLIND=1
c2hls --bench aes --model deepseek-flash --output-dir ./runs/aes
```

The default hardware target is the Alveo U280 at 3.33 ns. Check each benchmark's metadata when selecting a different target. Translation invokes the configured model and compiler; the registered cleanup campaign below enforces campaign-wide budgets.

Use `--bench-dir` for an external benchmark. Use `--multistep --strategy dynamic` for routed optimization. Run `c2hls --help` for search settings and existing options.

## Prompts

`paper-sigmetrics-2026` preserves all 23 original templates and remains the default. `researcher-v1` offers rewritten prompts with the same placeholders and interface requirements:

```bash
c2hls --bench aes --model deepseek-flash --prompt-profile researcher-v1 --output-dir ./runs/aes-readable
```

Profiles have distinct content hashes and run identities. Select one profile per process. New prompt results need their own correctness and QoR evaluation.

## Reproduce and validate

```bash
python scripts/check_offline.py -q
python reproducibility/paper/replay.py
python scripts/compare_v6_replay.py --help
python -m hlsmith.experiments --help
```

The offline suite blocks actual network connections and FPGA jobs. Hardware integration tests require explicit activation. Portable paper replay verifies preserved evidence and regenerates the reported numerical summaries without model calls.

The fresh comparison uses DeepSeek V4.1-Flash-1017 through `deepseek-flash`: frozen runtime/paper prompts, clean runtime/paper prompts, and clean runtime/readable prompts, with 26 kernels, seven requested configurations and three repetitions (1,638 cells). The API alias is mutable; run records retain the declared release, requested/returned model identifiers and timestamps. Some inherited single-shot skill settings are inactive in the frozen runtime and are explicitly identified in the study registration.

Registration and cost estimation run without API credentials or tool jobs. Launch requires numeric spending, compute and input-token limits plus a pricing recheck. See [the experiment workflow](docs/cleanup_comparison.md) for commands and validation rules.

## Code and research records

- [Architecture and compatibility](docs/architecture.md)
- [Paper contract and evidence limitations](docs/paper-contract.md)
- [Validation report and pending gates](docs/validation-report.md)
- [Deterministic replay validation](docs/deterministic-replay-validation.md)
- [Historical branches and changelogs](docs/history-branches.md)
- [Local migration and archival](docs/migration.md)
- [Campaign status and operation](docs/campaign-operations.md)
- [Research command index](scripts/README.md)

Implementation lives in `hlsmith/`. Root modules are compatibility entrypoints. Packaged benchmark, skill and configuration files retain their original bytes. Generated results are stored outside the tracked source tree.

The `v8` rename changes documentation and branch identity, not runtime behavior
or experimental acceptance. Unresolved hardware comparisons and pending
fresh-model gates remain qualified in their records; the rename does not
establish paper-wide equivalence. GitHub's default remains `v1`. The protected
collaborator branch and the separate C-to-CUDA project are preserved unchanged.
