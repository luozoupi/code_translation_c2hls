# Deterministic v6 compatibility replay

The frozen v6 runtime and the cleaned runtime produce the same normalized
result on the archived boundary fixture. The comparison includes 18 extracted
source-contract helper ASTs, five representative translation/repair paths and
a seven-configuration GEMM feasible-winner restoration.

The frozen implementation is commit
`42de088a7716a5d00ec69696d07b1b8b1b75840f`. The canonical result SHA-256 is
`1da9bda6aa0a7dc809b39cb1d94b7369827f1d64ae906e3638e5755bdf028ee6`.
Each runtime is loaded in its own subprocess with a controlled configuration.
The expected result was captured from the frozen runtime, independently of
the cleaned implementation.

| Saved candidate | Independent C-output status | RTL status | Controller behavior preserved |
| --- | --- | --- | --- |
| Sonnet / 2mm / flash all recipes | Pass | Pass | Accepts the Phase B candidate |
| Opus / GEMM / multi-step recipes disabled | Fail | Not run | Requests repair; synthesis remains gated |
| Sonnet / correlation / multi-step recipes disabled | Fail | Not run | Requests repair; synthesis remains gated |
| GPT / Cholesky / multi-step recipes disabled | Pass | Timeout | Retains timeout as inconclusive; no measured RTL eligibility |
| GPT / FDTD-2D / flash all recipes | Pass | Fail | Requests correctness repair; candidate remains failed |

Compared fields include rendered request hashes, benchmark-context hashes,
extracted and preflight source hashes, controller history/turn hashes,
numerical report fields, tool-dispatch ordering, feasibility reasons, test
statuses, valid RTL-measurement checks, public header contracts and selected
winner source/report. The derived split-control-bundle probe compares ABI
normalization separately from experimental outcomes.

Run the portable regression without the old checkout:

```sh
python -m unittest discover -s tests -p test_v6_deterministic_replay.py -v
```

For a new direct comparison against an independently frozen v6 checkout:

```sh
python scripts/compare_v6_replay.py --frozen /path/to/frozen-v6
```

The fixture and expected result are checksum-bound. The subprocesses reject
network connections and unexpected tool-process launches. They instantiate
agents without creating provider clients and replace tool calls with saved
records. C++ compilation is stubbed. Repair calls repeat the saved response;
the replay is not a reconstruction of the original full optimization search.

These checks establish compatibility for the exercised controller and helper
paths. They do not establish whole-corpus equivalence, new C/RTL correctness,
fresh model QoR, adapter-training equivalence or historical hosted-weight
reproducibility. Use the full campaign and its frozen acceptance thresholds
for those claims. The separate paper replay checks the published numerical
exports and their validation boundaries.
