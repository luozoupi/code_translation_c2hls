# Historical branches

Versions `v1` through `v6` retain the original ancestry and authors. Each received
a single documentation commit changing only `README.md` and `CHANGELOG.md`; its
annotated snapshot tag points to the original code commit. On 2026-10-27, the
later `hlsmith-clean` research implementation was renamed `v8` at the user's
request. `v7` is deliberately unassigned.

The minimal standalone successor is
[`HLSmiths-nano`](https://github.com/luozoupi/code_translation_c2hls/tree/HLSmiths-nano).
It is a separate workflow, not a reproduction of the SIGMETRICS experiments.
`v8` retains the full SIGMETRICS/Vitis implementation and its research records.

| Version | Original branch | Original code | Documented tip | Snapshot tag |
| --- | --- | --- | --- | --- |
| v1 | `main` | `630ce11e808b` | `cf363c08e757` | `snapshot/20261018/v1` |
| v2 | `codex/hls-validation-pipeline-cleanup` | `9c9dfdc32748` | `9c535186a069` | `snapshot/20261018/v2` |
| v3 | `c2hls_enhanced` | `1164311e34d5` | `a32dd21eb4fe` | `snapshot/20261018/v3` |
| v4 | `c2hls_enhanced_l` | `ebe0675b16dc` | `7f9cace837d7` | `snapshot/20261018/v4` |
| v5 | `hpca2027-paper-eval` | `a778848863a3` | `82601b2ebefe` | `snapshot/20261018/v5` |
| v6 | `sigmetrics2027-validation (local preservation)` | `42de088a7716` | `95b9a966b6ef` | `snapshot/20261018/v6` |
| v7 | Unassigned; the user selected the label `v8` | — | — | — |
| v8 | `hlsmith-clean` | `3b6081b02ae4` | Documentation update on `v8` | `snapshot/20261027/v8` |

Version order follows recorded development milestones, not inferred remote branch creation
times. Git does not preserve online creation dates. Local reflogs show the HPCA branch
created July 13, 2026, and the SIGMETRICS branch September 28, 2026; older dates are unknown.

- **v1** (2026-03-31 milestone): Initial C-to-HLS pipeline, 4-ns evaluation target and recorded turn history.
- **v2** (2026-04-02 milestone): Stabilized validation outcomes and interactive pipeline visualization.
- **v3** (2026-06-02 milestone): Enhanced agentic translation, scope feedback, resource guards and correctness repair.
- **v4** (2026-07-24 milestone): Commercial campaign tooling, trajectory exports, portable Vitis runner and resumable sweeps.
- **v5** (2026-08-25 milestone): Reference-isolated HPCA evaluation, skill routing, setup ranking and QoR sweeps.
- **v6** (2026-10-18 milestone): Preserved SIGMETRICS implementation, validation scheduling and controlled recipe-study tooling.
- **v8** (2026-10-27 rename): Preserved the full packaged research implementation,
  paper/readable prompt profiles, numerical and captured-response replay, and
  controlled hardware/fresh-model comparison tooling. The rename changes no
  runtime behavior and does not assert that outstanding acceptance gates passed.

The protected `c2hls_enhanced_l_pc2_api_layout` branch remains at
`2ef022410fa6186ff6e16be25f79da13951bf9b7`. For the `v1`–`v6` publication,
GitHub's native branch rename operation preserved redirects, then each version
received its documentation commit by fast-forward. Those published tips were
checked against the 2026-10-18 index.
Renaming `main` made `v1` the default; it remains GitHub's default through the
2026-10-27 branch reorganization.

[history-branches.json](history-branches.json) is the unchanged **2026-10-18
receipt**, containing full SHAs, tag objects, source refs, and validation of the
`v1`–`v6` publication. Its `hlsmith-clean` successor and remote-branch fields
describe that dated snapshot, not the current branch list. The later `v8`
snapshot preserves the full original tip
`3b6081b02ae4daf9022e843cacb46ee850a61314`; this Markdown index records the
2026-10-27 rename and nano successor separately.

Historical results keep their source, inputs, tool/model/prompt identity, failures and timeout
policy. Hosted-model endpoints can change; branch labels do not guarantee fresh experimental
performance. Numerical replay and new end-to-end runs remain separate validation gates.

Archive roots are `/mnt/data1/luo00466/c2hls_archive_20260811` and
`/mnt/data1/luo00466/c2hls_archive_20261018`. Their manifests define actual artifact coverage.
Manuscript prose, PDFs and private review material are excluded from the public code bundle.
