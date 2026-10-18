# Historical branches

Six historical versions retain the original ancestry and authors. Each branch adds a
single documentation commit changing only `README.md` and `CHANGELOG.md`; the annotated
snapshot tag points to its original code commit. Continue development on `hlsmith-clean`.

| Version | Original branch | Original code | Documented tip | Snapshot tag |
| --- | --- | --- | --- | --- |
| v1 | `main` | `630ce11e808b` | `cf363c08e757` | `snapshot/20261018/v1` |
| v2 | `codex/hls-validation-pipeline-cleanup` | `9c9dfdc32748` | `9c535186a069` | `snapshot/20261018/v2` |
| v3 | `c2hls_enhanced` | `1164311e34d5` | `a32dd21eb4fe` | `snapshot/20261018/v3` |
| v4 | `c2hls_enhanced_l` | `ebe0675b16dc` | `7f9cace837d7` | `snapshot/20261018/v4` |
| v5 | `hpca2027-paper-eval` | `a778848863a3` | `82601b2ebefe` | `snapshot/20261018/v5` |
| v6 | `sigmetrics2027-validation (local preservation)` | `42de088a7716` | `95b9a966b6ef` | `snapshot/20261018/v6` |

Version order follows recorded development milestones, not inferred remote branch creation
times. Git does not preserve online creation dates. Local reflogs show the HPCA branch
created July 13, 2026, and the SIGMETRICS branch September 28, 2026; older dates are unknown.

- **v1** (2026-03-31 milestone): Initial C-to-HLS pipeline, 4-ns evaluation target and recorded turn history.
- **v2** (2026-04-02 milestone): Stabilized validation outcomes and interactive pipeline visualization.
- **v3** (2026-06-02 milestone): Enhanced agentic translation, scope feedback, resource guards and correctness repair.
- **v4** (2026-07-24 milestone): Commercial campaign tooling, trajectory exports, portable Vitis runner and resumable sweeps.
- **v5** (2026-08-25 milestone): Reference-isolated HPCA evaluation, skill routing, setup ranking and QoR sweeps.
- **v6** (2026-10-18 milestone): Preserved SIGMETRICS implementation, validation scheduling and controlled recipe-study tooling.

The protected `c2hls_enhanced_l_pc2_api_layout` branch is unchanged. These refs are prepared
locally; no remote branches have been renamed/deleted/pushed and the remote default branch
is unchanged. The active branch and working index were not used by the plumbing transaction.
Full SHAs, tag objects, source refs and validation receipts are in
[history-branches.json](history-branches.json).

Historical results keep their source, inputs, tool/model/prompt identity, failures and timeout
policy. Hosted-model endpoints can change; branch labels do not guarantee fresh experimental
performance. Numerical replay and new end-to-end runs remain separate validation gates.

Archive roots are `/mnt/data1/luo00466/c2hls_archive_20260811` and
`/mnt/data1/luo00466/c2hls_archive_20261018`. Their manifests define actual artifact coverage.
Manuscript prose, PDFs and private review material are excluded from the public code bundle.
