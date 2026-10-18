# Historical Jacobi-1D numerical failures

All nine failures in the drained historical campaign have concrete buffer-state
or output-state defects. The original public program alternates A→B then B→A,
updates only indices 1 through 118, and preserves both arrays' boundary values.
The failures are not explained by floating-point tolerance alone.

| Validation ID prefix | Source defect |
|---|---|
| `01fa19f7d209` | Local B boundaries never loaded before neighboring A updates |
| `2e1a8c51c94b` | The second ping-pong bank's A and B boundaries are uninitialized |
| `3b372d3e1531` | Only bank zero is initialized; bank-one boundary values are read |
| `4e24d5656b98` | Local B boundaries never loaded |
| `751a5aa89ed8` | Local B boundaries never loaded; final B also lacks copy-back |
| `b854d1f0712b` | Local B boundaries never loaded |
| `dafd2f5d7e4a` | Local B boundaries never loaded; A remains unchanged because the final A buffer is written to B |
| `e6d9ff385214` | A reads B_next endpoints before initialization; preserving boundaries in a later copy loop is too late |
| `fa8a9bf12dee` | Odd stages read an uninitialized second A bank instead of the preceding B result; A never receives an update |

Separate diagnostic copies restore boundary initialization and the required
phase/copy-back semantics. **9/9 patched CPU executions match the original
public golden outputs with unchanged tolerance.** This is post-hoc causal
support on that harness, not proof of equivalence or an HLS performance result.
The public harness checks A; passing it alone does not establish every possible
observable state or input. Uninitialized reads also make original CPU behavior
undefined, so their particular erroneous numbers are not reproducibility claims.

Original selected sources, failed measurements and denominators are unchanged.
The diagnostic sources, native outputs, golden outputs, source hashes and exact
historical measurement links are stored in
[`diagnosis.json`](../artifacts/sigmetrics2027/jacobi_failure_diagnosis/diagnosis.json).
The adjacent `reproduce.py` is a CPU-only diagnostic script. These repairs are
not substituted into the historical experiment or injected into the frozen
recipe pilot.
