# Archived boundary fixtures

`snapshot.json` contains five exactly source-linked saved model replies and
their synthesis reports/public inputs. Independent October 8 validation records
supply one passing case, two failed C-output checks, one RTL timeout and one
failed RTL-output check. Seven saved Sonnet GEMM configurations supply a
separate feasible-winner restoration exercise.

Each fixture records source-result/history/reply/source/report hashes and the
public input hashes. Its origin is archive
`c2hls_newskills_commercial_sweep.tar.gz`, SHA-256
`1235bf8226e5bd84b87dc9458323b293856e8c3d987148e1538954d155dad80d`.
The selected model reply's final C++ fence matches the saved generated source
exactly after outer whitespace stripping. No source is repaired to manufacture
a passing experimental outcome.

`expected-v6.json` was captured by executing the frozen v6 runtime from
`42de088a7716a5d00ec69696d07b1b8b1b75840f` in a separate subprocess.
`manifest.json` binds both input and expected-output bytes. Capture is a
test-boundary execution with archived responses, not a new hardware run.

The harness runs the real translator, synthesis/repair controller,
source preflight, feasibility and best-state restoration. It checks all 18
extracted source-contract helper ASTs against the frozen implementation.
Network connections and unmocked subprocess execution are forbidden. Calls to
model, compiler, C simulation, synthesis and RTL tools receive saved records
or explicit stubs. A repeated archived reply stands in for repair responses;
this checks the current controller's deterministic behavior rather than
reconstructing the historical conversation.

The split-control-bundle probe is a clearly separated perturbation of saved
source, used to compare ABI normalization. It is not an experimental result.
Historical CSim completion is used for the seven-arm selection exercise and
does not upgrade that archive to independent correctness verification.
