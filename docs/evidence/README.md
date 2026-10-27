# Validation evidence

These receipts preserve the validation performed during development. They are
historical records, not a claim that each result applies to every later change.
The source commits and runtime hashes inside each receipt identify its scope.

| Receipt | Recorded scope | SHA-256 of unchanged bytes |
| --- | --- | --- |
| [VALIDATION.json](VALIDATION.json) | Initial native/Verilator and installed-package checks | `c0d37c3a6328dc6e54825e3387f7e86b1c93067b1456f6261e1cc06c4146b23e` |
| [LIGHTNINGSIM_VALIDATION.json](LIGHTNINGSIM_VALIDATION.json) | Optional trace integration; two bounded real checks timed out | `bd94014eb7d18726de0c5494cbb44e866440bbe9b5032183af05542a3432011b` |
| [HLSFACTORY_PILOT.json](HLSFACTORY_PILOT.json) | Three reduced-shape Bambu/Verilator portability pilots and preserved failures | `0677389210d6d962782f166307cd755eb89209b9bce59215ad68c507f67f8661` |
| [LIGHTNINGSIM_HOST_VALIDATION.json](LIGHTNINGSIM_HOST_VALIDATION.json) | Two fresh native/Vitis/LightningSim checks on the host | `62511f8ea0169cc6db48a99d1ce2db6513e57816c0919af50fee8486cf8003b3` |

Each JSON file moved here from the project root without editing its contents.
The later host-validation receipt was copied unchanged from its external run
directory. It identifies the restricted sandbox's socket-send denial as the
cause of the earlier LightningSim timeouts; the unmodified host runs completed.
Relative paths recorded inside them still resolve from the original project
root, not from `docs/evidence/`. Absolute paths retain the original machine
locations. The pre-cleanup Git snapshot is
`snapshots/minimal-before-cleanup-20261027` (source tip
`951d6f12748656a50927500a678082f9eefb7ec2`). It preserves the original example
layout, including demonstration files removed from the current working tree.

The [HLSFactory fixture manifest](../../examples/hlsfactory/manifest.json),
[native checks](../../examples/hlsfactory/native-checks.json), and
[extent-hook checks](../../examples/hlsfactory/native-extent-checks.json) remain
at their original paths with their original bytes. Their sources, notices, and
upstream licenses remain alongside the runnable examples.

Generated `runs/` and optional `.tools/` payloads are retained in external local
storage through compatibility links at those project-root paths. They are not
included in the Git repository or installed Python package. Receipt hashes bind
the original generated evidence; moving a directory does not create a new run.
The [cleanup record](../CLEANUP.md) lists the exact storage destinations and maps
retired build/wheel paths. [CLEANUP_VALIDATION.json](CLEANUP_VALIDATION.json)
records the archive verification and the post-migration checks.

Saved-response pilots made no provider requests. Native agreement and checked
RTL invocations cover their registered test cases. The receipts do not establish
paper reproduction, all-input equivalence, physical FPGA fit, or board performance.
