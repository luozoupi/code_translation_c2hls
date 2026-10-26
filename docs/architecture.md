# Implementation map

The October 23 [minimal implementation plan](minimal-core-plan.md) sets the next
development scope: an independent small runner, optional vendor tools and a
Verilator compatibility pilot. The map below describes the current runtime;
those planned changes are not yet implemented.

HLSmith translates public C/C++ inputs, checks functional behavior, synthesizes candidates, and selects feasible hardware designs. The controller keeps synthesis estimates separate from executed RTL measurements.

| Component | Location | Responsibility |
| --- | --- | --- |
| Controller and phase agents | `hlsmith.pipeline` | Translation, synthesis repair, optimization, best-state recovery and run serialization |
| Compiler interface | `hlsmith.toolchain` | Vitis execution, native checks, temporary directories and co-simulation monitoring |
| Correctness boundary | `hlsmith.validation` | Independent output checks, reference isolation and public interface contracts |
| Search and skills | `hlsmith.optimization` | Feedback, routing, candidate caches, hardware constraints and skill retrieval |
| Run identity | `hlsmith.provenance` | Source/input fingerprints, frozen skill identities and study manifests |
| Prompt profiles | `hlsmith.prompt_profiles` | Frozen study prompts and separately versioned researcher prompts |
| Registered comparisons | `hlsmith.experiments` | DeepSeek job registration, budgets, execution and reporting |
| Corpus and training | `dataset_pipeline`, `scripts/` | Result exports, role-attributed datasets, router evaluation and optional SFT tools |

Root Python files retain existing imports and script commands. Import adapters share the implementation module object, including patches applied by existing callers. New code should import the package modules.

The controller remains a substantial module. This migration extracts its source-contract helpers and relocates established subsystems first; further controller decomposition should follow the same captured-response regression tests.

Benchmark, header, skill and configuration bytes live under the package's `data/` directory. Root links retain the existing source checkout paths. Installed wheels use the packaged data independently of the working directory. `C2HLS_DATA_ROOT` selects an explicit input tree and `C2HLS_OUTPUT_ROOT` selects a writable output root.

Prompt profile selection is process-wide. Use a separate process for each profile, as the registered comparison runner does. The paper profile remains the default.

Historical source-text patching scripts remain available for their frozen snapshots. The new comparison runner uses explicit configuration and does not rewrite source code.
