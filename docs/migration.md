# Moving to the canonical checkout

The canonical development checkout is `/home/luo00466/code_translation_c2hls`, on `hlsmith-clean`. Its Git object store is independent of the historical linked worktrees.

The private preservation archive is `/mnt/data1/luo00466/c2hls_archive_20261018`. Its manifests identify original paths, source hashes, Git states, archived payloads and restore instructions. The August archive remains unchanged. Private environment files, peer-review notes and manuscript prose belong to the private archive, not the public source release.

The separate `/home/luo00466/code_translation` project is outside this migration. The collaborator branch `c2hls_enhanced_l_pc2_api_layout` is also unchanged.

The SIGMETRICS follow-up queue was still live during preservation. Its state/summary files require timestamped overlays, and its original checkout and shared Git owner remain available until the queue finishes. Do not delete those directories merely because the canonical checkout exists.

Historical evidence retains original absolute paths. Set `C2HLS_ARCHIVE_PATH_MAP` to a JSON file containing a `paths` object mapping old directory prefixes to restored directories; use `hlsmith.resources.resolve_historical_path` in new research commands. Do not rewrite old manifests to disguise relocation or change their recorded hashes.

Generated reports/results have been removed from the active tracked tree. They remain in historical Git snapshots and the preservation archive. Compact test fixtures and portable numerical evidence remain versioned so the regression suite does not depend on ignored experiment directories.

The source snapshot branch `v6` captures the previously uncommitted SIGMETRICS runtime. It excludes private review notes and machine-local secrets. The new branch adds package organization, prompt profiles, portable tests and the prospective DeepSeek comparison.

Default-branch promotion requires archive verification, deterministic replay, archived-design hardware checks and the fresh comparison. Paid API calls and long tool jobs require numeric API-dollar and compute-hour limits. Until those gates pass, `hlsmith-clean` is a development candidate.
