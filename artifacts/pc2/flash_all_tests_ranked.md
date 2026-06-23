# Flash synthesis — all tests ranked

**49 runs** in win pool · **30 variants** · 27 benches (`doitgen` excluded)

**Abs wins** = sole lowest latency among all runs. **Shared wins** = tied for lowest (within 0.1%). **Pooled†** = abs + ½ per 2-way tie, etc. (presentation metric).

| Rank | Variant | Geo-mean | Abs wins | Shared wins | Pooled† | OK | Faster GT | Slower GT | Skills file | # in file | Injected | Artifact |
|------|---------|----------|----------|-------------|---------|----|-----------|-----------|-------------|-----------|----------|----------|
| 1 | No avoids (old) | 0.0342 | 1 | 2 | 1.4 | 27/28 | 23 | 0 | `skills.json` | 55 (41+14) | 41 | `flash_all_skills_no_avoids_global_20260620_113247` |
| 2 | All+avoids (new) | 0.0397 | 1 | 1 | 1.1 | 27/28 | 25 | 0 | `skills_ii_target_miss_solutions_added.json` | 90 (66+24) | 90 | `flash_all_new_skills_avoids_global_20260623_024548` |
| 3 | No avoids (new) | 0.0474 | 1 | 0 | 1.0 | 27/28 | 22 | 2 | `skills_ii_target_miss_solutions_added.json` | 90 (66+24) | 66 | `flash_all_new_skills_no_avoids_global_20260621_075846` |
| 4 | Noskills (old) | 0.0528 | 0 | 0 | 0.0 | 27/28 | 25 | 0 | `—` | 0 | 0 | `flash_noskills_20260620_004507` |
| 5 | All+avoids (old) | 0.0620 | 0 | 1 | 0.1 | 27/28 | 22 | 0 | `skills.json` | 55 (41+14) | 55 | `flash_all_skills_avoids_global_20260621_075846` |
| 6 | HPC+ v1 noskills | 0.0627 | 0 | 2 | 0.3 | 27/28 | 24 | 0 | `—` | 0 | 0 | `flash_hpc_positive_v1_noskills_20260623_051145` |
| 7 | Curated All+avoids json+LLM (warnings) | 0.0644 | 1 | 0 | 1.0 | 27/28 | 24 | 1 | `skills_ii_target_miss_solutions_added.json` | 90 (66+24) | curated | `flash_curated_all_avoids_llm_warnings_20260621_104044` |
| 8 | Curated No avoids json+LLM (warnings) | 0.0645 | 1 | 1 | 1.1 | 27/28 | 23 | 1 | `skills_ii_target_miss_solutions_added.json` | 90 (66+24) | curated | `flash_curated_no_avoids_llm_warnings_20260621_104044` |
| 9 | Bn 2+2 (old) | 0.0653 | 0 | 1 | 0.5 | 27/28 | 26 | 0 | `skills.json` | 55 (41+14) | 4 | `flash_skills_20260620_004507` |
| 10 | Noskills (new) | 0.0670 | 0 | 1 | 0.2 | 27/28 | 25 | 0 | `—` | 0 | 0 | `flash_noskills_new_20260622_113723` |
| 11 | HPC+ v2 noskills | 0.0675 | 0 | 2 | 0.3 | 27/28 | 25 | 0 | `—` | 0 | 0 | `flash_hpc_positive_v2_noskills_20260623_132117` |
| 12 | HPC+ v1 all skills | 0.0709 | 0 | 0 | 0.0 | 27/28 | 24 | 1 | `skills_flash_hpc_positive_v1.json` | 30 (30+0) | 30 | `flash_hpc_positive_v1_all_skills_20260623_051145` |
| 13 | Curated All+avoids json_only (bottleneck) | 0.0719 | 1 | 0 | 1.0 | 27/28 | 23 | 1 | `skills_ii_target_miss_solutions_added.json` | 90 (66+24) | curated | `flash_curated_all_avoids_json_bottleneck_20260621_104044` |
| 14 | Curated Noskills (warnings) | 0.0749 | 0 | 2 | 0.3 | 27/28 | 23 | 1 | `—` | 0 | 0 | `flash_curated_noskills_warnings_20260621_104044` |
| 15 | Bn 4+2 (new) | 0.0767 | 1 | 0 | 1.0 | 27/28 | 25 | 1 | `skills_ii_target_miss_solutions_added.json` | 90 (66+24) | 6 | `flash_bn_skills_new_4_2_20260621_075846` |
| 16 | Curated No avoids json_only (combined) | 0.0812 | 0 | 0 | 0.0 | 27/28 | 22 | 1 | `skills_ii_target_miss_solutions_added.json` | 90 (66+24) | curated | `flash_curated_no_avoids_json_combined_20260621_104044` |
| 17 | Curated Noskills (bottleneck) | 0.0834 | 0 | 1 | 0.1 | 27/28 | 22 | 2 | `—` | 0 | 0 | `flash_curated_noskills_bottleneck_20260621_104044` |
| 18 | Bn 2+2 (new) | 0.0843 | 0 | 0 | 0.0 | 27/28 | 24 | 2 | `skills_ii_target_miss_solutions_added.json` | 90 (66+24) | 4 | `flash_bn_skills_new_2_2_20260621_075846` |
| 19 | Curated Noskills (combined) | 0.0914 | 0 | 0 | 0.0 | 27/28 | 26 | 0 | `—` | 0 | 0 | `flash_curated_noskills_combined_20260621_104044` |
| 20 | Curated All+avoids json+LLM (bottleneck) | 0.1000 | 0 | 0 | 0.0 | 27/28 | 22 | 0 | `skills_ii_target_miss_solutions_added.json` | 90 (66+24) | curated | `flash_curated_all_avoids_llm_bottleneck_20260621_104044` |
| 21 | Curated No avoids json+LLM (combined) | 0.1048 | 1 | 0 | 1.0 | 27/28 | 21 | 3 | `skills_ii_target_miss_solutions_added.json` | 90 (66+24) | curated | `flash_curated_no_avoids_llm_combined_20260621_104044` |
| 22 | HPC+ v1 Bn 4+2 | 0.1108 | 1 | 0 | 1.0 | 27/28 | 23 | 0 | `skills_flash_hpc_positive_v1.json` | 30 (30+0) | 6 | `flash_hpc_positive_v1_bn_4_2_20260623_051145` |
| 23 | Curated All+avoids json_only (combined) | 0.1158 | 0 | 0 | 0.0 | 27/28 | 22 | 0 | `skills_ii_target_miss_solutions_added.json` | 90 (66+24) | curated | `flash_curated_all_avoids_json_combined_20260621_104044` |
| 24 | Bn 6+2 (new) | 0.1199 | 0 | 1 | 0.1 | 27/28 | 23 | 1 | `skills_ii_target_miss_solutions_added.json` | 90 (66+24) | 8 | `flash_bn_skills_new_6_2_20260621_075846` |
| 25 | HPC+ v2 all skills | 0.1299 | 0 | 0 | 0.0 | 27/28 | 23 | 1 | `skills_flash_hpc_positive_v2.json` | 34 (34+0) | 34 | `flash_hpc_positive_v2_all_skills_20260623_132117` |
| 26 | Curated No avoids json+LLM (bottleneck) | 0.1306 | 0 | 0 | 0.0 | 27/28 | 22 | 0 | `skills_ii_target_miss_solutions_added.json` | 90 (66+24) | curated | `flash_curated_no_avoids_llm_bottleneck_20260621_104044` |
| 27 | Curated No avoids json_only (bottleneck) | 0.1329 | 1 | 1 | 1.1 | 27/28 | 17 | 1 | `skills_ii_target_miss_solutions_added.json` | 90 (66+24) | curated | `flash_curated_no_avoids_json_bottleneck_20260621_104044` |
| 28 | Curated All+avoids json+LLM (combined) | 0.1333 | 0 | 0 | 0.0 | 27/28 | 20 | 2 | `skills_ii_target_miss_solutions_added.json` | 90 (66+24) | curated | `flash_curated_all_avoids_llm_combined_20260621_104044` |
| 29 | Curated All+avoids json_only (warnings) | 0.1358 | 0 | 0 | 0.0 | 27/28 | 19 | 0 | `skills_ii_target_miss_solutions_added.json` | 90 (66+24) | curated | `flash_curated_all_avoids_json_warnings_20260621_104044` |
| 30 | Curated No avoids json_only (warnings) | 0.1564 | 0 | 0 | 0.0 | 27/28 | 21 | 3 | `skills_ii_target_miss_solutions_added.json` | 90 (66+24) | curated | `flash_curated_no_avoids_json_warnings_20260621_104044` |
