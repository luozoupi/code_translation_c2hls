# Judge skill libraries as they were on Fir, 2026-10-22

Working-tree copies of the three skill files in `code_translation_c2hls/hls_full_optimization_skills_schema_1_1_package/` that differ from the judge's last commit. The judge repo was not changed.

- Judge: https://github.com/luozoupi/code_translation_c2hls, branch `c2hls_enhanced_l_pc2_api_layout`, HEAD `2ef022410fa6186ff6e16be25f79da13951bf9b7`
- Why: the study scripts read `(90skills).json` and `(90skills)_plus_gemm_flatten_v2.json` by path. A fresh clone of the judge has the committed versions, which differ.
- Who or what changed them is not known; only the file dates are: see the table.
- On Otus, `otus/apply_otus_site.sh` copies them into the judge clone and checks the md5s.

| File | md5 | Modified on Fir | Diff against HEAD |
|---|---|---|---|
| `skills_ii_target_miss_solutions_added(90skills).json` | `47c5eca75ed76dbcb5d04e764d73b369` | 2026-10-14 08:53 | 1 insertion(+), 1 deletion(-) |
| `skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json` | `3edfd742b9e676696f1dd5c647ad4380` | 2026-10-21 08:21 | 1479 insertions(+), 211 deletions(-) |
| `skills_ii_target_miss_solutions_added(90skills)_plus_gemm_flatten_v2.json` | `aa54ac07ca390a038c9b9778ae4f7c15` | 2026-10-10 19:14 | 278 insertions(+), 271 deletions(-) |
