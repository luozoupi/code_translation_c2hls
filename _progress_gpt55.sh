#!/bin/bash
cd /mnt/e/courses/UMN/c2hls/code_translation_c2hls
echo "now = $(date '+%F %T')"
echo "=== gpt55 procs ==="
pgrep -af 'matrix_sweep|_run_gpt55_campaign' | grep -v pgrep || echo "  NONE RUNNING"
echo "=== gpt55 arm markers + exits ==="
grep -E 'CAMPAIGN (START|END)|ONE-SHOT START|MULTISTEP .* START|gpt55-.*-exit=' _gpt55_campaign.out 2>/dev/null
echo "--- last cell lines ---"
grep -E '\[[0-9]+/26\]|ok=|exit_|timeout_' _gpt55_campaign.out 2>/dev/null | tail -5
echo "=== gpt55 cells with result JSON ==="
for d in results_matrix_u280_ENH_oneshot_GPT55 results_matrix_u280_ENH_curated_multistep_skills_GPT55 results_matrix_u280_ENH_allpositive_multistep_skills_GPT55; do
  [ -d "$d" ] && echo "  $d: $(ls $d/hlsfactory_*/*/*_results.json 2>/dev/null | wc -l)/26" || echo "  $d: (not started)"
done
