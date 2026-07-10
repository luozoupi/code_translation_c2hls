#!/bin/bash
cd /mnt/e/courses/UMN/c2hls/code_translation_c2hls
echo "=== active procs ==="
pgrep -af 'matrix_sweep|_run_sonnet_campaign|_queue_gpt55|_run_gpt55' | grep -v pgrep || echo "  (none running)"
echo
echo "=== SONNET arm markers + exits ==="
grep -E 'CAMPAIGN (START|END)|MULTISTEP .* START|ONE-SHOT START|-exit=' _sonnet_campaign.out 2>/dev/null
echo "--- last 6 cell lines ---"
grep -E '\[[0-9]+/26\]|ok=|exit_|timeout_' _sonnet_campaign.out 2>/dev/null | tail -6
echo
echo "=== SONNET cells with result JSON (per arm) ==="
for d in results_matrix_u280_ENH_oneshot_SONNET results_matrix_u280_ENH_curated_multistep_skills_SONNET results_matrix_u280_ENH_allpositive_multistep_skills_SONNET; do
  n=$(ls "$d"/hlsfactory_*/*/*_results.json 2>/dev/null | wc -l)
  echo "  $d: $n/26"
done
echo
echo "=== GPT-5.5 queue state ==="
tail -3 _gpt55_campaign.out 2>/dev/null || echo "  (not started yet)"
