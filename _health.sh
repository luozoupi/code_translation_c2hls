#!/bin/bash
cd /mnt/e/courses/UMN/c2hls/code_translation_c2hls
D=results_matrix_u280_ENH_allpositive_multistep_skills_SONNET
echo "now = $(date '+%F %T')"
echo "=== campaign procs ==="
pgrep -af 'matrix_sweep|_run_sonnet_campaign|_queue_gpt55|vitis_hls' | grep -v pgrep || echo "  NONE RUNNING (dead!)"
echo "=== current cell: freshness (is it moving?) ==="
L=$(ls -t $D/hlsfactory_*/*/matrix_run.log 2>/dev/null | head -1)
echo "cell: $(basename $(dirname $(dirname "$L")))"
echo "log last-modified: $(stat -c %y "$L" 2>/dev/null)   (now $(date '+%F %T'))"
echo "retry-request lines in current cell: $(grep -c 'Retrying request' "$L" 2>/dev/null)"
tail -3 "$L" 2>/dev/null
echo "=== Argo reachable? ==="
KEY=$(cat /mnt/e/courses/UMN/c2hls/api-key.txt | tr -d '\r\n')
curl -sS -m 15 -H "x-api-key: $KEY" -H "anthropic-version: 2023-06-01" -H "Content-Type: application/json" -d '{"model":"claude-sonnet-4-6","max_tokens":5,"messages":[{"role":"user","content":"ok"}]}' https://apps.inside.anl.gov/argoapi/v1/messages 2>&1 | grep -oE '"stop_reason":"[^"]*"' || echo "  ARGO UNREACHABLE"
echo "=== disk free ==="
df -h /mnt/e / /tmp 2>/dev/null | awk 'NR==1 || /\/mnt\/e|\/$|\/tmp/'
