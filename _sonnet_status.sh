#!/bin/bash
cd /mnt/e/courses/UMN/c2hls/code_translation_c2hls
echo "=== running procs ==="
pgrep -af matrix_sweep | grep -v pgrep || echo "  no matrix_sweep"
pgrep -af _run_sonnet_campaign | grep -v pgrep || echo "  no campaign script"
echo "=== _sonnet_campaign.out tail ==="
tail -18 _sonnet_campaign.out 2>/dev/null || echo "(no log yet)"
