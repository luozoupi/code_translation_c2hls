#!/bin/bash
# Wait for the Sonnet campaign to finish (serial machine), then run GPT-5.5.
cd /mnt/e/courses/UMN/c2hls/code_translation_c2hls
echo "GPT55 QUEUE: waiting for Sonnet campaign to finish...  $(date '+%F %T')"
while pgrep -f _run_sonnet_campaign >/dev/null 2>&1; do sleep 300; done
while pgrep -x vitis_hls >/dev/null 2>&1; do sleep 120; done
echo "GPT55 QUEUE: Sonnet done, starting GPT-5.5  $(date '+%F %T')"
bash _run_gpt55_campaign.sh
