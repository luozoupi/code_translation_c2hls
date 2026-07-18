#!/bin/bash
cd /mnt/e/courses/UMN/c2hls/code_translation_c2hls
echo "===== RESUME $(date '+%F %T') =====" >> _sonnet_campaign.out
if pgrep -f _run_sonnet_campaign >/dev/null 2>&1; then echo "sonnet already running"; else
  setsid bash _run_sonnet_campaign.sh >> _sonnet_campaign.out 2>&1 < /dev/null &
  echo "relaunched sonnet campaign pid $!"
fi
sleep 2
if pgrep -f _queue_gpt55_after_sonnet >/dev/null 2>&1; then echo "gpt55 waiter already running"; else
  setsid bash _queue_gpt55_after_sonnet.sh >> _gpt55_campaign.out 2>&1 < /dev/null &
  echo "relaunched gpt55 waiter pid $!"
fi
