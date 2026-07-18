#!/bin/bash
cd /mnt/e/courses/UMN/c2hls/code_translation_c2hls
if pgrep -f '_run_gpt55_campaign|matrix_sweep.py' >/dev/null 2>&1; then
  echo "already running:"; pgrep -af 'matrix_sweep.py|_run_gpt55_campaign' | grep -v pgrep; exit 0
fi
nohup setsid bash _run_gpt55_campaign.sh > _gpt55_campaign.out 2>&1 < /dev/null &
disown
echo "launched _run_gpt55_campaign.sh; holding session until matrix_sweep establishes..."
for i in $(seq 1 40); do
  sleep 3
  if pgrep -f 'matrix_sweep.py' >/dev/null 2>&1; then echo "matrix_sweep UP after $((i*3))s"; break; fi
done
echo "=== procs ==="; pgrep -af 'matrix_sweep.py|_run_gpt55_campaign' | grep -v pgrep || echo "FAILED TO START"
echo "=== log head ==="; head -8 _gpt55_campaign.out
