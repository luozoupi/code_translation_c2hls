#!/bin/bash
# start_ds_proxy_otus.sh [port] : on an Otus login node, start the OpenAI-compatible DeepSeek proxy the controlled
# DeepSeek runs use (controlled/ds_proxy.py; Fir: login5:18094) and write its URL to controlled/ds_proxy_url.txt.
# The key comes from DeepSeek_API in ~/.bashrc (never printed or logged); compute nodes reach the proxy by host name.
set -euo pipefail
CC=$(dirname "$(dirname "$(readlink -f "$0")")")
PORT=${1:-18094}
KEY=$(bash -ic 'printf %s "$DeepSeek_API"' 2>/dev/null)
[ -n "$KEY" ] || { echo "DeepSeek_API is not set in ~/.bashrc"; exit 1; }
cd "$CC"
DEEPSEEK_API_KEY="$KEY" setsid nohup python3 controlled/ds_proxy.py --port "$PORT" --workers 8 \
  --log controlled/ds_proxy_calls.jsonl >> "logs/ds_proxy_$(hostname -s).out" 2>&1 < /dev/null &
sleep 3
echo "http://$(hostname -s):$PORT/v1" > controlled/ds_proxy_url.txt
echo "proxy pid $!; url $(cat controlled/ds_proxy_url.txt)"; tail -1 "logs/ds_proxy_$(hostname -s).out"
