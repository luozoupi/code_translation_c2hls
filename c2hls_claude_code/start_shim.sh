#!/bin/bash
# Start claude_code_shim.py on this login node (internet + Claude Code login) on a free port.
#   start_shim.sh [workers]            plain mode: every call is a tool-less `claude -p`
#   AGENT=1 start_shim.sh [workers]    agent mode for flash-step calls (FLASH MODE / HARD REJECT /
#                                      quality repair); other calls stay plain. Needs BENCH_DIR or the
#                                      default n1024 autosa_mm bench. MAX_CHECKS (12) caps checks per session.
# Prints the URL compute nodes use; writes shim[_agent]_url.txt and shim[_agent].pid. Logs: logs/shim[_agent]_<stamp>/.
# Stop: kill "$(cat shim.pid)"  or  kill "$(cat shim_agent.pid)"
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ "$(hostname -s)" == login* ]] || { echo "run this on a login node (compute nodes have no internet)" >&2; exit 1; }
NAME=shim; EXTRA=()
if [[ "${AGENT:-0}" == 1 ]]; then
  NAME=shim_agent
  BENCH_DIR="${BENCH_DIR:-/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/code_translation_c2hls/artifacts/pc2/autosa_mm_ijk_benches/n1024/autosa_mm}"
  EXTRA=(--agent-match '^(FLASH MODE:|HARD REJECT —|The current HLS code already synthesizes, but the controller flagged)'
         --agent-bench-dir "$BENCH_DIR" --agent-max-checks "${MAX_CHECKS:-12}")
fi
if [[ -f "$D/$NAME.pid" ]] && kill -0 "$(cat "$D/$NAME.pid")" 2>/dev/null; then
  echo "$NAME already running (pid $(cat "$D/$NAME.pid")): $(cat "$D/${NAME}_url.txt")"; exit 0
fi
PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1])")
LOG="$D/logs/${NAME}_$(date +%Y%m%dT%H%M%S)"; mkdir -p "$LOG"
nohup python3 "$D/claude_code_shim.py" --port "$PORT" --workers "${1:-2}" --log-dir "$LOG" "${EXTRA[@]}" \
  > "$LOG/server.log" 2>&1 &
echo $! > "$D/$NAME.pid"
URL="http://$(hostname -s):$PORT/v1"; echo "$URL" > "$D/${NAME}_url.txt"
sleep 2; curl -sf "$URL/health" >/dev/null || { echo "$NAME failed to start; see $LOG/server.log" >&2; exit 1; }
echo "$URL"
