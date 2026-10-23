#!/bin/bash
# run_agent_ds.sh <workspace> [hours=10]: run_agent.sh with the Claude Code harness driven by DeepSeek (experiment D).
# Claude Code talks to the local proxy study/exp_hlsmith/ds_anthropic_proxy.py (login5:18095), which adds the DeepSeek
# key and refuses any request containing the user's email. A fresh, empty CLAUDE_CONFIG_DIR per workspace means no
# Anthropic account is logged in (so no account email in context); non-essential traffic is off.
# RESUME=1 continues the session in transcript.jsonl (e.g. after a usage-limit stop) into transcript_resumeN.jsonl.
# The body is one { ...; exit; } block: bash parses it whole before running, so editing this file while a session
# runs cannot change or break that session (bash otherwise keeps reading the file at a byte offset).
{
set -uo pipefail
WS=$(cd "$1" && pwd); H=${2:-10}; V=$(cat $WS/variant.txt)
# budget mode (task.json "budget".hours): the session cap comes from the task, not the caller
BH=$(python3 -c "import json;print((json.load(open('$WS/task.json')).get('budget') or {}).get('hours',''))"); [ -n "$BH" ] && H=$BH
TOOLS="Read,Write,Edit,Glob,Grep,Bash"
ALLOW="Read,Write,Edit,Glob,Grep,Bash(bin/hls:*),Bash(./bin/hls:*),Bash(ls:*),Bash(cat:*),Bash(head:*),Bash(tail:*),Bash(grep:*),Bash(diff:*),Bash(wc:*),Bash(cd:*),Bash(sed -n:*),Bash(awk:*),Bash(sort:*),Bash(python3 -c:*)"
case $V in
  know) TOOLS="$TOOLS,WebSearch,WebFetch"; ALLOW="$ALLOW,WebSearch,WebFetch";;
  team) TOOLS="$TOOLS,WebSearch,WebFetch,Agent"; ALLOW="$ALLOW,WebSearch,WebFetch,Agent";;
esac
cd $WS
PROMPT="Do the task described in CLAUDE.md. Work autonomously until you are done."; OUTF=transcript.jsonl; RES=()
if [ "${RESUME:-0}" = 1 ]; then
  SID=$(head -1 transcript.jsonl | python3 -c "import json,sys; print(json.load(sys.stdin)['session_id'])")
  n=1; while [ -e transcript_resume$n.jsonl ]; do n=$((n+1)); done; OUTF=transcript_resume$n.jsonl; RES=(--resume "$SID")
  PROMPT="Your session was stopped by a usage limit and is now resumed. Jobs you submitted may have finished or timed out; check them with bin/hls and continue the task in CLAUDE.md until you are done."
  date -u +%FT%TZ > resumed_at_$n
else
  date -u +%FT%TZ > started_at
fi
export CLAUDE_CONFIG_DIR=$WS/.claude_cfg ANTHROPIC_BASE_URL=http://login5:18095 ANTHROPIC_AUTH_TOKEN=local-proxy
export ANTHROPIC_MODEL=deepseek-v4-flash ANTHROPIC_SMALL_FAST_MODEL=deepseek-v4-flash CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_TELEMETRY=1
unset ANTHROPIC_API_KEY
mkdir -p $CLAUDE_CONFIG_DIR
timeout ${H}h /home/asa582/.local/bin/claude -p "$PROMPT" "${RES[@]}" \
  --model deepseek-v4-flash --output-format stream-json --verbose \
  --tools "$TOOLS" --allowedTools "$ALLOW" --permission-mode acceptEdits \
  --strict-mcp-config --setting-sources "" --disable-slash-commands \
  > $OUTF 2>> stderr.txt
echo $? > exit_code; date -u +%FT%TZ > ended_at
exit 0
}
