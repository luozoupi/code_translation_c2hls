#!/bin/bash
# pack_for_otus.sh : on Fir, after the freeze (OTUS_MIGRATION.md step 1), pack everything the Otus copy needs into a few
# tarballs for a Globus transfer (Otus needs a VPN, so no ssh/rsync between the clusters). Unpack on Otus with
# `otus/otus_setup.sh unpack`.
#   OUT=<dir> (default /scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/otus_transfer)   OPTIONAL=1: Fir's LightningSim source and AutoSA
#   COPY_ENV=1: the judge's .env (secrets: tick "encrypt transfer" in Globus)   FORCE=1: skip the freeze checks
#   LIVE=1: Fir keeps running (contingency): no freeze checks; files a running job changes mid-pack are kept as read
#           (tar warns, the pack goes on); writes fir_queue_at_pack.txt and PACKED_AT for a later delta transfer
# Writes MANIFEST.tsv (tarball, where it unpacks on Otus) and SHA256SUMS next to the tarballs.
set -euo pipefail
[ "${CC_CLUSTER:-}" = fir ] || { echo "run this on Fir"; exit 1; }
OUT=${OUT:-/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/otus_transfer}
WS=${PACK_WS:-/scratch/hpc-prf-llmfpga/asa582/projects/workspaces} TMP=${PACK_TMP:-/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls}   # PACK_*: for tests
MEM=${PACK_MEM:-$HOME/.claude/projects/-scratch-asa582-workspaces-dse-v4-claude}
if [ "${FORCE:-0}" != 1 ] && [ "${LIVE:-0}" != 1 ]; then
  n=$(squeue -u "$USER" -h | wc -l)
  p=$(ssh -o ConnectTimeout=20 login5 'ps -u $USER -o cmd' | grep -cE 'python3 driver\.py|claude_code_shim\.py|ds_anthropic_proxy\.py|run_agent\.sh' || true)
  e=$WS/c2hls_claude_code/study/agentfw/runs/mm1024_hlsm_r4/exit_code
  echo "Fir: $n jobs queued; driver/shim/proxy/agent processes on login5: $p; C r4 exit_code: $([ -f "$e" ] && cat "$e" || echo missing)"
  [ "$n" = 0 ] && [ "$p" = 0 ] && [ -f "$e" ] || { echo "freeze Fir first (OTUS_MIGRATION.md step 1); FORCE=1 overrides"; exit 1; }
fi
mkdir -p "$OUT"; cd "$OUT"
[ -z "$(ls -A)" ] || { echo "$OUT is not empty: move its contents away first"; exit 1; }
# Vitis internals only (rebuilt by any rerun); every *_work/ check kernel and csynth.xml report stays
X=(--exclude=c2hls_claude_code/.git --exclude=__pycache__ --exclude=.autopilot --exclude=verilog --exclude=vhdl
   --exclude=systemc --exclude=impl --exclude='*/csim/build' --exclude='*.o' --exclude='*.vcd' --exclude='*.wdb')
GZ=(-I pigz); command -v pigz >/dev/null || GZ=(-z)
: > MANIFEST.tsv
date -u +%Y-%m-%dT%H:%M:%SZ > PACKED_AT
squeue -u "$USER" -h -o '%i|%j|%T|%M|%Z' > fir_queue_at_pack.txt || true
pack() {  # pack <tarball> <unpack dir on Otus> <tar args...>
  local f=$1 to=$2; shift 2
  echo "== $f -> $to"
  # GNU tar exits 1 when a file changed while it was read (a running Fir job); the pack keeps that copy
  tar --warning=no-file-changed -cf "$f" "$@" || [ $? = 1 ]; printf '%s\t%s\n' "$f" "$to" >> MANIFEST.tsv
  echo "   $(du -h "$f" | cut -f1), $(tar -tf "$f" | wc -l) entries"
}
# $J = <workspaces>/code_translation_c2hls (the clone), $W = <workspaces>, $B = <projects>, ~ = home
pack study_results.tar.gz '$J' "${GZ[@]}" "${X[@]}" -C "$WS" c2hls_claude_code
pack workspaces_extra.tar.gz '$W' "${GZ[@]}" "${X[@]}" -C "$WS" soda_src dse_v4_claude
pack tmp_c2hls.tar '$B/tmp/c2hls' -C "$TMP" vitis_hls_2023.2_include lsim/soda      # lsim/soda is already one tarball per project
pack claude_memory.tar.gz '~/.claude/projects/$KEY' "${GZ[@]}" -C "$MEM" memory
if [ "${OPTIONAL:-0}" = 1 ]; then
  pack lightningsim_fir_src.tar.gz '$B/tools' "${GZ[@]}" -C /scratch/asa582/tools/lightningsim --transform 's|^src|lightningsim_fir_src|' src
  pack autosa.tar '$W' -C "$WS" AutoSA
fi
if [ "${COPY_ENV:-0}" = 1 ]; then
  ( umask 077; pack judge_env.tar '$J' -C "$WS/code_translation_c2hls" .env )
fi
sha256sum $(cut -f1 MANIFEST.tsv) PACKED_AT fir_queue_at_pack.txt > SHA256SUMS
echo; echo "done: $OUT"; ls -l "$OUT"; du -sh "$OUT"
echo "Globus: transfer this whole folder to /scratch/hpc-prf-llmfpga/asa582/projects/transfer_from_fir/ on Otus,"
echo "then on Otus: otus/otus_setup.sh unpack"
