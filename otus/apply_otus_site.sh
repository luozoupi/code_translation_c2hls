#!/bin/bash
# apply_otus_site.sh [--dry-run] : on Otus, after cloning and rsyncing (OTUS_MIGRATION.md steps 5-6), install the Otus
# copies over the Fir-only pieces, in this clone only. Fir is never touched. Then review `git diff` and commit.
#   1. agent tool for free-agent sessions (C/D): study/agentfw/tools/hls_otus, hls_eval_job_otus.py -> tools/hls,
#      tools/hls_eval_job.py, and into bin/ of every workspace that has not finished (no exit_code; e.g. mm1024_hlsm_r4)
#   2. HLSmith-with-an-agent-backend checks: agent/hls_check_otus.{sh.template,py} -> agent/hls_check.{sh.template,py};
#      claude_code_shim.py --agent-account default def-zhenman -> hpc-prf-llmfpga
#   3. flash jobs (driver, sd_* runs): otus/sbatch_flash_cc_otus.sh -> sbatch_flash_cc.sh
#   4. claude binary path (/home/asa582/.local/bin/claude) -> $(command -v claude), in driver.py and the shim start scripts
#   5. judge skill libraries: study/judge_skills_fir_20261022/*.json -> the judge clone, md5-checked (Fir's working state)
#   6. hardened-bench headers: checks <projects>/tmp/c2hls/vitis_hls_2023.2_include (verify_hard.py's INC after the
#      step-6 path sed); if missing, links it to $XILINX_HLS/include (the same Vitis 2023.2 headers)
set -euo pipefail
CC=$(dirname "$(dirname "$(readlink -f "$0")")"); W=$(dirname "$CC"); B=$(dirname "$W")
DRY=0; [ "${1:-}" = --dry-run ] && DRY=1
[ "${CC_CLUSTER:-}" = fir ] && [ $DRY = 0 ] && { echo "this is Fir: only --dry-run here"; exit 1; }
cp_() { echo "  $1 -> $2"; [ $DRY = 1 ] || cp "$1" "$2"; }
cd "$CC"
echo "1. agent tool (free-agent sessions)"
cp_ study/agentfw/tools/hls_otus study/agentfw/tools/hls
cp_ study/agentfw/tools/hls_eval_job_otus.py study/agentfw/tools/hls_eval_job.py
for ws in study/agentfw/runs/*/; do
  [ -f "$ws/bin/hls" ] && [ ! -f "$ws/exit_code" ] || continue
  cp_ study/agentfw/tools/hls_otus "$ws/bin/hls"; cp_ study/agentfw/tools/hls_eval_job_otus.py "$ws/bin/hls_eval_job.py"
done
echo "2. agent-backend checks"
cp_ agent/hls_check_otus.sh.template agent/hls_check.sh.template
cp_ agent/hls_check_otus.py agent/hls_check.py
echo "  claude_code_shim.py: --agent-account default -> hpc-prf-llmfpga"
[ $DRY = 1 ] || sed -i 's/ap.add_argument("--agent-account", default="def-zhenman")/ap.add_argument("--agent-account", default="hpc-prf-llmfpga")/' claude_code_shim.py
echo "3. flash jobs"
cp_ otus/sbatch_flash_cc_otus.sh sbatch_flash_cc.sh
echo "4. claude binary"
CB=$(command -v claude || true); [ -n "$CB" ] || { echo "  claude not on PATH: install Claude Code first (step 4)"; exit 1; }
for f in study/exp_hlsmith/driver.py start_shim.sh logs/start_*.sh; do
  [ -f "$f" ] && grep -q /home/asa582/.local/bin/claude "$f" || continue
  echo "  $f: /home/asa582/.local/bin/claude -> $CB"; [ $DRY = 1 ] || sed -i "s|/home/asa582/.local/bin/claude|$CB|g" "$f"
done
echo "5. judge skill libraries"
K=$W/code_translation_c2hls/hls_full_optimization_skills_schema_1_1_package
S=study/judge_skills_fir_20261022
for f in "$S"/*.json; do cp_ "$f" "$K/$(basename "$f")"; done
[ $DRY = 1 ] || (cd "$K" && md5sum -c "$CC/$S/MD5SUMS")
echo "6. hardened-bench headers"
INC=$B/tmp/c2hls/vitis_hls_2023.2_include
if [ -f "$INC/ap_int.h" ]; then echo "  $INC present"
else
  set +u; source "${OTUS_VITIS_SETTINGS:-/opt/software/FPGA/Xilinx/Vitis/2023.2/settings64.sh}" >/dev/null 2>&1 || true; set -u
  [ -f "${XILINX_HLS:-}/include/ap_int.h" ] || { echo "  no $INC and no \$XILINX_HLS/include: rsync it from Fir (step 5)"; exit 1; }
  echo "  $INC -> $XILINX_HLS/include (symlink)"; [ $DRY = 1 ] || { mkdir -p "$(dirname "$INC")"; ln -sfn "$XILINX_HLS/include" "$INC"; }
fi
grep -q "$B/tmp/c2hls/vitis_hls_2023.2_include" study/bench_hard/verify_hard.py \
  || echo "  NOTE: study/bench_hard/verify_hard.py still has the Fir include path: run the step-6 path sed first"
echo; [ $DRY = 1 ] && echo "dry run: nothing changed" || { git -C "$CC" status --short | head -30; echo "review, then: git -C $CC commit -am 'Otus site: tools, checks, flash job, claude path'"; }
