#!/usr/bin/env bash
# Tier A ready flash smoke: 2 small kernels, 90-skills packaged library only.
#
# Default benches: spector_hls_dct, hp_fft_n256__UF1
# Skills: skills_ii_target_miss_solutions_added(90skills).json (all_skills_avoids_global)
#
# Usage (login node, repo root):
#   ./scripts/pc2/run_tier_a_flash_smoke.sh --dry-run
#   ./scripts/pc2/run_tier_a_flash_smoke.sh --submit
#   ./scripts/pc2/run_tier_a_flash_smoke.sh --submit --stamp 20260616_tier_a_smoke
#   ./scripts/pc2/run_tier_a_flash_smoke.sh --submit --benches spector_hls_dct,hp_fft_n256__UF1
#
# Environment overrides:
#   PC2_TIER_A_FLASH_WALLTIME   Slurm GPU+compute walltime (default 3:00:00)
#   C2HLS_TIER_A_FLASH_SMOKE_STAMP  artifact stamp
#   C2HLS_TURNS                   repair turns per phase (default 4)
#
# Per-step watchdogs (set in tier_a_flash_lib before worker runs):
#   C2HLS_SYNTH_TIMEOUT=1200  C2HLS_CSIM_TIMEOUT=180  C2HLS_LLM_TIMEOUT=900
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

PY="${C2HLS_PYTHON:-python3}"
SESSION_ID="${PC2_TIER_A_FLASH_SESSION_ID:-flash_tier_a_smoke_90skills}"
STAMP="${C2HLS_TIER_A_FLASH_SMOKE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
BENCHES="${C2HLS_TIER_A_FLASH_BENCHES:-spector_hls_dct,hp_fft_n256__UF1}"
DRY_RUN=0
SUBMIT=0
AUTO_STOP=1
FORCE=0

usage() {
  cat <<EOF
Usage: $0 [--dry-run | --submit] [options]

Modes (exactly one required):
  --dry-run    Preflight + write manifest plan; no Slurm jobs
  --submit     Preflight + submit one PC2 GPU+compute session

Options:
  --stamp STAMP       Artifact / env stamp (default: date-based)
  --benches A,B       Comma-separated tier_A_ready bench names
  --no-auto-stop      Do not stop GPU after compute worker succeeds
  --force             Re-run benches even when multistep results already exist
  -h, --help          Show this help

Defaults: benches=${BENCHES}
          walltime=\${PC2_TIER_A_FLASH_WALLTIME:-3:00:00}
          session_id=${SESSION_ID}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --submit) SUBMIT=1; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --benches) shift; BENCHES="$1"; shift ;;
    --no-auto-stop) AUTO_STOP=0; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${DRY_RUN}" -eq 1 && "${SUBMIT}" -eq 1 ]]; then
  echo "ERROR: use --dry-run or --submit, not both" >&2
  exit 2
fi
if [[ "${DRY_RUN}" -eq 0 && "${SUBMIT}" -eq 0 ]]; then
  echo "ERROR: specify --dry-run or --submit" >&2
  usage >&2
  exit 2
fi

export PC2_FORCE_WALLTIME="${PC2_TIER_A_FLASH_WALLTIME:-3:00:00}"
export C2HLS_TIER_A_FLASH_SMOKE_STAMP="${STAMP}"

run_batch() {
  "${PY}" scripts/pc2/run_tier_a_flash_smoke_batch.py --pc2 "$@"
}

echo "tier_A flash smoke stamp=${STAMP} walltime=${PC2_FORCE_WALLTIME}"
echo "benches=${BENCHES} session_id=${SESSION_ID}"
echo "corpus=related_work/benchmarks/HLSFactory_benchmarks/tier_A_ready"
echo ""

echo "=== preflight (skills + bench files) ==="
run_batch --verify-only --benches "${BENCHES}" --stamp "${STAMP}"
echo ""

echo "=== dry-run plan ==="
run_batch --dry-run --benches "${BENCHES}" --stamp "${STAMP}"
echo ""

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run complete: no Slurm jobs submitted"
  echo "Artifacts plan: artifacts/pc2/flash_tier_a_smoke_${STAMP}/"
  exit 0
fi

export PC2_SESSION_ID="${SESSION_ID}"
_pc2_configure_session_paths
if pgrep -u "$(whoami)" -f "watch_session.sh ${SESSION_ID}" >/dev/null 2>&1; then
  echo "stopping existing session ${SESSION_ID}"
  "${SCRIPT_DIR}/stop_session.sh" --session-id "${SESSION_ID}" || true
fi

worker_cmd=(
  "C2HLS_RECORD_FLOW=1"
  "C2HLS_TIER_A_FLASH_SMOKE_STAMP=${STAMP}"
  "${PY}" scripts/pc2/run_tier_a_flash_smoke_batch.py
  --pc2
  --benches "${BENCHES}"
  --stamp "${STAMP}"
)
if [[ "${FORCE}" -eq 1 ]]; then
  worker_cmd+=(--force)
fi
worker_cmd_str="${worker_cmd[*]}"

start_args=(--session-id "${SESSION_ID}" --worker-cmd "${worker_cmd_str}")
if [[ "${AUTO_STOP}" -eq 1 ]]; then
  start_args+=(--auto-stop-on-complete)
fi

echo "=== submitting PC2 session ${SESSION_ID} ==="
echo "worker: ${worker_cmd_str}"
"${SCRIPT_DIR}/start_session.sh" "${start_args[@]}"

echo ""
echo "Submitted tier_A flash smoke (1 GPU + 1 compute when GPU starts)."
echo "Monitor: tail -f artifacts/pc2/sessions/${SESSION_ID}/watch.log"
echo "Artifacts: artifacts/pc2/flash_tier_a_smoke_${STAMP}/"
