#!/usr/bin/env bash
# Post-flash DATAFLOW refactor on an existing flash matrix.
#
# Usage:
#   ./scripts/pc2/start_post_flash_dataflow.sh --dry-run
#   ./scripts/pc2/start_post_flash_dataflow.sh --submit --force
#   ./scripts/pc2/start_post_flash_dataflow.sh --submit --force --no-auto-stop-gpu
#   ./scripts/pc2/start_post_flash_dataflow.sh --submit --force --prompt-policy user_skills
#
# Prompt policies (C2HLS_POST_FLASH_PROMPT_POLICY or --prompt-policy):
#   system_skills  — legacy: mandatory rules + full skill catalog in system message
#   user_skills    — rules/checklist in system; skills + rich task brief in user message
#
# Results dirs are named: post_flash_dataflow_results_<stamp>_pp-<policy>_<suffix>
# Kernel bundle subdir: kernel_bundle_pp-<policy>
#   ./scripts/pc2/start_post_flash_dataflow.sh --show-prompts
#
# --submit starts a supervised PC2 session. By default it **borrows** an active
# GPU vLLM endpoint from another session/campaign when available (no extra gpu_h100
# job). Use --no-borrow-gpu to always request a dedicated GPU node.
#
# OPENAI_BASE_URL comes from llm_endpoint.json (borrowed or local GPU job).
#
# Skills: flash_no_RMW_m_axi_skill_entries.json (33 overlay skills) injected into
# the DATAFLOW system prompt. Override with C2HLS_DATAFLOW_SKILL_ENTRIES_JSON.
#
# Default matrix: flash_all_new_skills_avoids_global_20260623_024548
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

PY="${C2HLS_PYTHON:-python3}"
if [[ -z "${C2HLS_PYTHON:-}" && -n "${PC2_VLLM_VENV:-}" && -x "${PC2_VLLM_VENV}/bin/python3" ]]; then
  PY="${PC2_VLLM_VENV}/bin/python3"
fi
if [[ -x "${C2HLS_ROOT}/.venv/bin/python3" ]]; then
  PY="${C2HLS_ROOT}/.venv/bin/python3"
fi

MATRIX_ROOT="${C2HLS_POST_FLASH_MATRIX_ROOT:-artifacts/pc2/flash_all_new_skills_avoids_global_20260623_024548}"
BENCHES="${C2HLS_POST_FLASH_BENCHES:-}"
RESULTS_SUFFIX="${C2HLS_POST_FLASH_RESULTS_SUFFIX:-parallel_fix}"
PROMPT_POLICY="${C2HLS_POST_FLASH_PROMPT_POLICY:-system_skills}"
CONTRACT_TURNS="${C2HLS_DATAFLOW_CONTRACT_ROUNDS:-4}"
NO_CONTRACT_CHECK=0
SUBMIT=0
DRY=0
FORCE=0
SHOW_PROMPTS=0
BORROW_GPU=1
AUTO_STOP_GPU=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) SUBMIT=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --matrix-root) MATRIX_ROOT="$2"; shift 2 ;;
    --benches) BENCHES="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --show-prompts) SHOW_PROMPTS=1; shift ;;
    --prompt-policy) PROMPT_POLICY="$2"; shift 2 ;;
    --contract-turns) CONTRACT_TURNS="$2"; shift 2 ;;
    --no-contract-check) NO_CONTRACT_CHECK=1; shift ;;
    --borrow-gpu) BORROW_GPU=1; shift ;;
    --no-borrow-gpu) BORROW_GPU=0; shift ;;
    --auto-stop-gpu) AUTO_STOP_GPU=1; shift ;;
    --no-auto-stop-gpu) AUTO_STOP_GPU=0; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ "${SHOW_PROMPTS}" -eq 1 ]]; then
  exec "${PY}" scripts/pc2/run_post_flash_dataflow.py --show-prompts --prompt-policy "${PROMPT_POLICY}"
fi

ARGS=(--pc2 --matrix-root "${MATRIX_ROOT}" --results-suffix "${RESULTS_SUFFIX}" --prompt-policy "${PROMPT_POLICY}" --contract-turns "${CONTRACT_TURNS}")
[[ -n "${BENCHES}" ]] && ARGS+=(--benches "${BENCHES}")
[[ "${DRY}" -eq 1 ]] && ARGS+=(--dry-run)
[[ "${FORCE}" -eq 1 ]] && ARGS+=(--force)
[[ "${NO_CONTRACT_CHECK}" -eq 1 ]] && ARGS+=(--no-contract-check)

if [[ "${SUBMIT}" -eq 1 ]]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  SESSION_ID="post_flash_dataflow_${STAMP}"
  WORKER_CMD="${PY} scripts/pc2/run_post_flash_dataflow.py ${ARGS[*]}"
  pc2_log "submitting supervised session id=${SESSION_ID}"
  pc2_log "worker: ${WORKER_CMD}"
  BORROW_ARGS=()
  if [[ "${BORROW_GPU}" -eq 1 ]]; then
    BORROW_ARGS=(--borrow-gpu)
    pc2_log "GPU policy: borrow active endpoint if available, else submit gpu job"
  else
    BORROW_ARGS=(--no-borrow-gpu)
    pc2_log "GPU policy: dedicated gpu job"
  fi
  STOP_ARGS=()
  if [[ "${AUTO_STOP_GPU}" -eq 1 ]]; then
    STOP_ARGS=(--auto-stop-on-complete)
    pc2_log "GPU policy: scancel dedicated gpu job when worker finishes (delay via PC2_AUTO_STOP_DELAY_SEC)"
  fi
  exec "${SCRIPT_DIR}/start_session.sh" \
    --session-id "${SESSION_ID}" \
    --worker-cmd "${WORKER_CMD}" \
    "${BORROW_ARGS[@]}" \
    "${STOP_ARGS[@]}"
else
  exec "${PY}" scripts/pc2/run_post_flash_dataflow.py "${ARGS[@]}"
fi
