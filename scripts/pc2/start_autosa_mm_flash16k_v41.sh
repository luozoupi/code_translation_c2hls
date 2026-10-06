#!/usr/bin/env bash
# Flash-only replay of frozen mmflow 20260830 flash on today's V4.1-Flash alias.
#
# Pins: max_tokens=16384, frozen user-prompt skill order, aav_n_gf, no DSE/stream.
# Thinking ON omits the API thinking field (hosted default). OFF sends
# thinking.type=disabled.
#
# Usage:
#   ./scripts/pc2/start_autosa_mm_flash16k_v41.sh --thinking-on --endpoint-url http://login5:18092/v1
#   ./scripts/pc2/start_autosa_mm_flash16k_v41.sh --thinking-off --endpoint-url http://login5:18092/v1
#   ./scripts/pc2/start_autosa_mm_flash16k_v41.sh --thinking-on --dry-run --endpoint-url http://login5:18092/v1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

THINKING_MODE=""
ENDPOINT_URL_ARG=""
DRY_RUN=0
STAMP_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --thinking-on) THINKING_MODE="on"; shift ;;
    --thinking-off|--no-thinking) THINKING_MODE="off"; shift ;;
    --endpoint-url) shift; ENDPOINT_URL_ARG="$1"; shift ;;
    --stamp) shift; STAMP_ARG="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${THINKING_MODE}" ]]; then
  echo "ERROR: pass --thinking-on or --thinking-off" >&2
  exit 2
fi

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${C2HLS_PYTHON:-python3}"
fi

REPLAY_DIR="${C2HLS_ROOT}/artifacts/pc2/flash16k_v41_replay"
FROZEN_CELL="${C2HLS_ROOT}/artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf"
PACK_SRC="${C2HLS_ROOT}/hls_full_optimization_skills_schema_1_1_package/skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json"
OVERLAY_SRC="${C2HLS_ROOT}/hls_full_optimization_skills_schema_1_1_package/flash_no_RMW_m_axi_skill_entries.json"
mkdir -p "${REPLAY_DIR}"

mapfile -t _PIN < <("${PY}" - "${REPLAY_DIR}" "${FROZEN_CELL}" "${PACK_SRC}" "${OVERLAY_SRC}" <<'PY'
import hashlib, json, re, shutil, sys
from pathlib import Path

replay, cell, pack_src, overlay_src = map(Path, sys.argv[1:])
hist = json.loads((cell / "autosa_mm_history.json").read_text(encoding="utf-8"))
user = next(
    m["content"]
    for m in hist["messages"]
    if m.get("role") == "user" and "[Step: flash]" in (m.get("content") or "")
)
ids = re.findall(r"\[skill ([^\]]+)\]", user)
if len(ids) != 129 or len(set(ids)) != 129:
    raise SystemExit(f"expected 129 unique skill markers in frozen user prompt, got {len(ids)}")
order_path = replay / "flash_skill_order_from_frozen_user_prompt.json"
order_path.write_text(
    json.dumps(
        {
            "schema": "skill_prompt_order_v1",
            "source": "frozen autosa_mm_history.json flash user prompt [skill id] markers",
            "frozen_cell": str(cell),
            "skill_count": len(ids),
            "ids": ids,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
pack_dst = replay / pack_src.name
overlay_dst = replay / overlay_src.name
shutil.copy2(pack_src, pack_dst)
shutil.copy2(overlay_src, overlay_dst)
ss = cell / "skills_source.json"
manifest = {
    "pack_sha256": hashlib.sha256(pack_src.read_bytes()).hexdigest(),
    "frozen_skills_source_sha256": hashlib.sha256(ss.read_bytes()).hexdigest() if ss.is_file() else None,
    "pack_matches_frozen_skills_source": (
        ss.is_file() and pack_src.read_bytes() == ss.read_bytes()
    ),
    "overlay_sha256": hashlib.sha256(overlay_src.read_bytes()).hexdigest(),
    "prompt_skill_count": len(ids),
    "prompt_first_ids": ids[:4],
    "order_json": str(order_path),
    "pack_copy": str(pack_dst),
    "overlay_copy": str(overlay_dst),
}
(replay / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(order_path)
print(pack_dst)
print(manifest["pack_sha256"][:16])
print("true" if manifest["pack_matches_frozen_skills_source"] else "false")
PY
)

ORDER_JSON="${_PIN[0]}"
PACK_COPY="${_PIN[1]}"
PACK_SHA16="${_PIN[2]}"
PACK_MATCH="${_PIN[3]}"
if [[ "${PACK_MATCH}" != "true" ]]; then
  echo "ERROR: repo gemm_flatten pack does not match frozen skills_source.json (sha ${PACK_SHA16})" >&2
  exit 2
fi

if [[ "${THINKING_MODE}" == "on" ]]; then
  # Record intent; _thinking_env_mode("api_default") omits extra_body.thinking.
  export C2HLS_THINKING=api_default
  ARM="think"
  JOB_PREFIX="mmf16t"
else
  export C2HLS_THINKING=disabled
  ARM="nothink"
  JOB_PREFIX="mmf16n"
fi

if [[ -n "${STAMP_ARG}" ]]; then
  STAMP="${STAMP_ARG}"
else
  STAMP="$(date -u +%Y%m%d_%H%M%S)_flash16k_v41_${ARM}"
fi

unset C2HLS_DSE_V2 C2HLS_DSE_V2_CHAIN_FLASH C2HLS_DSE_V2_GRID C2HLS_PE_RECIPE || true
export C2HLS_FLASH_ONLY=1
export C2HLS_FLASH_MAX_TOKENS=16384
# Phase B on frozen mmflow used the orchestrator default 8192 (drain out=8191).
export C2HLS_LLM_MAX_TOKENS=8192
# Frozen flash was one completion; do not send continuation prompts.
export C2HLS_CPP_CONTINUATIONS=0
export C2HLS_SKILL_PROMPT_ORDER_JSON="${ORDER_JSON}"
export C2HLS_PACKAGED_SKILLS_JSON="${PACK_COPY}"
export C2HLS_PACKAGED_SKILLS_ONLY=1
export PC2_BATCH_JOB_PREFIX="${JOB_PREFIX}"
export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_autosa_mm_flow_aav_n_gf_flash16k_v41_${ARM}"
export BATCH_PARALLEL_STAMP="${STAMP}"

echo "=== flash16k V4.1 replay arm=${ARM} ==="
echo "order=${ORDER_JSON}"
echo "pack=${PACK_COPY} sha16=${PACK_SHA16} match_frozen=${PACK_MATCH}"
echo "flash_max_tokens=16384 llm_max_tokens=8192 continuations=0 thinking=${C2HLS_THINKING}"

FLOW_ARGS=(--stamp "${STAMP}")
if [[ "${THINKING_MODE}" == "off" ]]; then
  FLOW_ARGS+=(--no-thinking)
fi
if [[ -n "${ENDPOINT_URL_ARG}" ]]; then
  FLOW_ARGS+=(--endpoint-url "${ENDPOINT_URL_ARG}")
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
  FLOW_ARGS+=(--dry-run)
fi

"${SCRIPT_DIR}/start_autosa_mm_flow.sh" "${FLOW_ARGS[@]}"

CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/pc2/${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"
if [[ -d "${CAMPAIGN_ROOT}" ]]; then
  printf '%s\n' "${C2HLS_THINKING}" > "${CAMPAIGN_ROOT}/thinking.txt"
  printf '1\n' > "${CAMPAIGN_ROOT}/flash_only.txt"
  printf '16384\n' > "${CAMPAIGN_ROOT}/flash_max_tokens.txt"
  cat > "${CAMPAIGN_ROOT}/NOTES.md" <<EOF
# flash16k V4.1 ${ARM}

Flash-only isolation of frozen mmflow 139484/10. No DSE 2.0, no DSE v1, no stream.

Canonical notebook: \`docs/frozen-flash-139k-replication.md\`

- thinking.txt: \`${C2HLS_THINKING}\` (ON = omit API thinking field; OFF = thinking.type=disabled)
- skill order: \`${ORDER_JSON}\` (extracted from frozen flash **user prompt**, 129 ids)
- pack copy: \`${PACK_COPY}\` (byte-identical to frozen \`skills_source.json\`, sha16 ${PACK_SHA16})
- harvest QoR: cell \`variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/\`
  quote **latency_cycles / latency_cycles_worst + DSP** from \`autosa_mm_flash_opt_report.json\`
  and drain \`finish_reason\`, \`content_len\`, \`out=\`
EOF
fi
