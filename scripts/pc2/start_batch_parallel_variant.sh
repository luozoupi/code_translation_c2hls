#!/usr/bin/env bash
# Submit synth/cosim compute nodes for one variant.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

CAMPAIGN_ROOT="${BATCH_PARALLEL_CAMPAIGN_ROOT:?}"
VARIANT="${BATCH_PARALLEL_VARIANT:?}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

PY="${C2HLS_PYTHON:-python3}"
CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_pilot.json}"
export BATCH_PARALLEL_CONFIG="${CONFIG}"
read -r SYNTH_NODES SYNTH_WPN COSIM_NODES COSIM_WPN WORKER_CPUS WORKER_MEM <<<"$(
  "${PY}" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["BATCH_PARALLEL_CONFIG"])
d = json.loads(p.read_text())
print(d["synth_nodes_per_variant"], d["synth_workers_per_node"], d["cosim_nodes_per_variant"], d["cosim_workers_per_node"], d["worker_cpus"], d["worker_mem_gb"])
PY
)"

_register_compute_job() {
  local role="$1"
  local node_index="$2"
  local job_id="$3"
  "${PY}" - "${CAMPAIGN_ROOT}" "${VARIANT}" "${role}" "${node_index}" "${job_id}" <<'PY'
import json, sys
from pathlib import Path
root, variant, role, node_index, job_id = sys.argv[1:6]
p = Path(root) / "campaign.json"
doc = json.loads(p.read_text())
doc.setdefault("compute_jobs", []).append({
    "variant": variant,
    "role": role,
    "node_index": int(node_index),
    "slurm_job_id": job_id,
})
if doc.get("compute_state", "waiting_for_gpu") == "waiting_for_gpu":
    doc["compute_state"] = "submitted"
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
}

submit_node() {
  local role="$1"
  local node_index="$2"
  local workers="$3"
  local cpus=$((workers * WORKER_CPUS))
  local mem=$((workers * WORKER_MEM))
  local template="${SCRIPT_DIR}/batch_parallel_${role}.sbatch.sh"
  echo "submit ${role} node ${node_index}: ${cpus} cpu ${mem}G"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  local job_id
  local job_tag="${PC2_JOB_TAG:-$(basename "${CAMPAIGN_ROOT}")}"
  local job_prefix
  job_prefix="$(pc2_batch_job_prefix "${CAMPAIGN_ROOT}")"
  local enf_export
  enf_export="$("${PY}" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["BATCH_PARALLEL_CAMPAIGN_ROOT"]) / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}

def _flag(key, env, default="0"):
    if key in doc and doc[key] is not None:
        raw = str(doc[key]).strip().lower()
    else:
        raw = str(os.getenv(env) or default).strip().lower()
    return "1" if raw in {"1", "true", "yes", "on"} else "0"

on = _flag("enforcement", "C2HLS_ENFORCEMENT")
rounds = str(doc.get("enforcement_rounds") or os.getenv("C2HLS_ENFORCEMENT_ROUNDS") or "20")
to = str(doc.get("synth_timeout") or os.getenv("C2HLS_SYNTH_TIMEOUT") or "").strip()
bits = [
    f"C2HLS_ENFORCEMENT={on}",
    f"C2HLS_ENFORCEMENT_ROUNDS={rounds}",
    f"C2HLS_AUTOSA_FLOW={_flag('autosa_flow', 'C2HLS_AUTOSA_FLOW')}",
    f"C2HLS_POST_FLASH_DSE={_flag('post_flash_dse', 'C2HLS_POST_FLASH_DSE')}",
    f"C2HLS_DSE_CHAIN_FLASH={_flag('dse_chain_flash', 'C2HLS_DSE_CHAIN_FLASH')}",
    f"C2HLS_DSE_V2={_flag('dse_v2', 'C2HLS_DSE_V2')}",
    f"C2HLS_DSE_V2_CHAIN_FLASH={_flag('dse_v2_chain_flash', 'C2HLS_DSE_V2_CHAIN_FLASH')}",
    f"C2HLS_POST_FLASH_STREAM={_flag('post_flash_stream', 'C2HLS_POST_FLASH_STREAM')}",
    f"C2HLS_STREAM_CHAIN_FLASH={_flag('stream_chain_flash', 'C2HLS_STREAM_CHAIN_FLASH')}",
    f"C2HLS_POST_FLASH_NO_SKILLS={_flag('post_flash_no_skills', 'C2HLS_POST_FLASH_NO_SKILLS')}",
    f"C2HLS_SKIP_PHASE_B={_flag('skip_phase_b', 'C2HLS_SKIP_PHASE_B')}",
    f"C2HLS_ONE_SHOT={_flag('one_shot', 'C2HLS_ONE_SHOT')}",
    f"C2HLS_SKIP_FLASH={_flag('skip_flash', 'C2HLS_SKIP_FLASH')}",
    f"C2HLS_PP_LOAD_B_IN_DF={_flag('pp_load_b_in_dataflow', 'C2HLS_PP_LOAD_B_IN_DF')}",
]
seed_dir = str(doc.get("flash_seed_dir") or os.getenv("C2HLS_FLASH_SEED_DIR") or "").strip()
if seed_dir:
    bits.append(f"C2HLS_FLASH_SEED_DIR={seed_dir}")
mode = str(doc.get("flash_opt_prompt_mode") or os.getenv("C2HLS_FLASH_OPT_PROMPT_MODE") or "").strip()
if mode:
    bits.append(f"C2HLS_FLASH_OPT_PROMPT_MODE={mode}")
turns = doc.get("turns")
if turns is None:
    turns = os.getenv("C2HLS_TURNS") or ""
if str(turns).strip():
    bits.append(f"C2HLS_TURNS={str(turns).strip()}")
if to:
    bits.append(f"C2HLS_SYNTH_TIMEOUT={to}")
pe = str(doc.get("pe_recipe") or os.getenv("C2HLS_PE_RECIPE") or "").strip()
if pe:
    bits.append(f"C2HLS_PE_RECIPE={pe}")
flash_min = str(doc.get("flash_min_dsp") or os.getenv("C2HLS_FLASH_MIN_DSP") or "").strip()
if flash_min.isdigit():
    bits.append(f"C2HLS_FLASH_MIN_DSP={flash_min}")
flash_max = str(doc.get("flash_max_dsp") or os.getenv("C2HLS_FLASH_MAX_DSP") or "").strip()
if flash_max.isdigit():
    bits.append(f"C2HLS_FLASH_MAX_DSP={flash_max}")
flash_redo = str(doc.get("flash_dsp_redo") or os.getenv("C2HLS_FLASH_DSP_REDO") or "").strip().lower()
if flash_redo in {"1", "true", "yes", "on"}:
    bits.append("C2HLS_FLASH_DSP_REDO=1")
flash_fill = str(doc.get("flash_dsp_fill_pct") or os.getenv("C2HLS_FLASH_DSP_FILL_PCT") or "").strip()
if flash_fill.isdigit():
    bits.append(f"C2HLS_FLASH_DSP_FILL_PCT={flash_fill}")
cands = str(doc.get("candidates_per_step") or os.getenv("C2HLS_CANDIDATES_PER_STEP") or "").strip()
if cands:
    bits.append(f"C2HLS_CANDIDATES_PER_STEP={cands}")
flash_kt = str(doc.get("flash_k_tile") or os.getenv("C2HLS_FLASH_K_TILE") or "").strip()
if flash_kt.isdigit():
    bits.append(f"C2HLS_FLASH_K_TILE={flash_kt}")
flash_ot = str(doc.get("flash_onchip_tile") or os.getenv("C2HLS_FLASH_ONCHIP_TILE") or "").strip().lower()
if flash_ot in {"1", "true", "yes", "on"}:
    bits.append("C2HLS_FLASH_ONCHIP_TILE=1")
ref_only = str(doc.get("reference_only") or os.getenv("C2HLS_REFERENCE_ONLY") or "").strip().lower()
if ref_only in {"1", "true", "yes", "on"}:
    bits.append("C2HLS_REFERENCE_ONLY=1")
dse_pack = str(doc.get("dse_skill_entries_json") or os.getenv("C2HLS_DSE_SKILL_ENTRIES_JSON") or "").strip()
if dse_pack:
    bits.append(f"C2HLS_DSE_SKILL_ENTRIES_JSON={dse_pack}")
dse_v2_grid = str(doc.get("dse_v2_grid") or os.getenv("C2HLS_DSE_V2_GRID") or "").strip()
if dse_v2_grid:
    bits.append(f"C2HLS_DSE_V2_GRID={dse_v2_grid}")
stream_pack = str(doc.get("stream_skill_entries_json") or os.getenv("C2HLS_STREAM_SKILL_ENTRIES_JSON") or "").strip()
if stream_pack:
    bits.append(f"C2HLS_STREAM_SKILL_ENTRIES_JSON={stream_pack}")
dse_min = str(doc.get("dse_min_dsp") or os.getenv("C2HLS_DSE_MIN_DSP") or "").strip()
if dse_min.lstrip("-").isdigit():
    bits.append(f"C2HLS_DSE_MIN_DSP={dse_min}")
flash_row = str(doc.get("flash_row_uf") or os.getenv("C2HLS_FLASH_ROW_UF") or "").strip()
if flash_row.isdigit():
    bits.append(f"C2HLS_FLASH_ROW_UF={flash_row}")
flash_pe = str(doc.get("flash_pe_blk") or os.getenv("C2HLS_FLASH_PE_BLK") or "").strip()
if flash_pe.isdigit():
    bits.append(f"C2HLS_FLASH_PE_BLK={flash_pe}")
flash_pp = str(doc.get("flash_tile_pp") or os.getenv("C2HLS_FLASH_TILE_PP") or "").strip().lower()
if flash_pp in {"1", "true", "yes", "on"}:
    bits.append("C2HLS_FLASH_TILE_PP=1")
flash_oc = str(doc.get("flash_onchip") or os.getenv("C2HLS_FLASH_ONCHIP") or "").strip().lower()
if flash_oc in {"1", "true", "yes", "on"}:
    bits.append("C2HLS_FLASH_ONCHIP=1")
skill_bin = str(doc.get("flash_skill_bin") or os.getenv("C2HLS_FLASH_SKILL_BIN") or "").strip()
if skill_bin:
    bits.append(f"C2HLS_FLASH_SKILL_BIN={skill_bin}")
pack = str(doc.get("packaged_skills_json") or os.getenv("C2HLS_PACKAGED_SKILLS_JSON") or "").strip()
if pack:
    bits.append(f"C2HLS_PACKAGED_SKILLS_JSON={pack}")
order = str(doc.get("skill_prompt_order_json") or os.getenv("C2HLS_SKILL_PROMPT_ORDER_JSON") or "").strip()
if order:
    bits.append(f"C2HLS_SKILL_PROMPT_ORDER_JSON={order}")
only = str(doc.get("packaged_skills_only") or os.getenv("C2HLS_PACKAGED_SKILLS_ONLY") or "").strip().lower()
if only in {"1", "true", "yes", "on"}:
    bits.append("C2HLS_PACKAGED_SKILLS_ONLY=1")
flash_tok = str(doc.get("flash_max_tokens") or os.getenv("C2HLS_FLASH_MAX_TOKENS") or "").strip()
if flash_tok.isdigit():
    bits.append(f"C2HLS_FLASH_MAX_TOKENS={flash_tok}")
llm_tok = str(doc.get("llm_max_tokens") or os.getenv("C2HLS_LLM_MAX_TOKENS") or "").strip()
if llm_tok.isdigit():
    bits.append(f"C2HLS_LLM_MAX_TOKENS={llm_tok}")
cont = str(doc.get("cpp_continuations") or os.getenv("C2HLS_CPP_CONTINUATIONS") or "").strip()
if cont.isdigit():
    bits.append(f"C2HLS_CPP_CONTINUATIONS={cont}")
thinking = str(doc.get("thinking") or os.getenv("C2HLS_THINKING") or "").strip()
if thinking:
    bits.append(f"C2HLS_THINKING={thinking}")
print(",".join(bits))
PY
)"
  job_id="$(
    sbatch --parsable \
      --chdir="${C2HLS_ROOT}" \
      --export="ALL,BATCH_PARALLEL_CAMPAIGN_ROOT=${CAMPAIGN_ROOT},BATCH_PARALLEL_VARIANT=${VARIANT},BATCH_PARALLEL_NODE_INDEX=${node_index},BATCH_PARALLEL_CONFIG=${CONFIG},${enf_export}" \
      --job-name="${job_prefix}-${role}-n${node_index}-${job_tag}" \
      --partition="${PC2_COMPUTE_PARTITION}" \
      --cpus-per-task="${cpus}" \
      --mem="${mem}G" \
      --time="${PC2_WALLTIME:-12:00:00}" \
      "${template}"
  )"
  job_id="${job_id%%;*}"
  _register_compute_job "${role}" "${node_index}" "${job_id}"
  echo "${job_id}"
}

for i in $(seq 0 $((SYNTH_NODES - 1))); do
  submit_node synth "${i}" "${SYNTH_WPN}"
done
if [[ "${COSIM_NODES}" -gt 0 ]]; then
  for i in $(seq 0 $((COSIM_NODES - 1))); do
    submit_node cosim "${i}" "${COSIM_WPN}"
  done
else
  echo "cosim: 0 nodes (skipping cosim submit; combined-HLS or cosim disabled)"
fi
