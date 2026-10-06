#!/usr/bin/env bash
# Parallel HLSFactory post-flash dataflow: one Slurm job per bench (max parallelism).
#
# Each bench gets its own Slurm job (16c/64G by default) and, by default, its own
# DeepSeek queue proxy on the login node (one port per bench) so LLM calls are not
# serialized behind a shared workers=1 proxy.
#
# Usage:
#   ./scripts/pc2/start_hlsfactory_parallel_dataflow.sh \
#       --campaign-root artifacts/pc2/batch_parallel_hlsfactory_ds_v4f_noskills_STAMP \
#       --benches hlsfactory_gemm,hlsfactory_gemver,...
#   ./scripts/pc2/start_hlsfactory_parallel_dataflow.sh --campaign-root ... --remaining
#   ./scripts/pc2/start_hlsfactory_parallel_dataflow.sh --campaign-root ... --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

CAMPAIGN_ROOT=""
BENCHES=""
REMAINING=0
DRY_RUN=0
CANCEL_POST_JOB=""
APPEND=0
SKIP_EXPORT=0
WALLTIME="${PC2_FORCE_WALLTIME:-72:00:00}"
WORKER_CPUS="${C2HLS_DATAFLOW_WORKER_CPUS:-16}"
WORKER_MEM_GB="${C2HLS_DATAFLOW_WORKER_MEM_GB:-64}"
EXCLUSIVE="${C2HLS_DATAFLOW_EXCLUSIVE:-0}"
JOB_PREFIX="${PC2_BATCH_JOB_PREFIX:-bphfpdf}"
PER_BENCH_PROXY="${C2HLS_PER_BENCH_PROXY:-1}"
PROXY_BASE_PORT="${C2HLS_PER_BENCH_PROXY_BASE_PORT:-18200}"
PROXY_BACKEND="${C2HLS_PROXY_BACKEND:-deepseek}"  # deepseek | anthropic
ANTHROPIC_PROXY_WORKERS="${C2HLS_ANTHROPIC_QUEUE_WORKERS:-16}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign-root) shift; CAMPAIGN_ROOT="$1"; shift ;;
    --benches) shift; BENCHES="$1"; shift ;;
    --remaining) REMAINING=1; shift ;;
    --cancel-post-job) shift; CANCEL_POST_JOB="$1"; shift ;;
    --no-cancel-post) CANCEL_POST_JOB=""; shift ;;
    --append) APPEND=1; shift ;;
    --skip-export) SKIP_EXPORT=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --walltime) shift; WALLTIME="$1"; shift ;;
    --cpus) shift; WORKER_CPUS="$1"; shift ;;
    --mem-gb) shift; WORKER_MEM_GB="$1"; shift ;;
    --no-exclusive) EXCLUSIVE=0; shift ;;
    --exclusive) EXCLUSIVE=1; shift ;;
    --per-bench-proxy) PER_BENCH_PROXY=1; shift ;;
    --shared-proxy) PER_BENCH_PROXY=0; shift ;;
    --proxy-base-port) shift; PROXY_BASE_PORT="$1"; shift ;;
    --proxy-backend) shift; PROXY_BACKEND="$1"; shift ;;
    --job-prefix) shift; JOB_PREFIX="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${CAMPAIGN_ROOT}" ]]; then
  echo "ERROR: --campaign-root required" >&2
  exit 2
fi
if [[ "${CAMPAIGN_ROOT}" != /* ]]; then
  CAMPAIGN_ROOT="${C2HLS_ROOT}/${CAMPAIGN_ROOT}"
fi
if [[ ! -d "${CAMPAIGN_ROOT}" ]]; then
  echo "ERROR: campaign root missing: ${CAMPAIGN_ROOT}" >&2
  exit 2
fi

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
[[ -x "${PY}" ]] || PY=python3

if [[ "${REMAINING}" -eq 1 || -z "${BENCHES}" ]]; then
  BENCHES="$("${PY}" - <<PY
import json, re
from pathlib import Path
root = Path("${CAMPAIGN_ROOT}")
# Prefer skills-complete summary sibling for canonical bench order, else matrix.
stamp = root.name.split("_")[-1] if False else None
log = root / "flow" / "post_flash_dataflow_watcher.log"
done = set()
if log.is_file():
    for line in log.read_text(errors="replace").splitlines():
        m = re.search(r"DONE (hlsfactory_\S+)", line)
        if m:
            done.add(m.group(1))
# Also treat existing per-bench success summaries under results if present
for p in root.glob("post_flash_dataflow_summary_*.json"):
    if "_meta_" in p.name:
        continue
    try:
        doc = json.loads(p.read_text())
    except Exception:
        continue
    if isinstance(doc, list):
        for row in doc:
            if isinstance(row, dict) and row.get("bench") and row.get("success") is True:
                done.add(row["bench"])
# Already-submitted streaming DF jobs
jobs = root / "flow" / "parallel_dataflow" / "jobs.jsonl"
if jobs.is_file():
    for line in jobs.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        b = row.get("bench")
        if b:
            done.add(b)
# Cells that already have a dataflow result
for p in root.glob("variants/*/hlsfactory_*/*/*_dataflow_result.json"):
    done.add(p.parent.parent.name)
# Matrix / cells
benches = []
mj = root / "matrix.json"
if mj.is_file():
    rows = json.loads(mj.read_text())
    if isinstance(rows, list):
        for r in rows:
            b = r.get("bench") if isinstance(r, dict) else None
            if b and b not in benches:
                benches.append(b)
if not benches:
    # fall back to variants/*/hlsfactory_* dirs
    for d in sorted((root / "variants").glob("*/*/")):
        name = d.name
        if name.startswith("hlsfactory_"):
            # cell dir is model__...; parent is bench
            pass
    for bench_dir in sorted((root / "variants").glob("*/*")):
        if bench_dir.is_dir() and bench_dir.name.startswith("hlsfactory_"):
            if bench_dir.name not in benches:
                benches.append(bench_dir.name)
remaining = [b for b in benches if b not in done]
print(",".join(remaining))
PY
)"
fi

IFS=',' read -r -a BENCH_ARR <<< "${BENCHES}"
BENCH_ARR=("${BENCH_ARR[@]// /}")
# drop empties
_tmp=()
for b in "${BENCH_ARR[@]}"; do
  [[ -n "${b}" ]] && _tmp+=("${b}")
done
BENCH_ARR=("${_tmp[@]}")

FLOW_DIR="${CAMPAIGN_ROOT}/flow"
DF_DIR="${FLOW_DIR}/parallel_dataflow"
mkdir -p "${DF_DIR}/logs" "${FLOW_DIR}"
JOB_LIST="${DF_DIR}/jobs.jsonl"

# Append mode: skip benches already recorded in jobs.jsonl (per-bench streaming).
# NOTE: pass wanted benches via env — do NOT nest $(...) inside an unquoted heredoc
# (shell would expand a JSON array into Python as a list, then json.loads(list) TypeErrors).
if [[ "${APPEND}" -eq 1 && -f "${JOB_LIST}" ]]; then
  WANTED_JSON="$(printf '%s\n' "${BENCH_ARR[@]}" | "${PY}" -c 'import json,sys; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))')"
  FILTERED="$(
    WANTED_JSON="${WANTED_JSON}" JOB_LIST="${JOB_LIST}" "${PY}" - <<'PY'
import json
import os
from pathlib import Path

already = set()
p = Path(os.environ["JOB_LIST"])
for line in p.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        row = json.loads(line)
    except Exception:
        continue
    if not isinstance(row, dict):
        continue
    b = row.get("bench")
    if b:
        already.add(b)
wanted = json.loads(os.environ["WANTED_JSON"])
keep = [b for b in wanted if b not in already]
print(",".join(keep))
PY
  )"
  if [[ -z "${FILTERED}" ]]; then
    echo "=== HLSFactory parallel dataflow ==="
    echo "campaign=${CAMPAIGN_ROOT}"
    echo "append: all requested benches already submitted; nothing to do"
    exit 0
  fi
  IFS=',' read -r -a BENCH_ARR <<< "${FILTERED}"
fi

if [[ "${#BENCH_ARR[@]}" -eq 0 ]]; then
  echo "ERROR: no benches to submit (BENCHES empty / remaining empty)" >&2
  exit 3
fi

echo "=== HLSFactory parallel dataflow ==="
echo "campaign=${CAMPAIGN_ROOT}"
echo "benches=${#BENCH_ARR[@]}: $(IFS=,; echo "${BENCH_ARR[*]}")"
echo "walltime=${WALLTIME} worker=${WORKER_CPUS}c/${WORKER_MEM_GB}G exclusive=${EXCLUSIVE}"
echo "per_bench_proxy=${PER_BENCH_PROXY} proxy_backend=${PROXY_BACKEND} proxy_base_port=${PROXY_BASE_PORT}"
echo "cancel_post=${CANCEL_POST_JOB:-<none>} append=${APPEND} skip_export=${SKIP_EXPORT}"
echo "dry_run=${DRY_RUN}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "(dry-run) would cancel post ${CANCEL_POST_JOB:-none}, start per-bench proxies=${PER_BENCH_PROXY} backend=${PROXY_BACKEND}, submit ${#BENCH_ARR[@]} df jobs + export"
  exit 0
fi

if [[ -n "${CANCEL_POST_JOB}" ]]; then
  echo "[1/4] scancel serial post ${CANCEL_POST_JOB}"
  scancel "${CANCEL_POST_JOB}" || true
else
  echo "[1/4] skip post cancel"
fi

PROXY_MAP=""
SHARED_EP_URL=""
# openai/xai (Luna/Grok) only support shared campaign proxies; never fall through to DeepSeek per-bench.
if [[ "${PROXY_BACKEND}" == "openai" || "${PROXY_BACKEND}" == "xai" || "${PROXY_BACKEND}" == "grok" ]]; then
  if [[ "${PER_BENCH_PROXY}" == "1" ]]; then
    echo "WARNING: PROXY_BACKEND=${PROXY_BACKEND} does not support per-bench proxies; forcing shared campaign proxy" >&2
  fi
  PER_BENCH_PROXY=0
fi
if [[ "${PER_BENCH_PROXY}" == "1" ]]; then
  PROXY_ROOT="${DF_DIR}/proxies"
  BENCH_CSV="$(IFS=,; echo "${BENCH_ARR[*]}")"
  if [[ "${PROXY_BACKEND}" == "anthropic" ]]; then
    echo "[2/4] start one Anthropic proxy per bench (base_port=${PROXY_BASE_PORT} workers=${ANTHROPIC_PROXY_WORKERS})"
    ANTHROPIC_PROXY_MODEL="${C2HLS_MODEL:-claude-sonnet-5}" \
    C2HLS_ANTHROPIC_QUEUE_WORKERS="${ANTHROPIC_PROXY_WORKERS}" \
    bash "${SCRIPT_DIR}/start_hlsfactory_per_bench_anthropic_proxies.sh" \
      --proxy-root "${PROXY_ROOT}" \
      --benches "${BENCH_CSV}" \
      --base-port "${PROXY_BASE_PORT}" \
      --login-host "${CHATHLS_LOGIN_HOST:-login5}" \
      --workers "${ANTHROPIC_PROXY_WORKERS}" \
      --model "${C2HLS_MODEL:-claude-sonnet-5}"
  elif [[ "${PROXY_BACKEND}" == "deepseek" ]]; then
    echo "[2/4] start one DeepSeek proxy per bench (base_port=${PROXY_BASE_PORT})"
    DEEPSEEK_PROXY_MODEL=deepseek-v4-flash \
    bash "${SCRIPT_DIR}/start_hlsfactory_per_bench_proxies.sh" \
      --proxy-root "${PROXY_ROOT}" \
      --benches "${BENCH_CSV}" \
      --base-port "${PROXY_BASE_PORT}" \
      --login-host "${CHATHLS_LOGIN_HOST:-login5}"
  else
    echo "ERROR: unsupported PROXY_BACKEND=${PROXY_BACKEND} for per-bench proxies" >&2
    exit 4
  fi
  PROXY_MAP="${PROXY_ROOT}/port_map.json"
  if [[ ! -f "${PROXY_MAP}" ]]; then
    echo "ERROR: missing ${PROXY_MAP}" >&2
    exit 4
  fi
else
  echo "[2/4] shared campaign proxy"
  if [[ ! -f "${CAMPAIGN_ROOT}/llm_endpoint.json" ]]; then
    echo "ERROR: missing ${CAMPAIGN_ROOT}/llm_endpoint.json" >&2
    exit 4
  fi
  SHARED_EP_URL="$("${PY}" -c "import json;print(json.load(open('${CAMPAIGN_ROOT}/llm_endpoint.json'))['url'])")"
  if ! curl -sf --max-time 10 "${SHARED_EP_URL}/models" >/dev/null; then
    echo "ERROR: endpoint not reachable: ${SHARED_EP_URL}" >&2
    exit 4
  fi
fi

EXCL_ARGS=()
if [[ "${EXCLUSIVE}" == "1" ]]; then
  EXCL_ARGS=(--exclusive)
fi

echo "[3/4] submit ${#BENCH_ARR[@]} dataflow jobs (prefix=${JOB_PREFIX})"
DF_JOBS=()
JOB_LIST="${DF_DIR}/jobs.jsonl"
if [[ "${APPEND}" -eq 1 ]]; then
  touch "${JOB_LIST}"
else
  : > "${JOB_LIST}"
fi
# Honor caller RUN_COSIM (lat/RAG2 waiter sets 0 + ranked cosim after dataflow).
DF_RUN_COSIM="${C2HLS_RUN_COSIM:-1}"
DF_REF_COSIM="${C2HLS_REFERENCE_COSIM:-${DF_RUN_COSIM}}"
# Avoid "C2HLS_RAG_MODE set but C2HLS_RAG is not enabled" when RAG is off.
if [[ "${C2HLS_RAG2:-0}" == "1" || "${C2HLS_RAG:-0}" == "1" || "${C2HLS_RAG_ENABLE:-0}" == "1" ]]; then
  RAG_EXPORT="C2HLS_RAG=${C2HLS_RAG:-0},C2HLS_RAG_ENABLE=${C2HLS_RAG_ENABLE:-0},C2HLS_RAG2=${C2HLS_RAG2:-0},C2HLS_RAG_MODE=${C2HLS_RAG_MODE:-everywhere},C2HLS_RAG2_OPT_CORPUS=${C2HLS_RAG2_OPT_CORPUS:-},C2HLS_RAG2_REPAIR_CORPUS=${C2HLS_RAG2_REPAIR_CORPUS:-}"
else
  RAG_EXPORT="C2HLS_RAG=0,C2HLS_RAG_ENABLE=0,C2HLS_RAG_SCRAPE=0,C2HLS_RAG2=0,C2HLS_RAG_MODE=,C2HLS_RAG2_OPT_CORPUS=,C2HLS_RAG2_REPAIR_CORPUS="
fi
# Propagate completion-token caps into DF workers (Luna/Grok need >>8192; missing env
# previously defaulted DF LLM calls to 8192 and produced empty finish_reason=length replies).
EXPORT_BASE="ALL,BATCH_PARALLEL_CAMPAIGN_ROOT=${CAMPAIGN_ROOT},C2HLS_RUN_COSIM=${DF_RUN_COSIM},C2HLS_REFERENCE_COSIM=${DF_REF_COSIM},C2HLS_COSIM_REQUIRED=0,C2HLS_COSIM_XELAB_MT_OFF=1,C2HLS_FLASH_MAX_TOKENS=${C2HLS_FLASH_MAX_TOKENS:-65536},C2HLS_LLM_MAX_TOKENS=${C2HLS_LLM_MAX_TOKENS:-${C2HLS_FLASH_MAX_TOKENS:-65536}},C2HLS_REASONING_EFFORT=${C2HLS_REASONING_EFFORT:-},C2HLS_OPENAI_UPSTREAM=${C2HLS_OPENAI_UPSTREAM:-},C2HLS_XAI_HOSTED_URL=${C2HLS_XAI_HOSTED_URL:-},Grok_API=${Grok_API:-${OPENAI_API_KEY:-}},XAI_API_KEY=${XAI_API_KEY:-${OPENAI_API_KEY:-}},C2HLS_DATAFLOW_REPAIR_ROUNDS=${C2HLS_DATAFLOW_REPAIR_ROUNDS:-4},C2HLS_DATAFLOW_CONTRACT_ROUNDS=${C2HLS_DATAFLOW_CONTRACT_ROUNDS:-4},C2HLS_POST_FLASH_RESULTS_SUFFIX=${C2HLS_POST_FLASH_RESULTS_SUFFIX:-hlsfactory_cosim_repairs},C2HLS_POST_FLASH_MATRIX_ROOT=${CAMPAIGN_ROOT},C2HLS_POST_FLASH_PROMPT_POLICY=${C2HLS_POST_FLASH_PROMPT_POLICY:-system_skills},C2HLS_TMP_RUN=$(basename "${CAMPAIGN_ROOT}"),C2HLS_DATAFLOW_NO_SKILLS=${C2HLS_DATAFLOW_NO_SKILLS:-0},C2HLS_BARE_OPT_PROMPTS=${C2HLS_BARE_OPT_PROMPTS:-0},C2HLS_PACKAGED_SKILLS_JSON=${C2HLS_PACKAGED_SKILLS_JSON:-},C2HLS_PACKAGED_SKILLS_ONLY=${C2HLS_PACKAGED_SKILLS_ONLY:-0},C2HLS_FORCE_SKILL_PROMPTS=${C2HLS_FORCE_SKILL_PROMPTS:-0},C2HLS_SKILL_PROMPT_MODE=${C2HLS_SKILL_PROMPT_MODE:-},C2HLS_POST_FLASH_LATENCY_OPT=${C2HLS_POST_FLASH_LATENCY_OPT:-0},C2HLS_LATENCY_OPT_CHAIN_FLASH=${C2HLS_LATENCY_OPT_CHAIN_FLASH:-0},C2HLS_LATENCY_OPT_CHAIN_DATAFLOW=${C2HLS_LATENCY_OPT_CHAIN_DATAFLOW:-0},C2HLS_LATENCY_OPT_ROUNDS=${C2HLS_LATENCY_OPT_ROUNDS:-3},C2HLS_LATENCY_OPT_REPAIR_ROUNDS=${C2HLS_LATENCY_OPT_REPAIR_ROUNDS:-3},C2HLS_RANKED_COSIM_AFTER_DATAFLOW=${C2HLS_RANKED_COSIM_AFTER_DATAFLOW:-0},${RAG_EXPORT},C2HLS_MODEL=${C2HLS_MODEL:-deepseek-v4-flash},OPENAI_API_KEY=${OPENAI_API_KEY:-},CHATHLS_API_KEY=${CHATHLS_API_KEY:-${OPENAI_API_KEY:-}},ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}"

for bench in "${BENCH_ARR[@]}"; do
  short="${bench#hlsfactory_}"
  log_out="${DF_DIR}/logs/${bench}.%j.out"
  log_err="${DF_DIR}/logs/${bench}.%j.err"
  if [[ -n "${PROXY_MAP}" ]]; then
    BENCH_URL="$("${PY}" -c "import json;print(json.load(open('${PROXY_MAP}'))['${bench}']['url'])")"
  else
    BENCH_URL="${SHARED_EP_URL}"
  fi
  EXPORT_ENV="${EXPORT_BASE},OPENAI_BASE_URL=${BENCH_URL},CHATHLS_API_BASE=${BENCH_URL},BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL=${BENCH_URL}"
  job_id="$(
    sbatch --parsable \
      --chdir="${C2HLS_ROOT}" \
      --job-name="${JOB_PREFIX}-df-${short}" \
      --output="${log_out}" \
      --error="${log_err}" \
      --account="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}" \
      --partition="${PC2_COMPUTE_PARTITION}" \
      --cpus-per-task="${WORKER_CPUS}" \
      --mem="${WORKER_MEM_GB}G" \
      --time="${WALLTIME}" \
      "${EXCL_ARGS[@]}" \
      --export="${EXPORT_ENV}" \
      --wrap="bash ${SCRIPT_DIR}/run_hlsfactory_dataflow_bench.sh ${bench}"
  )"
  job_id="${job_id%%;*}"
  DF_JOBS+=("${job_id}")
  echo "{\"bench\":\"${bench}\",\"job_id\":\"${job_id}\",\"openai_base_url\":\"${BENCH_URL}\"}" >> "${JOB_LIST}"
  echo "  ${bench} -> ${job_id}  ${BENCH_URL}"
done

# Collect all DF job ids (prior append + new) for export dependency.
ALL_DF_JOBS="$("${PY}" - <<PY
import json
from pathlib import Path
ids=[]
for line in Path("${JOB_LIST}").read_text(encoding="utf-8").splitlines():
    line=line.strip()
    if not line: continue
    try:
        row=json.loads(line)
    except Exception:
        continue
    j=row.get("job_id")
    if j: ids.append(str(j))
print(",".join(ids))
PY
)"
dep_csv="${ALL_DF_JOBS}"

EXPORT_JOB=""
if [[ "${SKIP_EXPORT}" -eq 1 ]]; then
  echo "[4/4] skip export (streaming mid-campaign; final export later)"
else
  echo "[4/4] submit export job after dataflow jobs: ${dep_csv}"
  FLASH_BUNDLE="${C2HLS_ROOT}/artifacts/pc2/flash_selected_bundle/$(basename "${CAMPAIGN_ROOT}")"
  DATAFLOW_BUNDLE="${C2HLS_ROOT}/artifacts/pc2/dataflow_selected_bundle/$(basename "${CAMPAIGN_ROOT}")"
  EXPORT_JOB="$(
    sbatch --parsable \
      --chdir="${C2HLS_ROOT}" \
      --job-name="${JOB_PREFIX}-df-export" \
      --output="${DF_DIR}/export-%j.out" \
      --error="${DF_DIR}/export-%j.err" \
      --account="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}" \
      --partition="${PC2_COMPUTE_PARTITION}" \
      --cpus-per-task=2 \
      --mem=8G \
      --time=4:00:00 \
      --dependency="afterany:${dep_csv}" \
      --export=ALL \
      --wrap="mkdir -p '${DATAFLOW_BUNDLE}' && ${PY} '${C2HLS_ROOT}/scripts/pc2/export_post_flash_dataflow_csynth_bundle.py' --matrix-root '${CAMPAIGN_ROOT}' --flash-bundle-root '${FLASH_BUNDLE}' --kernel-bundle '${DATAFLOW_BUNDLE}' --force || true; ln -sfn '${DATAFLOW_BUNDLE}' '${CAMPAIGN_ROOT}/dataflow_selected'; echo exported > '${DF_DIR}/export.done'"
  )"
  EXPORT_JOB="${EXPORT_JOB%%;*}"
  echo "  export_job=${EXPORT_JOB}"
fi

"${PY}" - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
prev = doc.get("dataflow_parallel") if isinstance(doc.get("dataflow_parallel"), dict) else {}
benches = []
job_ids = []
for line in Path("${JOB_LIST}").read_text(encoding="utf-8").splitlines():
    line=line.strip()
    if not line: continue
    try:
        row=json.loads(line)
    except Exception:
        continue
    b=row.get("bench"); j=row.get("job_id")
    if b and b not in benches: benches.append(b)
    if j: job_ids.append(str(j))
export_job = "${EXPORT_JOB}".strip() or prev.get("export_job_id")
doc["dataflow_parallel"] = {
    "scheme": "one_slurm_job_per_bench",
    "submitted_at": datetime.now(timezone.utc).isoformat(),
    "export_job_id": export_job,
    "dataflow_job_ids": job_ids,
    "benches": benches,
    "worker_cpus": int("${WORKER_CPUS}"),
    "exclusive": bool(int("${EXCLUSIVE}")),
    "per_bench_proxy": bool(int("${PER_BENCH_PROXY}")),
    "proxy_map": "${PROXY_MAP}",
    "append": bool(int("${APPEND}")),
    "skip_export": bool(int("${SKIP_EXPORT}")),
}
p.write_text(json.dumps(doc, indent=2) + "\n")
Path("${DF_DIR}/launch.json").write_text(json.dumps(doc["dataflow_parallel"], indent=2) + "\n")
PY

echo
echo "dataflow_jobs=${dep_csv}"
echo "export=${EXPORT_JOB:-<skipped>}"
echo "launch=${DF_DIR}/launch.json"
echo "squeue: squeue -u \$USER | rg '${JOB_PREFIX}'"
