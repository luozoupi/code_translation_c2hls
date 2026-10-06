#!/usr/bin/env bash
# Start one DSE v4 config the way DSE v3 started one PE×SIMD job.
# On the login node: one dedicated DeepSeek proxy, health check, then
# sbatch --export=ALL with OPENAI_BASE_URL set to that proxy.
# The model is deepseek-v4-flash. The public DeepSeek URL is not used.
#
# Usage:
#   ./scripts/pc2/start_dse_v4_one.sh --config st0_c1 --out-dir DIR
#   ./scripts/pc2/start_dse_v4_one.sh --config st0_c1 --out-dir DIR --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

CONFIG_ID=""
OUT_DIR=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) shift; CONFIG_ID="$1"; shift ;;
    --out-dir) shift; OUT_DIR="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${CONFIG_ID}" || -z "${OUT_DIR}" ]]; then
  echo "ERROR: --config and --out-dir are required" >&2
  exit 2
fi

case "${OUT_DIR}" in
  *autosa_mm_variant_sweep_20260918*|*20260919*)
    echo "ERROR: refusing to write into ${OUT_DIR}" >&2
    exit 2
    ;;
esac

PY="${C2HLS_PYTHON:-python3}"
"${PY}" - "${CONFIG_ID}" <<'PY'
import sys
import post_flash_dse_v4 as v4
v4.load_v4_config(sys.argv[1])
PY

echo "config=${CONFIG_ID}"
echo "model=deepseek-v4-flash"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  bash "${SCRIPT_DIR}/write_dse_v4_sbatch.sh" --out-dir "${OUT_DIR}" --config "${CONFIG_ID}"
  echo "proxy=no"
  echo "submit=no"
  exit 0
fi

# Noctua logins are login*. Otus frontends are fe*.lerna / otus*.
case "$(hostname -s)" in
  login*|fe*|lerna*|otus*) ;;
  *)
    echo "ERROR: dedicated proxy must start on a login node, host=$(hostname -s)" >&2
    exit 1
    ;;
esac
if [[ "$(whoami)" != "haqc2" ]]; then
  echo "ERROR: unexpected user $(whoami)" >&2
  exit 1
fi
if squeue -u haqc2 -n "${CONFIG_ID}" -h | grep -q .; then
  echo "ERROR: job name ${CONFIG_ID} is already queued" >&2
  exit 1
fi

if [[ "${OPENAI_API_KEY:-}" == "EMPTY" || "${OPENAI_API_KEY:-}" == "empty" ]]; then
  unset OPENAI_API_KEY || true
fi
if [[ "${CHATHLS_API_KEY:-}" == "EMPTY" || "${CHATHLS_API_KEY:-}" == "empty" ]]; then
  unset CHATHLS_API_KEY || true
fi
# shellcheck disable=SC1091
source /scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/scripts/pc2/setup_deepseek_api.sh
"${PY}" - <<'PY'
import os, sys
value = os.environ.get("OPENAI_API_KEY") or ""
if len(value) < 20 or value.lower() == "empty":
    print("API key missing or too short; not submitting", file=sys.stderr)
    sys.exit(1)
print(f"api_key_loaded len={len(value)}")
PY

# setup_deepseek_api.sh points OPENAI_BASE_URL at the public API when it is
# unset. The compute node cannot use that. Replace it with this job's proxy.
export DEEPSEEK_PROXY_MODEL=deepseek-v4-flash
export CHATHLS_DEEPSEEK_QUEUE_WORKERS=1
# Longer than one csynth plus one csim, so the proxy stays up while Vitis runs.
export CHATHLS_DEEPSEEK_IDLE_EXIT_S=259200
unset C2HLS_VITIS_JOBS || true
unset C2HLS_VITIS_JOBS_FROM_SLURM || true
unset C2HLS_DSE_V3 || true
unset C2HLS_DSE_V3_HARNESS || true
unset C2HLS_DSE_SOURCE_KERNEL || true

proxy_dir="${OUT_DIR}/.llm"
url_file="${OUT_DIR}/proxy_url.txt"
err_file="${OUT_DIR}/proxy_start.err"
mkdir -p "${proxy_dir}"
if ! timeout 180 setsid -w nohup "${SCRIPT_DIR}/start_dedicated_deepseek_proxy.sh" "${proxy_dir}" >"${url_file}" 2>"${err_file}" < /dev/null; then
  echo "ERROR: proxy start failed for ${CONFIG_ID}" >&2
  exit 1
fi

url="$("${PY}" - "${url_file}" <<'PY'
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(errors="replace")
match = re.search(r"https?://[A-Za-z0-9._-]+:\d+/v1", text)
if not match:
    raise SystemExit(2)
url = match.group(0)
if "api.deepseek.com" in url:
    raise SystemExit(3)
print(url)
PY
)" || {
  echo "ERROR: proxy did not print a login-node URL" >&2
  exit 1
}

base="${url%/v1}"
hostport="${url#http://}"
hostport="${hostport%/v1}"
if ! curl -sf --max-time 15 "${base}/health" >/dev/null; then
  echo "ERROR: proxy health failed at ${hostport}" >&2
  exit 1
fi
"${PY}" - "${proxy_dir}/llm_endpoint.json" <<'PY'
import json, sys
from pathlib import Path
doc = json.loads(Path(sys.argv[1]).read_text())
if doc.get("workers") != 1 or doc.get("model") != "deepseek-v4-flash":
    raise SystemExit(f"workers={doc.get('workers')} model={doc.get('model')}")
PY

export OPENAI_BASE_URL="${url}"
export CHATHLS_API_BASE="${url}"
export C2HLS_MODEL=deepseek-v4-flash
export C2HLS_DSE_MODEL=deepseek-v4-flash

bash "${SCRIPT_DIR}/write_dse_v4_sbatch.sh" \
  --out-dir "${OUT_DIR}" \
  --config "${CONFIG_ID}" \
  --endpoint-url "${url}"

job_file="${OUT_DIR}/${CONFIG_ID}.sbatch.sh"
submit_out="${OUT_DIR}/submit_stdout.txt"
if ! sbatch --export=ALL "${job_file}" >"${submit_out}" 2>"${OUT_DIR}/submit_stderr.txt"; then
  echo "ERROR: sbatch failed for ${CONFIG_ID}" >&2
  exit 1
fi
jobid="$(awk '/Submitted batch job/{print $4}' "${submit_out}")"
if [[ -z "${jobid}" ]]; then
  echo "ERROR: sbatch produced no job id" >&2
  exit 1
fi
"${PY}" - "${OUT_DIR}" "${CONFIG_ID}" "${jobid}" "${hostport}" <<'PY'
import json, sys
from pathlib import Path
out, config_id, jobid, hostport = sys.argv[1:]
doc = {
    "schema": "dse_v4_one_launch_v1",
    "config_id": config_id,
    "job_id": jobid,
    "model": "deepseek-v4-flash",
    "proxy": hostport,
    "proxy_workers": 1,
    "submitted": "yes",
}
(Path(out) / "launch.json").write_text(json.dumps(doc, indent=2) + "\n")
PY
echo "proxy=${hostport}"
echo "submit=yes"
echo "job=${jobid}"
