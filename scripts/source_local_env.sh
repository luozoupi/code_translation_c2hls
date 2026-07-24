#!/bin/bash
# Source from other scripts: exports C2HLS_ROOT; loads local.env only on PC2.
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export C2HLS_ROOT="${C2HLS_ROOT:-$_REPO_ROOT}"
if [[ "${C2HLS_SITE:-team}" == "pc2" && -f "${C2HLS_ROOT}/local.env" ]]; then
  # local.env assigns plain "VAR=value" (no ${VAR:-...} guards), so re-sourcing
  # it in a nested script's own `source common.sh` (e.g. start_batch_parallel_
  # campaign.sh invoked as a subprocess) would silently clobber deliberate
  # caller overrides (e.g. C2HLS_MODEL=deepseek-chat, C2HLS_RUN_COSIM=1) back
  # to site defaults. Snapshot already-exported vars and restore them after
  # sourcing so local.env only fills in values the caller hasn't already set.
  _pc2_env_snapshot="$(mktemp)"
  for _pc2_var in $(compgen -e); do
    declare -p "${_pc2_var}" 2>/dev/null
  done > "${_pc2_env_snapshot}"
  set -a
  # shellcheck disable=SC1091
  source "${C2HLS_ROOT}/local.env"
  set +a
  # shellcheck disable=SC1091
  source "${_pc2_env_snapshot}"
  rm -f "${_pc2_env_snapshot}"
  unset _pc2_var _pc2_env_snapshot
fi
