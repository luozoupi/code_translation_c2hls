#!/bin/bash
# Source from other scripts: exports C2HLS_ROOT; loads local.env only on PC2.
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export C2HLS_ROOT="${C2HLS_ROOT:-$_REPO_ROOT}"
if [[ "${C2HLS_SITE:-team}" == "pc2" && -f "${C2HLS_ROOT}/local.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${C2HLS_ROOT}/local.env"
  set +a
fi
