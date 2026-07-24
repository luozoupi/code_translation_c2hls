#!/bin/bash
# Source from other scripts: exports C2HLS_ROOT; loads site-specific env files.
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export C2HLS_ROOT="${C2HLS_ROOT:-$_REPO_ROOT}"

_site="${C2HLS_SITE:-team}"
_env_file=""
case "${_site}" in
  pc2) _env_file="${C2HLS_ROOT}/local.env" ;;
  fir) _env_file="${C2HLS_ROOT}/fir.env" ;;
esac

if [[ -n "${_env_file}" && -f "${_env_file}" ]]; then
  # Site env files assign plain "VAR=value" (no ${VAR:-...} guards), so re-sourcing
  # in a nested script would clobber caller overrides. Snapshot exports and restore
  # after source so the file only fills unset values.
  _pc2_env_snapshot="$(mktemp)"
  for _pc2_var in $(compgen -e); do
    declare -p "${_pc2_var}" 2>/dev/null
  done > "${_pc2_env_snapshot}"
  set -a
  # shellcheck disable=SC1091
  source "${_env_file}"
  set +a
  # shellcheck disable=SC1091
  source "${_pc2_env_snapshot}"
  rm -f "${_pc2_env_snapshot}"
  unset _pc2_var _pc2_env_snapshot _site _env_file
fi
