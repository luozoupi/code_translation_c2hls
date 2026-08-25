#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_ROOT="${C2HLS_ARCHIVE_ROOT:-/mnt/data1/luo00466/c2hls_archive_20260811}"
OLD_REPO="/home/luo00466/code_translation-c2hls"
NEW_REPO="/home/luo00466/code_translation-c2hls-hpca2027"
MANIFEST="${ARCHIVE_ROOT}/manifest.jsonl"
TRANSFER_LOG="${ARCHIVE_ROOT}/transfer.log"
ZSTD_BIN="${ZSTD_BIN:-$(command -v zstd)}"

case "${ARCHIVE_ROOT}" in
  /mnt/data1/luo00466/*) ;;
  *)
    echo "Refusing archive root outside /mnt/data1/luo00466: ${ARCHIVE_ROOT}" >&2
    exit 2
    ;;
esac

entries=(
  "${OLD_REPO}/artifacts/vllm_cosim_compare|old_branch/artifacts/vllm_cosim_compare|completed Vitis COSIM work products"
  "${OLD_REPO}/artifacts/vllm_vitis_smoke|old_branch/artifacts/vllm_vitis_smoke|completed Vitis smoke work products"
  "${OLD_REPO}/artifacts/rl_corpus|old_branch/artifacts/rl_corpus|inactive historical SFT corpus"
  "${OLD_REPO}/results_sweeps|old_branch/results_sweeps|historical sweep result directories"
  "${NEW_REPO}/artifacts/vllm_vitis_smoke|hpca2027_branch/artifacts/vllm_vitis_smoke|completed pre-HPCA Vitis work products"
  "${NEW_REPO}/artifacts/storage|hpca2027_branch/artifacts/storage|completed storage-relocation logs and staging data"
)

count_entries() {
  find "$1" -printf '.' | wc -c
}

count_type() {
  find "$1" -type "$2" -printf '.' | wc -c
}

regular_bytes() {
  find "$1" -type f -printf '%s\n' \
    | awk '{total += $1} END {printf "%.0f", total + 0}'
}

disk_bytes() {
  du -s -B1 "$1" | awk '{print $1}'
}

mkdir -p "${ARCHIVE_ROOT}"
touch "${MANIFEST}" "${TRANSFER_LOG}"
exec > >(tee -a "${TRANSFER_LOG}") 2>&1

echo "archive_started=$(date -Is)"
echo "archive_root=${ARCHIVE_ROOT}"
echo "archive_format=tar.zst"

for entry in "${entries[@]}"; do
  IFS='|' read -r source relative reason <<<"${entry}"
  destination="${ARCHIVE_ROOT}/${relative}"
  payload="${destination}/payload.tar.zst"
  payload_partial="${payload}.partial"
  metadata="${destination}/metadata.json"
  restore_notes="${destination}/RESTORE.txt"
  source_parent="$(dirname "${source}")"
  source_name="$(basename "${source}")"
  backup="${source}.archive_move_20260811"

  if [[ -d "${source}" && -s "${payload}" && -f "${metadata}" ]] \
      && jq -e --arg source "${source}" \
        '.source == $source
         and .source_replaced_by_symlink == false
         and .source_retention.mode == "tracked_metadata_retained_locally"' \
        "${metadata}" >/dev/null; then
    retained_files_present=true
    while IFS= read -r retained_file; do
      if [[ ! -f "${source}/${retained_file}" ]]; then
        retained_files_present=false
        break
      fi
    done < <(jq -r '.source_retention.retained_files[]' "${metadata}")
    if [[ "${retained_files_present}" == true ]]; then
      echo "already_archived_with_tracked_files source=${source} destination=${destination}"
      continue
    fi
  fi

  if [[ -L "${source}" ]]; then
    current_target="$(readlink -f "${source}")"
    expected_target="$(readlink -f "${destination}")"
    if [[ "${current_target}" != "${expected_target}" || ! -s "${payload}" ]]; then
      echo "Unexpected or incomplete archive symlink: ${source} -> ${current_target}" >&2
      exit 3
    fi
    echo "already_archived source=${source} destination=${destination}"
    continue
  fi

  if [[ ! -d "${source}" ]]; then
    echo "Missing archive source: ${source}" >&2
    exit 4
  fi
  if [[ -e "${backup}" || -L "${backup}" ]]; then
    echo "Unexpected archive backup path already exists: ${backup}" >&2
    exit 5
  fi

  mkdir -p "${destination}"
  unexpected_destination_entry="$(find "${destination}" -mindepth 1 -maxdepth 1 \
    ! -name 'payload.tar.zst.partial' -print -quit)"
  if [[ -n "${unexpected_destination_entry}" ]]; then
    echo "Archive destination is not empty: ${destination}" >&2
    exit 6
  fi
  rm -f -- "${payload_partial}"

  source_entries="$(count_entries "${source}")"
  source_files="$(count_type "${source}" f)"
  source_dirs="$(count_type "${source}" d)"
  source_links="$(count_type "${source}" l)"
  source_bytes="$(regular_bytes "${source}")"
  source_disk_bytes="$(disk_bytes "${source}")"
  echo "bundle_start source=${source} destination=${destination} entries=${source_entries} bytes=${source_bytes}"

  tar --sparse --acls --xattrs -C "${source_parent}" -cf - "${source_name}" \
    | nice -n 10 "${ZSTD_BIN}" -q -T2 -3 -o "${payload_partial}"

  "${ZSTD_BIN}" -q -t "${payload_partial}"
  archive_entries="$(tar --use-compress-program="${ZSTD_BIN} -q" -tf "${payload_partial}" | wc -l)"
  if [[ "${archive_entries}" != "${source_entries}" ]]; then
    echo "Archive entry-count verification failed for ${source}: source=${source_entries} archive=${archive_entries}" >&2
    exit 7
  fi

  verification_log="$(mktemp /tmp/c2hls-archive-compare.XXXXXX)"
  if ! tar --sparse --acls --xattrs --use-compress-program="${ZSTD_BIN} -q" \
      -df "${payload_partial}" -C "${source_parent}" "${source_name}" \
      >"${verification_log}" 2>&1; then
    echo "Archive content verification failed for ${source}:" >&2
    cat "${verification_log}" >&2
    exit 8
  fi
  if [[ -s "${verification_log}" ]]; then
    echo "Archive content verification reported differences for ${source}:" >&2
    cat "${verification_log}" >&2
    exit 9
  fi
  rm -f "${verification_log}"

  mv "${payload_partial}" "${payload}"
  archive_sha256="$(sha256sum "${payload}" | awk '{print $1}')"
  archive_bytes="$(stat -c '%s' "${payload}")"

  jq -n \
    --arg archived_at "$(date -Is)" \
    --arg source "${source}" \
    --arg destination "${destination}" \
    --arg payload "${payload}" \
    --arg reason "${reason}" \
    --arg archive_sha256 "${archive_sha256}" \
    --argjson entries "${source_entries}" \
    --argjson regular_files "${source_files}" \
    --argjson directories "${source_dirs}" \
    --argjson symlinks "${source_links}" \
    --argjson regular_bytes "${source_bytes}" \
    --argjson disk_bytes "${source_disk_bytes}" \
    --argjson archive_bytes "${archive_bytes}" \
    '{
      archived_at: $archived_at,
      source: $source,
      destination: $destination,
      payload: $payload,
      reason: $reason,
      source_stats: {
        entries: $entries,
        regular_files: $regular_files,
        directories: $directories,
        symlinks: $symlinks,
        regular_bytes: $regular_bytes,
        disk_bytes: $disk_bytes
      },
      archive: {
        format: "tar.zst",
        bytes: $archive_bytes,
        sha256: $archive_sha256,
        zstd_test: "passed",
        tar_entry_count_match: true,
        tar_compare: "passed"
      },
      source_replaced_by_symlink: true
    }' >"${metadata}.partial"
  mv "${metadata}.partial" "${metadata}"

  printf '%s\n' \
    "Archived source: ${source}" \
    "Reason: ${reason}" \
    "SHA-256: ${archive_sha256}" \
    "" \
    "Restore after removing the archive symlink at the source path:" \
    "  tar --acls --xattrs --use-compress-program='${ZSTD_BIN} -q' -xf '${payload}' -C '${source_parent}'" \
    >"${restore_notes}"

  mv "${source}" "${backup}"
  ln -s "${destination}" "${source}"
  rm -rf -- "${backup}"

  jq -c . "${metadata}" >>"${MANIFEST}"
  echo "bundle_complete source=${source} destination=${destination} archive_bytes=${archive_bytes} sha256=${archive_sha256}"
done

echo "archive_completed=$(date -Is)"
