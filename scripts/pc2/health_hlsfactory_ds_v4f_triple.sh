#!/usr/bin/env bash
# One health snapshot for the HLSFactory DeepSeek-v4-flash triple campaign.
set -euo pipefail
STAMP="${1:?stamp}"
ROOT="/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2"
MANIFEST="${ROOT}/hlsfactory_ds_v4f_triple_${STAMP}/triple_manifest.json"
echo "=== health $(date -u +%Y-%m-%dT%H:%M:%SZ) stamp=${STAMP} ==="
if [[ -f "${MANIFEST}" ]]; then
  python3 - "${MANIFEST}" <<'PY'
import json, sys
from pathlib import Path
doc = json.loads(Path(sys.argv[1]).read_text())
for name, meta in (doc.get("flavors") or {}).items():
    root = Path(meta["campaign_root"])
    camp = {}
    cj = root / "campaign.json"
    if cj.is_file():
        camp = json.loads(cj.read_text())
    q = root / "queue" / "jobs.jsonl"
    pending = running = done = failed = 0
    if q.is_file():
        for line in q.read_text().splitlines():
            if not line.strip():
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            st = j.get("status") or j.get("state") or ""
            if st in ("pending", "queued"):
                pending += 1
            elif st in ("running", "claimed"):
                running += 1
            elif st in ("done", "complete", "completed"):
                done += 1
            elif st in ("failed", "error"):
                failed += 1
    print(f"{name}: status={camp.get('campaign_status')} skip_peak={camp.get('skip_peak_pause')} "
          f"jobs pending={pending} running={running} done={done} failed={failed} root={root}")
PY
fi
echo "--- squeue synth summary ---"
squeue -u "$USER" -h -o '%j %T' 2>/dev/null | awk '
  /bphfs-synth/ {s[$2]++} /bphfn-synth/ {n[$2]++} /bphfb-synth/ {b[$2]++}
  END {
    printf "skills_synth:"; for (k in s) printf " %s=%d", k, s[k]; print ""
    printf "noskills_synth:"; for (k in n) printf " %s=%d", k, n[k]; print ""
    printf "bare_synth:"; for (k in b) printf " %s=%d", k, b[k]; print ""
  }'
echo "--- proxies ---"
for p in 18092 18093 18094; do
  if curl -sf --max-time 5 "http://127.0.0.1:${p}/v1/models" >/dev/null 2>&1; then
    echo "port ${p}: OK"
  else
    echo "port ${p}: DOWN"
  fi
done
echo "--- recent errors (if any) ---"
for f in skills noskills bare; do
  case $f in
    skills) C=$ROOT/batch_parallel_hlsfactory_ds_v4f_skills_$STAMP ;;
    noskills) C=$ROOT/batch_parallel_hlsfactory_ds_v4f_noskills_$STAMP ;;
    bare) C=$ROOT/batch_parallel_hlsfactory_ds_v4f_bare_$STAMP ;;
  esac
  if [[ -f "$C/flow/events.jsonl" ]]; then
    err=$(grep -iE 'error|fail|abort' "$C/flow/events.jsonl" 2>/dev/null | tail -3 || true)
    if [[ -n "${err}" ]]; then
      echo "[$f]"; echo "${err}"
    fi
  fi
done
