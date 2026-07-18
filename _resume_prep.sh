#!/bin/bash
cd /mnt/e/courses/UMN/c2hls/code_translation_c2hls
D=results_matrix_u280_ENH_allpositive_multistep_skills_SONNET
cp "$D/matrix.json" "$D/matrix.json.prereboot.bak"
python3 - <<'PY'
import json
p="results_matrix_u280_ENH_allpositive_multistep_skills_SONNET/matrix.json"
c=json.load(open(p))
ok=[x for x in c if x.get("status")=="ok"]
drop=sorted(x["bench"].replace("hlsfactory_","") for x in c if x.get("status")!="ok")
json.dump(ok, open(p,"w"), indent=2)
print("matrix.json: kept", len(ok), "ok ; dropped-for-retry", len(drop), ":", drop)
PY
n=0
for d in $D/hlsfactory_*/; do
  if ! ls "$d"*/*_multistep_results.json >/dev/null 2>&1; then rm -rf "$d"; n=$((n+1)); fi
done
echo "deleted $n failed/partial cell dirs"
echo "allpositive cell dirs remaining (should be 9 ok): $(ls -d $D/hlsfactory_*/ 2>/dev/null | wc -l)"
echo "oneshot/curated intact: $(ls -d results_matrix_u280_ENH_oneshot_SONNET/hlsfactory_*/ 2>/dev/null | wc -l) / $(ls -d results_matrix_u280_ENH_curated_multistep_skills_SONNET/hlsfactory_*/ 2>/dev/null | wc -l)"
