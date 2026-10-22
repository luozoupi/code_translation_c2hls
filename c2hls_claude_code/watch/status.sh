#!/bin/bash
# One-shot status of every overnight job (read-only). Prints job states, key progress lines, anomalies.
CC=/scratch/asa582/workspaces/c2hls_claude_code
date
echo "== Slurm states (JobName State Elapsed MaxRSS)"
IDS=$(awk '{print $2}' $CC/cosim/jobs.txt $CC/cosim/fastcosim/jobs.txt | paste -sd,),63754829,63755599,$(cat $CC/xclbin/*/job_id.txt | paste -sd,)
sacct -X -n -P -j "$IDS" -o JobID,JobName,State,Elapsed | column -t -s'|'
echo "== Finished cosim verdicts"
for f in $CC/cosim/*/cosim_summary.json $CC/cosim/fastcosim/[0-9]*/fastcosim_summary.json; do
  [ -f "$f" ] || continue; n=$(basename $(dirname $f)); [[ $n == selftest* ]] && continue
  echo "$n: $(grep -oE '"(correct|rtl_check_passed|kernel_runtime_cycles|rtl_cycles)": [^,]*' $f | tr '\n' ' ')"
done
echo "== Fast cosim heartbeats (last)"
for l in /scratch/asa582/tmp/c2hls/fastcosim_*/vivado.log /localscratch/*/c2hls/fastcosim_*/vivado.log; do
  [ -f "$l" ] && echo "$(basename $(dirname $l)): $(grep -c FASTCOSIM_HEARTBEAT $l) beats, last: $(grep FASTCOSIM_ $l | tail -1)"; done 2>/dev/null
echo "== xclbin build (last stage lines)"
for b in $CC/xclbin/*/; do j=$(cat $b/job_id.txt); echo " [$(basename $b) job $j]"; grep -hE "^== |ERROR|reusing|done$" $b/logs/build-$j.out 2>/dev/null | tail -4; done
grep -iE "error" $CC/xclbin/ladder_agent_r2_L3/logs/build-*.err | tail -3
X=$(ls -d /scratch/asa582/tmp/c2hls/xclbin_ladder_agent_r2_L3_* 2>/dev/null | head -1)
[ -n "$X" ] && for r in $X/_x_l/link/vivado/vpl/prj/prj.runs/*; do [ -d "$r" ] && echo "  run $(basename $r): $(ls $r/*.dcp 2>/dev/null | wc -l) dcp, $(tail -1 $r/runme.log 2>/dev/null | cut -c1-120)"; done
echo "== Window r1 / agent ladder r3"
ls -d $CC/runs/skills_win7080_r1_*/autosa_mm/autosa_mm_selected_report.json $CC/runs/ladder_agent_r3_*/skills_L3_*/autosa_mm/autosa_mm_selected_report.json 2>/dev/null
echo "== Leftover work dirs of finished/killed jobs in /scratch/asa582/tmp/c2hls (cosim/fastcosim/xclbin)"
for d in /scratch/asa582/tmp/c2hls/cosim_* /scratch/asa582/tmp/c2hls/fastcosim_* /scratch/asa582/tmp/c2hls/xclbin_*; do [ -d "$d" ] && echo "  $(du -sh $d 2>/dev/null | cut -f1) $(basename $d)"; done
echo "== /scratch quota"; diskusage_report 2>/dev/null | grep -i scratch
