# Running the post-hoc csynths on Otus

> **Superseded for the full move** (`study/OTUS_MIGRATION.md`): use `posthoc_csim/resubmit_csynth_otus.sh` (`--check`, `--compare`, then no option). This file stays for the case where only these csynths run on Otus and everything else stays on Fir.

This runbook covers the 21 post-hoc csynths and the team's 2 pending ones. The same job header works for any other `hls_eval_job.py` csim or csynth. The Fir jobs stay as they are.

Replace `<user>@<otus-host>` with your Otus login. `$P` is the project folder on Otus.

## 1. Copy the files to Otus (from Fir)

```bash
P=/scratch/hpc-prf-llmfpga/asa582/projects/c2hls_judge_20261022
OT=<user>@<otus-host>
ssh $OT "mkdir -p $P/tmp $P/code_translation_c2hls $P/check_v3"
# the judge: hls_eval needs only these four files (standard library only), not the 13 GB repo
cd /scratch/asa582/workspaces/code_translation_c2hls
rsync -a c2hls_temp.py hls_eval.py hls_feedback.py hls_log_compact.py $OT:$P/code_translation_c2hls/
# the job runner
rsync -a /scratch/asa582/workspaces/c2hls_claude_code/study/agentfw/tools/hls_eval_job.py $OT:$P/
# the 21 kernels and the bench files (task.json, kernel.h, testbench.cpp)
rsync -a --exclude '*_work' --exclude '*.out' --exclude '*.err' \
  /scratch/asa582/workspaces/c2hls_claude_code/study/posthoc_csim/ $OT:$P/posthoc_csim/
# the team's two pending csynths
rsync -a --exclude '*_work' \
  /scratch/asa582/workspaces/c2hls_claude_code/study/agentfw/runs/mm1024_team_r1/exp_post/v4_ti36_csynth \
  /scratch/asa582/workspaces/c2hls_claude_code/study/agentfw/runs/mm1024_team_r1/exp_post/v4_ti40_csynth \
  $OT:$P/posthoc_csim/csynth/mm1024_team_r1/
# the known design for the equivalence check (team v3: 527,571 cycles on Fir)
rsync -a /scratch/asa582/workspaces/c2hls_claude_code/study/agentfw/runs/mm1024_team_r1/exp_post/v3_csynth/kernel.cpp \
  $OT:$P/check_v3/
```

If Fir can't reach Otus over ssh, run the same `rsync` lines from Otus with source and destination swapped.

## 2. On Otus: point the runner at Otus

```bash
P=/scratch/hpc-prf-llmfpga/asa582/projects/c2hls_judge_20261022
cd $P
# the runner hard-codes the Fir path of the judge
sed -i "s|/scratch/asa582/workspaces/code_translation_c2hls|$P/code_translation_c2hls|" hls_eval_job.py
grep -n "sys.path.insert" hls_eval_job.py          # should now show $P
# Python, and a check that the judge imports
module load lang Python/3.11.5-GCCcore-13.2.0
python3 -c "import sys; sys.path.insert(0,'$P/code_translation_c2hls'); import hls_eval; print('ok')"
```

It should print `ok`. No venv is needed.

`keep_copy.sh` only exists on Fir. The job header below unsets `SLURM_TMPDIR`, so Vitis runs in place and the runner never calls it.

## 3. The job script (`$P/job_one.sh`)

```bash
cat > $P/job_one.sh <<'EOF'
#!/bin/bash
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# usage: sbatch -o DIR/csynth-%j.out job_one.sh csynth|csim DIR
set -eo pipefail
P=/scratch/hpc-prf-llmfpga/asa582/projects/c2hls_judge_20261022
module load lang Python/3.11.5-GCCcore-13.2.0
export C2HLS_VITIS_SETTINGS=/opt/software/FPGA/Xilinx/Vitis/2023.2/settings64.sh
export C2HLS_TMP_ROOT=$P/tmp C2HLS_CSIM_TIMEOUT=200000 C2HLS_SYNTH_TIMEOUT=200000
unset SLURM_TMPDIR HLS_LOCAL_ROOT C2HLS_USE_CONTAINER C2HLS_DSE_V3 C2HLS_DSE_V3_HARNESS C2HLS_DSE_V4 C2HLS_SITE C2HLS_VITIS_JOBS C2HLS_VITIS_JOBS_FROM_SLURM
python3 $P/hls_eval_job.py "$1" "$2" $P/posthoc_csim/ws/task.json
EOF
```

## 4. Check that Otus matches Fir, before anything else

```bash
cd $P && sbatch -o check_v3/csynth-%j.out job_one.sh csynth $P/check_v3
# when it finishes:
head -12 check_v3/csynth_summary.txt
```

Fir's csynth of team v3 (`exp_post/v3_csynth/csynth_summary.txt`) gave:

| Field | Value |
|---|---|
| latency_cycles (best and worst) | 527,571 |
| DSP | 8,192 |
| LUT | 984,616 (75.5% of 1,303,680) |
| FF | 1,706,888 |
| BRAM_18K | 2,197 |
| URAM | 0 |

Otus should match exactly. If anything differs, stop: the two Vitis installs disagree, and the Otus results shouldn't be mixed with Fir's.

## 5. Submit the 23 csynths

```bash
cd $P
for d in posthoc_csim/csynth/*/*/; do
  [ -f "$d/kernel.cpp" ] && [ ! -f "$d/csynth_summary.json" ] && \
    echo "$d $(sbatch --parsable -o $d/csynth-%j.out job_one.sh csynth $P/$d)"
done | tee submitted_otus.txt
squeue -u $USER
```

## 6. Bring the results back to Fir (from Fir)

```bash
rsync -a --exclude '*_work' $OT:$P/posthoc_csim/csynth/ \
  /scratch/asa582/workspaces/c2hls_claude_code/study/posthoc_csim/csynth/
```

The team's two (v4_ti36 and v4_ti40) come back to `posthoc_csim/csynth/mm1024_team_r1/`. The page code already reads that folder, so they don't need to move.

**If both clusters finish the same job:** the Fir jobs are still queued, so whichever cluster finishes second writes into the same folder. The results should be identical if step 4 passed. To keep them apart, change the destination in step 6 to `posthoc_csim/csynth_otus/`; the page code would then need to read from there.

## 7. After the results are back (on Fir)

1. `sbatch study/posthoc_csim/job_hard.sh` (hardened bench on the new designs)
2. `python3 controlled/story/make_story_v14.py`
3. Update the survey numbers to match, then publish both pages.
