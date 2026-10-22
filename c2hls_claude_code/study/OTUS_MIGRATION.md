# Moving the study from Fir to Otus

Status checked on 2026-10-22: 113 Slurm jobs are pending on Fir and none is running. Two processes run on login2:
- `study/exp_hlsmith/driver.py` (experiment A/B queue; next up are `pl_refv2_cc_r2` and then C r4);
- the plain Claude shim on port 38585.

Git and rsync carry different things:
- **Git** carries the code: the judge and the study scripts.
- **rsync** carries the results: 27 GB in about 140k files, mostly `logs/` (15 GB) and `study/` (7.7 GB).

On Otus everything goes under `W=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces`. That folder takes the place of `/scratch/asa582/workspaces`, so a single `sed` fixes most paths.

**The Otus side is one script, `otus/otus_setup.sh <stage>`.** Run the stages in this order; each one prints the next:

| Stage | Plan step | What it does |
|---|---|---|
| `check` | 0 | network, Vitis, U280 platform, a 5-minute test job, ssh to Fir, disk and file quota; changes nothing |
| `clone` | 5 | makes `$W` and `$B/{tmp/c2hls,cache,packages,tools}`, clones the combined branch, links `c2hls_claude_code` |
| `home` | 4 | Claude Code if missing, the non-secret `~/.bashrc` lines, checks `DeepSeek_API` is there (never printed) |
| `copy` | 5 | every rsync from Fir; refuses while Fir has jobs queued or the driver running (`SKIP_FIR_CHECK=1` overrides) |
| `site` | 6 | the path/account/python/login-host `sed` (`OTUS_LOGIN=<node>` required), `apply_otus_site.sh`, one commit |
| `venv`, `lsim` | 6 | the c2hls venv; LightningSim with the fork (into `$LSIM_T`, default `$B/tools/lightningsim`) |
| `verify`, `compare` | 6 | team v3 csynth on Otus, then the field-by-field comparison with Fir; stop if it differs |
| `dryrun` | 7 | `--dry-run` of every resubmit script and the driver-state preview |

Before the clone exists, fetch the script alone (the repo is public):
`curl -fsSLO https://raw.githubusercontent.com/luozoupi/code_translation_c2hls/c2hls_enhanced_l_pc2_api_layout_study/c2hls_claude_code/otus/otus_setup.sh`. Then `bash otus_setup.sh check` and `bash otus_setup.sh clone`; after that, use `$W/c2hls_claude_code/otus/otus_setup.sh`. Logging in to Claude Code, adding the DeepSeek key, freezing Fir (step 1) and the services and real resubmits (step 7) stay manual. The sections below explain each step.

## 0. Check Otus first (about 10 minutes)

These checks decide whether the agent runs can work on Otus at all. Do them before you stop anything on Fir.

```bash
# on an Otus login node
curl -sI https://api.anthropic.com | head -1          # Claude Code needs this
curl -sI https://api.deepseek.com  | head -1          # DeepSeek proxy (experiment D)
srun -A hpc-prf-llmfpga -p normal -t 5 bash -c 'hostname; echo TMP=$SLURM_TMPDIR; df -h ${SLURM_TMPDIR:-/tmp} | tail -1; curl -s -m5 http://$SLURM_SUBMIT_HOST:1/ ; echo rc=$?'
ls /opt/software/FPGA/Xilinx/Vitis/2023.2/settings64.sh
ls /opt/software/FPGA/Xilinx/platforms 2>/dev/null | grep -i u280    # only xclbin needs this
```

What each check needs to show:
- **Compute nodes reach a login-node port.** The `curl` from the job should fail with rc=7 ("connection refused") or similar, not time out. A time-out means the firewall blocks the port, so the shim can't stay on a login node: it would have to run inside each job.
- **`SLURM_TMPDIR` is set.** If it's empty, the scripts that run Vitis in `$SLURM_TMPDIR` need a node-local path, such as `/tmp/$SLURM_JOB_ID` or whatever Otus documents.
- **Long-running login processes are allowed.** The shim, the DeepSeek proxy and the driver live for days on Fir's login2. Check Otus's rules, or plan to run them in `tmux`/`screen`.

## 1. Freeze Fir (you run these: I can't cancel jobs)

```bash
# first save every pending job's command, arguments, working folder and --export (step 7 reads them; the copy carries them)
squeue -u asa582 -h -o '%i|%j|%o|%Z' > /scratch/asa582/workspaces/c2hls_claude_code/study/fir_pending_20261022.txt
for j in $(squeue -u asa582 -h -o %i); do scontrol show job -o $j; done > /scratch/asa582/workspaces/c2hls_claude_code/study/fir_pending_20261022_full.txt
ssh login2 'pkill -f "python3 driver.py"; pkill -f claude_code_shim.py; pkill -f ds_anthropic_proxy.py'
scancel -u asa582 --state=PENDING          # all 113 jobs; every one is resubmitted on Otus in step 7
squeue -u asa582                           # should be empty
```

**The order matters: kill the driver before the `scancel`.** A driver that's still running sees `pl_refv2_cc_r2` end as CANCELLED, records it as final, and submits C r4 on Fir. (`study/exp_hlsmith/otus_reset_state.py` reopens a CANCELLED tag anyway, but nothing would stop that C r4 submission on Fir.)

If you'd rather let the 19+2 post-hoc csynths finish on Fir, cancel by name instead and keep `posthoc_csynth` and `afw_post`.

## 2. The judge (`code_translation_c2hls`): no commit

The judge checkout on Fir stays as it is: nothing is committed or pushed from it. Its remote is `git@github.com:luozoupi/code_translation_c2hls.git`, branch `c2hls_enhanced_l_pc2_api_layout` (`2ef0224`), with no unpushed commits. On Otus the judge comes from the combined branch `c2hls_enhanced_l_pc2_api_layout_study` (step 3): that commit's files plus the study in `c2hls_claude_code/`.

Three skill libraries in its working tree differ from that commit. The study reads two of them by path. A snapshot is in the study repo, `study/judge_skills_fir_20261022/`, with a README, the md5s and the diff sizes:
- `(90skills).json`: only its `saved_at` timestamp differs;
- `(90skills)_plus_gemm_flatten_v2.json`: +278/−271 lines;
- `(90skills)_gemm_flatten_v1.json`: +1,479/−211 lines.

Who or what changed them isn't known. `otus/apply_otus_site.sh` (step 6) copies the snapshot into the Otus clone and checks the md5s, so Otus runs use the same libraries as Fir.

The judge's other local changes (`fir.env.example`, `nibi.env`, `scripts/fir/gpu_serve.sbatch.sh`, `rl/eval/batch_parallel_heldout_u280_fir.json`, untracked `experiments_registry.json`) aren't used by the study and aren't carried over. Never commit `.env`; it's git-ignored and holds secrets. If the c2hls flow needs it on Otus, copy it by `scp` (step 4). The study's refined skill files (`…_refined.json`, `…_refined_v2.json`, `…_refined_v2_neutral.json`) are in the study tree.

## 3. Git: the study (`c2hls_claude_code`), a new private repo

**Done on 2026-10-22.** The repo is local, on branch `main`, with a first commit (3,131 files, 2.2 MB packed). Its `.gitignore` keeps out:
- logs (but not `logs/start_*.sh`), run outputs (`runs*/`) and `ablation/` results;
- Vitis work trees, reports (`*.xml`, `*.rpt`) and logs, including gzipped ones;
- the generated SODA and LUD designs;
- transcripts (`*.jsonl`) and agent configs;
- `.bak_*`, `.out`, `.err` and `.pid` files and archives;
- the clean AutoSA clone, whose commit is recorded in `study/lud/upstream/AutoSA.commit`.

Two files hold personal data, so they stay out of git; rsync carries them:
- `study/exp_hlsmith/ds_anthropic_proxy.py`: its privacy guard spells out your addresses;
- `cosim/lsim/otus/lightningsim_vitis2023-axi.bundle`: its commits carry the author's email.

The fork patches name the author with a noreply address and rebuild exactly Fir's tree (`da0dae7f`, tested). The commit identity is repo-local: `ahmadsb101 <ahmadsb101@users.noreply.github.com>`.

On GitHub (`luozoupi/code_translation_c2hls`, **a public repo**), two branches; `c2hls_enhanced_l_pc2_api_layout` isn't touched:
- `c2hls_claude_code_study`: the study alone, with its own history. Local `main` tracks it.
- `c2hls_enhanced_l_pc2_api_layout_study`: **the one Otus uses.** A merge of `c2hls_enhanced_l_pc2_api_layout` (`2ef0224`) and the study, with the study in the subfolder `c2hls_claude_code/`. Both histories are kept.

```bash
cd /scratch/asa582/workspaces/c2hls_claude_code
# later edits on Fir before the move: commit and push the study ...
git add -A && git commit -m "..." && git push    # goes to origin/c2hls_claude_code_study
# ... then merge it into the combined branch again, in a clone of that branch:
#   git pull -s subtree origin c2hls_claude_code_study && git push
```

To use your real address on the commits instead: `git config user.email <address>` and `git commit --amend --reset-author --no-edit`.

## 4. Home directory on Otus

```bash
# Claude Code: install, then log in interactively once
curl -fsSL https://claude.ai/install.sh | bash
claude            # then /login
# the account switcher you run on Fir (cswap)
uv tool install claude-swap 2>/dev/null || pip install --user claude-swap
# DeepSeek key: add this line to ~/.bashrc yourself (same non-exported name; never paste it into a chat)
#   DeepSeek_API=<key>
# the other ~/.bashrc lines Fir has, re-pointed at Otus:
#   export SBATCH_ACCOUNT=hpc-prf-llmfpga SALLOC_ACCOUNT=hpc-prf-llmfpga SLURM_ACCOUNT=hpc-prf-llmfpga
#   export TMPDIR=$W/../tmp XDG_CACHE_HOME=$W/../cache PIP_CACHE_DIR=$W/../cache/pip
# judge secrets, if needed
scp fir:/scratch/asa582/workspaces/code_translation_c2hls/.env $W/code_translation_c2hls/.env
```

My memory files go across with rsync in step 5. Claude Code keys project memory by the working folder, so on Otus they sit under a new folder name. Start `claude` once in `$W/dse_v4_claude` on Otus and look at `ls ~/.claude/projects/` to get that name.

## 5. Clone and copy the results (on Otus)

```bash
W=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces; mkdir -p $W ../tmp/c2hls ../cache
cd $W
# one clone: the judge, with the study in c2hls_claude_code/; the link gives the Fir layout the scripts expect
git clone -b c2hls_enhanced_l_pc2_api_layout_study git@github.com:luozoupi/code_translation_c2hls.git
ln -s code_translation_c2hls/c2hls_claude_code c2hls_claude_code
FIR=asa582@fir.alliancecan.ca
# results on top of the clone (keeps .git). *_work/ folders are copied: the page builders and posthoc_csim/enumerate*.py
# read checks/N_*_work/kernel.cpp and N_*_work/hls_proj/*/syn/report/csynth.xml. Only Vitis internals are left out.
rsync -a --info=progress2 --exclude '.git' --exclude '__pycache__' \
  --exclude '.autopilot/' --exclude 'verilog/' --exclude 'vhdl/' --exclude 'systemc/' --exclude 'impl/' \
  --exclude 'csim/build/' --exclude '*.o' --exclude '*.vcd' --exclude '*.wdb' \
  $FIR:/scratch/asa582/workspaces/c2hls_claude_code/ $W/c2hls_claude_code/
# also needed: SODA sources (SODA runs), AutoSA (only if you regenerate designs), dse_v4_claude
rsync -a $FIR:/scratch/asa582/workspaces/soda_src $FIR:/scratch/asa582/workspaces/dse_v4_claude $W/
# Vitis 2023.2 headers the hardened bench compiles against (study/bench_hard/verify_hard.py; 5.4 MB, outside workspaces/).
# If you skip this, step 6's apply_otus_site.sh links that path to Otus's $XILINX_HLS/include instead.
mkdir -p ${W%/workspaces}/tmp/c2hls
rsync -a $FIR:/scratch/asa582/tmp/c2hls/vitis_hls_2023.2_include ${W%/workspaces}/tmp/c2hls/
# packed SODA projects the LightningSim jobs read (3.8 GB, outside workspaces/; 158 projects, one tarball each)
mkdir -p ${W%/workspaces}/tmp/c2hls/lsim
rsync -a $FIR:/scratch/asa582/tmp/c2hls/lsim/soda ${W%/workspaces}/tmp/c2hls/lsim/
# the LightningSim fork (3 commits on the local branch vitis2023-axi, never pushed) travels inside the study tree:
# cosim/lsim/otus/{fork_patches/,lightningsim_vitis2023-axi.bundle,fork_tree.txt,*_env_explicit.txt}. As a second
# copy, also keep Fir's source (50 MB, 598 files; no llvm-project):
rsync -a $FIR:/scratch/asa582/tools/lightningsim/src/ ${W%/workspaces}/tools/lightningsim_fir_src/
# Claude memory for this project
rsync -a $FIR:.claude/projects/-scratch-asa582-workspaces-dse-v4-claude/memory/ ~/.claude/projects/<otus-key>/memory/
```

I kept the `*_work/` and `hls_proj/` folders on purpose. The page builders choose each run's counted design and its fallback from the session checks: the kernel in `checks/N_*_work/kernel.cpp` and the worst-case latency from `N_*_work/hls_proj/*/syn/report/csynth.xml`. On Fir that's 212 check kernels and 104 reports. Dropping `*_work/` would have lost all of them and quietly changed numbers on the pages. With only the Vitis internals left out, the copy is about 108k files and 5.5 GB (a dry run on Fir). Check your file quota on Otus first. If that's too many, tar `logs/shim_*` per folder, but the pages read `logs/<shim>/call_*/workspace/checks/`, so keep those unpacked.

## 6. Make the code Otus-native (one commit on Otus)

```bash
OTUS_LOGIN=<otus login node> $W/c2hls_claude_code/otus/otus_setup.sh site
```

What it does, in the clone only (tested on a copy of the combined layout on Fir: 992 files changed, no Fir path or `login2` host left):
- **paths**: `/scratch/asa582/workspaces` → `$W` and `/scratch/asa582/tmp` → `$B/tmp`, through placeholders so neither replacement rewrites the other. `logs/` (15 GB of transcripts) and `*.jsonl` are left out, except `logs/start_*.sh`.
- **account**: `--account=def-zhenman…` → `--account=hpc-prf-llmfpga` plus `#SBATCH --partition=normal` on `#SBATCH` lines, and `--account=hpc-prf-llmfpga --partition=normal` inline (`srun`, `sbatch` command lines).
- **python**: the Compute Canada `python3` path → `python3`.
- **login host**: the bare word `login2` → `$OTUS_LOGIN` (`driver.py` HOST, `shim_url.txt`, `controlled/ds_proxy_url.txt`, `serial_driver.py`, `run_agent_ds.sh`). Names that contain it, such as `logs/shim_20261020_login2/` and `start_shim_38585_login2.sh`, stay, because the pages read those folders; `*.md` files keep Fir's history as written.
- `otus/apply_otus_site.sh --dry-run`, then for real: agent tools and checks, flash job, claude path, judge skill libraries (md5-checked), hardened-bench headers.
- one commit of the tracked changes, the judge skill libraries included.

`otus_setup.sh`, this file, `OTUS_RUNBOOK.md` and `judge_skills_fir_20261022/` aren't touched: they hold Fir-side commands and records.

The `sed` can't fix these. Edit them by hand:

| Fir piece | Where | Otus replacement |
|---|---|---|
| Vitis apptainer container (`apptainer exec "$XILINX_SIF"`, `APPTAINER_BINDPATH`) | `study/soda/lsim/job_proj.sh` (SODA cosims: done in `study/soda/dse/job_soda_otus.sh`), `cosim/lsim/sbatch_lsim.sh`, `xclbin/v4_systolic_st0_c1/sbatch_build.sh` | `source /opt/software/FPGA/Xilinx/Vitis/2023.2/settings64.sh` and run the command directly |
| Vitis for the c2hls flow | `sbatch_flash_cc.sh`, every `job.sh` from `hls_eval_job.py` | `export C2HLS_VITIS_SETTINGS=…/settings64.sh`; `unset C2HLS_USE_CONTAINER` (see `study/posthoc_csim/OTUS_RUNBOOK.md`) |
| `module load` lines (Fir modules) | the job scripts | `module load lang Python/3.11.5-GCCcore-13.2.0` |
| `$SLURM_TMPDIR` | `sbatch_flash_cc.sh`, `study/lud/job_vitis.sh`, SODA/lsim/xclbin jobs | keep it if step 0 showed it's set; otherwise a node-local path |
| LightningSim fork (built on Fir) | `/scratch/asa582/tools/lightningsim` | rebuild once on an Otus login node: `bash cosim/lsim/otus/setup_lsim_otus.sh`. It builds the same envs from Fir's exact package lists, clones upstream, applies the 3 fork commits, checks the source tree hash against Fir's, and rebuilds the Rust core. Then `cosim/lsim/otus/resubmit_lsim_otus.sh --check`: the cycle count must equal Fir's. |
| U280 platform for `v++` | xclbin job | only if step 0 found it (`sbatch_build_otus.sh` searches the usual dirs, or set `OTUS_U280_XPFM`) |
| c2hls venv (`/scratch/asa582/packages/c2hls-venv`) | every flash job (`sbatch_flash_cc.sh`, the driver, `sd_heat3d_ds`) | `bash otus/setup_venv_otus.sh` (exact versions from Fir, no torch). `otus/apply_otus_site.sh` installs `otus/sbatch_flash_cc_otus.sh` as `sbatch_flash_cc.sh`: the same command line, native Vitis, paths from the submit dir. |
| Agent-session checks: `study/agentfw/tools/hls` (free agent, C/D; a copy already sits in `runs/mm1024_hlsm_r4/bin/`) and `agent/hls_check.sh.template` + `hls_check.py` (HLSmith with an agent backend, experiment B) | C r4, experiment B, any new agent run | `otus/apply_otus_site.sh` installs the `_otus` copies: Otus account and partition, native Vitis, paths from their own location. It also installs them into unfinished workspaces (`mm1024_hlsm_r4`, `smoke_base`) and sets the shim's `--agent-account` default. |
| DeepSeek proxy for the controlled runs (`controlled/ds_proxy.py`, Fir login2:18094, **not running on Fir now**) | `sd_*_ds` runs | `bash otus/start_ds_proxy_otus.sh` (writes `controlled/ds_proxy_url.txt`) |
| `--claude-bin /home/asa582/.local/bin/claude` | `logs/start_shim_*.sh`, `start_shim.sh` | `$(command -v claude)` |

Before resubmitting anything, run the equivalence check in `study/posthoc_csim/OTUS_RUNBOOK.md` step 4 (team v3 must give 527,571 cycles, 8,192 DSP and 984,616 LUT). If it doesn't match, stop.

## 7. Start the services and resubmit (on Otus)

```bash
cd $W/c2hls_claude_code
tmux new -s study        # or screen
setsid nohup bash start_shim.sh                       >> logs/shim_otus.out 2>&1 &   # plain shim, port 38585
# agent shim (port in driver.py AGENT_PORT) and the DeepSeek proxy, if experiment D continues:
DEEPSEEK_API_KEY="$DeepSeek_API" setsid nohup python3 study/exp_hlsmith/ds_anthropic_proxy.py --port 18095 \
   --log study/exp_hlsmith/ds_anthropic_calls.jsonl >> study/exp_hlsmith/ds_anthropic_proxy.out 2>&1 &
curl -s http://localhost:38585/health
cd study/exp_hlsmith && setsid nohup python3 driver.py >> driver.out 2>&1 &   # resumes from driver_state.json
```

Before starting the driver, clear the Fir job id it would otherwise wait on (`pl_refv2_cc_r2`, Fir job 66683742):

```bash
python3 study/exp_hlsmith/otus_reset_state.py           # shows: pl_refv2_cc_r2: drop Fir job 66683742 (tier 2, tries 3)
python3 study/exp_hlsmith/otus_reset_state.py --apply   # backup: driver_state.json.bak_fir
```

The run restarts at the 6 h tier it reached on Fir.

Then resubmit the other job groups. Each line below loops over the folders that have no result yet:

| Jobs on Fir | Count | Resubmit with |
|---|---|---|
| `posthoc_csynth` | 19 | `posthoc_csim/resubmit_csynth_otus.sh`. Run `--check` first, then `--compare`: team v3 must match Fir in every latency and resource field. Then run it with no option. It submits `job_csynth_otus.sh` → `agentfw/tools/hls_eval_job_otus.py` (8 cpus, 64G, 12 h, as on Fir) and skips dirs that already have a successful csynth. |
| `afw_post` (team v4_ti36/ti40) | 2 | same script: they run in their own `agentfw/runs/mm1024_team_r1/exp_post/` dirs with that run's `task.json`, as on Fir |
| `soda_cosim`, `soda_pat_cosim` | 37 + 48 | `study/soda/dse/resubmit_soda_otus.sh` (try `--dry-run` first). It reads `otus_soda_resubmit.txt` and submits `job_soda_otus.sh`: the Fir script with the Otus account and native Vitis, and a node-local fallback when `$SLURM_TMPDIR` is unset. It skips designs that already have a cosim report. |
| `lsim_soda2`, `lsim_soda_pat`, `lsim_proj` | 3 | `cosim/lsim/otus/resubmit_lsim_otus.sh` (after `setup_lsim_otus.sh` and `--check`). It submits `otus/sbatch_lsim_otus.sh` and `study/soda/lsim/job_proj_otus.sh`. It leaves out the 19 projects Fir already finished; `ALL=1` reruns them. |
| `lud_csynth` | 1 | `study/lud/resubmit_lud_otus.sh` (submits `job_vitis_otus.sh`, 192G, 2 days) |
| `sd_heat3d_ds` | 1 | `otus/resubmit_sd_heat3d_otus.sh`. It needs the c2hls venv (`otus/setup_venv_otus.sh`, once) and the DeepSeek proxy (`otus/start_ds_proxy_otus.sh`, port 18094). It submits `otus/sbatch_flash_cc_otus.sh` with Fir's exact `--export` (12 h tier, 192G); only the paths, the proxy URL and the include dir change. |
| `xclbin_v4_systolic_st0_c1` | 1 | `xclbin/v4_systolic_st0_c1/resubmit_xclbin_otus.sh` (32 cpus, 448G, 2 days; `MEM=`, `TIME=`, `PARTITION=` override). `sbatch_build_otus.sh` stops early if it can't find the U280 platform; set `OTUS_U280_XPFM` if it lives elsewhere. |
| `pl_refv2_cc_r2`, then C r4 and the rest of A/B | queue | `driver.py` |

Not every SODA, lsim and flash job script records its submit arguments. Step 1 saves them on Fir before the `scancel`, in `study/fir_pending_20261022.txt` and `study/fir_pending_20261022_full.txt`; the step 5 copy carries them.

## 8. After the move

Open Claude Code on Otus in the same project and point it at `study/PROGRESS.md` and this file. The page builders (`controlled/story/make_story_v14.py`) and the two artifact URLs don't depend on the cluster.
