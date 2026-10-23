#!/bin/bash
# otus_setup.sh <stage> : the Otus side of study/OTUS_MIGRATION.md (steps 0 and 4-7), one stage at a time, in order:
#   check    step 0: Anthropic/DeepSeek reachable, Vitis, U280 platform, a 5-min test job (SLURM_TMPDIR, login-node
#            port from a compute node), ssh to Fir, disk and file quota. Changes nothing.
#   clone    step 5a: make $W and $B/{tmp/c2hls,cache,packages,tools}, clone the combined branch as
#            $W/code_translation_c2hls (judge + study in c2hls_claude_code/), link $W/c2hls_claude_code to the subfolder
#   home     step 4: Claude Code (install if missing), the non-secret ~/.bashrc lines (account, W, B, TMPDIR, caches,
#            LSIM_T), and a check that DeepSeek_API is in ~/.bashrc (never printed). Log in with `claude` yourself.
#   copy     step 5b: rsync the results and the extra folders from Fir. Refuses while Fir still has jobs queued or the
#            driver running (step 1); SKIP_FIR_CHECK=1 overrides. Optional: OPTIONAL=1 (Fir's LightningSim source,
#            AutoSA), COPY_ENV=1 (the judge's .env).
#   unpack   step 5b without ssh (Otus needs a VPN, so Fir can't reach it): unpack the tarballs that otus/pack_for_otus.sh
#            made on Fir and Globus brought to $B/transfer_from_fir (or the folder given as $2); checks SHA256SUMS first
#   site     step 6: the path/account/python/login-host sed (OTUS_LOGIN=<login node> required), apply_otus_site.sh,
#            one commit in the clone
#   venv     the c2hls venv (otus/setup_venv_otus.sh)
#   lsim     rebuild LightningSim with the fork (cosim/lsim/otus/setup_lsim_otus.sh) into $LSIM_T
#   verify   the team v3 csynth that must match Fir (posthoc_csim/resubmit_csynth_otus.sh --check)
#   compare  after that job: every field must equal Fir's (--compare). Stop here if it doesn't.
#   dryrun   --dry-run of every resubmit script, and the driver-state preview
# Env: W (default /scratch/hpc-prf-llmfpga/asa582/projects/workspaces; B is the folder above it), FIR (default
# asa582@fir.alliancecan.ca). The ssh to Fir is multiplexed, so Fir's MFA prompt comes once per stage.
# Before the clone exists, fetch this script alone:
#   curl -fsSLO https://raw.githubusercontent.com/luozoupi/code_translation_c2hls/c2hls_enhanced_l_pc2_api_layout_study/c2hls_claude_code/otus/otus_setup.sh
set -euo pipefail
[ "${CC_CLUSTER:-}" = fir ] && { echo "this is Fir: otus_setup.sh runs on Otus"; exit 1; }
W=${W:-/scratch/hpc-prf-llmfpga/asa582/projects/workspaces}; B=${W%/workspaces}
[ "$B" != "$W" ] || { echo "W must end in /workspaces (B is the folder above it): $W"; exit 1; }
FIR=${FIR:-asa582@fir.alliancecan.ca}
REPO=${REPO:-https://github.com/luozoupi/code_translation_c2hls.git} BRANCH=${BRANCH:-c2hls_enhanced_l_pc2_api_layout_study}   # public: no GitHub key
J=$W/code_translation_c2hls CC=$W/c2hls_claude_code
export LSIM_T=${LSIM_T:-$B/tools/lightningsim}
ACCT=hpc-prf-llmfpga VITIS=/opt/software/FPGA/Xilinx/Vitis/2023.2/settings64.sh
SSH="ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-%r@%h:%p -o ControlPersist=1h"
ok() { echo "  [ok]   $*"; }
bad() { echo "  [FAIL] $*"; }
need_clone() { [ -f "$CC/otus/apply_otus_site.sh" ] || { echo "no clone at $CC: run '$0 clone' first"; exit 1; }; }

case "${1:-}" in
check)
  echo "step 0 checks (nothing is changed)"
  for u in https://api.anthropic.com https://api.deepseek.com; do
    c=$(curl -s -m10 -o /dev/null -w '%{http_code}' "$u" || true); [ "$c" != 000 ] && ok "$u reachable (HTTP $c)" || bad "$u not reachable"
  done
  [ -f "$VITIS" ] && ok "Vitis 2023.2: $VITIS" || bad "no $VITIS"
  P=$(ls -d /opt/software/FPGA/Xilinx/platforms/*u280* 2>/dev/null | head -1 || true)
  [ -n "$P" ] && ok "U280 platform: $P" || echo "  [warn] no U280 platform under /opt/software/FPGA/Xilinx/platforms (only xclbin needs it; OTUS_U280_XPFM)"
  $SSH -o ConnectTimeout=20 -o BatchMode=yes "$FIR" true 2>/dev/null && ok "ssh to $FIR" \
    || echo "  [warn] no ssh to $FIR: use Globus (otus/pack_for_otus.sh on Fir, then the unpack stage) instead of copy"
  git ls-remote "$REPO" "refs/heads/$BRANCH" >/dev/null 2>&1 && ok "GitHub: $BRANCH" || bad "GitHub not reachable: the clone stage needs it"
  echo "  disk and quota for $(dirname "$B"):"; df -h "$(dirname "$B")" | tail -1 | sed 's/^/    /'
  { lfs quota -h -u "$USER" "$(dirname "$B")" 2>/dev/null || quota -s 2>/dev/null || true; } | sed 's/^/    /'
  echo "  the copy needs about 9.7 GB and 124k files (the study results alone: 5.5 GB, 108k files)"
  echo "  a 5-minute test job (account $ACCT, partition normal):"
  srun -A $ACCT -p normal -t 5 bash -c 'echo "    node=$(hostname) SLURM_TMPDIR=${SLURM_TMPDIR:-<unset>}"; df -h ${SLURM_TMPDIR:-/tmp} | tail -1 | sed "s/^/    local disk: /"; curl -s -m5 http://$SLURM_SUBMIT_HOST:1/ >/dev/null; echo "    curl to the login node port: rc=$? (7 = refused = reachable; 28 = timed out = blocked)"' \
    || bad "srun failed"
  echo "  also check yourself: Otus allows long-running login processes (shim, proxy, driver), or use tmux/screen" ;;

clone)
  mkdir -p "$W" "$B/tmp/c2hls" "$B/cache/pip" "$B/packages" "$B/tools"
  if [ -d "$J/.git" ]; then echo "clone exists: $J ($(git -C "$J" rev-parse --abbrev-ref HEAD) $(git -C "$J" rev-parse --short HEAD))"
  else git clone -b $BRANCH $REPO "$J"; fi
  if [ -L "$CC" ]; then echo "link exists: $CC -> $(readlink "$CC")"
  elif [ -e "$CC" ]; then echo "$CC exists and is not a link: move it away first"; exit 1
  else ln -s code_translation_c2hls/c2hls_claude_code "$CC"; echo "linked $CC -> code_translation_c2hls/c2hls_claude_code"; fi
  git -C "$J" config user.name ahmadsb101; git -C "$J" config user.email ahmadsb101@users.noreply.github.com
  echo "next: $CC/otus/otus_setup.sh home" ;;

home)
  if command -v claude >/dev/null; then echo "claude: $(command -v claude)"
  else curl -fsSL https://claude.ai/install.sh | bash; echo "installed Claude Code: run 'claude' once and /login"; fi
  add() { grep -qF "$1" ~/.bashrc 2>/dev/null && echo "  have: $1" || { echo "$1" >> ~/.bashrc; echo "  added: $1"; }; }
  echo "~/.bashrc:"
  add "export SBATCH_ACCOUNT=$ACCT SALLOC_ACCOUNT=$ACCT SLURM_ACCOUNT=$ACCT"
  add "export W=$W B=$B LSIM_T=$LSIM_T"
  add "export TMPDIR=$B/tmp XDG_CACHE_HOME=$B/cache PIP_CACHE_DIR=$B/cache/pip"
  grep -q '^DeepSeek_API=' ~/.bashrc 2>/dev/null && echo "  have: DeepSeek_API (not printed)" \
    || echo "  missing: add the line DeepSeek_API=<key> to ~/.bashrc yourself (not exported, as on Fir)"
  echo "next: '$0 copy' after Fir is frozen (OTUS_MIGRATION.md step 1)" ;;

copy)
  need_clone
  if [ "${SKIP_FIR_CHECK:-0}" != 1 ]; then
    n=$($SSH "$FIR" 'squeue -u $USER -h | wc -l'); d=$($SSH "$FIR" 'ssh login2 "pgrep -fc \"python3 [d]river.py\"" || true')
    echo "Fir: $n jobs in the queue, driver processes on login2: ${d:-0}"
    [ "$n" = 0 ] && [ "${d:-0}" = 0 ] || { echo "freeze Fir first (step 1: kill the driver, then scancel); SKIP_FIR_CHECK=1 overrides"; exit 1; }
  fi
  R=(rsync -a --info=progress2 -e "$SSH")
  echo "1/6 study results onto the clone (keeps .git; only Vitis internals left out)"
  "${R[@]}" --exclude .git --exclude __pycache__ \
    --exclude .autopilot/ --exclude verilog/ --exclude vhdl/ --exclude systemc/ --exclude impl/ \
    --exclude csim/build/ --exclude '*.o' --exclude '*.vcd' --exclude '*.wdb' \
    "$FIR:/scratch/asa582/workspaces/c2hls_claude_code/" "$CC/"
  echo "2/6 soda_src and dse_v4_claude"
  "${R[@]}" "$FIR:/scratch/asa582/workspaces/soda_src" "$FIR:/scratch/asa582/workspaces/dse_v4_claude" "$W/"
  echo "3/6 Vitis 2023.2 headers for the hardened bench"
  "${R[@]}" "$FIR:/scratch/asa582/tmp/c2hls/vitis_hls_2023.2_include" "$B/tmp/c2hls/"
  echo "4/6 packed SODA projects for LightningSim (3.8 GB)"
  mkdir -p "$B/tmp/c2hls/lsim"; "${R[@]}" "$FIR:/scratch/asa582/tmp/c2hls/lsim/soda" "$B/tmp/c2hls/lsim/"
  echo "5/6 Claude memory for the dse_v4_claude project"
  K=$(echo "$W/dse_v4_claude" | sed 's|[^A-Za-z0-9]|-|g'); mkdir -p ~/.claude/projects/"$K"
  "${R[@]}" "$FIR:.claude/projects/-scratch-asa582-workspaces-dse-v4-claude/memory" ~/.claude/projects/"$K/"
  echo "  -> ~/.claude/projects/$K/memory (start claude in $W/dse_v4_claude once and check the name in ~/.claude/projects/)"
  echo "6/6 optional"
  if [ "${OPTIONAL:-0}" = 1 ]; then
    "${R[@]}" "$FIR:/scratch/asa582/tools/lightningsim/src/" "$B/tools/lightningsim_fir_src/"
    "${R[@]}" "$FIR:/scratch/asa582/workspaces/AutoSA" "$W/"
  else echo "  skipped Fir's LightningSim source and AutoSA (OPTIONAL=1)"; fi
  if [ "${COPY_ENV:-0}" = 1 ]; then "${R[@]}" "$FIR:/scratch/asa582/workspaces/code_translation_c2hls/.env" "$J/.env"; chmod 600 "$J/.env"
  else echo "  skipped the judge's .env (COPY_ENV=1)"; fi
  echo "next: OTUS_LOGIN=<login node> $0 site" ;;

unpack)
  need_clone; IN=${2:-$B/transfer_from_fir}; cd "$IN"
  [ -f MANIFEST.tsv ] && [ -f SHA256SUMS ] || { echo "no MANIFEST.tsv/SHA256SUMS in $IN"; exit 1; }
  echo "checking SHA256SUMS in $IN"; sha256sum -c SHA256SUMS
  KEY=$(echo "$W/dse_v4_claude" | sed 's|[^A-Za-z0-9]|-|g')
  while IFS=$'\t' read -r f to; do
    to=${to//'$J'/$J}; to=${to//'$W'/$W}; to=${to//'$B'/$B}; to=${to//'$KEY'/$KEY}; to=${to/#\~/$HOME}
    echo "== $f -> $to"; mkdir -p "$to"; tar -xf "$IN/$f" -C "$to"
  done < MANIFEST.tsv
  [ -f "$J/.env" ] && chmod 600 "$J/.env"
  echo "memory -> ~/.claude/projects/$KEY/memory (start claude in $W/dse_v4_claude once and check the name in ~/.claude/projects/)"
  echo "study files now: $(find "$CC/" | wc -l)"
  echo "the tarballs in $IN can go once 'verify' and 'compare' pass"
  echo "next: OTUS_LOGIN=<login node> $0 site" ;;

site)
  need_clone; : "${OTUS_LOGIN:?set OTUS_LOGIN to the Otus login node the shim, proxy and driver run on}"
  cd "$CC"
  # Fir-side commands and records stay as written
  KEEP='^\./(otus/otus_setup\.sh|study/OTUS_MIGRATION\.md|study/posthoc_csim/OTUS_RUNBOOK\.md|study/judge_skills_fir_20261022/)'
  # files that contain any of the patterns; logs/ (15 GB of transcripts) and *.jsonl are left out, but not logs/start_*.sh
  files() { local a=(); for p; do a+=(-e "$p"); done
    { grep -rlI --exclude-dir=.git --exclude-dir=logs --exclude='*.jsonl' "${a[@]}" . ; grep -lI "${a[@]}" logs/start_*.sh 2>/dev/null | sed 's|^|./|'; } \
      | grep -vE "$KEEP" | sort -u || true; }
  # via placeholders, so a target path that contains a Fir path is not rewritten twice
  echo "paths";       files /scratch/asa582/workspaces /scratch/asa582/tmp \
                        | xargs -r -d '\n' sed -i -e 's|/scratch/asa582/workspaces|@OTUS_W@|g' -e 's|/scratch/asa582/tmp|@OTUS_T@|g' \
                                                -e "s|@OTUS_W@|$W|g" -e "s|@OTUS_T@|$B/tmp|g"
  echo "account";     files def-zhenman | xargs -r -d '\n' sed -i \
                        -e "/^#SBATCH/ s|--account=def-zhenman[_a-z]*|--account=$ACCT\n#SBATCH --partition=normal|" \
                        -e "s|--account=def-zhenman[_a-z]*|--account=$ACCT --partition=normal|g"
  echo "python";      files /cvmfs/soft.computecanada.ca | xargs -r -d '\n' sed -i 's|/cvmfs/soft.computecanada.ca/[^ "]*/python3|python3|g'
  # the host only: names such as logs/shim_20261020_login2/ and start_shim_38585_login2.sh stay; *.md keep Fir's history
  echo "login host";  files login2 | { grep -v '\.md$' || true; } | xargs -r -d '\n' sed -i -E "s/(^|[^A-Za-z0-9_])login2([^A-Za-z0-9_.]|\$)/\1$OTUS_LOGIN\2/g"
  otus/apply_otus_site.sh --dry-run && otus/apply_otus_site.sh
  git -C "$J" add -u
  git -C "$J" diff --cached --stat | tail -3
  git -C "$J" diff --cached --quiet || git -C "$J" commit -qm "Otus site: paths, account, python, login host, Otus tools" && git -C "$J" log --oneline -1
  echo "next: $0 venv; $0 lsim; $0 verify" ;;

venv)    need_clone; bash "$CC/otus/setup_venv_otus.sh" ;;
lsim)    need_clone; bash "$CC/cosim/lsim/otus/setup_lsim_otus.sh"; echo "then: $CC/cosim/lsim/otus/resubmit_lsim_otus.sh --check (cycles must equal Fir's)" ;;
verify)  need_clone; bash "$CC/study/posthoc_csim/resubmit_csynth_otus.sh" --check ;;
compare) need_clone; bash "$CC/study/posthoc_csim/resubmit_csynth_otus.sh" --compare ;;

dryrun)
  need_clone; cd "$CC"
  for s in study/posthoc_csim/resubmit_csynth_otus.sh study/soda/dse/resubmit_soda_otus.sh cosim/lsim/otus/resubmit_lsim_otus.sh \
           study/lud/resubmit_lud_otus.sh otus/resubmit_sd_heat3d_otus.sh xclbin/v4_systolic_st0_c1/resubmit_xclbin_otus.sh; do
    echo "== $s --dry-run"; bash "$s" --dry-run 2>&1 | tail -4 || echo "  (failed: see above)"
  done
  echo "== driver state"; module load lang Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true; python3 study/exp_hlsmith/otus_reset_state.py
  echo "the services, the driver and the real resubmits: OTUS_MIGRATION.md step 7" ;;

*) sed -n '2,24p' "$0"; exit 1 ;;
esac
