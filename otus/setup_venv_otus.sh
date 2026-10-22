#!/bin/bash
# setup_venv_otus.sh : on an Otus login node, make the c2hls venv the flash jobs use (Fir: /scratch/asa582/packages/c2hls-venv).
#   VENV=<dir> bash setup_venv_otus.sh     (default <projects>/packages/c2hls-venv, next to workspaces/)
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")")          # c2hls_claude_code/otus
W=$(dirname "$(dirname "$HERE")")             # workspaces
[ -f "$W/hls_eval.py" ] && W=$(dirname "$W")   # combined branch: the study is a subfolder of the judge clone
VENV=${VENV:-$(dirname "$W")/packages/c2hls-venv}
module load lang Python/3.11.5-GCCcore-13.2.0
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q -r "$HERE/requirements_c2hls_venv.txt"
cd "$W/code_translation_c2hls" && "$VENV/bin/python" -c "import c2hls, hls_eval; print('c2hls venv ok:', '$VENV')"
