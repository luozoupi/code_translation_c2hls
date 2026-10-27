#!/bin/bash
# apply_frp_fir.sh : put the FRP fix (fork patch 0004, commit 6cf23e7ef on Otus) into Fir's LightningSim and reinstall it.
# Run on a Fir login node:   bash apply_frp_fir.sh
# 1. checks that no LightningSim job runs and the fork is the clean pre-FRP tree (da0dae7f...);
# 2. backs up the installed package to $T/tmp/lightningsim_site_backup_pre_frp_<date>;
# 3. git am the embedded patch, checks the tree is Otus's (6c1cbd54...), rebuilds the Rust core and reinstalls;
# 4. submits one validation job (pat_heat3d_u1 expect 3770048, pat_seidel2d_u1 expect 1067879) unless VALIDATE=0.
# Undo: cp -a the backup back over the site-packages lightningsim dir, and git -C $T/src reset --hard HEAD~1.
set -euo pipefail
T=/scratch/asa582/tools/lightningsim
TMPB=/scratch/asa582/tmp/c2hls/lsim
LSIMDIR=/scratch/asa582/workspaces/c2hls_claude_code/cosim/lsim
ROOT=${ROOT:-$TMPB/soda}
PRE=da0dae7ff11104eb2b138e07794ed4b4676a1c01
POST=6c1cbd549ab1037cc654833dab08a220ebf09df5
stamp=$(date +%Y%m%d_%H%M)

if squeue -u "$USER" -h -t R -o '%j' | grep -q lsim; then
  echo "a LightningSim job is running on Fir; reinstalling now could break it. Try again when it ends:"; squeue -u "$USER" -t R | grep lsim; exit 1
fi
TREE=$(git -C "$T/src" rev-parse 'HEAD^{tree}')
if [ "$TREE" = "$POST" ]; then echo "fork already has the FRP patch (tree $POST); only reinstalling"
else
  [ "$TREE" = "$PRE" ] || { echo "fork tree $TREE is neither pre-FRP $PRE nor $POST; stop"; exit 1; }
  [ -z "$(git -C "$T/src" status --porcelain --untracked-files=no)" ] || { echo "fork has local changes:"; git -C "$T/src" status --short; exit 1; }
  mkdir -p "$T/tmp"
  cat > "$T/tmp/0004-frp.patch" <<'EOF_FRP_PATCH'
From 6cf23e7ef1642195271bc56b28682d8bc98d46a9 Mon Sep 17 00:00:00 2001
From: "Claude (for asa582)" <asa582@localhost>
Date: Mon, 26 Oct 2026 19:43:54 +0100
Subject: [PATCH] frp: read-after-write +1 on FIFOs read by Vitis free-running
 pipelines

FRP pipelines issue only when num_data_valid covers outstanding requests + 1 and read at stage 1. Input FIFOs are found
from ap_frp_data_req_* in the RTL (lightningsim/frp.py); LSIM_FRP=0 turns it off. SODA heat3d/jacobi3d vs Vitis 2023.2
cosim: -14..-20% before, +0.003..+0.047% after; designs without FRP unchanged.
---
 backend/lightningsim-core/edge.rs  |  2 +-
 backend/lightningsim-core/lib.rs   |  7 ++++++
 backend/lightningsim/frp.py        | 35 ++++++++++++++++++++++++++++++
 backend/lightningsim/main.py       |  1 +
 backend/lightningsim/runner.py     |  2 ++
 backend/lightningsim/trace_file.py |  2 ++
 6 files changed, 48 insertions(+), 1 deletion(-)
 create mode 100644 backend/lightningsim/frp.py

diff --git a/backend/lightningsim-core/edge.rs b/backend/lightningsim-core/edge.rs
index 240f58823..a30ffeff4 100644
--- a/backend/lightningsim-core/edge.rs
+++ b/backend/lightningsim-core/edge.rs
@@ -95,7 +95,7 @@ impl Edge {
                 let width = parameters.get_fifo_width(fifo)?;
                 let depth = parameters.get_fifo_depth(fifo)?;
                 let fifo_type = FifoType::from_width_and_depth(width, depth);
-                Ok(Some(u + fifo_type.raw_delay()))
+                Ok(Some(u + (fifo_type.raw_delay() + parameters.get_fifo_raw_extra(fifo))))
             }
             Edge::FifoWar { fifo, index } => {
                 let width = parameters.get_fifo_width(fifo)?;
diff --git a/backend/lightningsim-core/lib.rs b/backend/lightningsim-core/lib.rs
index c7a47e7f8..e47e47294 100644
--- a/backend/lightningsim-core/lib.rs
+++ b/backend/lightningsim-core/lib.rs
@@ -77,6 +77,8 @@ pub struct SimulationParameters {
     fifo_widths: FxHashMap<FifoId, u32>,
     axi_delays: FxHashMap<AxiAddress, ClockCycle>,
     ap_ctrl_chain_top_port_count: Option<u32>,
+    /// Extra read-after-write delay per FIFO (FIFOs read by a Vitis free-running pipeline get 1, see frp.py).
+    fifo_raw_extra: FxHashMap<FifoId, ClockCycle>,
 }
 
 #[pyclass(module = "lightningsim._core")]
@@ -258,6 +260,7 @@ impl CompiledSimulation {
                         fifo_widths: base_config.fifo_widths.clone(),
                         axi_delays: base_config.axi_delays.clone(),
                         ap_ctrl_chain_top_port_count: base_config.ap_ctrl_chain_top_port_count,
+                        fifo_raw_extra: base_config.fifo_raw_extra.clone(),
                     };
                     let latency = match self.resolve(&parameters) {
                         Ok(node_cycles) => Ok(Some(self.top_module.end.resolve(&node_cycles))),
@@ -423,6 +426,10 @@ impl SimulationParameters {
             .ok_or(SimulationError::FifoWidthNotProvided(fifo.id))
     }
 
+    pub(crate) fn get_fifo_raw_extra(&self, fifo: Fifo) -> ClockCycle {
+        self.fifo_raw_extra.get(&fifo.id).copied().unwrap_or(0)
+    }
+
     pub(crate) fn get_axi_delay(
         &self,
         interface: AxiInterface,
diff --git a/backend/lightningsim/frp.py b/backend/lightningsim/frp.py
new file mode 100644
index 000000000..f134d5279
--- /dev/null
+++ b/backend/lightningsim/frp.py
@@ -0,0 +1,35 @@
+"""Vitis HLS free-running pipelines (FRP, `frp_pipeline_valid` + `frp_fifoout` in the RTL).
+
+An FRP pipeline issues an iteration only when every input FIFO's num_data_valid covers the reads it has already
+requested (`ap_frp_data_req_<fifo>`) plus this one, and it reads one cycle later (stage 1). An item therefore has to be
+in the FIFO one cycle earlier than for a stall pipeline: read-after-write delay + 1 on every FIFO an FRP pipeline reads.
+Checked against Vitis 2023.2 cosim on the SODA heat3d/jacobi3d designs (10 designs, +0.003% to +0.047%).
+LSIM_FRP=0 turns it off.
+"""
+
+import os
+import re
+import sys
+from pathlib import Path
+from typing import Dict, Iterable
+
+_REQ = re.compile(r"reg\s+\[\d+:0\]\s+ap_frp_data_req_(\w+?)(?<!_op\d)(?<!_op\d\d)(?<!_op\d\d\d);")
+
+
+def frp_input_fifos(solution_dir: Path) -> set[str]:
+    names: set[str] = set()
+    for v in (solution_dir / "syn" / "verilog").glob("*.v"):
+        names |= set(_REQ.findall(v.read_text(errors="replace")))
+    return names
+
+
+def frp_raw_extra(solution_dir: Path, fifos: Iterable) -> Dict[int, int]:
+    if os.environ.get("LSIM_FRP", "1") == "0":
+        return {}
+    rtl = frp_input_fifos(Path(solution_dir))
+    if not rtl:
+        return {}
+    extra = {f.id: 1 for f in fifos if any(f.name == n or f.name.endswith(n) for n in rtl)}
+    print(f"[lsim-frp] {len(extra)} FIFOs read by free-running pipelines (RTL names {len(rtl)}): read-after-write +1",
+          file=sys.stderr)
+    return extra
diff --git a/backend/lightningsim/main.py b/backend/lightningsim/main.py
index 010268bed..30205a26b 100755
--- a/backend/lightningsim/main.py
+++ b/backend/lightningsim/main.py
@@ -338,6 +338,7 @@ class Server:
                 fifo_widths=self.trace.params.fifo_widths,
                 axi_delays=self.trace.params.axi_delays,
                 ap_ctrl_chain_top_port_count=self.trace.params.ap_ctrl_chain_top_port_count,
+                fifo_raw_extra=self.trace.params.fifo_raw_extra,
             ),
             fifos=self.trace.fifos,
             axi_interfaces=self.trace.axi_interfaces,
diff --git a/backend/lightningsim/runner.py b/backend/lightningsim/runner.py
index 171e26ed7..cb9757097 100644
--- a/backend/lightningsim/runner.py
+++ b/backend/lightningsim/runner.py
@@ -35,6 +35,7 @@ from typing import (
     TypeVar,
 )
 from .model import Solution, ProjectFile
+from .frp import frp_raw_extra
 from .trace_file import ResolvedTrace, await_trace_functions, read_trace, resolve_trace
 
 CONDA_PREFIX = Path(r"""/opt/anaconda1anaconda2anaconda3""")
@@ -606,6 +607,7 @@ class Runner:
 
         with self.steps[RunnerStep.RESOLVING_TRACE] as step:
             trace = await resolve_trace(trace, progress_callback=step.set_progress)
+            trace.params.fifo_raw_extra = frp_raw_extra(self.solution.path, trace.fifos)
             self.trace = trace
 
         return trace
diff --git a/backend/lightningsim/trace_file.py b/backend/lightningsim/trace_file.py
index 787c620cb..e8c0a70ed 100644
--- a/backend/lightningsim/trace_file.py
+++ b/backend/lightningsim/trace_file.py
@@ -314,6 +314,8 @@ class SimulationParameters:
     fifo_widths: Mapping[int, int]
     axi_delays: Mapping[int, int]
     ap_ctrl_chain_top_port_count: int | None
+    # extra read-after-write delay per FIFO id (Vitis free-running pipelines, see frp.py)
+    fifo_raw_extra: Mapping[int, int] = field(default_factory=dict)
 
 
 @dataclass(slots=True)
-- 
2.52.0

EOF_FRP_PATCH
  git -C "$T/src" -c user.name="$USER" -c user.email="$USER@localhost" am "$T/tmp/0004-frp.patch"
  TREE=$(git -C "$T/src" rev-parse 'HEAD^{tree}')
  [ "$TREE" = "$POST" ] || { echo "after git am the tree is $TREE, expected $POST; stop (undo: git -C $T/src reset --hard HEAD~1)"; exit 1; }
  echo "fork patched: $(git -C "$T/src" log --oneline -1)"
fi

SITE=$("$T/env/bin/python" -c 'import lightningsim, os; print(os.path.dirname(lightningsim.__file__))')
BK="$T/tmp/lightningsim_site_backup_pre_frp_$stamp"
cp -a "$SITE" "$BK"; echo "backup of $SITE -> $BK"

# same steps as cosim/lsim/build_lsim.sh, with Fir paths
export PATH="$T/rustenv/bin:$T/env/bin:$PATH"
export CARGO_HOME="$T/cargo" CARGO_TARGET_DIR="$TMPB/cargo-target" TMPDIR="$TMPB/tmp"
unset PYTHONPATH PIP_CONFIG_FILE
mkdir -p "$TMPDIR"
cd "$T/src/backend"
sed -i "s#^CONDA_PREFIX = Path(r\"\"\"[^\"]*\"\"\")#CONDA_PREFIX = Path(r\"\"\"$T/env\"\"\")#" lightningsim/runner.py
"$T/env/bin/python" -m pip install --no-deps --no-build-isolation --ignore-installed . 2>&1 | tail -n 3
sed -i "s#^CONDA_PREFIX = Path(r\"\"\"$T/env\"\"\")#CONDA_PREFIX = Path(r\"\"\"/opt/anaconda1anaconda2anaconda3\"\"\")#" lightningsim/runner.py
cd /
"$T/env/bin/python" - <<'EOF'
import dataclasses, lightningsim._core as c, lightningsim.runner as r, lightningsim.frp as f
from lightningsim.trace_file import SimulationParameters
assert "fifo_raw_extra" in {x.name for x in dataclasses.fields(SimulationParameters)}, "trace_file.py not patched"
assert b"fifo_raw_extra" in open(c.__file__, "rb").read(), "Rust core was not rebuilt with fifo_raw_extra"
print("installed with FRP rule:", c.__file__, r.CONDA_PREFIX)
EOF

[ "${VALIDATE:-1}" = 1 ] || exit 0
for n in pat_heat3d_u1 pat_seidel2d_u1; do [ -f "$ROOT/$n/proj.tar.gz" ] || { echo "no $ROOT/$n/proj.tar.gz; skip the validation job (set ROOT=...)"; exit 0; }; done
J=$(sbatch --parsable --export=ALL,ROOT="$ROOT",ONLY=pat_heat3d_u1:pat_seidel2d_u1,TAG=frpfir,PAR=2 "$LSIMDIR/sbatch_lsim.sh")
echo "validation job $J. When it ends:"
echo "  grep -h '\"end\"' $ROOT/pat_heat3d_u1/lsim_frpfir.json $ROOT/pat_seidel2d_u1/lsim_frpfir.json   # expect 3770048 and 1067879"
echo "  grep -h lsim-frp $ROOT/pat_heat3d_u1/lsim_frpfir.log                                          # expect 35 FIFOs"
