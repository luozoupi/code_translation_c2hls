"""Environment helpers for batch_parallel workers."""

from __future__ import annotations

import os


def configure_synth_env(*, cosim_timeout_s: int) -> None:
    os.environ["C2HLS_RUN_COSIM"] = "0"
    os.environ.setdefault("C2HLS_SYNTH_TIMEOUT", "7200")
    os.environ.setdefault("C2HLS_CSIM_TIMEOUT", "600")
    # Prefer gold-check TB for functional csim (dump PolyBench TB always returns 0).
    os.environ.setdefault("C2HLS_CSIM_USE_COSIM_TB", "1")
    os.environ.setdefault("C2HLS_COSIM_TIMEOUT", str(cosim_timeout_s))


def configure_cosim_env(*, cosim_timeout_s: int) -> None:
    os.environ["C2HLS_RUN_COSIM"] = "1"
    os.environ["C2HLS_COSIM_TIMEOUT"] = str(cosim_timeout_s)
    os.environ.setdefault("C2HLS_SYNTH_TIMEOUT", "7200")
    os.environ.setdefault("C2HLS_CSIM_TIMEOUT", "600")
    # Mitigate XSIM 43-3316 xelab SIGSEGV on HPC (multi-thread xelab + module env).
    # cosim_design -setup → patch xelab -mt off → sim.sh (see hls_eval.run_cosim).
    os.environ.setdefault("C2HLS_COSIM_XELAB_MT_OFF", "1")
