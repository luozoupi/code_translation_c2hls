"""Environment helpers for batch_parallel workers."""

from __future__ import annotations

import os


def configure_synth_env(*, cosim_timeout_s: int) -> None:
    os.environ["C2HLS_RUN_COSIM"] = "0"
    os.environ.setdefault("C2HLS_SYNTH_TIMEOUT", "7200")
    os.environ.setdefault("C2HLS_CSIM_TIMEOUT", "600")
    os.environ.setdefault("C2HLS_COSIM_TIMEOUT", str(cosim_timeout_s))


def configure_cosim_env(*, cosim_timeout_s: int) -> None:
    os.environ["C2HLS_RUN_COSIM"] = "1"
    os.environ["C2HLS_COSIM_TIMEOUT"] = str(cosim_timeout_s)
    os.environ.setdefault("C2HLS_SYNTH_TIMEOUT", "7200")
    os.environ.setdefault("C2HLS_CSIM_TIMEOUT", "600")
