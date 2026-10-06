#!/usr/bin/env python3
"""Run one DSE v4 config. This script does not submit Slurm jobs.

The n1024 bench is the csim golden. ``C2HLS_DSE_SOURCE_KERNEL`` is cleared so
the flash nest cannot enter the prompt. ``C2HLS_DSE_V3`` stays unset.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import post_flash_dse_v4 as v4

_FROZEN_MARKERS = (
    "autosa_mm_variant_sweep_20260918",
    "20260919",
)
_N1024 = REPO / "artifacts/pc2/autosa_mm_ijk_benches/n1024"


def _bench_dir() -> Path:
    ready = os.getenv("C2HLS_AUTOSA_READY_ROOT", "").strip()
    if ready:
        root = Path(ready)
        if (root / "testbench.cpp").is_file():
            return root
        nested = root / "autosa_mm"
        if (nested / "testbench.cpp").is_file():
            return nested
    return _N1024 / "autosa_mm"


def _refuse_frozen(out_dir: Path) -> None:
    text = str(out_dir)
    for marker in _FROZEN_MARKERS:
        if marker in text:
            raise SystemExit(f"refusing to write DSE v4 into {out_dir}")


def _apply_env(config_id: str) -> None:
    os.environ["C2HLS_DSE_V4"] = "1"
    os.environ["C2HLS_DSE_V4_CONFIG"] = config_id
    os.environ.setdefault("C2HLS_DSE_V4_REPAIR_ROUNDS", "12")
    os.environ["C2HLS_RUN_COSIM"] = "0"
    os.environ["C2HLS_FOCUS_GROUP"] = "0"
    os.environ["C2HLS_AUTOSA_READY_ROOT"] = str(_N1024)
    os.environ["C2HLS_SYNTH_TIMEOUT"] = str(v4.N1024_SYNTH_TIMEOUT_S)
    os.environ["C2HLS_CSIM_TIMEOUT"] = str(v4.N1024_CSIM_TIMEOUT_S)
    for key in (
        "C2HLS_DSE_V3",
        "C2HLS_DSE_V3_HARNESS",
        "C2HLS_DSE_SOURCE_KERNEL",
    ):
        os.environ.pop(key, None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pc2", action="store_true")
    parser.add_argument("--config", default="")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.pc2:
        from c2hls_paths import configure_site

        configure_site("pc2")

    config_id = args.config.strip() or os.getenv("C2HLS_DSE_V4_CONFIG", "st0_c1")
    _apply_env(config_id)
    cfg = v4.load_v4_config(config_id)
    out_dir = args.out_dir.resolve()
    _refuse_frozen(out_dir)
    bench = _bench_dir()
    header = bench / "kernel.h"
    testbench = bench / "testbench.cpp"
    if not header.is_file() or not testbench.is_file():
        raise SystemExit(f"n1024 bench is missing under {bench}")

    print(
        f"config={cfg['id']} pe={v4.pe_count(cfg)} "
        f"repairs={v4.repair_round_limit()} bench={bench}"
    )
    print("submit=no")
    if args.dry_run:
        return 0

    from c2hls import C2HLSOrchestrator, DEFAULT_MODEL_ID
    from post_flash_dse import dse_max_tokens

    model = os.getenv("C2HLS_MODEL", "").strip() or DEFAULT_MODEL_ID
    orch = C2HLSOrchestrator(
        gpt_model=model,
        turns_limitation=v4.repair_round_limit() + 1,
        max_completion_tokens=dse_max_tokens(),
    )
    payload = v4.run_v4_config(
        config=cfg,
        out_dir=out_dir,
        orchestrator=orch,
        header_code=header.read_text(encoding="utf-8"),
        header_name="kernel.h",
        testbench_code=testbench.read_text(encoding="utf-8"),
        part=v4.FPGA_PART,
        clock_ns=v4.CLOCK_NS,
    )
    print(f"legal={payload.get('legal')} attempts={len(payload.get('attempts') or [])}")
    return 0 if payload.get("legal") else 1


if __name__ == "__main__":
    raise SystemExit(main())
