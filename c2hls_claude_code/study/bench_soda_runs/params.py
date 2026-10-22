#!/usr/bin/env python3
"""--export for one SODA kernel of the Refined + Res.-Req study: bench_wide_runs/params.py with the bench_soda root
and floors_soda.json (approved 2026-10-16). Tags sd_<kernel>_<cc|ds>.
  params.py KERNEL MODEL(cc|ds) TAG
"""
import importlib.util, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bw_params", HERE.parent / "bench_wide_runs/params.py")
bw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bw)
bw.BW = HERE.parent / "bench_soda"
bw.OUT = HERE / "runs"
_orig_read = Path.read_text


def export(bench, model, tag):
    # bw.export reads <BW>/floors_final.json; point it at floors_soda.json
    def read(p, *a, **k):
        if p == bw.BW / "floors_final.json":
            p = bw.BW / "floors_soda.json"
        return _orig_read(p, *a, **k)
    Path.read_text = read
    try:
        return bw.export(bench, model, tag)
    finally:
        Path.read_text = _orig_read


if __name__ == "__main__":
    print(export(*sys.argv[1:4]))
