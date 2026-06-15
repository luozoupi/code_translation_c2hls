"""Generalized version of _strip_syrk_from_matrix.py — pass bench name as arg."""
import json
import shutil
import sys
import time
from pathlib import Path

if len(sys.argv) < 2:
    print("usage: _strip_bench_from_matrix.py <bench> [<matrix_dir>]")
    sys.exit(2)
bench = sys.argv[1]
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "results_matrix_u280_multistep_old_skills")
matrix = OUT / "matrix.json"

ts = time.strftime("%Y%m%d_%H%M%S")
backup = matrix.with_name(f"matrix.json.bak.{ts}")
shutil.copy2(matrix, backup)
print(f"backup: {backup}")

m = json.loads(matrix.read_text())
before = len(m)
m = [e for e in m if e.get("bench") != bench]
print(f"removed {before - len(m)} {bench} entries (was {before}, now {len(m)})")
matrix.write_text(json.dumps(m, indent=2, default=str))

cell = OUT / bench
if cell.exists():
    shutil.rmtree(cell)
    print(f"removed {cell}")
