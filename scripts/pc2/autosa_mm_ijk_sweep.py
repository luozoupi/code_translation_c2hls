#!/usr/bin/env python3
"""Stage, submit, and harvest flash / DSE v1 / DSE v2 at I=J=K in {512,1024,2048,4096}.

Does not edit related_work/benchmarks/autosa_ready/autosa_mm (the live 64³ bench).
Does not write autosa_mm_variant_sweep_20260918.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from autosa_flash_lib import resolve_autosa_ready_root as _flash_ready_root  # noqa: E402
from autosa_mm_variant_sweep import (  # noqa: E402
    DEFAULT_ENDPOINT,
    cell_complete,
    find_variant_cell,
    inflight_count,
    load_cells,
    parse_waves,
    qor_from_report,
    refuse_frozen,
    run_sweep,
    stage_report_paths,
    write_cells,
)
from harvest_autosa_mm_variant_sweep import harvest_cell  # noqa: E402

IJK_SIZES = (512, 1024, 2048, 4096)
SWEEP_STAMP = "20260923"
SOURCE_BENCH = REPO / "related_work" / "benchmarks" / "autosa_ready" / "autosa_mm"
BENCHES_ROOT = REPO / "artifacts" / "pc2" / "autosa_mm_ijk_benches"
SWEEP_PARENT = REPO / "artifacts" / "pc2" / f"autosa_mm_ijk_sweep_{SWEEP_STAMP}"

_DEFINE_RE = re.compile(r"^(\s*)#\s*define\s+([IJK])\s+(\d+)\b")

HEAP_TESTBENCH = """#include "kernel.h"
extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]);

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  data_t (*A)[K] = (data_t (*)[K])malloc(sizeof(data_t) * (size_t)I * (size_t)K);
  data_t (*B)[K] = (data_t (*)[K])malloc(sizeof(data_t) * (size_t)J * (size_t)K);
  data_t (*C)[J] = (data_t (*)[J])malloc(sizeof(data_t) * (size_t)I * (size_t)J);
  data_t (*C_golden)[J] = (data_t (*)[J])malloc(sizeof(data_t) * (size_t)I * (size_t)J);
  if (!A || !B || !C || !C_golden) {
    printf("Failed to allocate\\n");
    return 1;
  }

  for (int i = 0; i < I; i++)
    for (int k = 0; k < K; k++) {
      A[i][k] = (data_t)rand() / RAND_MAX;
    }

  for (int j = 0; j < J; j++)
    for (int k = 0; k < K; k++) {
      B[j][k] = (data_t)rand() / RAND_MAX;
    }

  autosa_mm(A, B, C);

  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {
      C_golden[i][j] = 0;
      for (int k = 0; k < K; k++) {
        C_golden[i][j] = C_golden[i][j] + A[i][k] * B[j][k];
      }
    }

  int err = 0;
  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {
      if (fabs((float)C_golden[i][j] - (float)C[i][j]) > 0.001)
        err++;
    }

  free(A);
  free(B);
  free(C);
  free(C_golden);

  if (err)
    printf("Failed with %d errors!\\n", err);
  else
    printf("Passed!\\n");

  return err ? 1 : 0;
}
"""

BEST64 = {
    "flash": {
        "cell_id": "flash_90_f1_r08",
        "latency_cycles_worst": 3135,
        "dsp": 384,
        "note": "20260918 harvest",
    },
    "dse_v1": {
        "cell_id": "dse1_gf_f0_r02",
        "latency_cycles_worst": 5590,
        "dsp": 352,
        "note": "20260918 harvest; predecessor flash_gf_f0_r02",
    },
    "dse_v2": {
        "cell_id": "pe64_simd16",
        "latency_cycles_worst": 1113,
        "dsp": 5248,
        "note": "20260914_084009 real grid; 20260918 sweep DSE v2 cells were flash clones",
    },
}


def resolve_autosa_ready_root() -> Path:
    return _flash_ready_root()


def ready_root_for(n: int, benches_root: Path = BENCHES_ROOT) -> Path:
    return Path(benches_root) / f"n{int(n)}"


def sweep_root_for_n(n: int, repo: Path = REPO) -> Path:
    root = Path(repo) / "artifacts" / "pc2" / f"autosa_mm_ijk_sweep_{SWEEP_STAMP}" / f"n{int(n)}"
    refuse_frozen(root)
    return root


_IJK_USE_RE = re.compile(r"\b[IJK]\b")


def rewrite_ijk_defines(text: str, n: int) -> str:
    """Comment out active I/J/K defines and insert the new ones before first use.

    The signature `A[I][K]` is not a use of a later `#define`. Appending the
    active defines at EOF leaves I/J/K undeclared and the gold csynth fails.
    """
    rewritten: list[str] = []
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("//"):
            rewritten.append(line)
            continue
        m = _DEFINE_RE.match(line)
        if m:
            rewritten.append(f"{m.group(1)}//{m.group(0).strip()}\n")
            continue
        rewritten.append(line)
    block = f"#define I {n}\n#define J {n}\n#define K {n}\n"
    out: list[str] = []
    inserted = False
    for line in rewritten:
        stripped = line.lstrip()
        if (
            not inserted
            and stripped
            and not stripped.startswith("//")
            and _IJK_USE_RE.search(line)
        ):
            if out and not out[-1].endswith("\n"):
                out.append("\n")
            out.append(block)
            inserted = True
        out.append(line)
    if not inserted:
        if out and not out[-1].endswith("\n"):
            out.append("\n")
        out.append("\n" + block)
    return "".join(out)


def stage_ijk_bench(n: int, dest_root: Path, *, source: Path = SOURCE_BENCH) -> Path:
    if int(n) not in IJK_SIZES and int(n) != 64:
        if int(n) < 1:
            raise ValueError(f"invalid IJK {n}")
    dest = Path(dest_root) / f"n{int(n)}" / "autosa_mm"
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "plain.cpp",
        "gold_hls_source.cpp",
        "hls_baseline.cpp",
        "kernel.h",
        "metadata.json",
    ):
        src = source / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    header = rewrite_ijk_defines((source / "kernel.h").read_text(encoding="utf-8"), int(n))
    (dest / "kernel.h").write_text(header, encoding="utf-8")
    (dest / "testbench.cpp").write_text(HEAP_TESTBENCH, encoding="utf-8")
    meta_path = dest / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    meta["csim_timeout_s"] = 86400 if int(n) >= 1024 else 7200
    meta["synth_timeout_s"] = 86400 if int(n) >= 1024 else 28800
    meta["ijk"] = int(n)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return dest


def stage_all_benches(benches_root: Path = BENCHES_ROOT) -> list[Path]:
    refuse_frozen(benches_root)
    return [stage_ijk_bench(n, benches_root) for n in IJK_SIZES]


def pick_best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    legal: list[dict[str, Any]] = []
    for row in rows:
        if (row.get("csim") or "") == "fail":
            continue
        worst = row.get("latency_cycles_worst")
        if worst is None:
            continue
        legal.append(row)
    if not legal:
        return None
    return min(
        legal,
        key=lambda r: (
            int(r["latency_cycles_worst"]),
            int(r.get("dsp") if r.get("dsp") is not None else 10**9),
            str(r.get("cell_id") or ""),
        ),
    )


def _dse_v2_trial_id(campaign_root: Path) -> str:
    variant = find_variant_cell(campaign_root)
    if variant is None:
        return ""
    for name in (
        "autosa_mm_post_flash_dse_v2.json",
        "autosa_mm_dse_result.json",
    ):
        path = variant / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        winner = str(data.get("winner_trial_id") or "")
        if winner.startswith("pe") and "simd" in winner:
            return winner
    return ""


def collect_size_rows(sweep_root: Path) -> list[dict[str, Any]]:
    cells_path = sweep_root / "cells.jsonl"
    if not cells_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for cell in load_cells(cells_path):
        harvested = harvest_cell(cell)
        trial = ""
        if cell.get("family") == "dse_v2":
            trial = _dse_v2_trial_id(Path(cell.get("campaign_root") or ""))
        if harvested:
            for item in harvested:
                item["cell_id"] = cell.get("cell_id")
                item["dse_v2_trial"] = trial
                rows.append(item)
            continue
        rows.append(
            {
                "cell_id": cell.get("cell_id"),
                "family": cell.get("family"),
                "skill": cell.get("skill"),
                "floor": int(cell.get("floor") or 0),
                "rep": int(cell.get("rep") or 0),
                "latency_cycles_worst": None,
                "dsp": None,
                "csim": "",
                "dse_v2_trial": trial,
                "status": cell.get("status") or "pending",
            }
        )
    return rows


def harvest_size(sweep_root: Path) -> dict[str, Any]:
    rows = collect_size_rows(Path(sweep_root))
    fail_counts: dict[str, int] = {"flash": 0, "dse_v1": 0, "dse_v2": 0}
    pending_counts: dict[str, int] = {"flash": 0, "dse_v1": 0, "dse_v2": 0}
    by_family: dict[str, list[dict[str, Any]]] = {"flash": [], "dse_v1": [], "dse_v2": []}
    skill_floor: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        family = str(row.get("family") or "")
        if family not in by_family:
            continue
        ok = (
            row.get("latency_cycles_worst") is not None
            and (row.get("csim") or "") != "fail"
            and (family != "dse_v2" or bool(row.get("dse_v2_trial")))
        )
        if not ok:
            status = str(row.get("status") or "")
            no_qor = row.get("latency_cycles_worst") is None and (row.get("csim") or "") == ""
            if no_qor and status not in {"complete", "failed"}:
                pending_counts[family] += 1
            else:
                fail_counts[family] += 1
            continue
        by_family[family].append(row)
        key = f"{family}|{row.get('skill')}|{int(row.get('floor') or 0)}"
        skill_floor.setdefault(key, []).append(row)
    return {
        "best": {
            "flash": pick_best(by_family["flash"]),
            "dse_v1": pick_best(by_family["dse_v1"]),
            "dse_v2": pick_best(by_family["dse_v2"]),
        },
        "best_skill_floor": {
            key: pick_best(group) for key, group in sorted(skill_floor.items())
        },
        "fail_counts": fail_counts,
        "pending_counts": pending_counts,
        "n_rows": len(rows),
        "rows": rows,
    }


def harvest_all(parent: Path = SWEEP_PARENT) -> dict[str, Any]:
    sizes: dict[str, Any] = {
        "64": {"best": BEST64, "fail_counts": {}, "note": "existing 64³; not rerun"}
    }
    for n in IJK_SIZES:
        sizes[str(n)] = harvest_size(sweep_root_for_n(n, repo=REPO))
        sizes[str(n)].pop("rows", None)
    doc = {
        "schema": "autosa_mm_ijk_sweep_harvest_v1",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "score": "min latency_cycles_worst, then min DSP",
        "sizes": sizes,
    }
    return doc


def submit_size(
    n: int,
    *,
    reps: int,
    endpoint: str,
    max_inflight: int,
    submit: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    ready = ready_root_for(n)
    if not (ready / "autosa_mm" / "metadata.json").is_file():
        stage_ijk_bench(n, ready.parent)
    os.environ["C2HLS_AUTOSA_READY_ROOT"] = str(ready)
    os.environ["C2HLS_COMPUTE_WALLTIME"] = "2:00:00"
    os.environ["PC2_BATCH_PARALLEL_WALLTIME"] = "2:00:00"
    # Login-node proxy, not a Slurm allocation. Outlive a 2-day csim that
    # makes no LLM call, so the proxy does not exit and force a new billed call.
    os.environ["CHATHLS_DEEPSEEK_IDLE_EXIT_S"] = "259200"
    root = sweep_root_for_n(n)
    refuse_frozen(root)
    return run_sweep(
        date=SWEEP_STAMP,
        reps=reps,
        waves=parse_waves("flash,dse"),
        endpoint=endpoint,
        max_inflight=max_inflight,
        dry_run=dry_run,
        submit=submit and not dry_run,
        sweep_root=root,
        prefix_tag=f"n{n}",
    )


def follow_sizes(
    sizes: tuple[int, ...] = IJK_SIZES,
    *,
    reps: int,
    endpoint: str,
    max_inflight: int,
    sleep_s: int = 120,
) -> None:
    while True:
        pending = 0
        for n in sizes:
            cells = submit_size(
                n,
                reps=reps,
                endpoint=endpoint,
                max_inflight=max_inflight,
                submit=True,
                dry_run=False,
            )
            selected = [
                c
                for c in cells
                if c.get("family") in {"flash", "dse_v1", "dse_v2"}
            ]
            n_complete = sum(1 for c in selected if c.get("status") == "complete" or cell_complete(c))
            n_failed = sum(1 for c in selected if int(c.get("fail_count") or 0) >= 3)
            n_pending = len(selected) - n_complete - n_failed
            pending += n_pending
            print(
                f"n={n} complete={n_complete} pending={n_pending} failed={n_failed} "
                f"inflight={inflight_count(cells)}"
            )
        if pending <= 0:
            print("follow: no pending IJK cells left")
            return
        time.sleep(max(15, int(sleep_s)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--harvest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--max-inflight", type=int, default=80)
    parser.add_argument("--sleep", type=int, default=120)
    parser.add_argument("--endpoint-url", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--sizes",
        default=",".join(str(n) for n in IJK_SIZES),
        help="comma-separated I=J=K values",
    )
    args = parser.parse_args()
    sizes = tuple(int(p.strip()) for p in args.sizes.split(",") if p.strip())
    if args.stage_only or args.submit or args.follow or args.dry_run:
        stage_all_benches()
        print(f"staged benches under {BENCHES_ROOT}")
        source_h = SOURCE_BENCH / "kernel.h"
        text = source_h.read_text(encoding="utf-8")
        if "#define I 64" not in text:
            raise SystemExit(f"live 64³ header changed unexpectedly: {source_h}")
    if args.dry_run or (args.submit and not args.follow):
        for n in sizes:
            submit_size(
                n,
                reps=args.reps,
                endpoint=args.endpoint_url,
                max_inflight=args.max_inflight,
                submit=args.submit,
                dry_run=args.dry_run,
            )
    if args.follow and not args.dry_run:
        follow_sizes(
            sizes,
            reps=args.reps,
            endpoint=args.endpoint_url,
            max_inflight=args.max_inflight,
            sleep_s=args.sleep,
        )
    if args.harvest or args.follow:
        doc = harvest_all()
        out = SWEEP_PARENT / "ijk_sweep_harvest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"harvest={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
