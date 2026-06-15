"""Emit a per-cell <bench>_hls_config.tcl that reproduces the Vitis HLS run
the c2hls orchestrator drove during the sweep.

Mirrors hls_eval.run_cosim's TCL builder exactly:
    open_project hls_proj
    set_top <top_function>
    add_files <hls_source.cpp>[ -cflags "..."]
    add_files <header.h>
    add_files -tb <testbench.cpp>[ -cflags "..."]
    add_files -tb <extra .cpp files if any>
    open_solution "sol1" -flow_target vitis
    set_part {<part>}
    create_clock -period <clock_ns> -name default
    csynth_design
    cosim_design
    exit

For multistep cells, ALSO emits steps/<i>_<step>_hls_config.tcl per step
pointing at that step's intermediate cpp.

Paths in the TCL are relative to the cell directory (where the .tcl lives),
so a collaborator can `cd` into the cell dir and run
`vitis-run --tcl --input_file <bench>_hls_config.tcl` to reproduce.

Usage:
  python3 _emit_cell_hls_config_tcl.py <results_dir> [<results_dir> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PART = "xcu280-fsvh2892-2L-e"
DEFAULT_CLOCK = 3.33


def _bench_dir_rel_from(cell_dir: Path, bench_short: str) -> str:
    """Path from cell_dir back to benchmarks/<bench_short>/ as ../../../benchmarks/...
    (cell_dir lives at <results_root>/<bench>/<setup>/; benchmarks/ is at REPO_ROOT)."""
    try:
        rel = Path("..") / ".." / ".." / "benchmarks" / f"hlsfactory_{bench_short}"
        return rel.as_posix()
    except Exception:
        return f"benchmarks/hlsfactory_{bench_short}"


def _read_metadata(bench: str) -> dict:
    """bench like 'hlsfactory_2mm'."""
    p = REPO_ROOT / "benchmarks" / bench / "metadata.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _compose_tcl(*, top_function: str, hls_src: str, header: str | None,
                 testbench: str | None, extra_tb_cpps: list[str],
                 part: str, clock_ns: float, comment_block: str) -> str:
    lines: list[str] = []
    if comment_block:
        for c in comment_block.splitlines():
            lines.append(f"# {c}")
        lines.append("")
    lines.append("open_project hls_proj")
    lines.append(f"set_top {top_function}")
    lines.append(f"add_files {hls_src}")
    if header:
        lines.append(f"add_files {header}")
    if testbench:
        lines.append(f"add_files -tb {testbench}")
    for extra in extra_tb_cpps:
        lines.append(f"add_files -tb {extra}")
    lines.append('open_solution "sol1" -flow_target vitis')
    lines.append(f"set_part {{{part}}}")
    lines.append(f"create_clock -period {clock_ns} -name default")
    lines.append("csynth_design")
    lines.append("cosim_design")
    lines.append("exit")
    lines.append("")
    return "\n".join(lines)


def emit_for_cell(cell_dir: Path) -> int:
    """Returns number of .tcl files written for this cell (1 main + N step TCLs)."""
    bench = cell_dir.parent.name  # e.g. hlsfactory_2mm
    bench_short = bench.replace("hlsfactory_", "")
    setup = cell_dir.name          # e.g. sonnet__multistep__skills
    is_multistep = "multistep" in setup

    meta = _read_metadata(bench)
    top = meta.get("hls_top") or f"kernel_{bench_short.replace('-', '_')}"
    header_name = meta.get("header_file") or f"{bench_short}.h"
    tb_name = meta.get("testbench_file") or "testbench.cpp"
    cosim_tb_name = meta.get("cosim_testbench_file") or tb_name

    bench_rel = _bench_dir_rel_from(cell_dir, bench_short)
    header_path = f"{bench_rel}/{header_name}"
    tb_path = f"{bench_rel}/{cosim_tb_name}"

    extras: list[str] = []
    # The cosim flow also brings in any .cpp from metadata.support_files
    for sf in meta.get("support_files") or []:
        if sf.endswith(".cpp"):
            extras.append(f"{bench_rel}/{sf}")

    # Main / final TCL — points at the cell's final HLS source
    main_src = f"{bench}_final.cpp" if is_multistep else f"{bench}_generated.cpp"
    main_src_path = cell_dir / main_src
    n_written = 0
    if main_src_path.exists():
        tcl_text = _compose_tcl(
            top_function=top,
            hls_src=main_src,
            header=header_path,
            testbench=tb_path,
            extra_tb_cpps=extras,
            part=DEFAULT_PART,
            clock_ns=DEFAULT_CLOCK,
            comment_block=(
                f"Auto-generated hls_config (TCL) for c2hls cell\n"
                f"  bench: {bench}\n"
                f"  setup: {setup}\n"
                f"  source: {main_src}\n"
                f"  target: {DEFAULT_PART} @ {DEFAULT_CLOCK} ns\n"
                f"To reproduce: `cd` into this cell dir and run\n"
                f"  vitis-run --tcl --input_file {bench}_hls_config.tcl"
            ),
        )
        (cell_dir / f"{bench}_hls_config.tcl").write_text(tcl_text, encoding="utf-8")
        n_written += 1

    # Per-step TCLs for multistep cells
    if is_multistep:
        steps_dir = cell_dir / "steps"
        if steps_dir.is_dir():
            # Bench dir relative to steps/ is one level deeper
            bench_rel_steps = f"../{bench_rel}"
            header_path_steps = f"{bench_rel_steps}/{header_name}"
            tb_path_steps = f"{bench_rel_steps}/{cosim_tb_name}"
            extras_steps = [f"{bench_rel_steps}/{sf.split('/')[-1] if '/' in sf else sf}"
                            for sf in (meta.get("support_files") or []) if sf.endswith(".cpp")]
            for step_cpp in sorted(steps_dir.glob("*_*.cpp")):
                step_stem = step_cpp.stem  # e.g. 0_tiling
                step_src = step_cpp.name
                tcl_text = _compose_tcl(
                    top_function=top,
                    hls_src=step_src,
                    header=header_path_steps,
                    testbench=tb_path_steps,
                    extra_tb_cpps=extras_steps,
                    part=DEFAULT_PART,
                    clock_ns=DEFAULT_CLOCK,
                    comment_block=(
                        f"Auto-generated hls_config (TCL) for c2hls multistep step\n"
                        f"  bench: {bench}\n"
                        f"  setup: {setup}\n"
                        f"  step:   {step_stem}\n"
                        f"  source: {step_src}\n"
                        f"  target: {DEFAULT_PART} @ {DEFAULT_CLOCK} ns"
                    ),
                )
                (steps_dir / f"{step_stem}_hls_config.tcl").write_text(tcl_text, encoding="utf-8")
                n_written += 1
    return n_written


def main() -> int:
    roots = [Path(a) for a in sys.argv[1:]] or [
        REPO_ROOT / "results_matrix_u280_fullcosim",
        REPO_ROOT / "results_matrix_u280_fullcosim_extended",
        REPO_ROOT / "results_matrix_u280_multistep_old_skills",
    ]
    total = 0
    for root in roots:
        if not root.is_dir():
            print(f"skip: {root} (no dir)"); continue
        n_root = 0
        for bench_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("hlsfactory_")):
            for cell_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
                n_root += emit_for_cell(cell_dir)
        print(f"{root.name}: wrote {n_root} hls_config.tcl files")
        total += n_root
    print(f"\nTotal: {total} TCL files written across {len(roots)} sweep dirs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
