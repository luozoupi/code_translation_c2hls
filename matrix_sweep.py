"""
Run a comparison matrix over (benchmark, model, mode, skills) cells.

Each cell launches c2hls.py as a subprocess so failures don't cross-contaminate.
Per-cell results are saved to results_matrix/<bench>/<model>__<mode>__<skills>/
and aggregated into results_matrix/matrix.json at the top level.

Usage examples:
    # default small slice — 2mm only, both models, single-shot + multistep,
    # with and without skills (8 cells)
    python3 matrix_sweep.py

    # broader: 3 benches
    python3 matrix_sweep.py --benches hlsfactory_2mm,hlsfactory_atax,hlsfactory_gemm

    # only one dimension
    python3 matrix_sweep.py --skip-multistep --skip-noskills

Env vars needed at runtime (the script inherits them):
    ANTHROPIC_BASE_URL=https://apps.inside.anl.gov/argoapi   (for Argo)
    VITIS_SETTINGS=/tools/Xilinx/Vitis/2023.2/settings64.sh
    C2HLS_SKILLS_PATH=<path-to-skills.json>                  (auto-set for skill cells)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parent
DEFAULT_SKILLS_PATH = "/mnt/e/courses/UMN/c2hls/hls_full_optimization_skills_schema_1_1_package/skills.json"

# Short labels for filesystem-friendly output dirs
MODEL_LABEL = {
    "claude-haiku-4-5-20251001": "haiku",
    "claude-sonnet-4-6": "sonnet",
    "claude-opus-4-8": "opus",
    # ALCF Sophia
    "meta-llama/Meta-Llama-3.1-70B-Instruct": "llama3.1-70b",
    "meta-llama/Llama-3.3-70B-Instruct": "llama3.3-70b",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "llama4-scout",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": "llama4-mav",
    "mistralai/Mistral-Large-Instruct-2407": "mistral-large",
    "mistralai/Mixtral-8x22B-Instruct-v0.1": "mixtral-8x22b",
    "openai/gpt-oss-120b": "gpt-oss-120b",
    "openai/gpt-oss-20b": "gpt-oss-20b",
    "google/gemma-3-27b-it": "gemma3-27b",
    "argonne/AuroraGPT-IT-v4-0125": "auroragpt",
}


def _label_model(model_id: str) -> str:
    return MODEL_LABEL.get(model_id, model_id.replace("/", "_"))


def _cell_dir(out_root: Path, bench: str, model_id: str, mode: str, with_skills: bool) -> Path:
    model = _label_model(model_id)
    skills_tag = "skills" if with_skills else "noskills"
    return out_root / bench / f"{model}__{mode}__{skills_tag}"


def _run_cell(bench: str, model_id: str, mode: str, with_skills: bool,
              out_root: Path, turns: int, skills_path: str,
              cell_timeout: int = 3600,
              quality_repair_turns: int | None = None) -> dict:
    cell_dir = _cell_dir(out_root, bench, model_id, mode, with_skills)
    cell_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(REPO / "c2hls.py"),
        "--bench", bench,
        "--model", model_id,
        "--turns", str(turns),
        "--output-dir", str(cell_dir),
    ]
    if quality_repair_turns is not None:
        cmd += ["--quality-repair-turns", str(quality_repair_turns)]
    if mode == "multistep":
        cmd.append("--multistep")

    env = os.environ.copy()
    if with_skills:
        env["C2HLS_SKILLS_PATH"] = skills_path
    else:
        env.pop("C2HLS_SKILLS_PATH", None)

    log_path = cell_dir / "matrix_run.log"
    t0 = time.time()
    with log_path.open("w") as logf:
        try:
            result = subprocess.run(
                cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                cwd=str(REPO), check=False, timeout=cell_timeout,
            )
            elapsed = round(time.time() - t0, 1)
            status = "ok" if result.returncode == 0 else f"exit_{result.returncode}"
        except subprocess.TimeoutExpired:
            elapsed = round(time.time() - t0, 1)
            status = f"timeout_{cell_timeout}s"

    # Pick up the per-cell result JSON if c2hls produced one
    summary: dict = {}
    if mode == "multistep":
        candidates = list(cell_dir.glob("*_multistep_results.json"))
    else:
        candidates = list(cell_dir.glob("*_results.json"))
    if candidates:
        try:
            summary = json.loads(candidates[0].read_text())
        except Exception:
            summary = {"_parse_error": str(candidates[0])}

    return {
        "bench": bench,
        "model": model_id,
        "mode": mode,
        "skills": "on" if with_skills else "off",
        "status": status,
        "wallclock_s": elapsed,
        "cell_dir": str(cell_dir),
        "summary": summary,
    }


def _extract_metrics(cell: dict) -> dict:
    """Pull a flat set of comparable metrics from per-cell summary."""
    s = cell.get("summary") or {}
    rep = s.get("synth_report") or s.get("final_report") or s.get("report") or {}
    cmp = (s.get("comparison") or {}).get("comparison") if isinstance(s.get("comparison"), dict) else {}
    if not isinstance(cmp, dict):
        cmp = {}

    def _f(x):
        try:
            return float(x) if x is not None else None
        except (TypeError, ValueError):
            return None

    def _ratio(key: str):
        return _f((cmp.get(key) or {}).get("ratio"))

    return {
        "phase_success": s.get("success"),
        "phase": s.get("phase"),
        "latency_cycles": _f(rep.get("latency_cycles")),
        "latency_ns": _f(rep.get("latency_ns")),
        "fmax_mhz": _f(rep.get("fmax_mhz")),
        "slack_ns": _f(rep.get("slack_ns")),
        "bram": _f(rep.get("bram")),
        "dsp": _f(rep.get("dsp")),
        "ff": _f(rep.get("ff")),
        "lut": _f(rep.get("lut")),
        "ratio_latency_ns": _ratio("latency_ns"),
        "ratio_fmax": _ratio("fmax_mhz"),
        "ratio_bram": _ratio("bram"),
        "ratio_dsp": _ratio("dsp"),
        "ratio_ff": _ratio("ff"),
        "ratio_lut": _ratio("lut"),
    }


def _print_table(rows: list[dict]):
    cols = ["bench", "model", "mode", "skills", "status", "ok", "lat_cyc", "lat_ns", "fmax", "bram", "dsp", "ff", "lut", "secs"]
    widths = {c: max(len(c), 5) for c in cols}
    formatted = []
    for r in rows:
        m = _extract_metrics(r)
        row = {
            "bench": r["bench"].replace("hlsfactory_", "hf_"),
            "model": _label_model(r["model"]),
            "mode": r["mode"],
            "skills": r["skills"],
            "status": r["status"],
            "ok": "Y" if m.get("phase_success") else "N",
            "lat_cyc": str(int(m["latency_cycles"])) if m.get("latency_cycles") else "-",
            "lat_ns": f"{m['latency_ns']:.0f}" if m.get("latency_ns") is not None else "-",
            "fmax": f"{m['fmax_mhz']:.1f}" if m.get("fmax_mhz") else "-",
            "bram": str(int(m["bram"])) if m.get("bram") is not None else "-",
            "dsp": str(int(m["dsp"])) if m.get("dsp") is not None else "-",
            "ff": str(int(m["ff"])) if m.get("ff") is not None else "-",
            "lut": str(int(m["lut"])) if m.get("lut") is not None else "-",
            "secs": f"{r['wallclock_s']:.0f}",
        }
        formatted.append(row)
        for c in cols:
            widths[c] = max(widths[c], len(row[c]))
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for row in formatted:
        print(" | ".join(row[c].ljust(widths[c]) for c in cols))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benches", default="hlsfactory_2mm",
                   help="comma-separated benchmark names")
    p.add_argument("--models", default="claude-haiku-4-5-20251001,claude-sonnet-4-6",
                   help="comma-separated model IDs")
    p.add_argument("--modes", default="flash,multistep",
                   help="comma-separated: flash (single-shot), multistep")
    p.add_argument("--skills-modes", default="on,off",
                   help="comma-separated: on, off")
    p.add_argument("--turns", type=int, default=2)
    p.add_argument("--quality-repair-turns", type=int, default=None,
                   help="If set, forwarded to c2hls.py as --quality-repair-turns")
    p.add_argument("--skills-path", default=DEFAULT_SKILLS_PATH)
    p.add_argument("--out", default="results_matrix",
                   help="output root dir under repo")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cell-timeout", type=int, default=3600,
                   help="per-cell wallclock cap in seconds (default 3600)")
    args = p.parse_args()

    benches = [b.strip() for b in args.benches.split(",") if b.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    skills_modes = [s.strip() for s in args.skills_modes.split(",") if s.strip()]

    out_root = REPO / args.out
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    cells = list(itertools.product(benches, models, modes, skills_modes))
    print(f"Matrix: {len(cells)} cells")
    print(f"  benches={benches}  models={[_label_model(m) for m in models]}  modes={modes}  skills={skills_modes}")
    print(f"  out_root={out_root}")
    if args.dry_run:
        for b, m, mode, sk in cells:
            cd = _cell_dir(out_root, b, m, mode, sk == "on")
            print(f"  - {b} | {_label_model(m)} | {mode} | skills={sk} -> {cd.name}")
        return

    matrix_path = out_root / "matrix.json"
    results: list[dict] = []
    if matrix_path.exists():
        try:
            results = json.loads(matrix_path.read_text())
        except Exception:
            results = []
    done_keys = {(r["bench"], r["model"], r["mode"], r["skills"]) for r in results}

    for i, (b, m, mode, sk) in enumerate(cells, 1):
        with_skills = (sk == "on")
        key = (b, m, mode, "on" if with_skills else "off")
        if key in done_keys:
            print(f"[{i}/{len(cells)}] SKIP already done: {key}")
            continue
        label = f"{b} | {_label_model(m)} | {mode} | skills={sk}"
        print(f"[{i}/{len(cells)}] START {label}", flush=True)
        cell = _run_cell(b, m, mode, with_skills, out_root, args.turns, args.skills_path,
                         cell_timeout=args.cell_timeout,
                         quality_repair_turns=args.quality_repair_turns)
        results.append(cell)
        matrix_path.write_text(json.dumps(results, indent=2, default=str))
        metrics = _extract_metrics(cell)
        print(f"    {cell['status']}  ok={metrics.get('phase_success')}  "
              f"cycles={metrics.get('latency_cycles')}  "
              f"ns={metrics.get('latency_ns')}  "
              f"fmax={metrics.get('fmax_mhz')}  "
              f"FF={metrics.get('ff')}  LUT={metrics.get('lut')}  "
              f"({cell['wallclock_s']}s)")

    print()
    print("=== MATRIX SUMMARY ===")
    _print_table(results)
    print()
    print(f"Saved to {matrix_path}")


if __name__ == "__main__":
    main()
