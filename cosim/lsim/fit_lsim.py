#!/usr/bin/env python3
"""Fit the v2023 AXI delay constants of the LightningSim fork to Vitis 2023.2 cosim.

  fit_lsim.py MB_ROOT VCD_ROOT OUT.json [--procs N] [--max 24]

Runs inside the Vitis container with the LightningSim env (see sbatch_fit.sh). Workers unpack every
MB_ROOT/NAME/proj.tar.gz, build the LightningSim trace once (LSIM_AXI_MODEL=v2023, per-interface config from
axi_cfg.py) and return the pickled compiled graph. The parent then searches the constants by coordinate descent
(every constant over 0..max with the others fixed, repeated until nothing improves), re-executing the graphs via
lightningsim._core.set_axi_constants. Objective: sum |predicted report - cosim report| (the reference, see
compare.report_map), tie-break sum |end - DUT ap_done| (from VCD_ROOT/NAME/axi_timing.json).
"""
import argparse, asyncio, json, os, pickle, shutil, subprocess, sys, tarfile, tempfile
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare import cosim_cycles, report_map, saxi_args  # noqa: E402


def build(args):
    name, mb_root, work = args
    os.environ["LSIM_AXI_MODEL"] = "v2023"
    d = Path(work) / name
    d.mkdir(parents=True, exist_ok=True)
    with tarfile.open(Path(mb_root) / name / "proj.tar.gz") as t:
        t.extractall(d)
    sol = d / "hls_proj/sol1"
    cfg = d / "axi_cfg.json"
    subprocess.run([sys.executable, str(HERE / "axi_cfg.py"), str(sol), str(cfg)], check=True, capture_output=True)
    os.environ["LSIM_AXI_IFCFG"] = str(cfg)
    from lightningsim.model import Solution
    from lightningsim.runner import Runner

    trace = asyncio.run(Runner(Solution(sol)).run())
    blob = pickle.dumps((trace.compiled, trace.params))
    shutil.rmtree(d, ignore_errors=True)
    return name, blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mb_root")
    ap.add_argument("vcd_root")
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--max", type=int, default=24)
    a = ap.parse_args()
    os.environ["LSIM_AXI_MODEL"] = "v2023"
    import lightningsim._core as core

    _, start = core.get_axi_model()
    keys = [k for k, _ in start]
    best = [v for _, v in start]
    names = sorted(p.parent.name for p in Path(a.mb_root).glob("*/proj.tar.gz")
                   if (Path(a.vcd_root) / p.parent.name / "axi_timing.json").exists())
    done = {n: json.load(open(Path(a.vcd_root) / n / "axi_timing.json"))["ap_done"] for n in names}
    report = {n: cosim_cycles(Path(a.mb_root) / n) for n in names}
    nargs = {n: saxi_args(Path(a.mb_root) / n) for n in names}
    work = tempfile.mkdtemp(prefix="lsim_fit_", dir=os.environ.get("TMPDIR"))
    with Pool(a.procs) as pool:
        graphs = {n: pickle.loads(b) for n, b in pool.map(build, [(n, a.mb_root, work) for n in names])}
    print(f"{len(graphs)} graphs built; start {dict(zip(keys, best))}", flush=True)

    cache = {}

    def ends(c):
        c = tuple(c)
        if c not in cache:
            core.set_axi_constants(list(c))
            out = {}
            for n, (compiled, params) in graphs.items():
                try:
                    out[n] = compiled.execute(params).top_module.end
                except ValueError:
                    out[n] = None
            cache[c] = out
        return cache[c]

    def score(c):
        e = ends(c)
        if any(v is None for v in e.values()):
            return (10**12, 10**12)
        return (sum(abs(report_map(e[n], nargs[n]) - report[n]) for n in names),
                sum(abs(e[n] - done[n]) for n in names))

    cur = score(best)
    print(f"start score {cur}", flush=True)
    for rnd in range(12):
        improved = False
        for i, k in enumerate(keys):
            if k == "wl":
                continue  # lower bound only; kept at its default
            for v in range(0, a.max + 1):
                cand = list(best)
                cand[i] = v
                s = score(cand)
                if s < cur:
                    cur, best, improved = s, cand, True
        print(f"round {rnd}: {dict(zip(keys, best))} score {cur}", flush=True)
        if not improved:
            break
    e = ends(best)
    per = {n: {"dut_done": done[n], "lsim_end": e[n], "err": e[n] - done[n], "report": report[n],
               "pred": report_map(e[n], nargs[n]), "report_err": report_map(e[n], nargs[n]) - report[n]} for n in names}
    out = {"constants_order": keys, "best": dict(zip(keys, best)), "best_sum_abs_report_err": cur[0],
           "best_sum_abs_done_err": cur[1],
           "best_max_abs_report_err": max(abs(v["report_err"]) for v in per.values()), "per_config": per}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("best", "best_sum_abs_report_err", "best_sum_abs_done_err",
                                         "best_max_abs_report_err")}))
    for n, v in per.items():
        if v["report_err"]:
            print(f"{n:20s} done={v['dut_done']:5d} lsim={v['lsim_end']} err={v['err']:+d} "
                  f"report={v['report']} pred={v['pred']} rerr={v['report_err']:+d}")


if __name__ == "__main__":
    main()
