#!/usr/bin/env python3
"""Compare held-out base vs DPO campaign cells with hybrid latency policy.

Usage:
  python rl/scripts/compare_heldout_ab.py \\
    --base-campaign artifacts/pc2/batch_parallel_..._heldout_base_... \\
    --dpo-campaign  artifacts/pc2/batch_parallel_..._heldout_dpo_... \\
    --out rl/eval/heldout_ab_*/compare.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


def _status_passed(status: Any) -> Optional[bool]:
    if isinstance(status, dict):
        if "passed" in status:
            return bool(status.get("passed"))
        gen = status.get("generated")
        if isinstance(gen, str):
            g = gen.lower()
            if g == "passed":
                return True
            if g in ("failed", "error"):
                return False
            return None
    if isinstance(status, bool):
        return status
    return None


def _tier(csim: Optional[bool], cosim: Optional[bool], synth_ok: bool) -> int:
    if csim is True and cosim is True:
        return 4
    if cosim is True:
        return 3
    if csim is True:
        return 2
    if synth_ok:
        return 1
    return 0


def _latency(meta: dict) -> tuple[Optional[float], str]:
    if meta.get("cosim_passed") is True and meta.get("cosim_cycles") is not None:
        return float(meta["cosim_cycles"]), "cosim"
    if meta.get("latency_cycles") is not None:
        return float(meta["latency_cycles"]), "csynth"
    return None, "none"


def load_arm(campaign: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for res in campaign.rglob("*_multistep_results.json"):
        try:
            r = json.loads(res.read_text())
        except Exception:
            continue
        if not isinstance(r, dict):
            continue
        bench = str(r.get("benchmark") or res.parent.name)
        csim = _status_passed(r.get("csim") or r.get("csim_status"))
        cosim = _status_passed(r.get("cosim") or r.get("cosim_status"))
        synth_ok = bool(r.get("success")) or bool(r.get("final_report") or r.get("synth_report"))
        rep = r.get("final_report") or r.get("synth_report") or {}
        lat = None
        if isinstance(rep, dict) and rep.get("latency_cycles") is not None:
            lat = float(rep["latency_cycles"])
        cosim_cycles = None
        cos = r.get("cosim") or {}
        if isinstance(cos, dict):
            for k in ("kernel_runtime_cycles", "latency_cycles"):
                if cos.get(k) is not None:
                    cosim_cycles = float(cos[k])
                    break
            meas = cos.get("measured") or {}
            if isinstance(meas, dict) and meas.get("latency_cycles_avg") is not None:
                cosim_cycles = float(meas["latency_cycles_avg"])
        # dataflow sibling
        stem = res.name.replace("_multistep_results.json", "")
        dfp = res.parent / f"{stem}_dataflow_result.json"
        if dfp.exists():
            try:
                df = json.loads(dfp.read_text())
                if isinstance(df, dict):
                    if (df.get("csim") or {}).get("passed") is True:
                        csim = True
                    if (df.get("cosim") or {}).get("passed") is True:
                        cosim = True
                    if df.get("latency_cycles") is not None:
                        lat = float(df["latency_cycles"])
            except Exception:
                pass
        meta = {
            "csim_passed": csim,
            "cosim_passed": cosim,
            "synth_ok": synth_ok,
            "latency_cycles": lat,
            "cosim_cycles": cosim_cycles,
            "path": str(res),
        }
        meta["tier"] = _tier(csim, cosim, synth_ok)
        eff, src = _latency(meta)
        meta["effective_latency"] = eff
        meta["latency_source"] = src
        # keep best tier / latency per bench
        prev = out.get(bench)
        if prev is None or meta["tier"] > prev["tier"] or (
            meta["tier"] == prev["tier"]
            and (meta["effective_latency"] or float("inf"))
            < (prev.get("effective_latency") or float("inf"))
        ):
            out[bench] = meta
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-campaign", type=Path, default=None)
    p.add_argument("--dpo-campaign", type=Path, default=None)
    p.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help="If set, read ab_meta.json for base/dpo campaign paths; default --out under this root.",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.eval_root is not None:
        meta = json.loads((args.eval_root / "ab_meta.json").read_text())
        args.base_campaign = args.base_campaign or Path(meta["base_campaign"])
        args.dpo_campaign = args.dpo_campaign or Path(meta["dpo_campaign"])
        args.out = args.out or (args.eval_root / "compare.md")
    if args.base_campaign is None or args.dpo_campaign is None or args.out is None:
        p.error("need --base-campaign/--dpo-campaign/--out, or --eval-root with ab_meta.json")

    base = load_arm(args.base_campaign)
    dpo = load_arm(args.dpo_campaign)
    benches = sorted(set(base) | set(dpo))

    lines = [
        "# Held-out A/B: base Devstral vs DPO",
        "",
        f"- base campaign: `{args.base_campaign}`",
        f"- dpo campaign: `{args.dpo_campaign}`",
        "",
        "Policy: tier (csim∧cosim > cosim > csim > synth), then hybrid latency.",
        "",
        "| bench | base tier | dpo tier | base lat | dpo lat | winner |",
        "|---|---:|---:|---:|---:|---|",
    ]
    wins = {"dpo": 0, "base": 0, "tie": 0, "missing": 0}
    for b in benches:
        bb, dd = base.get(b), dpo.get(b)
        if not bb or not dd:
            wins["missing"] += 1
            lines.append(
                f"| {b} | {bb['tier'] if bb else '-'} | {dd['tier'] if dd else '-'} | "
                f"{bb.get('effective_latency') if bb else '-'} | {dd.get('effective_latency') if dd else '-'} | missing |"
            )
            continue
        winner = "tie"
        if dd["tier"] != bb["tier"]:
            winner = "dpo" if dd["tier"] > bb["tier"] else "base"
        else:
            bl = bb.get("effective_latency")
            dl = dd.get("effective_latency")
            if bl is not None and dl is not None and bl != dl:
                winner = "dpo" if dl < bl else "base"
        wins[winner] += 1
        lines.append(
            f"| {b} | {bb['tier']} | {dd['tier']} | "
            f"{bb.get('effective_latency')} ({bb.get('latency_source')}) | "
            f"{dd.get('effective_latency')} ({dd.get('latency_source')}) | {winner} |"
        )

    lines += ["", "## Summary", f"- wins: {wins}", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    (args.out.with_suffix(".json")).write_text(
        json.dumps({"wins": wins, "base": base, "dpo": dpo}, indent=2) + "\n"
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
