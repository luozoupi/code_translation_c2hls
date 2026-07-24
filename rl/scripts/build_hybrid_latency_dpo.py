#!/usr/bin/env python3
"""Build DPO pairs with hybrid latency:

  effective_latency =
      cosim_cycles   if cosim_passed and cosim_cycles is not None
      else latency_cycles   # csynth

Correctness tiers (revised — csim and cosim are NOT nested):
  cosim synthesizes to RTL then simulates; it implies synth, NOT csim.
  So we rank joint evidence, not a false total order cosim > csim:

  tier 4 = csim_pass AND cosim_pass   (best functional evidence)
  tier 3 = cosim_pass only            (RTL verified; csim missing/fail)
  tier 2 = csim_pass only             (C verified; no cosim)
  tier 1 = synth_pass only
  tier 0 = else

Chosen must be strictly better on (tier, then lower latency).
Pairs are within the same benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


DEFAULT_SYSTEM = (
    "You are an expert in Xilinx Vitis HLS. Convert the given plain C/C++ kernel "
    "into synthesizable, high-performance Vitis HLS for the AMD Alveo U280 "
    "(xcu280-fsvh2892-2L-e). Preserve correctness; prefer lower latency when "
    "possible. Return a complete HLS source in a single ```cpp code fence."
)


def _tier(meta: dict) -> int:
    """Joint pass tier. Cosim ⇒ synth/RTL path, but does not imply csim."""
    csim = meta.get("csim_passed") is True
    cosim = meta.get("cosim_passed") is True
    synth = meta.get("synth_passed") is True or cosim  # cosim required RTL synth
    if csim and cosim:
        return 4
    if cosim:
        return 3
    if csim:
        return 2
    if synth:
        return 1
    return 0


def effective_latency(meta: dict) -> tuple[Optional[float], Optional[str]]:
    """Cosim latency if available+passed, else csynth latency_cycles."""
    if meta.get("cosim_passed") is True and meta.get("cosim_cycles") is not None:
        return float(meta["cosim_cycles"]), "cosim"
    lat = meta.get("latency_cycles")
    if lat is not None:
        return float(lat), "csynth"
    return None, None


def _assistant_content(row: dict) -> Optional[str]:
    if row.get("completion"):
        return str(row["completion"]).strip()
    msgs = row.get("messages") or []
    for m in reversed(msgs):
        if m.get("role") == "assistant" and m.get("content"):
            return str(m["content"]).strip()
    return None


def _prompt_messages(row: dict, system: str) -> list[dict]:
    """Use all messages before the final assistant turn as the prompt."""
    msgs = row.get("messages") or []
    if not msgs:
        return [{"role": "system", "content": system}]
    # drop trailing assistant
    cut = len(msgs)
    if msgs and msgs[-1].get("role") == "assistant":
        cut -= 1
    prompt = list(msgs[:cut])
    if not any(m.get("role") == "system" for m in prompt):
        prompt = [{"role": "system", "content": system}] + prompt
    return prompt


def _load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _candidate(row: dict) -> Optional[dict[str, Any]]:
    meta = row.get("metadata") or {}
    lat, source = effective_latency(meta)
    if lat is None:
        return None
    content = _assistant_content(row)
    if not content:
        return None
    tier = _tier(meta)
    if tier < 1:
        return None  # require at least synth pass
    return {
        "benchmark": row.get("benchmark") or meta.get("benchmark"),
        "split": row.get("split", "train"),
        "tier": tier,
        "latency": lat,
        "latency_source": source,
        "content": content,
        "code_sha256": meta.get("code_sha256"),
        "prompt": _prompt_messages(row, DEFAULT_SYSTEM),
        "cosim_cycles": meta.get("cosim_cycles"),
        "latency_cycles": meta.get("latency_cycles"),
        "cosim_passed": meta.get("cosim_passed"),
        "csim_passed": meta.get("csim_passed"),
        "synth_passed": meta.get("synth_passed"),
    }


def build_pairs(
    candidates: list[dict],
    max_pairs_per_bench: int,
    min_speedup: float,
    seed: int,
) -> list[dict]:
    by_bench: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    for c in candidates:
        key = (c["benchmark"], c.get("code_sha256") or hash(c["content"]))
        if key in seen:
            continue
        seen.add(key)
        by_bench[c["benchmark"]].append(c)

    rng = random.Random(seed)
    pairs = []
    for bench, items in sorted(by_bench.items()):
        # better = higher tier, then lower latency
        items.sort(key=lambda x: (-x["tier"], x["latency"]))
        local = []
        for i, chosen in enumerate(items):
            for rejected in items[i + 1 :]:
                # chosen must be strictly better
                if chosen["tier"] < rejected["tier"]:
                    continue
                if chosen["tier"] == rejected["tier"]:
                    if chosen["latency"] <= 0 or rejected["latency"] <= 0:
                        continue
                    speedup = rejected["latency"] / chosen["latency"]
                    if speedup < min_speedup:
                        continue
                else:
                    speedup = None
                local.append((chosen, rejected, speedup))
        rng.shuffle(local)
        for chosen, rejected, speedup in local[:max_pairs_per_bench]:
            pairs.append(
                {
                    "prompt": chosen["prompt"],
                    "chosen": [{"role": "assistant", "content": chosen["content"]}],
                    "rejected": [{"role": "assistant", "content": rejected["content"]}],
                    "bench": bench,
                    "split": chosen.get("split", "train"),
                    "chosen_tier": chosen["tier"],
                    "rejected_tier": rejected["tier"],
                    "chosen_latency": chosen["latency"],
                    "rejected_latency": rejected["latency"],
                    "chosen_latency_source": chosen["latency_source"],
                    "rejected_latency_source": rejected["latency_source"],
                    "speedup": speedup,
                    "policy": "tier_then_hybrid_latency(cosim_if_pass_else_csynth)",
                }
            )
    return pairs


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            root / "extracted/rl_corpus/agentic_trl_chat_v1/train.jsonl",
            root / "extracted/rl_corpus/agentic_trl_chat_v1/val.jsonl",
            root / "extracted/rl_corpus/agentic_trl_chat_v1/cosim_observed.jsonl",
        ],
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=root / "prepared" / "dpo_hybrid",
    )
    p.add_argument("--max-pairs-per-bench", type=int, default=40)
    p.add_argument("--min-speedup", type=float, default=1.05,
                   help="When tiers tie, require rejected/chosen latency >= this")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    args = p.parse_args()

    rows = _load_rows([x for x in args.inputs if x.exists()])
    cands = []
    skipped = 0
    source_counts = defaultdict(int)
    for r in rows:
        c = _candidate(r)
        if c is None:
            skipped += 1
            continue
        source_counts[c["latency_source"]] += 1
        cands.append(c)

    pairs = build_pairs(
        cands,
        max_pairs_per_bench=args.max_pairs_per_bench,
        min_speedup=args.min_speedup,
        seed=args.seed,
    )

    # split by bench
    by_bench: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        by_bench[pair["bench"]].append(pair)
    rng = random.Random(args.seed)
    train, val = [], []
    for bench, items in sorted(by_bench.items()):
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * args.val_frac))) if len(items) >= 8 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def write(path: Path, data: list[dict]) -> None:
        with path.open("w") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write(args.output_dir / "train.jsonl", train)
    write(args.output_dir / "val.jsonl", val)

    # also refresh prepared/dpo for the default DPO slurm path if requested via symlink-like copy note in manifest
    manifest = {
        "policy": {
            "latency": "cosim_cycles if cosim_passed else latency_cycles (csynth)",
            "rank": "higher joint pass tier first, then lower effective latency",
            "tiers": {
                "4": "csim_pass AND cosim_pass",
                "3": "cosim_pass only (does not imply csim)",
                "2": "csim_pass only",
                "1": "synth_pass only",
                "0": "fail",
            },
            "note": "Cosim implies RTL synth+sim, not csim. Prefer both when available.",
            "min_speedup_same_tier": args.min_speedup,
            "min_tier": 1,
        },
        "inputs": [str(x) for x in args.inputs],
        "candidates": len(cands),
        "skipped_rows": skipped,
        "latency_source_counts": dict(source_counts),
        "pairs_train": len(train),
        "pairs_val": len(val),
        "benches": len(by_bench),
        "files": {
            "train": str(args.output_dir / "train.jsonl"),
            "val": str(args.output_dir / "val.jsonl"),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
