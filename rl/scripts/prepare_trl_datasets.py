#!/usr/bin/env python3
"""Normalize team RL exports into TRL-ready SFT / DPO JSONL.

Default source: extracted/rl_dataset (cosim-verified, small, clean).

Outputs under --output (default: rl/prepared):
  sft/train.jsonl   {"messages": [...]}
  sft/val.jsonl
  dpo/train.jsonl   {"prompt": [...], "chosen": [...], "rejected": [...]}
  dpo/val.jsonl
  manifest.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

DEFAULT_SYSTEM = (
    "You are an expert in Xilinx Vitis HLS. Convert the given plain C/C++ kernel "
    "into synthesizable, high-performance Vitis HLS for the AMD Alveo U280 "
    "(xcu280-fsvh2892-2L-e). Preserve correctness; prefer lower latency when "
    "possible. Return a complete HLS source in a single ```cpp code fence."
)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _as_assistant(content: str) -> list[dict]:
    text = content.strip()
    if not text.startswith("```"):
        text = f"```cpp\n{text}\n```"
    return [{"role": "assistant", "content": text}]


def prepare_sft(sft_train: Path, sft_val: Path, out_dir: Path) -> dict:
    train_rows = [{"messages": r["messages"]} for r in _read_jsonl(sft_train)]
    val_rows = [{"messages": r["messages"]} for r in _read_jsonl(sft_val)]
    _write_jsonl(out_dir / "sft" / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "sft" / "val.jsonl", val_rows)
    return {"sft_train": len(train_rows), "sft_val": len(val_rows)}


def prepare_dpo(
    dpo_path: Path,
    out_dir: Path,
    val_frac: float,
    seed: int,
    system_prompt: str,
) -> dict:
    rows = _read_jsonl(dpo_path)
    by_bench: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_bench[r.get("bench", "unknown")].append(r)

    rng = random.Random(seed)
    train, val = [], []
    for bench, items in sorted(by_bench.items()):
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_frac))) if len(items) >= 5 else 0
        val_items = items[:n_val]
        train_items = items[n_val:]
        for r in train_items:
            train.append(_to_dpo_row(r, system_prompt))
        for r in val_items:
            val.append(_to_dpo_row(r, system_prompt))

    _write_jsonl(out_dir / "dpo" / "train.jsonl", train)
    _write_jsonl(out_dir / "dpo" / "val.jsonl", val)
    return {
        "dpo_train": len(train),
        "dpo_val": len(val),
        "dpo_benches": len(by_bench),
    }


def _to_dpo_row(r: dict, system_prompt: str) -> dict:
    prompt = r["prompt"]
    if isinstance(prompt, str):
        prompt_msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        prompt_msgs = prompt
    return {
        "prompt": prompt_msgs,
        "chosen": _as_assistant(r["chosen"] if isinstance(r["chosen"], str) else r["chosen"][-1]["content"]),
        "rejected": _as_assistant(
            r["rejected"] if isinstance(r["rejected"], str) else r["rejected"][-1]["content"]
        ),
        "bench": r.get("bench"),
        "chosen_cycles": r.get("chosen_cycles"),
        "rejected_cycles": r.get("rejected_cycles"),
        "speedup": r.get("speedup"),
    }


def prepare_agentic_sft(train: Path, val: Path, out_dir: Path, tag: str) -> dict:
    """Optional: large agentic_trl_chat_v1 export (messages field)."""
    train_rows = [{"messages": r["messages"]} for r in _read_jsonl(train) if r.get("messages")]
    val_rows = [{"messages": r["messages"]} for r in _read_jsonl(val) if r.get("messages")]
    _write_jsonl(out_dir / tag / "train.jsonl", train_rows)
    _write_jsonl(out_dir / tag / "val.jsonl", val_rows)
    return {f"{tag}_train": len(train_rows), f"{tag}_val": len(val_rows)}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=root / "extracted" / "rl_dataset",
        help="Path to extracted rl_dataset/",
    )
    p.add_argument(
        "--corpus-dir",
        type=Path,
        default=root / "extracted" / "rl_corpus",
        help="Path to extracted rl_corpus/ (optional agentic export)",
    )
    p.add_argument("--output", type=Path, default=root / "prepared")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--include-agentic",
        action="store_true",
        help="Also export agentic_trl_chat_v1 SFT splits (large).",
    )
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    args = p.parse_args()

    stats = {}
    stats.update(
        prepare_sft(
            args.dataset_dir / "sft.train.jsonl",
            args.dataset_dir / "sft.val.jsonl",
            args.output,
        )
    )
    stats.update(
        prepare_dpo(
            args.dataset_dir / "dpo.jsonl",
            args.output,
            val_frac=args.val_frac,
            seed=args.seed,
            system_prompt=args.system_prompt,
        )
    )
    if args.include_agentic:
        trl_dir = args.corpus_dir / "agentic_trl_chat_v1"
        stats.update(
            prepare_agentic_sft(
                trl_dir / "train.jsonl",
                trl_dir / "val.jsonl",
                args.output,
                tag="agentic_sft",
            )
        )

    manifest = {
        "source_dataset": str(args.dataset_dir),
        "output": str(args.output),
        "counts": stats,
        "notes": [
            "SFT rows are chat messages for TRL SFTTrainer.",
            "DPO rows use conversational prompt/chosen/rejected lists.",
            "Start with rl_dataset (cosim-verified). Scale later with --include-agentic.",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
