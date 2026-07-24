#!/usr/bin/env python3
"""Merge mined local SFT with the team-extracted agentic/trl corpora.

Writes a combined TRL-ready pack under rl/prepared/sft_combined/:
  train.jsonl / val.jsonl / test.jsonl / manifest.json

Dedupes by (benchmark, code_sha256) preferring higher correctness_tier /
validated_positive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

CODE_FENCE_RE = re.compile(r"```(?:cpp|c\+\+|c)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
VAL = {"StreamCluster", "viterbi"}
TEST = {"nw", "spmv_crs"}


def _sha_code(messages) -> str:
    text = ""
    if messages:
        m = CODE_FENCE_RE.search(messages[-1].get("content") or "")
        text = (m.group(1) if m else messages[-1].get("content") or "").strip()
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _split(bench: str) -> str:
    bare = bench
    for pref in ("hlsfactory_", "machsuite_", "rodinia_", "c2hlsc_", "autosa_"):
        if bare.startswith(pref):
            bare = bare[len(pref) :]
            break
    if bare in VAL or bench in VAL:
        return "val"
    if bare in TEST or bench in TEST:
        return "test"
    return "train"


def _load(path: Path, origin: str) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            msgs = r.get("messages")
            if not msgs:
                continue
            ql = r.get("quality_label") or (r.get("metadata") or {}).get("quality_label")
            if ql and ql not in ("validated_positive", "synth_positive"):
                # team trl export is already filtered positive; keep
                if origin.startswith("agentic") or origin.startswith("trl"):
                    pass
                else:
                    continue
            meta = dict(r.get("metadata") or {})
            meta.setdefault("code_sha256", _sha_code(msgs))
            bench = r.get("benchmark") or meta.get("benchmark") or "unknown"
            rows.append(
                {
                    "messages": msgs,
                    "benchmark": bench,
                    "split": r.get("split") or _split(str(bench)),
                    "quality_label": ql or "validated_positive",
                    "metadata": meta,
                    "origin": origin,
                }
            )
    return rows


def _better(a: dict, b: dict) -> dict:
    rank = {"validated_positive": 3, "synth_positive": 2}
    ra, rb = rank.get(a.get("quality_label"), 1), rank.get(b.get("quality_label"), 1)
    if ra != rb:
        return a if ra > rb else b
    ta = int((a.get("metadata") or {}).get("correctness_tier") or 0)
    tb = int((b.get("metadata") or {}).get("correctness_tier") or 0)
    if ta != tb:
        return a if ta > tb else b
    # prefer local mined when tied (fresher / our labels)
    if a.get("origin", "").startswith("mined") != b.get("origin", "").startswith("mined"):
        return a if a.get("origin", "").startswith("mined") else b
    return a


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mined", type=Path, default=root / "prepared" / "mined_sft" / "sft_positive.jsonl")
    p.add_argument(
        "--agentic-train",
        type=Path,
        default=root / "extracted" / "rl_corpus" / "agentic_trl_chat_v1" / "train.jsonl",
    )
    p.add_argument(
        "--agentic-val",
        type=Path,
        default=root / "extracted" / "rl_corpus" / "agentic_trl_chat_v1" / "val.jsonl",
    )
    p.add_argument(
        "--rl-dataset-train",
        type=Path,
        default=root / "prepared" / "sft" / "train.jsonl",
    )
    p.add_argument(
        "--rl-dataset-val",
        type=Path,
        default=root / "prepared" / "sft" / "val.jsonl",
    )
    p.add_argument("--output", type=Path, default=root / "prepared" / "sft_combined")
    p.add_argument("--include-synth-positive", action="store_true")
    args = p.parse_args()

    rows: list[dict] = []
    rows += _load(args.mined, "mined_local")
    rows += _load(args.rl_dataset_train, "rl_dataset_train")
    rows += _load(args.rl_dataset_val, "rl_dataset_val")
    rows += _load(args.agentic_train, "agentic_trl_train")
    rows += _load(args.agentic_val, "agentic_trl_val")

    if not args.include_synth_positive:
        rows = [r for r in rows if r.get("quality_label") != "synth_positive"]

    best: dict[tuple, dict] = {}
    for r in rows:
        bench = str(r["benchmark"])
        sha = (r.get("metadata") or {}).get("code_sha256") or _sha_code(r["messages"])
        k = (bench, sha)
        if k not in best:
            best[k] = r
        else:
            best[k] = _better(best[k], r)

    merged = list(best.values())
    by_split: dict[str, list] = defaultdict(list)
    for r in merged:
        by_split[r["split"]].append(r)

    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        with (args.output / f"{split}.jsonl").open("w") as f:
            for r in by_split.get(split, []):
                f.write(
                    json.dumps(
                        {
                            "messages": r["messages"],
                            "benchmark": r["benchmark"],
                            "split": r["split"],
                            "quality_label": r.get("quality_label"),
                            "metadata": r.get("metadata"),
                            "origin": r.get("origin"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    manifest = {
        "total": len(merged),
        "by_split": {s: len(by_split.get(s, [])) for s in ("train", "val", "test")},
        "by_origin": dict(Counter(r.get("origin") for r in merged)),
        "by_quality": dict(Counter(r.get("quality_label") for r in merged)),
        "inputs": {
            "mined": str(args.mined),
            "rl_dataset_train": str(args.rl_dataset_train),
            "agentic_train": str(args.agentic_train),
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
