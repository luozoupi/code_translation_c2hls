#!/usr/bin/env python3
"""Write the neutral refined skill pack: the refined v2 pack with every DSP/device-sizing push removed.

Source: flash_refined/..._refined_v2.json (138 skills; only read). Output: ..._refined_v2_neutral.json (136 skills).
  * drops hls-size-parallel-datapath-to-device-budget and avoid-idle-device-on-compute-bound-kernel;
  * rewrites the refined wordings that ask for "the device budget" / "most of the device budget" into a
    pure cap ("every resource below 100%");
  * balance skills keep their content but no longer assume a filled or device-sized datapath.
Hold-back removals (wide bus, separate bundles, BRAM capacity, "resource explosion" definition) are kept.
Every replacement is asserted to hit exactly once.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "skills_ii_target_miss_solutions_added(90skills)_plus_gemm_flatten_v2_refined_v2.json"
OUT = HERE / "skills_ii_target_miss_solutions_added(90skills)_plus_gemm_flatten_v2_refined_v2_neutral.json"
DROP = {"hls-size-parallel-datapath-to-device-budget", "avoid-idle-device-on-compute-bound-kernel"}
EDITS = [  # (skill id, field, old, new)
    ("hls-pipeline-unroll-small-inner-loops", "required_steps",
     "otherwise to the factor the device budget (DSP, buffer ports, timing) supports",
     "otherwise partially, with any factor that keeps every resource below 100% of the device"),
    ("hls-tile-compute-inner-parallelism", "required_steps",
     "sized to the device budget and the local buffer ports", "sized to the local buffer ports"),
    ("prompt-unroll", "strategy",
     "sized so the whole design uses most of the device budget without reaching 100% of any resource",
     "sized so no resource reaches 100% of the device"),
    ("hls-balance-compute-and-transfer-after-scaling", "pattern",
     "the parallel datapath has been scaled toward the device budget, so each tile's compute now takes few cycles",
     "the parallel datapath is wide enough that each tile's compute takes few cycles"),
    ("avoid-transfer-bound-filled-datapath", "pattern", "DSP is filled but latency", "latency"),
    ("avoid-transfer-bound-filled-datapath", "strategy", "do not stop after filling DSP: rebalance", "rebalance"),
    ("hls-balance-compute-and-transfer-after-scaling", "strategy", "after sizing the datapath, budget", "for the chosen datapath, budget"),
]
RENAME = {"hls-balance-compute-and-transfer-after-scaling": "hls-balance-compute-and-transfer",
          "avoid-transfer-bound-filled-datapath": "avoid-transfer-bound-datapath"}


def main():
    pack = json.loads(SRC.read_text())
    pack["skills"] = [s for s in pack["skills"] if s["id"] not in DROP]
    skills = {s["id"]: s for s in pack["skills"]}
    for sid, field, old, new in EDITS:
        v = skills[sid][field]
        if isinstance(v, list):
            hits = [i for i, x in enumerate(v) if old in x]
            assert len(hits) == 1, (sid, field, old)
            v[hits[0]] = v[hits[0]].replace(old, new)
        else:
            assert v.count(old) == 1, (sid, field, old)
            skills[sid][field] = v.replace(old, new)
    text = json.dumps(pack, indent=1)
    for old, new in RENAME.items():  # ids name the idea too; drop "after-scaling" / "filled"
        text = text.replace(old, new)
    pack = json.loads(text)
    for bad in ("most of the device", "device budget supports", "scaled toward the device", "fill target", "idle device", "filled-datapath", "after-scaling"):
        assert bad not in text, bad
    pack.setdefault("refined", {})["neutral"] = {"dropped": sorted(DROP), "edits": [e[:3] for e in EDITS]}
    OUT.write_text(json.dumps(pack, indent=1))
    print(OUT, len(pack["skills"]), "skills")


if __name__ == "__main__":
    main()
