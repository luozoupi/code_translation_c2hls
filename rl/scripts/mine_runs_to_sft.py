#!/usr/bin/env python3
"""Mine local c2hls runs into an offline SFT corpus.

Sources (read-only), ranked by label quality:
  1. results_matrix_u280_fullcosim{,_extended}/  — code + history + csim/cosim
  2. results/                                    — small, high-signal
  3. artifacts/pc2/**/variants/**                — campaign cells
       - prefer *_dataflow_result.json when present
       - else *_multistep_results.json (only if generated csim/cosim ran)

Does NOT mine c2hls_tmp/batch_parallel_* HLS workdirs (no durable history/pass
JSON). Those campaigns already land under artifacts/pc2/.

Quality labels (aligned with team agentic_sft policy):
  validated_positive — code present, synth OK, csim_passed
  synth_positive     — code present, synth OK, csim missing/unknown
  negative           — code present but csim failed (or synth failed)
  no_code            — missing assistant/hls code

Default SFT export keeps validated_positive only (use --include-synth-positive
to also keep synth_positive).

Output:
  <out>/sft_all.jsonl
  <out>/sft_positive.jsonl   (filtered)
  <out>/train.jsonl / val.jsonl / test.jsonl
  <out>/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

REPO = Path(__file__).resolve().parents[2]  # c2hls/
DEFAULT_SYSTEM = (
    "You are an expert in Xilinx Vitis HLS. Convert the given plain C/C++ kernel "
    "into synthesizable, high-performance Vitis HLS. Preserve correctness; prefer "
    "lower latency when possible. Return a complete HLS source in a single "
    "```cpp code fence."
)

VAL_BENCH = {"StreamCluster", "viterbi"}
TEST_BENCH = {"nw", "spmv_crs"}

CODE_FENCE_RE = re.compile(r"```(?:cpp|c\+\+|c)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class MinedRow:
    benchmark: str
    split: str
    source: str
    quality_label: str
    messages: list[dict]
    metadata: dict

    def to_json(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "split": self.split,
            "source": self.source,
            "quality_label": self.quality_label,
            "messages": self.messages,
            "metadata": self.metadata,
        }


def _split_for(bench: str) -> str:
    # strip suite prefixes for split policy match
    bare = bench
    for pref in ("hlsfactory_", "machsuite_", "rodinia_", "c2hlsc_", "autosa_"):
        if bare.startswith(pref):
            bare = bare[len(pref) :]
            break
    if bare in VAL_BENCH or bench in VAL_BENCH:
        return "val"
    if bare in TEST_BENCH or bench in TEST_BENCH:
        return "test"
    return "train"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _extract_code(text: str) -> Optional[str]:
    if not text:
        return None
    m = CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # raw cpp fallback
    if "#pragma HLS" in text or "extern \"C\"" in text or "void kernel_" in text:
        return text.strip()
    return None


def _normalize_messages(raw: Any, final_code: Optional[str] = None) -> Optional[list[dict]]:
    msgs: list[dict] = []
    if isinstance(raw, dict) and "messages" in raw:
        raw = raw["messages"]
    if isinstance(raw, list):
        for m in raw:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role in ("system", "user", "assistant") and isinstance(content, str) and content.strip():
                msgs.append({"role": role, "content": content})
    if not msgs:
        return None
    # Ensure trailing assistant holds the final code when provided
    if final_code:
        fence = f"```cpp\n{final_code.rstrip()}\n```"
        if msgs[-1]["role"] == "assistant":
            # replace if empty / no code
            if not _extract_code(msgs[-1]["content"]):
                msgs[-1] = {"role": "assistant", "content": fence}
        else:
            msgs.append({"role": "assistant", "content": fence})
    if msgs[-1]["role"] != "assistant":
        return None
    if not any(m["role"] == "user" for m in msgs):
        return None
    if not any(m["role"] == "system" for m in msgs):
        msgs = [{"role": "system", "content": DEFAULT_SYSTEM}] + msgs
    return msgs


def _status_passed(status: Any) -> Optional[bool]:
    if status is True:
        return True
    if status is False:
        return False
    if isinstance(status, dict):
        # prefer generated when present
        gen = status.get("generated")
        if isinstance(gen, str):
            g = gen.lower()
            if g == "passed":
                return True
            if g in ("failed", "error"):
                return False
            if g in ("not_run", "skipped", "unknown", ""):
                return None
        if "passed" in status:
            return bool(status.get("passed"))
    if isinstance(status, str):
        s = status.lower()
        if s == "passed":
            return True
        if s in ("failed", "error"):
            return False
    return None


def _latency_from_report(rep: Any) -> Optional[float]:
    if not isinstance(rep, dict):
        return None
    for k in ("latency_cycles", "Latency", "best_latency"):
        if rep.get(k) is not None:
            try:
                return float(rep[k])
            except (TypeError, ValueError):
                pass
    return None


def _cosim_cycles(cosim: Any) -> Optional[float]:
    if not isinstance(cosim, dict):
        return None
    for path in (
        ("kernel_runtime_cycles",),
        ("measured", "latency_cycles_avg"),
        ("measured", "latency_cycles"),
        ("latency_cycles",),
    ):
        cur: Any = cosim
        ok = True
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                ok = False
                break
            cur = cur[p]
        if ok and cur is not None:
            try:
                return float(cur)
            except (TypeError, ValueError):
                pass
    return None


def _label(synth_ok: bool, csim: Optional[bool], has_code: bool) -> str:
    if not has_code:
        return "no_code"
    if not synth_ok:
        return "negative"
    if csim is True:
        return "validated_positive"
    if csim is False:
        return "negative"
    return "synth_positive"


def _build_row(
    *,
    benchmark: str,
    source: str,
    messages: list[dict],
    synth_ok: bool,
    csim_passed: Optional[bool],
    cosim_passed: Optional[bool],
    latency_cycles: Optional[float],
    cosim_cycles: Optional[float],
    model: Optional[str],
    extra: Optional[dict] = None,
) -> Optional[MinedRow]:
    code = _extract_code(messages[-1]["content"])
    has_code = bool(code)
    quality = _label(synth_ok, csim_passed, has_code)
    tier = 0
    if cosim_passed is True and csim_passed is True:
        tier = 4
    elif cosim_passed is True:
        tier = 3
    elif csim_passed is True:
        tier = 2
    elif synth_ok:
        tier = 1
    meta = {
        "synth_passed": bool(synth_ok),
        "csim_passed": csim_passed,
        "cosim_passed": cosim_passed,
        "latency_cycles": latency_cycles,
        "cosim_cycles": cosim_cycles,
        "correctness_tier": tier,
        "code_sha256": _sha(code) if code else None,
        "model": model,
        "reward_policy": "tier then hybrid latency (cosim if pass else csynth)",
    }
    if extra:
        meta.update(extra)
    return MinedRow(
        benchmark=benchmark,
        split=_split_for(benchmark),
        source=source,
        quality_label=quality,
        messages=messages,
        metadata=meta,
    )


# ── miners ──────────────────────────────────────────────────────────────────

def mine_results_tree(results_dir: Path, tag: str) -> list[MinedRow]:
    rows: list[MinedRow] = []
    if not results_dir.is_dir():
        return rows
    for res_path in results_dir.rglob("*_results.json"):
        try:
            res = json.loads(res_path.read_text())
        except Exception:
            continue
        if isinstance(res, list):
            # occasionally a list of step records — take last dict-like entry
            res = next((x for x in reversed(res) if isinstance(x, dict)), None)
            if res is None:
                continue
        if not isinstance(res, dict):
            continue
        bench = str(res.get("benchmark") or res_path.parent.name)
        hist_path = res_path.with_name(res_path.name.replace("_results.json", "_history.json"))
        hist_raw: Any = None
        if hist_path.exists():
            try:
                hist_raw = json.loads(hist_path.read_text())
            except Exception:
                hist_raw = None
        if hist_raw is None:
            hist_raw = res.get("turn_history") or res.get("optimization_history")
        code = res.get("hls_code")
        msgs = _normalize_messages(hist_raw, final_code=code if isinstance(code, str) else None)
        if not msgs and isinstance(code, str) and code.strip():
            # minimal prompt-less record skipped — need user turn
            continue
        if not msgs:
            continue
        csim = _status_passed(res.get("csim")) 
        if csim is None:
            csim = _status_passed(res.get("csim_status"))
        cosim = _status_passed(res.get("cosim"))
        if cosim is None:
            cosim = _status_passed(res.get("cosim_status"))
        synth_ok = bool(res.get("success")) or bool(res.get("synth_report")) or bool(res.get("final_report"))
        lat = _latency_from_report(res.get("synth_report") or res.get("final_report"))
        cyc = _cosim_cycles(res.get("cosim") or {})
        model = None
        if isinstance(hist_raw, dict):
            model = hist_raw.get("model")
        row = _build_row(
            benchmark=bench,
            source=f"{tag}:{res_path.relative_to(results_dir.parent) if results_dir.parent.exists() else res_path}",
            messages=msgs,
            synth_ok=synth_ok,
            csim_passed=csim,
            cosim_passed=cosim,
            latency_cycles=lat,
            cosim_cycles=cyc,
            model=model,
            extra={"result_path": str(res_path)},
        )
        if row:
            rows.append(row)
    return rows


def mine_pc2_variants(pc2_root: Path) -> list[MinedRow]:
    rows: list[MinedRow] = []
    if not pc2_root.is_dir():
        return rows
    # cell dirs contain *_history.json
    for hist_path in pc2_root.rglob("*_history.json"):
        if "variants" not in hist_path.parts:
            continue
        # skip dataflow history handled with dataflow_result below (avoid double)
        if hist_path.name.endswith("_dataflow_history.json"):
            continue
        stem = hist_path.name[: -len("_history.json")]
        parent = hist_path.parent
        dataflow = parent / f"{stem}_dataflow_result.json"
        dataflow_hist = parent / f"{stem}_dataflow_history.json"
        multi = parent / f"{stem}_multistep_results.json"
        final_cpp = parent / f"{stem}_final.cpp"
        selected_cpp = parent / f"{stem}_selected.cpp"

        try:
            hist = json.loads(hist_path.read_text())
        except Exception:
            continue

        # Prefer dataflow outcome when available (stronger generated csim labels).
        if dataflow.exists():
            try:
                df = json.loads(dataflow.read_text())
            except Exception:
                df = None
            if isinstance(df, dict):
                dhist = hist
                if dataflow_hist.exists():
                    try:
                        dhist = json.loads(dataflow_hist.read_text())
                    except Exception:
                        pass
                code = None
                for cand in (parent / f"{stem}_dataflow.cpp", final_cpp, selected_cpp):
                    if cand.exists():
                        code = cand.read_text(errors="replace")
                        break
                msgs = _normalize_messages(dhist, final_code=code)
                if not msgs:
                    continue
                csim = _status_passed((df.get("csim") or {}))
                if csim is None:
                    csim = True if df.get("success") and (df.get("csim") or {}).get("passed") is True else _status_passed(df.get("csim"))
                cosim = _status_passed(df.get("cosim"))
                synth_ok = bool(df.get("success")) or bool(df.get("synth_report"))
                lat = _latency_from_report(df.get("synth_report") or df)
                row = _build_row(
                    benchmark=str(df.get("benchmark") or stem),
                    source=f"pc2_dataflow:{hist_path.relative_to(pc2_root)}",
                    messages=msgs,
                    synth_ok=synth_ok,
                    csim_passed=csim,
                    cosim_passed=cosim,
                    latency_cycles=lat,
                    cosim_cycles=_cosim_cycles(df.get("cosim") or {}),
                    model=(dhist.get("model") if isinstance(dhist, dict) else None),
                    extra={"cell": str(parent)},
                )
                if row:
                    rows.append(row)
                continue

        # Multistep cell
        if not multi.exists():
            continue
        try:
            res = json.loads(multi.read_text())
        except Exception:
            continue
        code = res.get("hls_code")
        if not isinstance(code, str) or not code.strip():
            for cand in (final_cpp, selected_cpp):
                if cand.exists():
                    code = cand.read_text(errors="replace")
                    break
        msgs = _normalize_messages(hist, final_code=code if isinstance(code, str) else None)
        if not msgs:
            continue
        csim = _status_passed(res.get("csim_status"))
        cosim = _status_passed(res.get("cosim_status"))
        # Many flash cells never ran generated csim — keep as synth_positive only if success
        synth_ok = bool(res.get("success")) or bool(res.get("final_report"))
        lat = _latency_from_report(res.get("final_report"))
        row = _build_row(
            benchmark=str(res.get("benchmark") or stem),
            source=f"pc2_multistep:{hist_path.relative_to(pc2_root)}",
            messages=msgs,
            synth_ok=synth_ok,
            csim_passed=csim,
            cosim_passed=cosim,
            latency_cycles=lat,
            cosim_cycles=None,
            model=(hist.get("model") if isinstance(hist, dict) else None),
            extra={"cell": str(parent)},
        )
        if row:
            rows.append(row)
    return rows


def dedupe(rows: Iterable[MinedRow]) -> list[MinedRow]:
    best: dict[tuple, MinedRow] = {}
    rank = {"validated_positive": 3, "synth_positive": 2, "negative": 1, "no_code": 0}

    def key(r: MinedRow):
        sha = (r.metadata or {}).get("code_sha256") or _sha(r.messages[-1]["content"])
        return (r.benchmark, sha)

    def better(a: MinedRow, b: MinedRow) -> MinedRow:
        # higher quality, then higher tier, then prefer cosim latency present
        ra, rb = rank.get(a.quality_label, 0), rank.get(b.quality_label, 0)
        if ra != rb:
            return a if ra > rb else b
        ta = int((a.metadata or {}).get("correctness_tier") or 0)
        tb = int((b.metadata or {}).get("correctness_tier") or 0)
        if ta != tb:
            return a if ta > tb else b
        # lower latency if both have it
        def eff(m):
            if m.get("cosim_passed") is True and m.get("cosim_cycles") is not None:
                return float(m["cosim_cycles"])
            if m.get("latency_cycles") is not None:
                return float(m["latency_cycles"])
            return float("inf")
        return a if eff(a.metadata) <= eff(b.metadata) else b

    for r in rows:
        k = key(r)
        if k not in best:
            best[k] = r
        else:
            best[k] = better(best[k], r)
    return list(best.values())


def write_corpus(rows: list[MinedRow], out_dir: Path, include_synth_positive: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "sft_all.jsonl"
    pos_path = out_dir / "sft_positive.jsonl"
    with all_path.open("w") as fall, pos_path.open("w") as fpos:
        n_all = n_pos = 0
        for r in rows:
            fall.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")
            n_all += 1
            keep = r.quality_label == "validated_positive" or (
                include_synth_positive and r.quality_label == "synth_positive"
            )
            if keep:
                fpos.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")
                n_pos += 1

    # split positive set
    by_split: dict[str, list[MinedRow]] = defaultdict(list)
    for r in rows:
        keep = r.quality_label == "validated_positive" or (
            include_synth_positive and r.quality_label == "synth_positive"
        )
        if keep:
            by_split[r.split].append(r)
    for split in ("train", "val", "test"):
        path = out_dir / f"{split}.jsonl"
        with path.open("w") as f:
            for r in by_split.get(split, []):
                # TRL-ready: messages only + light metadata sidecar fields
                f.write(
                    json.dumps(
                        {
                            "messages": r.messages,
                            "benchmark": r.benchmark,
                            "split": r.split,
                            "quality_label": r.quality_label,
                            "metadata": r.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    counts_q = Counter(r.quality_label for r in rows)
    counts_src = Counter(r.source.split(":")[0] for r in rows)
    counts_split = {s: len(by_split.get(s, [])) for s in ("train", "val", "test")}
    tier = Counter(int((r.metadata or {}).get("correctness_tier") or 0) for r in rows if r.quality_label == "validated_positive")
    manifest = {
        "record_count_all": n_all,
        "record_count_positive": n_pos,
        "counts_by_quality": dict(counts_q),
        "counts_by_source_prefix": dict(counts_src),
        "positive_by_split": counts_split,
        "validated_tier_counts": dict(tier),
        "include_synth_positive": include_synth_positive,
        "files": {
            "sft_all": all_path.name,
            "sft_positive": pos_path.name,
            "train": "train.jsonl",
            "val": "val.jsonl",
            "test": "test.jsonl",
        },
        "policy": {
            "validated_positive": "code + synth + csim_passed",
            "synth_positive": "code + synth, csim unknown",
            "tiers": "4=csim&cosim, 3=cosim only, 2=csim only, 1=synth only",
            "note": "c2hls_tmp HLS workdirs are not mined; use artifacts/pc2 cells",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=REPO)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dir (default: rl/prepared/mined_sft)",
    )
    p.add_argument("--include-synth-positive", action="store_true")
    p.add_argument("--skip-pc2", action="store_true")
    p.add_argument("--skip-matrix", action="store_true")
    args = p.parse_args()
    repo: Path = args.repo
    out = args.output or (repo / "rl" / "prepared" / "mined_sft")

    rows: list[MinedRow] = []
    if not args.skip_matrix:
        for name in (
            "results_matrix_u280_fullcosim",
            "results_matrix_u280_fullcosim_extended",
        ):
            d = repo / name
            print(f"mining {d} …")
            part = mine_results_tree(d, tag=name)
            print(f"  +{len(part)}")
            rows.extend(part)
        print("mining results/ …")
        part = mine_results_tree(repo / "results", tag="results")
        print(f"  +{len(part)}")
        rows.extend(part)

    if not args.skip_pc2:
        pc2 = repo / "artifacts" / "pc2"
        print(f"mining {pc2} variants (may take a few minutes) …")
        part = mine_pc2_variants(pc2)
        print(f"  +{len(part)}")
        rows.extend(part)

    print(f"raw rows: {len(rows)}; deduping …")
    rows = dedupe(rows)
    print(f"deduped: {len(rows)}")
    manifest = write_corpus(rows, out, include_synth_positive=args.include_synth_positive)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
