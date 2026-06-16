"""Per-cell HLS technique detector.

For every cell directory under a results dir, read <bench>_generated.cpp and
emit <bench>_techniques_detected.json. Output is structured so it's directly
comparable to the skills_log feature: an inferred_skill_ids list mirrors
skills_log.unique_skill_ids, while raw pragma + structural signals stay
available for audit.

This detects techniques PRESENT in the generated code; it cannot tell whether
the model arrived at them via the skill block or via its trained priors. The
comparison across skills_off / skills_on(basic) / skills_on(ext) tells you
whether the skill block actually shifts what the model emits.

Detection layers:
  1. Pragma regex     — PIPELINE, UNROLL, ARRAY_PARTITION, DATAFLOW, INLINE,
                        INTERFACE m_axi (with widening/burst args), DEPENDENCE,
                        LOOP_TRIPCOUNT, LATENCY, BIND_STORAGE, RESOURCE
  2. Structural       — local tile buffers (l_X[N][M] arrays), multi-gmem
                        bundles, ap_uint<512> wide-bus ABI, ping-pong arrays,
                        multiple partial-sum accumulators
  3. Avoid violations — UNROLL on m_axi-direct loop, m_axi widening with no
                        lane-parallel compute, UNROLL on FP reduction with no
                        tolerance check in TB (heuristic)
  4. Skill-ID infer   — pragma+structural signals → skill catalog IDs.

Usage:
  python3 _detect_techniques.py <results_dir>
        [--ext]                    # also match extension-skill IDs
        [--force]                  # overwrite existing sidecars
        [--print-summary]          # print per-bench summary at the end
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Regex toolkit. All HLS pragmas are matched on the leading `#pragma HLS`
# token, case-insensitive on the pragma name itself (Vitis accepts both).
# ---------------------------------------------------------------------------

PRAGMA_LINE_RE = re.compile(r"^\s*#\s*pragma\s+HLS\s+(?P<body>.+?)\s*$",
                            re.IGNORECASE | re.MULTILINE)


def _parse_pragmas(code: str) -> list[dict]:
    """Return one dict per `#pragma HLS ...` line: name, args (string), the
    line number (1-based), and an args dict of key=value pairs."""
    out = []
    for m in PRAGMA_LINE_RE.finditer(code):
        body = m.group("body")
        parts = body.split(None, 1)
        name = parts[0].upper()
        args_str = parts[1] if len(parts) > 1 else ""
        kv = {}
        for tok in args_str.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                kv[k.lower()] = v
            else:
                kv[tok.lower()] = True
        line_no = code.count("\n", 0, m.start()) + 1
        out.append({"name": name, "args": args_str, "kv": kv, "line": line_no})
    return out


def _enclosing_loop(code_lines: list[str], pragma_line: int) -> str | None:
    """Walk backward from pragma_line looking for the nearest `for (` or
    `while (`. Return that line stripped, or None."""
    for i in range(pragma_line - 1, max(0, pragma_line - 8), -1):
        ln = code_lines[i - 1] if i - 1 < len(code_lines) else ""
        s = ln.strip()
        if s.startswith("for ") or s.startswith("for(") or s.startswith("while "):
            return s
    return None


def _enclosing_body_lines(code_lines: list[str], pragma_line: int,
                          window: int = 12) -> list[str]:
    """Lines immediately after the pragma until the matching closing brace
    or `window` lines, whichever comes first. Used to inspect loop bodies."""
    out = []
    depth = 0
    started = False
    for i in range(pragma_line, min(len(code_lines), pragma_line + window)):
        ln = code_lines[i]
        out.append(ln)
        depth += ln.count("{") - ln.count("}")
        if "{" in ln:
            started = True
        if started and depth <= 0:
            break
    return out


# ---------------------------------------------------------------------------
# Structural detectors
# ---------------------------------------------------------------------------

LOCAL_BUFFER_RE = re.compile(
    r"^\s*(?:double|float|int|long|short|char|ap_(?:uint|int|fixed)<[^>]+>)\s+"
    r"(?P<name>[lL]_[A-Za-z0-9_]+|local_[A-Za-z0-9_]+|tile_[A-Za-z0-9_]+|"
    r"buf_[A-Za-z0-9_]+|[A-Za-z]_(?:buf|tile|local)[A-Za-z0-9_]*|"
    r"l[A-Z][A-Za-z0-9_]*)"  # also lA, lB, lTmp, etc. — common single-l prefix
    r"\s*\[",
    re.MULTILINE,
)

# True tiling requires a TILE-sized SUBSET of the original dim. Detect by
# looking for TILE/BLOCK/BLK/TS constants in local-array dimensions. Without
# one, what looks like "tiling" is actually whole-array local staging.
TILE_CONSTANT_RE = re.compile(
    r"\b(?:TILE|BLOCK|BLK|TS|BS|CHUNK|BSIZE|TILESIZE|TILE_SIZE|TILE_R|TILE_C|TILE_M|TILE_N|TILE_K)\b"
)

# Match a local-buffer declaration whose dimension expression contains a
# TILE-like constant, e.g. `double tile[TILE]` or `float buf[TILE_R][TILE_C]`.
TILED_BUFFER_RE = re.compile(
    r"^\s*(?:double|float|int|long|short|char|ap_(?:uint|int|fixed)<[^>]+>)\s+"
    r"\w+\s*\[[^\]]*(?:TILE|BLOCK|BLK|TS|BS|CHUNK|BSIZE)[^\]]*\]",
    re.MULTILINE | re.IGNORECASE,
)

WIDEBUS_RE = re.compile(r"\bap_uint\s*<\s*512\s*>|\bap_int\s*<\s*512\s*>")

PARTIAL_SUM_RE = re.compile(
    r"^\s*(?:double|float)\s+(?:partial|psum|tree)_?\d*\s*[\[=]", re.MULTILINE,
)
# Also matches arrays of accumulators like `double partial[8];`
PARTIAL_SUM_ARRAY_RE = re.compile(
    r"^\s*(?:double|float)\s+(?:partial|psum|tree|acc)\w*\s*\[\s*\d+\s*\]",
    re.MULTILINE,
)

# Heuristic for ping-pong: two local buffers with `_a`/`_b` or `_0`/`_1`
# suffix, OR explicit `ping`/`pong` naming.
PINGPONG_RE = re.compile(
    r"\b(?:ping|pong|buf_a|buf_b|buf0|buf1)\b", re.IGNORECASE,
)

SHIFT_REGISTER_RE = re.compile(
    r"\bshift_?reg\b|=\s*\w+\s*\[\s*(?:i|k|j)\s*[-+]\s*1\s*\]\s*;\s*\w+\s*\[\s*"
    r"(?:i|k|j)\s*\]\s*=",
    re.IGNORECASE,
)


def _detect_structural(code: str, pragmas: list[dict]) -> dict:
    """Return a dict of structural-pattern flags + supporting evidence."""
    local_bufs = LOCAL_BUFFER_RE.findall(code)
    has_widebus = bool(WIDEBUS_RE.search(code))
    has_partial_sum = bool(PARTIAL_SUM_RE.search(code) or PARTIAL_SUM_ARRAY_RE.search(code))
    has_pingpong = bool(PINGPONG_RE.search(code))
    has_shiftreg = bool(SHIFT_REGISTER_RE.search(code))

    # Multi-bank gmem detection: distinct bundle=gmemN literals
    bundles = set()
    for p in pragmas:
        if p["name"] == "INTERFACE":
            b = p["kv"].get("bundle")
            if b and (b.startswith("gmem") or b == "gmem"):
                bundles.add(b)
    has_multibank = len(bundles) > 1

    # True tiling: TILE-like constant appears in a local-buffer dimension.
    # Distinguishes real loop-nest tiling from "stage entire array into a
    # local copy" — both use local arrays but only the former is tiling.
    has_tiled_buffer = bool(TILED_BUFFER_RE.search(code))
    has_tile_constant = bool(TILE_CONSTANT_RE.search(code))

    # Stage-global-memory pattern: a PIPELINE'd loop body that assigns from
    # a kernel-arg array to a local buffer (load prologue) or the reverse
    # (store epilogue). This is a strong signal for hls-pipeline-stage-
    # global-memory regardless of whether actual tiling happens.
    arg_names = _extract_kernel_arg_names(code)
    has_load_prologue = False
    has_store_epilogue = False
    lines = code.splitlines()
    for p in pragmas:
        if p["name"] != "PIPELINE":
            continue
        body = "\n".join(_enclosing_body_lines(lines, p["line"]))
        for arg in arg_names:
            # local = arg[...]
            if re.search(rf"(?:l_|l[A-Z]|local_|tile_|buf_)\w*\s*\[[^\]]*\][^=]*=\s*{re.escape(arg)}\s*\[",
                         body):
                has_load_prologue = True
            # arg[...] = local
            if re.search(rf"{re.escape(arg)}\s*\[[^\]]*\][^=]*=\s*(?:l_|l[A-Z]|local_|tile_|buf_)",
                         body):
                has_store_epilogue = True

    return {
        "local_tile_buffers": sorted(set(local_bufs)),
        "n_local_buffers": len(set(local_bufs)),
        "has_widebus_abi": has_widebus,
        "has_partial_sum_accumulators": has_partial_sum,
        "has_pingpong_pattern": has_pingpong,
        "has_shift_register_pattern": has_shiftreg,
        "gmem_bundles": sorted(bundles),
        "has_multibank_gmem": has_multibank,
        "has_tiled_buffer": has_tiled_buffer,
        "has_tile_constant": has_tile_constant,
        "has_load_prologue": has_load_prologue,
        "has_store_epilogue": has_store_epilogue,
    }


# ---------------------------------------------------------------------------
# Avoid-rule violation detectors
# ---------------------------------------------------------------------------

def _detect_violations(code: str, pragmas: list[dict],
                       structural: dict) -> list[dict]:
    """Detect HLS patterns the skill catalog explicitly warns against."""
    lines = code.splitlines()
    out = []

    # 1. avoid-over-unroll-axi-dep / hls-avoid-unroll-memory-bound-loop:
    #    UNROLL on a loop that directly indexes a function-arg (m_axi) array
    #    rather than a local buffer.
    arg_names = _extract_kernel_arg_names(code)
    for p in pragmas:
        if p["name"] != "UNROLL":
            continue
        body = _enclosing_body_lines(lines, p["line"])
        body_str = "".join(body)
        # Direct access to a kernel arg from inside the unrolled loop body
        for arg in arg_names:
            if re.search(rf"\b{re.escape(arg)}\s*\[", body_str):
                out.append({
                    "rule": "avoid-over-unroll-axi-dep",
                    "line": p["line"],
                    "evidence": f"UNROLL on loop that directly indexes kernel arg `{arg}` (likely m_axi)",
                })
                break

    # 2. hls-avoid-coalescing-interface-only: m_axi widening present but no
    #    lane-parallel compute (no UNROLL on inner loops, no widebus, no
    #    partial-sum array).
    has_widening = any(
        p["name"] == "INTERFACE" and "max_widen_bitwidth" in p["kv"]
        for p in pragmas
    )
    if has_widening:
        unroll_present = any(p["name"] == "UNROLL" for p in pragmas)
        if not (unroll_present or structural["has_widebus_abi"]
                or structural["has_partial_sum_accumulators"]):
            out.append({
                "rule": "hls-avoid-coalescing-interface-only",
                "line": None,
                "evidence": "m_axi widening pragma present but compute remains scalar (no UNROLL, no wide-bus ABI, no partial-sum lanes)",
            })

    # 3. hls-guard-fp-reduction-order-preserving: UNROLL on inner loop whose
    #    body contains `+=` of a `*` expression (heuristic FP reduction). We
    #    can't reliably check the testbench from here, but the violation is
    #    flagged for downstream review.
    for p in pragmas:
        if p["name"] != "UNROLL":
            continue
        body = _enclosing_body_lines(lines, p["line"])
        body_str = "\n".join(body)
        if re.search(r"\w+\s*\+=\s*[^;]*\*", body_str):
            # Also check the loop variable looks like a typical reduction
            # accumulator (single name `sum`, `acc`, `tmp`, `result`).
            if re.search(r"\b(?:sum|acc|tmp|result|s)\s*\+=\s*[^;]*\*", body_str):
                out.append({
                    "rule": "hls-guard-fp-reduction-order-preserving",
                    "line": p["line"],
                    "evidence": "UNROLL on inner loop with scalar `+=` of multiplied terms (likely FP reduction; reassociation breaks bit-exact csim)",
                })

    return out


def _extract_kernel_arg_names(code: str) -> list[str]:
    """Best-effort: find the top-level kernel function's argument names."""
    m = re.search(
        r"extern\s+\"C\"\s*\{?\s*\n.*?void\s+\w+\s*\(([^)]*)\)",
        code, re.DOTALL,
    )
    if not m:
        m = re.search(r"void\s+kernel_\w+\s*\(([^)]*)\)", code, re.DOTALL)
    if not m:
        return []
    arglist = m.group(1)
    names = []
    for arg in arglist.split(","):
        arg = arg.strip()
        if not arg:
            continue
        # last identifier token before '[' or end is the arg name
        m2 = re.search(r"(\w+)\s*(?:\[[^\]]*\]|\b)\s*$", arg)
        if m2:
            names.append(m2.group(1))
    return names


# ---------------------------------------------------------------------------
# Skill-ID inference. The catalog uses semantic IDs so the mapping is
# tractable: a pragma+structural signal triggers one or more skill IDs.
# We aim for medium precision: only fire a skill ID when we have direct
# textual evidence in the generated code.
# ---------------------------------------------------------------------------

def _infer_skill_ids(pragmas: list[dict], structural: dict,
                     include_ext: bool = True) -> list[str]:
    out: set[str] = set()
    pnames = Counter(p["name"] for p in pragmas)
    has_pipeline = pnames["PIPELINE"] > 0
    has_unroll = pnames["UNROLL"] > 0
    has_dataflow = pnames["DATAFLOW"] > 0
    has_inline = pnames["INLINE"] > 0
    has_dep_false = any(
        p["name"] == "DEPENDENCE" and ("false" in p["args"].lower())
        for p in pragmas
    )
    has_tripcount = pnames["LOOP_TRIPCOUNT"] > 0

    partition_pragmas = [p for p in pragmas if p["name"] == "ARRAY_PARTITION"]
    has_part_cyclic = any("cyclic" in p["args"].lower() for p in partition_pragmas)
    has_part_complete = any("complete" in p["args"].lower() for p in partition_pragmas)
    has_part_block = any(re.search(r"\bblock\b", p["args"], re.IGNORECASE)
                         for p in partition_pragmas)

    interface_pragmas = [p for p in pragmas if p["name"] == "INTERFACE"]
    has_widening = any("max_widen_bitwidth" in p["kv"] for p in interface_pragmas)
    has_burst_pragmas = any(
        "max_read_burst_length" in p["kv"] or "max_write_burst_length" in p["kv"]
        or "num_read_outstanding" in p["kv"] or "num_write_outstanding" in p["kv"]
        for p in interface_pragmas
    )

    # === prompt-* (initial-translation skills) =============================
    if has_pipeline:
        out.add("prompt-pipeline")
        out.add("hls-pipeline-hot-loop-achieve-ii")
        out.add("hls-pipeline-realistic-ii-selection")
    if has_unroll:
        out.add("prompt-unroll")
        out.add("hls-unroll-independent-loop")
    if structural["n_local_buffers"] >= 1:
        # Local buffers WITHOUT a TILE constant in their dims = whole-array
        # staging, not loop-nest tiling. Stage-skill fires for both; tile-
        # specific skills only fire when a tile-sized SUBSET is staged.
        out.add("local-axi-staging-for-ii")
        if structural["has_tiled_buffer"]:
            out.add("prompt-tiling")
            out.add("hls-tile-1d-reuse-and-compute-restructure")
            if structural["n_local_buffers"] >= 2:
                out.add("hls-tile-2d-locality-and-halo")
                out.add("hls-pipeline-local-compute-after-tiling")
                if has_part_cyclic:
                    out.add("hls-tile-partition-local-buffers")
    if structural.get("has_load_prologue") or structural.get("has_store_epilogue"):
        # explicit load/store loop that crosses gmem<->local boundary
        out.add("hls-pipeline-stage-global-memory")
    if has_dataflow or structural["has_pingpong_pattern"]:
        out.add("prompt-doublebuffer")
        out.add("hls-doublebuffer-load-compute-store")
        if has_dataflow:
            out.add("hls-doublebuffer-dataflow-stage-split")
        if structural["has_pingpong_pattern"]:
            out.add("hls-doublebuffer-pingpong-local-buffers")
    if has_widening or has_burst_pragmas:
        out.add("prompt-coalescing")
        if has_widening:
            out.add("axi-burst-coalescing-narrow-safe")
            out.add("hls-coalescing-512-compound-transform")
            out.add("hls-coalescing-contiguous-access-rewrite")
            if structural["has_widebus_abi"]:
                out.add("axi-burst-widening-512")
            if has_unroll:
                out.add("hls-coalescing-compute-lane-parallelism")
                out.add("hls-coalescing-lane-parallel-reduction")

    # === partition-related ==================================================
    if has_part_cyclic:
        out.add("partition-cyclic-on-port-conflict")
        out.add("hls-partition-select-complete-cyclic-block")
        if has_unroll:
            out.add("hls-unroll-with-array-partition")
        if has_pipeline:
            out.add("hls-pipeline-bank-local-buffers")
        if has_widening and has_pipeline:
            # coalescing-partition-lane-buffers is specifically about banking
            # the LANE buffers in a coalesced design — requires widening context
            out.add("hls-coalescing-partition-lane-buffers")
        if structural["has_tiled_buffer"]:
            # tile-partition-local-buffers requires actual tile buffers, not
            # whole-array staging (the latter is just bank-local-buffers)
            out.add("hls-tile-partition-local-buffers")
    if has_part_complete or has_part_block:
        out.add("hls-partition-select-complete-cyclic-block")

    # === dependence / tripcount ===========================================
    if has_dep_false:
        out.add("dependence-inter-false-on-accum")
        out.add("hls-pipeline-resolve-false-dependence")
    if has_tripcount:
        out.add("loop-tripcount-when-bound-runtime")
        if include_ext:
            out.add("hls-translation-loop-tripcount-always")

    # === recurrence / partial-sum patterns ================================
    if structural["has_shift_register_pattern"]:
        out.add("hls-pipeline-recurrence-with-shift-register")
        out.add("hls-pipeline-handle-true-recurrence")
    if structural["has_partial_sum_accumulators"]:
        out.add("hls-unroll-reduction-partial-sums")
        if has_unroll:
            out.add("hls-unroll-independent-tasks-processing-elements")

    # === multi-bank ========================================================
    if structural["has_multibank_gmem"]:
        out.add("hls-multibank-separate-independent-arrays")
        out.add("hls-multibank-balance-memory-traffic")

    return sorted(out)


# ---------------------------------------------------------------------------
# Top-level: walk a results dir, write sidecars
# ---------------------------------------------------------------------------

def _process_cell(cell_dir: Path, bench: str, include_ext: bool,
                  force: bool) -> tuple[str, dict | None]:
    # Flash cells write *_generated.cpp; multistep cells write *_final.cpp
    # (the merged kernel after all opt steps). Try both.
    cpp = cell_dir / f"{bench}_generated.cpp"
    if not cpp.exists():
        cpp = cell_dir / f"{bench}_final.cpp"
    if not cpp.exists():
        return "skipped_no_cpp", None
    sidecar = cell_dir / f"{bench}_techniques_detected.json"
    if sidecar.exists() and not force:
        return "skipped_exists", None

    code = cpp.read_text(encoding="utf-8", errors="ignore")
    pragmas = _parse_pragmas(code)
    structural = _detect_structural(code, pragmas)
    violations = _detect_violations(code, pragmas, structural)
    inferred_ids = _infer_skill_ids(pragmas, structural, include_ext=include_ext)

    pragma_counts = dict(Counter(p["name"] for p in pragmas).most_common())
    pragma_details = [
        {"name": p["name"], "args": p["args"], "line": p["line"]}
        for p in pragmas
    ]

    # Lightweight category roll-up (mirrors what comparison aggregates on)
    categories = {
        "pipeline": int(pragma_counts.get("PIPELINE", 0) > 0),
        "unroll": int(pragma_counts.get("UNROLL", 0) > 0),
        "partition_cyclic": int(any("cyclic" in p["args"].lower()
                                    for p in pragmas if p["name"] == "ARRAY_PARTITION")),
        "partition_complete": int(any("complete" in p["args"].lower()
                                      for p in pragmas if p["name"] == "ARRAY_PARTITION")),
        "dataflow": int(pragma_counts.get("DATAFLOW", 0) > 0),
        "inline": int(pragma_counts.get("INLINE", 0) > 0),
        "dependence_false": int(any("false" in p["args"].lower()
                                    for p in pragmas if p["name"] == "DEPENDENCE")),
        "loop_tripcount": int(pragma_counts.get("LOOP_TRIPCOUNT", 0) > 0),
        "interface_widening": int(any("max_widen_bitwidth" in p["kv"]
                                      for p in pragmas if p["name"] == "INTERFACE")),
        "interface_burst": int(any(("max_read_burst_length" in p["kv"]
                                    or "max_write_burst_length" in p["kv"])
                                   for p in pragmas if p["name"] == "INTERFACE")),
        "tiling_buffers": int(structural["n_local_buffers"] >= 1),
        "multibank": int(structural["has_multibank_gmem"]),
        "widebus": int(structural["has_widebus_abi"]),
        "partial_sums": int(structural["has_partial_sum_accumulators"]),
        "shift_register": int(structural["has_shift_register_pattern"]),
        "pingpong": int(structural["has_pingpong_pattern"]),
    }

    payload = {
        "detector_version": "1.0",
        "generated_file": cpp.name,
        "source_lines": code.count("\n") + 1,
        "pragma_counts": pragma_counts,
        "pragma_details": pragma_details,
        "structural": structural,
        "violations": violations,
        "categories": categories,
        "inferred_skill_ids": inferred_ids,
        "n_inferred_skill_ids": len(inferred_ids),
    }

    sidecar.write_text(json.dumps(payload, indent=2))
    return "wrote", payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--ext", action="store_true",
                    help="Also infer extension-skill IDs (e.g. hls-translation-loop-tripcount-always)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing *_techniques_detected.json")
    ap.add_argument("--print-summary", action="store_true")
    args = ap.parse_args()

    if not args.results_dir.is_dir():
        print(f"not a dir: {args.results_dir}", file=sys.stderr)
        return 2

    counts = Counter()
    rows: list[tuple[str, str, dict]] = []  # (bench, cell_name, payload)
    for bench_dir in sorted(args.results_dir.iterdir()):
        if not bench_dir.is_dir():
            continue
        bench = bench_dir.name
        for cell_dir in sorted(bench_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            status, payload = _process_cell(cell_dir, bench, args.ext, args.force)
            counts[status] += 1
            if payload is not None:
                rows.append((bench, cell_dir.name, payload))

    print(f"Processed {sum(counts.values())} cells under {args.results_dir}:")
    for k, n in counts.items():
        print(f"  {k:<30}: {n}")

    if args.print_summary and rows:
        print()
        print(f"{'bench':<28}{'cell':<28}{'n_skills':>10}{'n_pragmas':>11}  top_pragmas")
        print("-" * 105)
        for bench, cell, p in rows:
            top = ", ".join(f"{k}={v}" for k, v in
                            list(p["pragma_counts"].items())[:4])
            print(f"{bench:<28}{cell:<28}{p['n_inferred_skill_ids']:>10}"
                  f"{sum(p['pragma_counts'].values()):>11}  {top}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
