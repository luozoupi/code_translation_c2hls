#!/usr/bin/env python3
"""Ingest ChatHLS benchmark_optimization kernels into chathls_ready/chathls_*.

Uses adapt_external_kernel for plain/gold packaging. When no usable testbench
exists, emits a smoke TB that zero-inits arrays and calls the top (csim gate).
Cosim is enabled when a TB exists (harness may still skip if unsupported).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dataset_pipeline.external_adapter import (  # noqa: E402
    adapt_external_kernel,
    infer_cosim_depths,
    infer_top_function,
)

DEFAULT_SRC = Path(
    "/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/"
    "ChatHLS-ACL-26/benchmark/benchmark_optimization"
)
DEFAULT_OUT = REPO / "related_work/benchmarks/HLSFactory_benchmarks/chathls_ready"

PART = "xcu280-fsvh2892-2L-e"
CLOCK_NS = 3.33

# ChatHLS dir name -> optional overrides
BENCH_SPECS: dict[str, dict[str, Any]] = {
    "atax": {},
    "bicg": {},
    "covariance": {},
    "gemm": {},
    "gemm_blocked": {},
    "gemm_ncubed": {},
    "gesummv": {},
    "kernel_2mm": {},
    "kernel_3mm": {},
    "kernel_symm": {},
    "kernel_syr2k": {},
    "kernel_syrk": {},
    "matmul": {},
    "mobilenet": {"headers": ["mobilenet.h", "weights.h"], "tb": "mobilenet_tb.cpp"},
    "mvt": {},
    "transformer": {
        "headers": ["transformer.h"],
        "tb": "tb_transformer.cpp",
        "extra_globs": ["DRAM_*.txt"],
    },
}


def _top_from_tcl(src_dir: Path) -> str | None:
    tcl = src_dir / "run_hls.tcl"
    if not tcl.is_file():
        return None
    m = re.search(r"set_top\s+(\S+)", tcl.read_text(encoding="utf-8", errors="ignore"))
    if not m:
        return None
    return m.group(1).strip().strip("\r")


def _find_kernel(src_dir: Path, name: str) -> Path | None:
    for cand in (src_dir / f"{name}.cpp", src_dir / f"{name}.c"):
        if cand.is_file():
            return cand
    cpp = sorted(src_dir.glob("*.cpp"))
    cpp = [p for p in cpp if not re.search(r"(^|_)tb(_|\.|$)|testbench", p.name, re.I)]
    return cpp[0] if cpp else None


_PARAM_SPLIT = re.compile(r",(?![^<>]*>)")


def _parse_top_signature(src: str, top: str) -> tuple[str, list[dict[str, str]]] | None:
    """Return (return_typeish, [{type, name, dims}]) for top's params."""
    func_re = re.compile(
        rf"(?:extern\s+\"C\"\s*)?(?P<ret>[\w:<>,~*&\s]+?)\b{re.escape(top)}\s*"
        r"\((?P<params>.*?)\)\s*(?:\{|;)",
        re.DOTALL,
    )
    m = func_re.search(src)
    if not m:
        return None
    params_raw = m.group("params").strip()
    if not params_raw or params_raw == "void":
        return "void", []
    params: list[dict[str, str]] = []
    for chunk in _PARAM_SPLIT.split(params_raw):
        chunk = re.sub(r"/\*.*?\*/", " ", chunk).strip()
        if not chunk:
            continue
        dims = re.findall(r"\[([^\]]*)\]", chunk)
        name_m = re.search(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)*$", chunk)
        if not name_m:
            continue
        name = name_m.group(1)
        typ = chunk[: name_m.start()].strip()
        typ = re.sub(r"\s+", " ", typ)
        params.append({"type": typ, "name": name, "dims": dims})
    return "void", params


_TYPEDEF_RE = re.compile(r"^\s*typedef\s+.+$", re.MULTILINE)
_DEFINE_RE = re.compile(r"^\s*#\s*define\s+\w+.*$", re.MULTILINE)


def _kernel_preamble_for_tb(
    kernel_src: str, *, bench_name: str, header_texts: list[str] | None = None
) -> list[str]:
    """Pull typedefs/#defines the TB needs without pulling huge headers (e.g. weights.h)."""
    lines: list[str] = []
    blob = "\n".join([kernel_src] + list(header_texts or []))
    if "ap_fixed" in blob or "ap_int" in blob:
        lines.append('#include "ap_fixed.h"')
    for m in _TYPEDEF_RE.finditer(blob):
        text = m.group(0).rstrip()
        if text not in lines:
            lines.append(text)
    for m in _DEFINE_RE.finditer(kernel_src):
        text = m.group(0).rstrip()
        if re.search(r"#\s*define\s+\w+_H\b", text):
            continue
        lines.append(text)
    if bench_name == "mobilenet":
        lines += [
            "#ifndef INPUT_IMG_SIZE",
            "#define IMG_DIM 128",
            "#define IMG_CH 3",
            "#define INPUT_IMG_SIZE (IMG_DIM * IMG_DIM * IMG_CH)",
            "#define NUM_CLASSES 5",
            "#endif",
        ]
    if bench_name == "transformer" and not any("data_t" in ln for ln in lines):
        lines.append("typedef ap_fixed<16, 5> data_t;")
    return lines


def _top_already_extern_c(src: str, top: str) -> bool:
    """True if `top`'s definition sits under an `extern \"C\"` linkage."""
    m = re.search(rf"(?m)^[ \t]*(?:[\w:<>,~*&]+[ \t]+)+{re.escape(top)}\s*\(", src)
    if not m:
        return False
    window = src[max(0, m.start() - 120) : m.start()]
    return bool(re.search(r'extern\s*"C"', window))


def _find_function_body_span(src: str, top: str) -> tuple[int, int] | None:
    """Return [start, end) spanning the top function definition (sig + body)."""
    # Anchor at line start so we don't pull a preceding typedef into the sig.
    pat = re.compile(
        rf"(?m)^(?P<sig>[ \t]*(?:[\w:<>,~*&]+[ \t]+)+{re.escape(top)}\s*\((?:[^;]*?)\))\s*\{{",
        re.DOTALL,
    )
    m = pat.search(src)
    if not m:
        return None
    start = m.start("sig")
    brace_open = m.end() - 1
    depth = 0
    i = brace_open
    while i < len(src):
        ch = src[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, i + 1
        i += 1
    return None


def _ensure_extern_c_top(src: str, top: str) -> str:
    """Wrap the top function definition in `extern \"C\" { ... }` if missing.

    Flash codegen emits `extern \"C\"` tops; gold/plain must match so one smoke
    TB (also `extern \"C\"` prototypes) links for both gold-gate and flash csim.
    """
    if _top_already_extern_c(src, top):
        return src
    span = _find_function_body_span(src, top)
    if span is None:
        return src
    start, end = span
    body = src[start:end].rstrip() + "\n"
    wrapped = f'extern "C" {{\n{body}}}\n'
    return src[:start] + wrapped + src[end:]


def _ensure_extern_c_header_decl(text: str, top: str) -> str:
    """Ensure the header's top prototype uses C linkage.

    Gold/baseline wrap ``void top(...) { ... }`` in ``extern "C"``. If the
    accompanying header declares the same symbol with C++ linkage, Vitis fails
    with HLS 207-2538 (different language linkage) — the transformer failure mode.
    """
    if re.search(
        rf'extern\s*"C"\s*(?:\{{[\s\S]*?\b{re.escape(top)}\s*\(|{re.escape(top)}\s*\()',
        text,
    ):
        return text
    pat = re.compile(
        rf"(?m)^(?P<indent>[ \t]*)(?P<decl>(?:[\w:<>,~*&]+[ \t]+)+{re.escape(top)}\s*\([^;]*\);)"
    )
    m = pat.search(text)
    if not m:
        return text
    indent = m.group("indent")
    decl = m.group("decl")
    wrapped = (
        f"{indent}#ifdef __cplusplus\n"
        f'{indent}extern "C" {{\n'
        f"{indent}#endif\n"
        f"{indent}{decl}\n"
        f"{indent}#ifdef __cplusplus\n"
        f"{indent}}}\n"
        f"{indent}#endif"
    )
    return text[: m.start()] + wrapped + text[m.end() :]


def _emit_smoke_tb(
    out_dir: Path,
    top: str,
    kernel_src: str,
    includes: list[str],
    *,
    bench_name: str,
    header_texts: list[str] | None = None,
) -> Path:
    """Emit a csim-safe smoke TB: static buffers, kernel typedefs/defines, no huge headers."""
    parsed = _parse_top_signature(kernel_src, top)
    lines = [
        "#include <stdio.h>",
        "#include <string.h>",
        "#include <stdlib.h>",
        "#include <stdint.h>",
    ]
    safe_includes = [
        i
        for i in includes
        if i not in {"weights.h", "mobilenet.h", "transformer.h"}
    ]
    for inc in safe_includes:
        lines.append(f'#include "{inc}"')
    lines.extend(
        _kernel_preamble_for_tb(
            kernel_src, bench_name=bench_name, header_texts=header_texts
        )
    )
    lines.append("")
    # Match flash/gold `extern "C"` tops so csim links (C++ TB vs C linkage).
    lines.append('extern "C" {')

    if parsed is None:
        lines += [
            f"void {top}();",
            "}",
            "",
            "int main() {",
            f"  {top}();",
            '  printf("PASS smoke\\n");',
            "  return 0;",
            "}",
            "",
        ]
        path = out_dir / "testbench.cpp"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    _, params = parsed
    decl_parts = []
    for p in params:
        d = "".join(f"[{d}]" if d else "[]" for d in p["dims"]) if p["dims"] else ""
        decl_parts.append(f"{p['type']} {p['name']}{d}")
    lines.append(f"void {top}({', '.join(decl_parts)});")
    lines.append("}")
    lines.append("")

    # Static/file-scope buffers avoid csim link/stack blowups on large PolyBench arrays.
    call_args: list[str] = []
    static_decls: list[str] = []
    main_body: list[str] = []
    for i, p in enumerate(params):
        v = f"v{i}_{p['name']}"
        if p["dims"]:
            dim_s = "".join(f"[{d if d.strip() else '1'}]" for d in p["dims"])
            static_decls.append(f"static {p['type']} {v}{dim_s};")
            main_body.append(f"  memset({v}, 0, sizeof({v}));")
            call_args.append(v)
        else:
            typ = p["type"].replace("&", "").strip()
            # Prefer >=2 so kernels like covariance's (float_n - 1) don't /0.
            if "ap_fixed" in typ or typ in {"float", "double", "t_ap_fixed", "data_t"}:
                main_body.append(f"  {typ} {v} = ({typ})32;")
            elif typ in {"int", "long", "short", "unsigned", "size_t"} or "int" in typ:
                main_body.append(f"  {typ} {v} = ({typ})32;")
            else:
                main_body.append(f"  {typ} {v} = ({typ})0;")
            call_args.append(v)

    lines.extend(static_decls)
    lines.append("")
    lines.append("int main() {")
    lines.extend(main_body)
    lines.append(f"  {top}({', '.join(call_args)});")
    lines.append('  printf("PASS smoke\\n");')
    lines.append("  return 0;")
    lines.append("}")
    lines.append("")
    path = out_dir / "testbench.cpp"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _materialize_one(name: str, src_root: Path, out_root: Path, *, force_smoke_tb: bool) -> dict[str, Any]:
    spec = BENCH_SPECS[name]
    src_dir = src_root / name
    if not src_dir.is_dir():
        return {"bench": name, "status": "missing_src"}

    top = _top_from_tcl(src_dir) or name
    kernel = _find_kernel(src_dir, name)
    if kernel is None:
        return {"bench": name, "status": "no_kernel"}

    bench_name = f"chathls_{name}"
    out_dir = out_root / bench_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    headers = [src_dir / h for h in spec.get("headers", []) if (src_dir / h).is_file()]
    header_path = headers[0] if headers else None

    # Always generate our own smoke TB — upstream ChatHLS TBs are missing types,
    # call wrong tops (transformer→top), or pull duplicate weight tables.
    tb_path = None

    extra: list[Path] = []
    for h in headers[1:]:
        extra.append(h)
    for pat in spec.get("extra_globs", []):
        extra.extend(sorted(src_dir.glob(pat)))

    info = adapt_external_kernel(
        kernel_path=kernel,
        header_path=header_path,
        testbench_path=tb_path,
        root_support_paths=extra or None,
        bench_name=bench_name,
        output_dir=out_dir,
        source_repo="ChatHLS-ACL-26",
        top_function=top,
    )

    # Align gold/plain with flash: wrap top in extern "C" so smoke TB links.
    for fname in ("hls_baseline.cpp", "plain.cpp"):
        fpath = out_dir / fname
        if fpath.is_file():
            text = fpath.read_text(encoding="utf-8", errors="ignore")
            fpath.write_text(_ensure_extern_c_top(text, top), encoding="utf-8")

    # Always keep a gold_hls_source.cpp alias expected by some loaders.
    shutil.copy2(out_dir / "hls_baseline.cpp", out_dir / "gold_hls_source.cpp")

    # Header prototype must match extern "C" gold/flash definitions.
    if header_path is not None:
        header_out = out_dir / header_path.name
        if header_out.is_file():
            header_out.write_text(
                _ensure_extern_c_header_decl(
                    header_out.read_text(encoding="utf-8", errors="ignore"),
                    top,
                ),
                encoding="utf-8",
            )

    # Copy remaining headers (weights.h needed by kernel compile, not by TB)
    for h in headers[1:]:
        shutil.copy2(h, out_dir / h.name)

    kernel_src = kernel.read_text(encoding="utf-8", errors="ignore")
    incs = [h.name for h in headers]
    header_texts = [
        h.read_text(encoding="utf-8", errors="ignore")
        for h in headers
        if h.name != "weights.h"
    ]
    _emit_smoke_tb(
        out_dir,
        top,
        kernel_src,
        incs,
        bench_name=name,
        header_texts=header_texts,
    )
    has_tb = True
    smoke_tb = True
    meta_path = out_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(
        {
            "benchmark": bench_name,
            "chathls_source_dir": str(src_dir.resolve()),
            "gold_hls_source_file": "gold_hls_source.cpp",
            "gold_hls_baseline_file": "hls_baseline.cpp",
            "plain_c_file": "plain.cpp",
            "kernel_file": "hls_baseline.cpp",
            "kernel_top": top,
            "hls_top": top,
            "translated_hls_top": top,
            "testbench_file": "testbench.cpp",
            "supports_csim": True,
            # Cosim when TB present; required=false at campaign level.
            "supports_cosim": True,
            "cosim_required": False,
            "target_part": PART,
            "target_clock_ns": CLOCK_NS,
            "cosim_depths": infer_cosim_depths(kernel_src, top) or meta.get("cosim_depths") or {},
            "smoke_testbench": smoke_tb,
            "extern_c_top": True,
        }
    )
    if header_path:
        meta["header_file"] = header_path.name
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # Minimal TCL helpers
    (out_dir / "dataset_hls.tcl").write_text(
        "\n".join(
            [
                "open_project -reset proj",
                "add_files hls_baseline.cpp",
                f"set_top {top}",
                "open_solution -reset solution",
                f"set_part {{{PART}}}",
                f"create_clock -period {CLOCK_NS} -name default",
                "csynth_design",
                "exit",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "bench": bench_name,
        "status": "ok",
        "top": top,
        "smoke_tb": smoke_tb,
        "output_dir": str(out_dir),
        **{k: info.get(k) for k in ("top_function", "plain_lines", "raw_lines")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--benches", default="", help="Comma list of ChatHLS dir names; default all")
    ap.add_argument("--force-smoke-tb", action="store_true")
    args = ap.parse_args()

    names = (
        [x.strip() for x in args.benches.split(",") if x.strip()]
        if args.benches
        else list(BENCH_SPECS.keys())
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        if name not in BENCH_SPECS:
            rows.append({"bench": name, "status": "unknown_spec"})
            continue
        rows.append(
            _materialize_one(
                name, args.source_root, args.output_root, force_smoke_tb=args.force_smoke_tb
            )
        )
        print(json.dumps(rows[-1]))

    manifest = {
        "schema": "chathls_ready_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "part": PART,
        "clock_ns": CLOCK_NS,
        "rows": rows,
    }
    (args.output_root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"done ok={ok}/{len(rows)} -> {args.output_root}")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
