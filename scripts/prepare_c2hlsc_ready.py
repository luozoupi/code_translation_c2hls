#!/usr/bin/env python3
"""Ingest Lucaz97/c2hlsc Option-A kernels into c2hlsc_ready/c2hlsc_*.

Naive no-pragma HLS baselines: merge yaml includes + orig_code, strip HLS
pragmas for plain.cpp, curate testbenches from upstream test_code.
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
)
from prepare_benchmarks import _strip_hls_constructs  # noqa: E402

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

DEFAULT_SRC = Path("/scratch/hpc-prf-llmfpga/asa582/projects/c2hlsc")
DEFAULT_OUT = REPO / "related_work/benchmarks/HLSFactory_benchmarks/c2hlsc_ready"

PART = "xcu280-fsvh2892-2L-e"
CLOCK_NS = 3.33

# Option A: yaml benches whose orig_code is real C/C++.
OPTION_A_BENCHES = [
    "ascon",
    "block",
    "cusums",
    "des",
    "filter",
    "four_parallel",
    "four_sequential",
    "kmp",
    "monobit",
    "nw",
    "overlapping",
    "present",
    "quicksort",
    "repeated_four_p",
    "repeated_four_s",
    "runs",
    "sha256",
    "two_parallel",
    "two_sequential",
]

_PRAGMA_RE = re.compile(r"^\s*(?://+\s*)?#pragma\s+(HLS|ACCEL)\b", re.IGNORECASE)
_FUNC_BODY_RE = re.compile(
    r"(?m)^(?P<sig>[ \t]*(?:[\w:<>,~*&]+[ \t]+)+{top}\s*\((?:[^;]*?)\))\s*\{{",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML required: pip install pyyaml")
    return yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or {}


def _resolve_cfg(src_root: Path, name: str) -> tuple[Path, dict[str, Any]]:
    bench_dir = src_root / "inputs" / name
    for cand in (
        bench_dir / f"config_{name}.yaml",
        bench_dir / "config.yaml",
        *sorted(bench_dir.glob("config*.yaml")),
    ):
        if cand.is_file():
            return cand, _load_yaml(cand)
    raise FileNotFoundError(f"no config yaml under {bench_dir}")


def _read_rel(src_root: Path, rel: str | None) -> str:
    if not rel:
        return ""
    path = Path(rel)
    if not path.is_absolute():
        path = src_root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def _path_rel(src_root: Path, rel: str | None) -> Path | None:
    if not rel:
        return None
    path = Path(rel)
    if not path.is_absolute():
        path = src_root / path
    return path if path.is_file() else None


def _top_already_extern_c(src: str, top: str) -> bool:
    m = re.search(rf"(?m)^[ \t]*(?:[\w:<>,~*&]+[ \t]+)+{re.escape(top)}\s*\(", src)
    if not m:
        return False
    window = src[max(0, m.start() - 120) : m.start()]
    return bool(re.search(r'extern\s*"C"', window))


def _find_function_body_span(src: str, top: str) -> tuple[int, int] | None:
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
    """Wrap the full translation unit in extern \"C\" for stable csim linkage."""
    if src.lstrip().startswith('extern "C"'):
        return src
    return f'extern "C" {{\n{src.rstrip()}\n}}\n'


def _count_pragmas(text: str) -> int:
    return sum(1 for ln in text.splitlines() if _PRAGMA_RE.match(ln))


def _build_kernel_text(src_root: Path, name: str, cfg: dict[str, Any]) -> tuple[str, list[Path]]:
    """Return (merged kernel source, extra support files to copy beside it)."""
    includes_path = _path_rel(src_root, cfg.get("includes"))
    orig_path = _path_rel(src_root, cfg.get("orig_code"))
    if orig_path is None:
        raise FileNotFoundError(f"{name}: orig_code missing")

    parts: list[str] = []
    support: list[Path] = []

    # Prefer .h/.hpp as a real header file; .txt includes are inlined.
    header_as_file = False
    if includes_path and includes_path.suffix.lower() in {".h", ".hh", ".hpp", ".hxx"}:
        header_as_file = True
        parts.append(f'#include "{includes_path.name}"\n')
        support.append(includes_path)
    elif includes_path:
        inc = includes_path.read_text(encoding="utf-8", errors="ignore")
        if name == "ascon":
            # Rewrite Catapult-style relative includes to local copies.
            inc = inc.replace('../include/ascon_permutation.h', 'ascon_permutation.h')
            inc = inc.replace('../include/hex_utils.h', 'hex_utils.h')
            inc = inc.replace('"../include/ascon_permutation.h"', '"ascon_permutation.h"')
            inc = inc.replace('"../include/hex_utils.h"', '"hex_utils.h"')
        parts.append(inc.rstrip() + "\n\n")

    if name == "ascon":
        # Upstream ascon.c is already an amalgam (hex_utils + permutation + cipher).
        # Only copy headers for include resolution; do not re-append .c helpers.
        include_dir = src_root / "include"
        for h in ("ascon_permutation.h", "hex_utils.h"):
            hp = include_dir / h
            if hp.is_file():
                support.append(hp)

    parts.append(f"\n/* ==== {orig_path.name} ==== */\n")
    orig = orig_path.read_text(encoding="utf-8", errors="ignore")
    if name == "sha256":
        # Upstream calls pass SHA256_CTX* where state_t* is required (C++ reject).
        orig = orig.replace(
            "sha256_transform(ctx, ctx->data);",
            "sha256_transform(&ctx->state, &ctx->data);",
        )
    parts.append(orig.rstrip() + "\n")

    _ = header_as_file
    return "".join(parts), support


_PARAM_SPLIT = re.compile(r",(?![^<>]*>)")


def _parse_top_signature(src: str, top: str) -> tuple[str, list[dict[str, str]]] | None:
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
        pname = name_m.group(1)
        typ = chunk[: name_m.start()].strip()
        typ = re.sub(r"\s+", " ", typ)
        params.append({"type": typ, "name": pname, "dims": dims})
    return "void", params


def _emit_smoke_tb(out_dir: Path, top: str, kernel_src: str) -> Path:
    parsed = _parse_top_signature(kernel_src, top)
    lines = [
        "#include <stdio.h>",
        "#include <string.h>",
        "#include <stdlib.h>",
        "#include <stdint.h>",
        "",
        'extern "C" {',
    ]
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
    else:
        _, params = parsed
        decl_parts = []
        for p in params:
            d = "".join(f"[{d}]" if d else "[]" for d in p["dims"]) if p["dims"] else ""
            decl_parts.append(f"{p['type']} {p['name']}{d}")
        lines.append(f"void {top}({', '.join(decl_parts)});")
        lines.append("}")
        lines.append("")
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
                if typ in {"float", "double"}:
                    main_body.append(f"  {typ} {v} = ({typ})32;")
                elif "int" in typ or typ in {"size_t", "unsigned"}:
                    main_body.append(f"  {typ} {v} = ({typ})32;")
                else:
                    main_body.append(f"  {typ} {v}{{}};")
                call_args.append(f"&{v}" if "*" in p["type"] else v)
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


def _tb_safe_includes_snippet(inc: str) -> list[str]:
    """Keep defines/typedefs/enums; convert file-scope globals to extern; drop fns."""
    lines_out: list[str] = []
    brace_depth = 0
    keep_aggregate = False
    skip_fn = False
    for ln in inc.splitlines():
        if re.match(r"^\s*#\s*include\s*", ln):
            continue
        if re.match(r"^\s*#\s*define\b", ln):
            lines_out.append(ln)
            continue

        opens = ln.count("{")
        closes = ln.count("}")

        if keep_aggregate or (
            brace_depth == 0
            and re.match(r"^\s*(typedef\s+)?(struct|enum|union)\b", ln)
        ):
            keep_aggregate = True
            lines_out.append(ln)
            brace_depth += opens - closes
            # typedef struct\n{ ... } name;  — stay open until braces close or ';'
            if ";" in ln and brace_depth <= 0:
                keep_aggregate = False
                brace_depth = 0
            continue

        if skip_fn:
            brace_depth += opens - closes
            if brace_depth <= 0:
                skip_fn = False
                brace_depth = 0
            continue

        if brace_depth == 0 and re.match(r"^\s*typedef\b", ln) and ";" in ln:
            lines_out.append(ln)
            continue

        m = re.match(
            r"^\s*(?:static\s+)?(?:const\s+)?([\w\s\*]+?)\s+([A-Za-z_]\w*)\s*(?:\[[^\]]*\])*\s*(=|;)",
            ln,
        )
        if m and not re.search(r"\b(void|return|if|for|while)\b", m.group(1)):
            typ = re.sub(r"\s+", " ", m.group(1)).strip()
            if typ not in {"struct", "enum", "typedef", "union"}:
                name = m.group(2)
                dims_all = "".join(re.findall(r"\[[^\]]*\]", ln.split("=")[0]))
                lines_out.append(f"extern {typ} {name}{dims_all};")
            if opens > closes:
                skip_fn = True
                brace_depth = opens - closes
            continue

        if re.search(r"\)\s*\{", ln) or (
            re.match(r"^\s*[\w:\*<>,\s]+\s+[A-Za-z_]\w*\s*\([^;]*\)\s*$", ln)
        ):
            if opens > closes or "{" in ln:
                skip_fn = True
                brace_depth = max(1, opens - closes)
            continue
    return lines_out


def _prototypes_from_kernel(kernel_src: str, names: set[str]) -> list[str]:
    out: list[str] = []
    for name in sorted(names):
        m = re.search(
            rf"(?m)^[ \t]*((?:[\w:<>,~*&]+\s+)+){re.escape(name)}\s*\(([^;{{]*)\)\s*\{{",
            kernel_src,
        )
        if not m:
            continue
        ret = re.sub(r"\s+", " ", m.group(1)).strip()
        params = re.sub(r"\s+", " ", m.group(2)).strip()
        out.append(f"{ret} {name}({params});")
    return out


def _curate_testbench(
    src_root: Path,
    name: str,
    cfg: dict[str, Any],
    out_dir: Path,
    *,
    top: str,
    kernel_src: str,
    header_names: list[str],
) -> tuple[Path, bool]:
    """Return (tb_path, smoke). Prefer upstream test_code; else smoke."""
    test_path = _path_rel(src_root, cfg.get("test_code"))
    if test_path is None:
        return _emit_smoke_tb(out_dir, top, kernel_src), True

    raw = test_path.read_text(encoding="utf-8", errors="ignore")
    if "main(" not in raw and "main (" not in raw:
        return _emit_smoke_tb(out_dir, top, kernel_src), True

    preamble = [
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <stdint.h>",
        "#include <string.h>",
        "#include <math.h>",
        "#include <memory.h>",
        "#include <stddef.h>",
    ]

    includes_path = _path_rel(src_root, cfg.get("includes"))
    # Prefer TB-safe snippets over #including definition-bearing headers (present/des).
    if includes_path:
        inc = includes_path.read_text(encoding="utf-8", errors="ignore")
        if name == "ascon":
            inc = inc.replace("../include/ascon_permutation.h", "ascon_permutation.h")
            inc = inc.replace("../include/hex_utils.h", "hex_utils.h")
        preamble.append("")
        preamble.append("/* TB-safe includes excerpt */")
        preamble.extend(_tb_safe_includes_snippet(inc))
    elif header_names:
        for h in header_names:
            preamble.append(f'#include "{h}"')

    preamble.append("")
    preamble.append('extern "C" {')
    parsed = _parse_top_signature(kernel_src, top)
    if parsed is not None:
        _, params = parsed
        decl_parts = []
        for p in params:
            d = "".join(f"[{d}]" if d else "[]" for d in p["dims"]) if p["dims"] else ""
            decl_parts.append(f"{p['type']} {p['name']}{d}")
        ret_m = re.search(
            rf"(?m)^[ \t]*((?:[\w:<>,~*&]+\s+)+){re.escape(top)}\s*\(",
            kernel_src,
        )
        ret = "void"
        if ret_m:
            ret = re.sub(r"\s+", " ", ret_m.group(1)).strip() or "void"
        preamble.append(f"{ret} {top}({', '.join(decl_parts)});")
    else:
        preamble.append(f"void {top}();")

    helper_names = set()
    if name == "ascon":
        helper_names |= {"hex_to_bytes", "bytes_to_hex", "decrypt"}
    if name == "des":
        helper_names |= {"des_key_setup", "des_crypt"}
    if name == "sha256":
        helper_names |= {"sha256_init", "sha256_update", "sha256_final", "sha256_transform"}
    if name == "present":
        helper_names |= {"present80_encryptBlock", "present80_decryptBlock"}
    preamble.extend(_prototypes_from_kernel(kernel_src, helper_names - {top}))
    preamble.append("}")
    preamble.append("")

    body_lines = []
    for ln in raw.splitlines():
        if re.match(r"^\s*#\s*include\s*", ln):
            continue
        # kmp: string literal is STRING_SIZE chars; need +1 for NUL
        if name == "kmp" and re.search(r"char\s+tr\[204\]", ln):
            ln = re.sub(r"char\s+tr\[204\]", "char tr[STRING_SIZE + 1]", ln)
        if name == "sha256":
            ln = ln.replace(
                "sha256_update(&(ctx.data) ,&(ctx.datalen), &(ctx.state), &(ctx.bitlen), text, strlen(text));",
                "sha256_update(&(ctx.data), &(ctx.datalen), &(ctx.state), &(ctx.bitlen), (data_t*)text, strlen((const char*)text));",
            )
        body_lines.append(ln)
    body = "\n".join(body_lines).strip() + "\n"

    text = "\n".join(preamble) + "\n" + body
    path = out_dir / "testbench.cpp"
    path.write_text(text, encoding="utf-8")
    return path, False


def _materialize_one(name: str, src_root: Path, out_root: Path) -> dict[str, Any]:
    cfg_path, cfg = _resolve_cfg(src_root, name)
    top = str(cfg.get("top_function") or name)
    orig = _path_rel(src_root, cfg.get("orig_code"))
    if orig is None or orig.suffix.lower() not in {".c", ".cc", ".cpp", ".cxx"}:
        return {"bench": name, "status": "skip_non_cxx", "orig": cfg.get("orig_code")}

    bench_name = f"c2hlsc_{name}"
    out_dir = out_root / bench_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    kernel_text, support_paths = _build_kernel_text(src_root, name, cfg)
    tmp_kernel = out_dir / "_kernel_merged.cpp"
    tmp_kernel.write_text(kernel_text, encoding="utf-8")

    header_path = None
    includes_path = _path_rel(src_root, cfg.get("includes"))
    if includes_path and includes_path.suffix.lower() in {".h", ".hh", ".hpp", ".hxx"}:
        header_path = includes_path

    # Copy support/headers first so adapt sees them if needed.
    root_support: list[Path] = []
    for sp in support_paths:
        if header_path and sp.resolve() == header_path.resolve():
            continue
        root_support.append(sp)

    info = adapt_external_kernel(
        kernel_path=tmp_kernel,
        header_path=header_path,
        testbench_path=None,
        root_support_paths=root_support or None,
        bench_name=bench_name,
        output_dir=out_dir,
        source_repo="Lucaz97/c2hlsc",
        top_function=top,
    )

    # Overwrite baseline with merged text (adapt already wrote it from tmp).
    # Ensure extern "C" on top for both gold and plain.
    for fname in ("hls_baseline.cpp", "plain.cpp"):
        fpath = out_dir / fname
        if not fpath.is_file():
            continue
        text = fpath.read_text(encoding="utf-8", errors="ignore")
        if fname == "hls_baseline.cpp":
            text = kernel_text
        else:
            plain, _ = _strip_hls_constructs(kernel_text)
            text = plain
        fpath.write_text(_ensure_extern_c_top(text, top), encoding="utf-8")

    shutil.copy2(out_dir / "hls_baseline.cpp", out_dir / "gold_hls_source.cpp")
    tmp_kernel.unlink(missing_ok=True)

    header_names = []
    if header_path:
        header_names.append(header_path.name)
    for sp in support_paths:
        if sp.suffix.lower() in {".h", ".hh", ".hpp", ".hxx"} and sp.name not in header_names:
            # already copied by adapt as root support
            if (out_dir / sp.name).is_file() and sp.name not in header_names:
                header_names.append(sp.name)

    kernel_src = (out_dir / "hls_baseline.cpp").read_text(encoding="utf-8", errors="ignore")
    _tb_path, smoke_tb = _curate_testbench(
        src_root,
        name,
        cfg,
        out_dir,
        top=top,
        kernel_src=kernel_src,
        header_names=header_names,
    )

    meta_path = out_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(
        {
            "benchmark": bench_name,
            "c2hlsc_source_dir": str((src_root / "inputs" / name).resolve()),
            "c2hlsc_config": str(cfg_path.resolve()),
            "gold_hls_source_file": "gold_hls_source.cpp",
            "gold_hls_baseline_file": "hls_baseline.cpp",
            "plain_c_file": "plain.cpp",
            "kernel_file": "hls_baseline.cpp",
            "kernel_top": top,
            "hls_top": top,
            "translated_hls_top": top,
            "testbench_file": "testbench.cpp",
            "supports_csim": True,
            "supports_cosim": True,
            "cosim_required": False,
            "target_part": PART,
            "target_clock_ns": CLOCK_NS,
            "cosim_depths": infer_cosim_depths(kernel_src, top)
            or meta.get("cosim_depths")
            or {},
            "smoke_testbench": smoke_tb,
            "extern_c_top": True,
            "c2hlsc_mode": cfg.get("mode"),
            "c2hlsc_hierarchical": cfg.get("hierarchical"),
            "naive_no_pragma_baseline": True,
            "pragma_hls_count_raw": _count_pragmas(kernel_text),
            "preferred_gt_file": "hls_baseline.cpp",
        }
    )
    if header_path:
        meta["header_file"] = header_path.name
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

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
        "pragma_hls_count": _count_pragmas(kernel_text),
        "output_dir": str(out_dir),
        **{k: info.get(k) for k in ("top_function", "plain_lines", "raw_lines")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--benches",
        default="",
        help="Comma list of c2hlsc input dir names; default Option-A set",
    )
    args = ap.parse_args()

    names = (
        [x.strip() for x in args.benches.split(",") if x.strip()]
        if args.benches
        else list(OPTION_A_BENCHES)
    )
    if not args.source_root.is_dir():
        print(f"ERROR: source root missing: {args.source_root}", file=sys.stderr)
        return 2

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        try:
            row = _materialize_one(name, args.source_root, args.output_root)
        except Exception as exc:  # noqa: BLE001 — collect per-bench failures
            row = {"bench": f"c2hlsc_{name}", "status": "error", "error": str(exc)}
        rows.append(row)
        print(json.dumps(row))

    manifest = {
        "schema": "c2hlsc_ready_manifest_v1",
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
