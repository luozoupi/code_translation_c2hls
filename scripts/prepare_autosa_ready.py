#!/usr/bin/env python3
"""Materialize AutoSA autosa_tests kernels into autosa_ready/ for c2hls flash."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUTOSA_ROOT = Path(
    __import__("os").environ.get(
        "AUTOSA_ROOT",
        "/scratch/hpc-prf-llmfpga/asa582/projects/AutoSA",
    )
)
DEFAULT_OUT = REPO / "related_work/benchmarks/autosa_ready"

KERNELS: list[tuple[str, str]] = [
    ("mm", "autosa_tests/mm"),
    ("mm_intel", "autosa_tests/mm_intel"),
    ("mm_int16", "autosa_tests/mm_int16"),
    ("mm_hcl", "autosa_tests/mm_hcl"),
    ("mm_hcl_intel", "autosa_tests/mm_hcl_intel"),
    ("mm_hbm", "autosa_tests/mm_hbm"),
    ("mm_getting_started", "autosa_tests/mm_getting_started"),
    ("mm_catapult", "autosa_tests/mm_catapult"),
    ("mm_block_sparse", "autosa_tests/mm_block_sparse"),
    ("cnn", "autosa_tests/cnn"),
    ("lu", "autosa_tests/lu"),
    ("dnn_ops", "autosa_tests/dnn_ops"),
    ("large_mm", "autosa_tests/large/mm"),
    ("large_mm_intel", "autosa_tests/large/mm_intel"),
    ("large_mm_int16", "autosa_tests/large/mm_int16"),
    ("large_mm_int8", "autosa_tests/large/mm_int8"),
    ("large_mm_block_sparse", "autosa_tests/large/mm_block_sparse"),
    ("large_cnn", "autosa_tests/large/cnn"),
    ("large_ttm", "autosa_tests/large/ttm"),
    ("large_ttmc", "autosa_tests/large/ttmc"),
    ("large_mttkrp", "autosa_tests/large/mttkrp"),
]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _collect_defines(*texts: str) -> dict[str, int | None]:
    """Map macro names to integer values (or None for empty #define)."""
    raw: dict[str, str] = {}
    define_re = re.compile(r"#\s*define\s+(\w+)(?:\s+(.*))?\s*$")
    for text in texts:
        for line in text.splitlines():
            match = define_re.match(line.strip())
            if not match:
                continue
            raw[match.group(1)] = (match.group(2) or "").strip()

    defines: dict[str, int | None] = {}
    for _ in range(max(1, len(raw) + 1)):
        for name, value in raw.items():
            if name in defines:
                continue
            if not value:
                defines[name] = None
                continue
            if re.fullmatch(r"\d+", value):
                defines[name] = int(value)
                continue
            if value in defines and defines[value] is not None:
                defines[name] = defines[value]
    return defines


def _token_value(token: str, defines: dict[str, int | None]) -> int | None:
    token = token.strip()
    if re.fullmatch(r"\d+", token):
        return int(token)
    if token in defines:
        return defines[token]
    return None


def _eval_if_expr(expr: str, defines: dict[str, int | None]) -> bool:
    expr = expr.strip()
    defined_match = re.fullmatch(r"defined\s*\(\s*(\w+)\s*\)", expr)
    if defined_match:
        return defined_match.group(1) in defines
    for op in ("==", "!="):
        if op in expr:
            left, right = [part.strip() for part in expr.split(op, 1)]
            left_val = _token_value(left, defines)
            right_val = _token_value(right, defines)
            if left_val is None or right_val is None:
                return False
            return (left_val == right_val) if op == "==" else (left_val != right_val)
    if re.fullmatch(r"\w+", expr):
        val = _token_value(expr, defines)
        return bool(val)
    return False


def _directive_condition(stripped: str, defines: dict[str, int | None]) -> bool | None:
    if stripped.startswith("#ifdef "):
        return stripped.split()[1] in defines
    if stripped.startswith("#ifndef "):
        return stripped.split()[1] not in defines
    if stripped.startswith("#if defined(") or stripped.startswith("#if defined ("):
        sym = stripped.split("(", 1)[1].split(")", 1)[0].strip()
        return sym in defines
    if stripped.startswith("#if "):
        return _eval_if_expr(stripped[4:].strip(), defines)
    return None


def _elif_condition(stripped: str, defines: dict[str, int | None]) -> bool | None:
    if stripped.startswith("#elif defined(") or stripped.startswith("#elif defined ("):
        sym = stripped.split("(", 1)[1].split(")", 1)[0].strip()
        return sym in defines
    if stripped.startswith("#elif "):
        return _eval_if_expr(stripped[6:].strip(), defines)
    return None


def _preprocess_ifdefs(text: str, kernel_h: Path) -> str:
    header_text = kernel_h.read_text(encoding="utf-8")
    defines = _collect_defines(header_text, text)
    lines = text.splitlines()
    out: list[str] = []
    stack: list[dict[str, bool]] = []

    def emitting() -> bool:
        return all(frame["emitting"] for frame in stack)

    for line in lines:
        stripped = line.strip()
        cond = _directive_condition(stripped, defines)
        if cond is not None:
            parent_emit = emitting() if stack else True
            take = parent_emit and cond
            stack.append({"parent_emit": parent_emit, "emitting": take, "matched": take})
            continue
        if stripped.startswith("#elif"):
            if not stack:
                out.append(line)
                continue
            frame = stack[-1]
            branch_cond = _elif_condition(stripped, defines)
            if frame["matched"]:
                frame["emitting"] = False
            else:
                take = frame["parent_emit"] and bool(branch_cond)
                frame["emitting"] = take
                if take:
                    frame["matched"] = True
            continue
        if stripped.startswith("#else"):
            if stack:
                frame = stack[-1]
                if frame["matched"]:
                    frame["emitting"] = False
                else:
                    frame["emitting"] = frame["parent_emit"]
                    frame["matched"] = True
            continue
        if stripped.startswith("#endif"):
            if stack:
                stack.pop()
            continue
        if emitting():
            out.append(line)
    return "\n".join(out) + "\n"


def _function_signature(func_body: str) -> str | None:
    head = func_body.split("{", 1)[0].strip()
    match = re.search(r"\((.*)\)\s*$", head, re.DOTALL)
    if not match:
        return None
    return re.sub(r"\s+", " ", match.group(1).strip())


def _remove_function_definition(text: str, func_name: str) -> str:
    pattern = re.compile(
        rf"(?:static\s+)?(?:inline\s+)?void\s+{re.escape(func_name)}\s*\([^)]*\)\s*\{{",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return text
    brace_start = text.find("{", match.end() - 1)
    depth = 1
    i = brace_start + 1
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[: match.start()] + text[i:]


def _c_linkage_decl(top: str, signature: str) -> str:
    """C linkage so a C++ testbench can link Phase B `extern "C"` kernels."""
    return f'extern "C" void {top}({signature});'


def _ensure_extern_c_definition(body: str) -> str:
    if re.search(r'extern\s*"C"', body):
        return body
    return re.sub(
        r"(?:static\s+)?(?:inline\s+)?void\s+",
        'extern "C" void ',
        body,
        count=1,
    )


def _append_c_linkage_prototype(header: str, top: str, signature: str) -> str:
    decl = _c_linkage_decl(top, signature)
    if decl in header:
        return header
    return header.rstrip() + "\n\n#ifdef __cplusplus\n" + decl + "\n#endif\n"


def _insert_forward_decl(text: str, top: str, signature: str) -> str:
    decl_line = _c_linkage_decl(top, signature) + "\n"
    lines = text.splitlines()
    out: list[str] = []
    inserted = False
    for line in lines:
        out.append(line)
        if not inserted and line.strip().startswith("#include"):
            out.append(decl_line.rstrip("\n"))
            inserted = True
    if not inserted:
        out.insert(0, decl_line.rstrip("\n"))
    return "\n".join(out) + "\n"


def _bench_name(kernel_id: str) -> str:
    return f"autosa_{kernel_id}"


def _top_name(kernel_id: str) -> str:
    if kernel_id == "lu":
        return "lu_device"
    return f"autosa_{kernel_id}"


def _find_scop_function(text: str) -> tuple[str, str] | None:
    fn_pattern = re.compile(
        r"(?:static\s+)?(?:inline\s+)?void\s+(\w+)\s*\([^)]*\)\s*\{",
        re.MULTILINE,
    )
    matches: list[tuple[str, str]] = []
    for fn_match in fn_pattern.finditer(text):
        name = fn_match.group(1)
        if name == "main":
            continue
        start = fn_match.start()
        brace_start = text.find("{", fn_match.end() - 1)
        depth = 1
        i = brace_start + 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[start:i]
        if "#pragma scop" not in body:
            continue
        if name in {"init_array", "lu_cpu"}:
            continue
        matches.append((name, body))
    if not matches:
        return None
    for preferred in ("lu_device",):
        for name, body in matches:
            if name == preferred:
                return name, body
    return matches[0]


def _extract_main_scop(text: str) -> str | None:
    main_match = re.search(r"\bint\s+main\s*\([^)]*\)\s*\{", text)
    if not main_match:
        return None
    start = main_match.end()
    depth = 1
    i = start
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    main_body = text[start : i - 1]
    scop = re.search(r"#pragma\s+scop([\s\S]*?)#pragma\s+endscop", main_body)
    if not scop:
        return None
    return scop.group(1).strip()


def _decls_before_scop(main_body: str) -> list[str]:
    scop_pos = main_body.find("#pragma scop")
    if scop_pos < 0:
        return []
    prefix = main_body[:scop_pos]
    decls: list[str] = []
    for line in prefix.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if re.match(r"data_t\s+\w+", stripped) or re.match(r"static\s+data_t", stripped):
            decls.append(stripped.rstrip(";"))
    return decls


def _params_from_decls(decls: list[str]) -> list[str]:
    params: list[str] = []
    for decl in decls:
        decl = re.sub(r"^static\s+", "", decl).strip()
        if not decl or "[" not in decl:
            continue
        type_match = re.match(r"((?:unsigned\s+)?(?:data_t|\w+))\s+(.+)", decl)
        if not type_match:
            continue
        base_type = type_match.group(1)
        rest = type_match.group(2)
        for part in rest.split(","):
            part = part.strip()
            if not part:
                continue
            if not re.match(r"(?:unsigned\s+)?data_t\s+", part):
                part = f"{base_type} {part}"
            params.append(part)
    return params


def _params_used_in_scop(scop_body: str, decls: list[str]) -> list[str]:
    params = _params_from_decls(decls)
    used: list[str] = []
    for param in params:
        name_match = re.search(r"(\w+)\s*\[", param)
        if name_match and re.search(rf"\b{name_match.group(1)}\b", scop_body):
            used.append(param)
    return used


def _replace_main_scop_with_call(text: str, top: str, params: list[str]) -> str:
    main_match = re.search(r"\bint\s+main\s*\([^)]*\)\s*\{", text)
    if not main_match:
        return text
    start = main_match.end()
    depth = 1
    i = start
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    main_body = text[start : i - 1]
    scop = re.search(r"#pragma\s+scop[\s\S]*?#pragma\s+endscop", main_body)
    if not scop:
        return text
    arg_names = []
    for param in params:
        m = re.search(r"(\w+)\s*\[", param)
        if m:
            arg_names.append(m.group(1))
    call = f"  {top}({', '.join(arg_names)});"
    new_main_body = main_body[: scop.start()] + call + main_body[scop.end() :]
    return text[:start] + new_main_body + text[i - 1 :]


def prepare_one(kernel_id: str, rel_dir: str, out_root: Path, *, dry_run: bool = False) -> dict:
    src_dir = AUTOSA_ROOT / rel_dir
    kernel_c = src_dir / "kernel.c"
    kernel_h = src_dir / "kernel.h"
    if not kernel_c.is_file():
        raise FileNotFoundError(kernel_c)
    if not kernel_h.is_file():
        raise FileNotFoundError(kernel_h)

    bench = _bench_name(kernel_id)
    top = _top_name(kernel_id)
    out_dir = out_root / bench
    text = kernel_c.read_text(encoding="utf-8")
    text = _preprocess_ifdefs(text, kernel_h)

    signature = ""
    scop_fn = _find_scop_function(text)
    if scop_fn:
        top, plain_body = scop_fn
        signature = _function_signature(plain_body) or ""
        plain_cpp = f'#include "kernel.h"\n\n{_ensure_extern_c_definition(plain_body)}\n'
        testbench_cpp = _remove_function_definition(text, top)
        if signature:
            testbench_cpp = _insert_forward_decl(testbench_cpp, top, signature)
    else:
        scop_body = _extract_main_scop(text)
        if not scop_body:
            raise ValueError(f"no #pragma scop found in {kernel_c}")
        main_match = re.search(r"\bint\s+main\s*\([^)]*\)\s*\{", text)
        assert main_match
        start = main_match.end()
        depth = 1
        i = start
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        main_body = text[start : i - 1]
        decls = _decls_before_scop(main_body)
        params = _params_used_in_scop(scop_body, decls)
        signature = ", ".join(params)
        plain_cpp = (
            '#include "kernel.h"\n\n'
            f'extern "C" void {top}({signature}) {{\n'
            f"{scop_body}\n"
            "}\n"
        )
        testbench_cpp = _replace_main_scop_with_call(text, top, params)
        prefix, _, _ = testbench_cpp.partition("int main")
        if f"void {top}(" not in prefix:
            testbench_cpp = _insert_forward_decl(testbench_cpp, top, signature)

    hls_baseline = plain_cpp
    meta = {
        "benchmark": bench,
        "source_repo": "AutoSA",
        "corpus": "autosa_ready",
        "algorithm_source_path": str(kernel_c.resolve()),
        "gold_hls_source_path": str(kernel_c.resolve()),
        "gold_hls_source_file": "gold_hls_source.cpp",
        "gold_hls_baseline_file": "hls_baseline.cpp",
        "kernel_file": "plain.cpp",
        "plain_c_file": "plain.cpp",
        "header_file": "kernel.h",
        "testbench_file": "testbench.cpp",
        "baseline_variant": f"{bench}_0_baseline",
        "translated_hls_top": top,
        "hls_top": top,
        "kernel_top": top,
        "support_files": [],
        "include_dirs": [],
        "supports_csim": True,
        "supports_cosim": False,
        "cosim_depths": {},
        "target_part": "xcu280-fsvh2892-2L-e",
        "target_clock_ns": 3.33,
        "synth_timeout_s": 14400,
        "csim_timeout_s": 1800,
        "skip_phase_a": False,
        "variants": [
            {
                "name": f"{bench}_0_baseline",
                "file": "hls_baseline.cpp",
                "source_path": str(kernel_c.resolve()),
            }
        ],
        "provenance": {
            "autosa_kernel_id": kernel_id,
            "autosa_rel_dir": rel_dir,
            "plain_c_sha256": _sha256(plain_cpp),
            "gold_hls_baseline_sha256": _sha256(hls_baseline),
        },
    }

    if dry_run:
        return {"benchmark": bench, "top": top, "out_dir": str(out_dir), "dry_run": True}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plain.cpp").write_text(plain_cpp, encoding="utf-8")
    (out_dir / "hls_baseline.cpp").write_text(hls_baseline, encoding="utf-8")
    (out_dir / "gold_hls_source.cpp").write_text(hls_baseline, encoding="utf-8")
    (out_dir / "testbench.cpp").write_text(testbench_cpp, encoding="utf-8")
    header_text = kernel_h.read_text(encoding="utf-8")
    if signature:
        header_text = _append_c_linkage_prototype(header_text, top, signature)
    (out_dir / "kernel.h").write_text(header_text, encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {"benchmark": bench, "top": top, "out_dir": str(out_dir), "ok": True}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare autosa_ready corpus for c2hls flash")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--kernel", action="append", dest="kernels", help="limit to kernel id(s)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = KERNELS
    if args.kernels:
        allowed = set(args.kernels)
        selected = [row for row in KERNELS if row[0] in allowed]

    results = []
    errors = []
    for kernel_id, rel_dir in selected:
        try:
            results.append(prepare_one(kernel_id, rel_dir, args.out, dry_run=args.dry_run))
        except Exception as exc:
            errors.append({"kernel_id": kernel_id, "error": str(exc)})

    summary = {"prepared": len(results), "errors": errors, "results": results}
    print(json.dumps(summary, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
