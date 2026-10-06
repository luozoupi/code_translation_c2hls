#!/usr/bin/env python3
"""Aggregate and strip AutoSA-generated multi-file HLS sources for plain.cpp."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from prepare_benchmarks import _strip_hls_constructs  # noqa: E402

_INCLUDE_RE = re.compile(r'^\s*#include\s+"kernel_kernel\.h"\s*$', re.MULTILINE)
_TOP_VOID_FN_RE = re.compile(
    r"^(?P<head>(?:template\s*<[^>]*>\s*)?(?:static\s+|inline\s+)*void\s+(?P<name>\w+)\s*\()",
    re.MULTILINE,
)


def _drop_include_kernel_kernel_h(text: str) -> str:
    return _INCLUDE_RE.sub("", text)


def _extract_void_function_names(text: str) -> list[str]:
    return [m.group("name") for m in _TOP_VOID_FN_RE.finditer(text or "")]


def dedupe_top_level_void_functions(text: str) -> tuple[str, dict]:
    """Keep the first body of each top-level ``void`` function; drop later copies.

    AutoSA plain packaging historically concatenated ``kernel_kernel_modules.cpp``
    with ``kernel_kernel.cpp``, but the kernel file already embeds the modules —
    producing redefinition errors that Phase B LLM repair then "fixes" by deleting
    real module bodies.
    """
    if not text:
        return text, {"kept": 0, "dropped": 0, "dropped_names": []}

    matches = list(_TOP_VOID_FN_RE.finditer(text))
    if not matches:
        return text, {"kept": 0, "dropped": 0, "dropped_names": []}

    seen: set[str] = set()
    drop_spans: list[tuple[int, int]] = []
    dropped_names: list[str] = []
    kept = 0

    for match in matches:
        name = match.group("name")
        start = match.start()
        brace = text.find("{", match.end() - 1)
        if brace < 0:
            continue
        depth = 0
        i = brace
        end = None
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    if end < len(text) and text[end] == "\n":
                        end += 1
                    break
            i += 1
        if end is None:
            continue

        if name in seen:
            drop_spans.append((start, end))
            dropped_names.append(name)
        else:
            seen.add(name)
            kept += 1

    if not drop_spans:
        return text, {"kept": kept, "dropped": 0, "dropped_names": []}

    parts: list[str] = []
    cursor = 0
    for start, end in drop_spans:
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    cleaned = re.sub(r"\n{3,}", "\n\n", "".join(parts))
    return cleaned, {
        "kept": kept,
        "dropped": len(dropped_names),
        "dropped_names": dropped_names,
    }


def aggregate_autosa_sources(
    modules_cpp: str,
    kernel_cpp: str,
    *,
    include_header: bool = True,
) -> str:
    """Merge modules + kernel into one translation unit without duplicating bodies.

    If ``kernel_cpp`` already defines the module functions (normal AutoSA export),
    use the kernel file alone. Only concatenate when the kernel is missing modules.
    """
    modules_names = set(_extract_void_function_names(modules_cpp))
    kernel_names = set(_extract_void_function_names(kernel_cpp))
    # Typical AutoSA: kernel embeds all modules + kernel0.
    if modules_names and modules_names.issubset(kernel_names):
        parts: list[str] = []
        if include_header:
            parts.append('#include "kernel_kernel.h"\n')
        parts.append(_drop_include_kernel_kernel_h(kernel_cpp).strip())
        return "\n\n".join(p for p in parts if p) + "\n"

    parts: list[str] = []
    if include_header:
        parts.append('#include "kernel_kernel.h"\n')
    if modules_cpp.strip():
        parts.append(_drop_include_kernel_kernel_h(modules_cpp).strip())
    parts.append(_drop_include_kernel_kernel_h(kernel_cpp).strip())
    merged = "\n\n".join(p for p in parts if p) + "\n"
    merged, _ = dedupe_top_level_void_functions(merged)
    return merged


def _strip_autosa_specific(text: str) -> tuple[str, dict]:
    """Remove AutoSA performance pragmas; keep ``hls::stream`` declarations.

    Stream locals are part of the AutoSA module graph. Dropping them (while
    leaving fifo uses in ``kernel0``) produces undeclared-identifier errors that
    Phase B repair cannot sanely recover from. With ``skip_phase_a``, plain must
    remain a compiling HLS-C++ skeleton.
    """
    report = {
        "removed_hls_stream_declarations": 0,
        "removed_hls_resource_pragmas": 0,
        "removed_hls_inline_off": 0,
        "removed_hls_array_partition": 0,
        "kept_hls_stream_declarations": True,
    }
    lines: list[str] = []
    for line in text.splitlines():
        if re.match(r"^\s*#pragma\s+HLS\s+RESOURCE\b", line, re.IGNORECASE):
            report["removed_hls_resource_pragmas"] += 1
            continue
        if re.match(r"^\s*#pragma\s+HLS\s+INLINE\s+OFF\b", line, re.IGNORECASE):
            report["removed_hls_inline_off"] += 1
            continue
        if re.match(r"^\s*#pragma\s+HLS\s+ARRAY_PARTITION\b", line, re.IGNORECASE):
            report["removed_hls_array_partition"] += 1
            continue
        lines.append(line)
    return "\n".join(lines), report


def strip_autosa_hls(
    text: str,
    *,
    keep_ap_includes: bool = True,
) -> tuple[str, dict]:
    """Strip HLS constructs from aggregated AutoSA optimized sources."""
    text, dedupe_report = dedupe_top_level_void_functions(text)
    text, autosa_report = _strip_autosa_specific(text)
    stripped, base_report = _strip_hls_constructs(text, keep_ap_includes=keep_ap_includes)
    report = {**base_report, **autosa_report, "dedupe": dedupe_report}
    report["plain_contains_hls_stream"] = bool(re.search(r"\bhls::stream\b", stripped))
    report["skip_phase_a_recommended"] = bool(
        report.get("plain_contains_hls_pragmas")
        or report.get("plain_contains_ap_uint")
        or report.get("plain_contains_hls_stream")
        or "ap_int.h" in stripped
    )
    return stripped, report


def build_plain_from_src_dir(src_dir: Path) -> tuple[str, dict]:
    modules = src_dir / "kernel_kernel_modules.cpp"
    kernel = src_dir / "kernel_kernel.cpp"
    if not kernel.is_file():
        # Packaged AutoSA_sources often rename kernel_kernel.cpp -> gold_hls_source.cpp
        for alt in ("gold_hls_source.cpp", "hls_baseline.cpp"):
            candidate = src_dir / alt
            if candidate.is_file():
                kernel = candidate
                break
    if not kernel.is_file():
        raise FileNotFoundError(f"missing kernel under {src_dir}")
    modules_text = modules.read_text(encoding="utf-8") if modules.is_file() else ""
    raw = aggregate_autosa_sources(
        modules_text,
        kernel.read_text(encoding="utf-8"),
    )
    return strip_autosa_hls(raw, keep_ap_includes=True)
