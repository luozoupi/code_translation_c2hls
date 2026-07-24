"""Deterministic HLS loop label injection for ChatHLS kernel_info.txt."""
from __future__ import annotations

import re
from typing import Tuple

# Plain ``for (`` / ``for(`` or C-named ``outer:for(`` / ``outer : for (`` (not ``L1:``).
_LOOP_STMT = re.compile(
    r"^(\s*)(?:(\w+)\s*:\s*)?(for\s*\(|while\s*\()(.*)$"
)


def _find_top_function_start(lines: list[str], top: str) -> int | None:
    """Return the line index of the top function definition, not a forward decl."""
    for i, line in enumerate(lines):
        if not re.search(rf"\b{re.escape(top)}\s*\(", line):
            continue
        if "{" in line:
            return i
        ahead = "".join(lines[i : i + 12])
        m = re.search(
            rf"\b{re.escape(top)}\s*\(.*?\)\s*(\{{|;)",
            ahead,
            re.S,
        )
        if m is None:
            return i
        if m.group(1) == "{":
            return i
    return None


def inject_loop_labels(source: str, *, top: str) -> Tuple[str, int]:
    """Insert L1:, L2:, ... before for/while lines inside the top function body.

    Only labels loops that are not already labeled (line does not already match
    ``^\\s*L\\d+:``). Returns (new_source, label_count).
    """
    lines = source.splitlines(keepends=True)
    start = _find_top_function_start(lines, top)
    if start is None:
        return source, 0

    # Brace depth from first { after start
    body_start = None
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if "{" in lines[i] and body_start is None:
            body_start = i
            break
    if body_start is None:
        return source, 0

    out: list[str] = []
    n = 0
    depth = 0
    in_body = False
    for i, line in enumerate(lines):
        if i == body_start:
            in_body = True
        if in_body:
            depth += line.count("{") - line.count("}")
        already = bool(re.match(r"^\s*L\d+\s*:", line))
        m = None if already else _LOOP_STMT.match(line.rstrip("\r\n"))
        if in_body and depth >= 1 and m:
            n += 1
            indent, _c_label, kw, rest = m.groups()
            ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
            out.append(f"{indent}L{n}: {kw}{rest}{ending}")
        else:
            out.append(line)
        if in_body and i > body_start and depth <= 0:
            in_body = False
    return "".join(out), n


def _extract_top_param_list(lines: list[str], start: int, top: str) -> str | None:
    """Return the parameter list text from the top function definition."""
    ahead = "".join(lines[start : start + 20])
    m = re.search(rf"\b{re.escape(top)}\s*\((.*?)\)\s*\{{", ahead, re.S)
    return m.group(1) if m else None


def build_kernel_info(labeled_source: str, *, top: str) -> str:
    """Build ChatHLS kernel_info.txt from labeled source."""
    rows = [top]
    for i, line in enumerate(labeled_source.splitlines(), start=1):
        m = re.match(r"^\s*(L\d+):\s*(for\s*\(|while\s*\()", line)
        if m:
            rows.append(f"{m.group(1)},loop,{i}")
    lines = labeled_source.splitlines(keepends=True)
    start = _find_top_function_start(lines, top)
    param_list = _extract_top_param_list(lines, start, top) if start is not None else None
    if param_list:
        loop_line = next(
            (r.split(",")[-1] for r in rows[1:] if r.startswith("L")),
            "0",
        )
        seen_arrays: set[str] = set()
        for part in param_list.split(","):
            part = part.strip()
            am = re.search(r"\b([A-Za-z_]\w*)\s*(\[|$)", part)
            if am and "[" in part:
                name = am.group(1)
                if name in seen_arrays:
                    continue
                seen_arrays.add(name)
                rows.append(f"{name},array,{loop_line}")
    return "\n".join(rows) + "\n"
