"""Normalize upstream HLSFactory Tier A sources to standard Vitis #pragma HLS form.

Pipeline:
  1. Parse simple #define knobs from headers
  2. Expand PRAGMA_HLS(...) macro calls to #pragma HLS lines
  3. Inline numeric/identifier defines inside pragma bodies
  4. Normalize casing and spacing for Vitis
"""

from __future__ import annotations

import ast
import operator
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_DEFINE_RE = re.compile(
    r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*(?://.*)?$",
    re.MULTILINE,
)
_PRAGMA_HLS_CALL_RE = re.compile(r"\bPRAGMA_HLS\s*\(", re.IGNORECASE)
# Active pragmas only — do not match commented-out lines like `// #pragma HLS ...`.
_PRAGMA_LINE_RE = re.compile(
    r"^\s*#pragma\s+HLS\b(.*)$",
    re.IGNORECASE | re.MULTILINE,
)
_PRAGMA_MACRO_DEF_RE = re.compile(
    r"^\s*#\s*define\s+PRAGMA_(?:HLS|SUB)\b.*$",
    re.MULTILINE | re.IGNORECASE,
)
_PRAGMA_SUB_CALL_RE = re.compile(r"\bPRAGMA_SUB\s*\(", re.IGNORECASE)
_VPRAGMA_HLS_RE = re.compile(
    r'^\s*_Pragma\s*\(\s*"(HLS[^"]*)"\s*\)\s*$',
    re.IGNORECASE | re.MULTILINE,
)

_PARTITION_TYPES = frozenset({"block", "cyclic", "complete"})

# Pragma attribute keys and bare tokens to lowercase; identifier *values* keep source casing.
_PRAGMA_KEY_TOKENS = frozenset(
    {
        "array_partition",
        "bind_op",
        "bind_storage",
        "block",
        "bundle",
        "complete",
        "cyclic",
        "dataflow",
        "dependence",
        "depth",
        "dim",
        "factor",
        "false",
        "ii",
        "inline",
        "interface",
        "inter",
        "intra",
        "loop_flatten",
        "loop_tripcount",
        "m_axi",
        "master",
        "max",
        "min",
        "offset",
        "performance",
        "pipeline",
        "port",
        "s_axilite",
        "slave",
        "stable",
        "stream",
        "true",
        "type",
        "unroll",
        "variable",
    }
)


def parse_defines(*texts: str) -> Dict[str, str]:
    """Collect simple #define NAME VALUE mappings from header/source text."""
    defines: Dict[str, str] = {}
    for text in texts:
        for match in _DEFINE_RE.finditer(text):
            name = match.group(1)
            if name.startswith("PRAGMA_"):
                continue
            value = match.group(2).strip()
            if value.endswith("\\"):
                continue
            defines[name] = value
    return defines


def _eval_define_expr(expr: str, defines: Dict[str, str], depth: int = 0) -> Optional[str]:
    if depth > 8:
        return None
    expr = expr.strip()
    if not expr:
        return None
    if expr in defines:
        return _eval_define_expr(defines[expr], defines, depth + 1) or defines[expr]
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    try:
        value = _eval_ast(node.body, defines, depth)
    except (ValueError, TypeError, ZeroDivisionError):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _eval_ast(node: ast.AST, defines: Dict[str, str], depth: int):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in defines:
            raise ValueError(node.id)
        inner = _eval_define_expr(defines[node.id], defines, depth + 1)
        if inner is None:
            raise ValueError(node.id)
        return _eval_ast(ast.parse(inner, mode="eval").body, defines, depth + 1)
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(op_type)
        left = _eval_ast(node.left, defines, depth)
        right = _eval_ast(node.right, defines, depth)
        if op_type in (ast.Div, ast.FloorDiv) and right == 0:
            raise ZeroDivisionError
        return _SAFE_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(op_type)
        return _SAFE_OPS[op_type](_eval_ast(node.operand, defines, depth))
    raise ValueError(type(node))


def inline_defines_in_text(text: str, defines: Dict[str, str]) -> str:
    """Replace define identifiers with evaluated literals where possible."""
    if not defines:
        return text

    def _resolve(name: str) -> Optional[str]:
        if name in defines:
            return _eval_define_expr(defines[name], defines)
        for key, value in defines.items():
            if key.lower() == name.lower():
                return _eval_define_expr(value, defines)
        return None

    def _sub_identifier(match: re.Match) -> str:
        name = match.group(0)
        evaluated = _resolve(name)
        if evaluated is not None:
            return evaluated
        return name

    return re.sub(r"\b[A-Za-z_]\w*\b", _sub_identifier, text)


def _fold_numeric_expr(expr: str) -> str:
    expr = expr.strip()
    try:
        value = _eval_ast(ast.parse(expr, mode="eval").body, {}, 0)
    except (ValueError, TypeError, SyntaxError, ZeroDivisionError):
        return expr
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _fold_pragma_arithmetic(body: str) -> str:
    def _repl(match: re.Match) -> str:
        key = match.group(1)
        folded = _fold_numeric_expr(match.group(2))
        return f"{key}={folded}"

    return re.sub(
        r"\b(ii|factor|depth|min|max)=([^ \t]+)",
        _repl,
        body,
        flags=re.IGNORECASE,
    )


def inline_pragmas_in_source(text: str, defines: Dict[str, str]) -> str:
    """Inline #define knobs inside #pragma HLS bodies only."""

    def _rewrite(match: re.Match) -> str:
        body = inline_defines_in_text(match.group(1).strip(), defines)
        body = _fold_pragma_arithmetic(body)
        return f"#pragma HLS {body}"

    return _PRAGMA_LINE_RE.sub(_rewrite, text)


def _extract_balanced_parens(text: str, open_idx: int) -> Tuple[str, int]:
    """Return (inner_content, index_after_closing_paren)."""
    depth = 0
    start = open_idx
    for idx in range(open_idx, len(text)):
        ch = text[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1 : idx], idx + 1
    return text[start + 1 :], len(text)


def expand_pragma_hls_macros(text: str, defines: Optional[Dict[str, str]] = None) -> str:
    """Replace PRAGMA_HLS(...) / PRAGMA_SUB(...) with #pragma HLS lines."""
    out: List[str] = []
    pos = 0
    while pos < len(text):
        match = _PRAGMA_HLS_CALL_RE.search(text, pos)
        sub_match = _PRAGMA_SUB_CALL_RE.search(text, pos) if not match else None
        if match and (not sub_match or match.start() <= sub_match.start()):
            out.append(text[pos : match.start()])
            open_paren = text.find("(", match.end() - 1)
            body, after = _extract_balanced_parens(text, open_paren)
            pragma_line = _pragma_hls_from_macro_body(body, defines)
            out.append(pragma_line)
            pos = after
            continue
        if sub_match:
            out.append(text[pos : sub_match.start()])
            open_paren = text.find("(", sub_match.end() - 1)
            body, after = _extract_balanced_parens(text, open_paren)
            pragma_line = _pragma_hls_from_macro_body(body.strip('"'), defines)
            out.append(pragma_line)
            pos = after
            continue
        out.append(text[pos:])
        break
    return "".join(out)


def _pragma_hls_from_macro_body(body: str, defines: Optional[Dict[str, str]] = None) -> str:
    cleaned = " ".join(body.replace("\n", " ").split())
    cleaned = re.sub(r"^HLS\s+", "", cleaned, flags=re.IGNORECASE)
    if defines:
        cleaned = inline_defines_in_text(cleaned, defines)
    cleaned = _fold_pragma_arithmetic(cleaned)
    cleaned = _normalize_pragma_body(cleaned)
    return f"#pragma HLS {cleaned}"


def _normalize_pragma_token(token: str) -> str:
    """Lowercase pragma keywords; preserve identifier values (port/variable names)."""
    if "=" not in token:
        return token.lower()
    key, value = token.split("=", 1)
    key = key.lower()
    if re.fullmatch(r"[A-Za-z_]\w*", value):
        return f"{key}={value}"
    if value.lower() in _PRAGMA_KEY_TOKENS:
        return f"{key}={value.lower()}"
    return f"{key}={value}"


def _normalize_pragma_tokens(body: str) -> str:
    """Normalize spacing and keyword casing without mangling C++ identifiers."""
    body = " ".join(body.split())
    body = re.sub(r"\s*=\s*", "=", body)
    return " ".join(_normalize_pragma_token(tok) for tok in body.split())


def _normalize_pragma_body(body: str) -> str:
    body = " ".join(body.split())
    body = re.sub(r"\s*=\s*", "=", body)
    lower = body.lower()

    # array_partition: trailing or embedded partition type keyword
    m = re.match(
        r"array_partition\s+variable=(\w+)\s+(\w+)\s+factor=(\S+)\s+dim=(\S+)",
        body,
        re.IGNORECASE,
    )
    if m and m.group(2).lower() in _PARTITION_TYPES:
        return (
            f"array_partition variable={m.group(1)} type={m.group(2).lower()} "
            f"factor={m.group(3)} dim={m.group(4)}"
        )

    m = re.match(
        r"array_partition\s+variable=(\w+)\s+factor=(\S+)\s+(\w+)",
        body,
        re.IGNORECASE,
    )
    if m and m.group(3).lower() in _PARTITION_TYPES:
        return (
            f"array_partition variable={m.group(1)} type={m.group(3).lower()} "
            f"factor={m.group(2)}"
        )

    m = re.match(
        r"array_partition\s+variable=(\w+)\s+(\w+)\s+factor=(\S+)",
        body,
        re.IGNORECASE,
    )
    if m and m.group(2).lower() in _PARTITION_TYPES:
        return (
            f"array_partition variable={m.group(1)} type={m.group(2).lower()} "
            f"factor={m.group(3)}"
        )

    # unroll factor
    m = re.match(r"unroll\s+factor=(\S+)", body, re.IGNORECASE)
    if m:
        return f"unroll factor={m.group(1)}"

    m = re.match(r"unroll\s+factor\s+(\S+)", body, re.IGNORECASE)
    if m:
        return f"unroll factor={m.group(1)}"

    # pipeline / inline / dataflow / interface / … — keywords only, ids preserved
    for directive in (
        "pipeline",
        "inline",
        "dataflow",
        "loop_flatten",
        "dependence",
        "interface",
        "stream",
        "bind_op",
        "bind_storage",
        "loop_tripcount",
        "stable",
        "performance",
    ):
        if lower.startswith(directive):
            return _normalize_pragma_tokens(body)

    return _normalize_pragma_tokens(body)


def normalize_vitis_pragmas(text: str, defines: Optional[Dict[str, str]] = None) -> str:
    """Normalize existing #pragma HLS lines (casing, spacing, inline knobs)."""

    def _rewrite(match: re.Match) -> str:
        body = match.group(1).strip()
        if defines:
            body = inline_defines_in_text(body, defines)
        body = _fold_pragma_arithmetic(body)
        body = _normalize_pragma_body(body)
        return f"#pragma HLS {body}"

    text = _PRAGMA_LINE_RE.sub(_rewrite, text)
    return text


def remove_pragma_macro_definitions(text: str) -> str:
    """Drop PRAGMA_HLS / PRAGMA_SUB #define lines from plain headers."""
    text = _PRAGMA_MACRO_DEF_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def normalize_gold_source(
    kernel_text: str,
    header_texts: Optional[Iterable[str]] = None,
) -> str:
    """Full gold normalization: expand macros, inline knobs, normalize pragmas."""
    headers = list(header_texts or [])
    defines = parse_defines(kernel_text, *headers)
    text = expand_pragma_hls_macros(kernel_text, defines)
    text = inline_pragmas_in_source(text, defines)
    text = normalize_vitis_pragmas(text, defines)
    return text


def normalize_gold_header(header_text: str) -> str:
    """Headers for gold: keep numeric defines, drop PRAGMA macro definitions."""
    return remove_pragma_macro_definitions(header_text)


def normalize_file_gold(path: Path, extra_headers: Optional[List[Path]] = None) -> str:
    kernel = path.read_text(encoding="utf-8", errors="ignore")
    headers = []
    for hp in extra_headers or []:
        if hp.is_file():
            headers.append(hp.read_text(encoding="utf-8", errors="ignore"))
    return normalize_gold_source(kernel, headers)


def expand_pragma_hls_in_strip_pass(text: str) -> str:
    """Expand PRAGMA_HLS to #pragma HLS before strip pass (for plain derivation)."""
    return expand_pragma_hls_macros(text)
