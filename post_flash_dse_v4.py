"""DSE v4: one AutoSA mm_config per run, checked in a fixed order.

``C2HLS_DSE_V4=1`` turns this harness on. It does not extend DSE v3. With the
flag unset, v2 and v3 stay as they are. If both v3 and v4 are set, v4 wins
and v3 rules are not appended.

The first user message has four labeled parts taken from the AutoSA docs.
It does not paste ``autosa_mm_final.cpp`` or the gold ``kernel_kernel.cpp``.
A repair round sends one check's fix text plus the previous kernel.

Structural checks 2-9 run before compile / csynth / csim. Cosim stays off.
This Vitis errors on ``config_compile -jobs``; v4 does not pass ``-jobs``.
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from post_flash_dataflow import extract_kernel_block

CHECK_IDS = (
    "extract",
    "kernel0_dataflow",
    "pe_calls",
    "streams",
    "dram_pack",
    "simd_port",
    "pe_count",
    "modules",
    "full_k",
    "compile",
    "csynth",
    "csim",
)
STRUCTURAL_CHECK_IDS = CHECK_IDS[1:9]
VITIS_CHECK_IDS = CHECK_IDS[9:]
DEFAULT_REPAIR_ROUNDS = 12
CLOCK_NS = 3.33
FPGA_PART = "xcu280-fsvh2892-2L-e"
REQUIRED_K_MAX = 1023
# Tool ceilings. The Slurm wall is 12 hours, so a long csim can be killed
# before the 86400-second tool timeout.
N1024_SYNTH_TIMEOUT_S = 86400
N1024_CSIM_TIMEOUT_S = 86400
JOB_WALL = "12:00:00"
FORBIDDEN_ST0_ST3 = ("C_IO_", "C_PE_dummy_out")
FORBIDDEN_ST4 = ("C_drain_",)

_REPAIR_TAIL = (
    "Return the whole kernel in one fence; do not drop a check that "
    "already passed."
)

_FOR_HEAD_RE = re.compile(r"\bfor\s*\(")
_STREAM_DECL_RE = re.compile(
    r"hls::stream\s*<([^>]+)>\s+([A-Za-z_]\w*)\s*;",
    re.M,
)
_INCLUDE_RE = re.compile(r'#include\s+"([^"]+)"')
_KERNEL0_RE = re.compile(r"\bvoid\s+kernel0\s*\(")
_DATAFLOW_RE = re.compile(r"#pragma\s+HLS\s+DATAFLOW\b", re.I)
_TOP_EXTERN_RE = re.compile(
    r'extern\s+"C"\s+void\s+([A-Za-z_]\w*)\s*\('
)
_PE_DEF_RE = re.compile(r"\bvoid\s+PE\s*\((.*?)\)", re.S)
_TYPEDEF512_RE = re.compile(
    r"typedef\s+ap_uint\s*<\s*512\s*>\s+(A_t16|B_t16|C_t16)\s*;"
)
_FOR_BOUND_RE = re.compile(
    r"for\s*\(\s*(?:[\w:<>,\s]*?\s+)?([A-Za-z_]\w*)\s*=\s*[^;]+;"
    r"\s*\1\s*(<=|<)\s*([^;]+);",
    re.S,
)
_RANGE_RE = re.compile(
    r"\|\s*`([A-Za-z_]\w*)`\s*\|\s*(\d+)\.\.(\d+)"
)
_S0_RE = re.compile(r"S_0\((.*)\)")
_MODULE_CELL_RE = re.compile(r"`([^`]+)`")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


ValidateFn = Callable[[str, dict[str, Any]], str]


def _env_flag(name: str) -> Optional[bool]:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return None
    return raw in {"1", "true", "yes", "on"}


def dse_v4_enabled() -> bool:
    return _env_flag("C2HLS_DSE_V4") is True


def repair_round_limit() -> int:
    raw = os.getenv("C2HLS_DSE_V4_REPAIR_ROUNDS", "").strip()
    if not raw:
        return DEFAULT_REPAIR_ROUNDS
    return int(raw)


def resolve_autosa_docs_dir() -> Path:
    raw = os.getenv("C2HLS_AUTOSA_DOCS_DIR", "").strip()
    if raw:
        return Path(raw)
    from c2hls_paths import AUTOSA_DOCS_DIR

    return AUTOSA_DOCS_DIR


def resolve_autosa_root() -> Path:
    return resolve_autosa_docs_dir().parent


def resolve_v4_configs_path() -> Path:
    raw = os.getenv("C2HLS_DSE_V4_CONFIGS_JSON", "").strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parent / "post_flash_dse_v4_configs.json"


def load_v4_configs(path: Optional[Path] = None) -> list[dict[str, Any]]:
    cfg_path = path or resolve_v4_configs_path()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(item) for item in data]
    if isinstance(data, dict) and isinstance(data.get("configs"), list):
        return [dict(item) for item in data["configs"]]
    if isinstance(data, dict) and data.get("id"):
        return [dict(data)]
    raise ValueError(f"dse_v4 configs must be a list of objects: {cfg_path}")


def load_v4_config(config_id: Optional[str] = None) -> dict[str, Any]:
    wanted = (config_id or os.getenv("C2HLS_DSE_V4_CONFIG", "st0_c1")).strip()
    for item in load_v4_configs():
        if str(item.get("id", "")).strip() == wanted:
            return item
    raise KeyError(f"unknown C2HLS_DSE_V4_CONFIG={wanted!r}")


def pe_count(config: dict[str, Any]) -> int:
    dims = config.get("pe") or []
    n = 1
    for dim in dims:
        n *= int(dim)
    return n


def first_simd(config: dict[str, Any]) -> int:
    simd = config.get("simd") or [8]
    return int(simd[0])


def resolve_instruction_path(config: dict[str, Any]) -> Path:
    rel = str(config["instruction"])
    return resolve_autosa_root() / rel


def resolve_gold_kernel_path(config: dict[str, Any]) -> Path:
    rel = str(config["gold_kernel"])
    return resolve_autosa_root() / rel


def _gold_src_dir(config: dict[str, Any]) -> Path:
    return resolve_gold_kernel_path(config).parent


def host_reorder_source(config: dict[str, Any]) -> str:
    """``host_serialize_*`` / ``host_deserialize_C`` for this config.

    These stay in the testbench. ``kernel0`` reads the packed buffers.
    """
    header = (_gold_src_dir(config) / "kernel_kernel.h").read_text(encoding="utf-8")
    start = header.find("inline void host_serialize_A")
    end = header.find("void kernel0(")
    if start < 0 or end < start:
        raise ValueError(f"no host reorder helpers for {config.get('id')}")
    body = header[start:end].strip()
    for name in ("host_serialize_A", "host_serialize_B", "host_deserialize_C"):
        if f"inline void {name}" not in body:
            raise ValueError(f"{config.get('id')} host header lacks {name}")
    return body + "\n"


def packed_float_counts(config: dict[str, Any]) -> tuple[int, int, int]:
    """Serialized A, B, and C lengths, in floats, from this config's host."""
    host = (_gold_src_dir(config) / "kernel_host.cpp").read_text(encoding="utf-8")
    counts: list[int] = []
    for name in ("dev_A", "dev_B", "dev_C"):
        match = re.search(
            rf"float \*{name} = \(float \*\)malloc\((\d+) \* sizeof\(float\)\)",
            host,
        )
        if not match:
            raise ValueError(f"{config.get('id')} host has no {name} malloc")
        counts.append(int(match.group(1)))
    return counts[0], counts[1], counts[2]


def apply_v4_timeouts() -> None:
    """n1024 csim/csynth budgets. ``CSIM_TIMEOUT`` is fixed at import."""
    os.environ["C2HLS_SYNTH_TIMEOUT"] = str(N1024_SYNTH_TIMEOUT_S)
    os.environ["C2HLS_CSIM_TIMEOUT"] = str(N1024_CSIM_TIMEOUT_S)
    import hls_eval

    hls_eval.CSIM_TIMEOUT = N1024_CSIM_TIMEOUT_S


def n1024_kernel0_testbench(config: dict[str, Any]) -> str:
    """n1024 golden that calls ``kernel0`` on this config's packed buffers.

    The shared ``testbench.cpp`` calls ``autosa_mm`` on unpacked matrices.
    That file stays as it is. This bench keeps its compare and ``Passed!``
    line, and it is the bench v4 csim compiles.
    """
    a_n, b_n, c_n = packed_float_counts(config)
    helpers = host_reorder_source(config)
    return f"""#include "kernel.h"
#include <ap_int.h>
#include <cstring>

typedef ap_uint<512> A_t16;
typedef ap_uint<512> B_t16;
typedef ap_uint<512> C_t16;

void kernel0(A_t16 *A, B_t16 *B, C_t16 *C);

{helpers}
int main(int argc, char **argv) {{
  (void)argc;
  (void)argv;
  data_t (*A)[K] = (data_t (*)[K])malloc(sizeof(data_t) * (size_t)I * (size_t)K);
  data_t (*B)[K] = (data_t (*)[K])malloc(sizeof(data_t) * (size_t)J * (size_t)K);
  data_t (*C)[J] = (data_t (*)[J])malloc(sizeof(data_t) * (size_t)I * (size_t)J);
  data_t (*C_golden)[J] = (data_t (*)[J])malloc(sizeof(data_t) * (size_t)I * (size_t)J);
  float *packed_A = (float *)malloc((size_t){a_n} * sizeof(float));
  float *packed_B = (float *)malloc((size_t){b_n} * sizeof(float));
  float *packed_C = (float *)malloc((size_t){c_n} * sizeof(float));
  if (!A || !B || !C || !C_golden || !packed_A || !packed_B || !packed_C) {{
    printf("Failed to allocate\\n");
    return 1;
  }}

  for (int i = 0; i < I; i++)
    for (int k = 0; k < K; k++) {{
      A[i][k] = (data_t)rand() / RAND_MAX;
    }}

  for (int j = 0; j < J; j++)
    for (int k = 0; k < K; k++) {{
      B[j][k] = (data_t)rand() / RAND_MAX;
    }}

  std::memset(packed_C, 0, (size_t){c_n} * sizeof(float));
  host_serialize_A(packed_A, (float *)A);
  host_serialize_B(packed_B, (float *)B);
  kernel0((A_t16 *)packed_A, (B_t16 *)packed_B, (C_t16 *)packed_C);
  host_deserialize_C((float *)C, packed_C);

  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {{
      C_golden[i][j] = 0;
      for (int k = 0; k < K; k++) {{
        C_golden[i][j] = C_golden[i][j] + A[i][k] * B[j][k];
      }}
    }}

  int err = 0;
  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {{
      if (fabs((float)C_golden[i][j] - (float)C[i][j]) > 0.001)
        err++;
    }}

  free(A);
  free(B);
  free(C);
  free(C_golden);
  free(packed_A);
  free(packed_B);
  free(packed_C);

  if (err)
    printf("Failed with %d errors!\\n", err);
  else
    printf("Passed!\\n");

  return 0;
}}
"""


def read_instruction(config: dict[str, Any]) -> str:
    return resolve_instruction_path(config).read_text(encoding="utf-8")


def read_factors_doc() -> str:
    return (resolve_autosa_docs_dir() / "mm_codegen_factors.md").read_text(
        encoding="utf-8"
    )


def read_stream_stitching() -> str:
    return (
        resolve_autosa_docs_dir() / "mm_configs" / "stream_stitching.md"
    ).read_text(encoding="utf-8")


def _heading_level(line: str) -> Optional[int]:
    match = _HEADING_RE.match(line.rstrip("\n"))
    if not match:
        return None
    return len(match.group(1))


def extract_md_section(text: str, title: str) -> str:
    """Return the heading that contains *title* through the next same-level heading."""
    lines = text.splitlines(keepends=True)
    start = None
    start_level = None
    for i, line in enumerate(lines):
        level = _heading_level(line)
        if level is None:
            continue
        if title in line:
            start = i
            start_level = level
            break
    if start is None:
        return ""
    end = len(lines)
    for j in range(start + 1, len(lines)):
        level = _heading_level(lines[j])
        if level is not None and level <= start_level:
            end = j
            break
    return "".join(lines[start:end]).rstrip() + "\n"


def extract_md_through(text: str, start_title: str, end_title: str) -> str:
    """Section *start_title* including the *end_title* subsection."""
    start = extract_md_section(text, start_title)
    if not start:
        return ""
    if end_title in start:
        # Keep everything through the end subsection (already inside start).
        return start
    extra = extract_md_section(text, end_title)
    return (start + extra).rstrip() + "\n"


def _md_tables(text: str) -> list[str]:
    lines = text.splitlines()
    tables: list[list[str]] = []
    buf: list[str] = []
    for line in lines:
        if line.startswith("|"):
            buf.append(line)
            continue
        if buf:
            tables.append(buf)
            buf = []
    if buf:
        tables.append(buf)
    return ["\n".join(rows) + "\n" for rows in tables]


def _keep_table_rows(table: str, keep: Callable[[str], bool]) -> str:
    out: list[str] = []
    for line in table.splitlines():
        stripped = line.strip()
        is_sep = bool(re.match(r"^\|[\s:|-]+\|$", stripped))
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        is_header = (not is_sep) and stripped.startswith("|") and not keep(line)
        if stripped.startswith("|") and not is_sep and cells:
            first = cells[0].strip().strip("`")
            if first and not keep(line) and not _looks_header_row(cells):
                continue
        out.append(line)
        _ = is_header
    return "\n".join(out) + "\n"


def _looks_header_row(cells: list[str]) -> bool:
    joined = " ".join(cells).lower()
    return any(token in joined for token in ("id", "space", "array", "module", "fifo"))


def _space_time_row(line: str, space_time: int) -> bool:
    cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
    return bool(cells) and cells[0] == str(space_time)


def _eval_affine(expr: str, values: dict[str, int]) -> int:
    class _Names(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id not in values:
                raise ValueError(node.id)
            return ast.copy_location(ast.Constant(values[node.id]), node)

    tree = ast.parse(expr, mode="eval")
    tree = _Names().visit(tree)
    ast.fix_missing_locations(tree)
    return int(eval(compile(tree, "<k>", "eval"), {"__builtins__": {}}))


def _split_top_args(text: str) -> list[str]:
    args: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        args.append("".join(buf).strip())
    return args


def s0_k_expression(config: dict[str, Any]) -> str:
    section = extract_md_section(read_instruction(config), "2. Loop order")
    match = _S0_RE.search(section) or _S0_RE.search(read_instruction(config))
    if not match:
        raise ValueError(f"no S_0(...) in instruction for {config.get('id')}")
    args = _split_top_args(match.group(1))
    if len(args) < 3:
        raise ValueError(f"S_0 does not have 3 arguments: {match.group(0)}")
    return args[2]


def instruction_loop_maxes(config: dict[str, Any]) -> dict[str, int]:
    section = extract_md_section(read_instruction(config), "2. Loop order")
    maxes: dict[str, int] = {}
    for match in _RANGE_RE.finditer(section):
        maxes[match.group(1)] = int(match.group(3))
    return maxes


def expected_k_max(config: dict[str, Any]) -> int:
    expr = s0_k_expression(config)
    maxes = instruction_loop_maxes(config)
    names = {
        node.id
        for node in ast.walk(ast.parse(expr, mode="eval"))
        if isinstance(node, ast.Name)
    }
    values = {name: maxes[name] for name in names if name in maxes}
    if set(values) != names:
        missing = ", ".join(sorted(names - set(values)))
        raise ValueError(f"missing loop bounds for {missing} in {expr}")
    return _eval_affine(expr, values)


def expected_module_calls(config: dict[str, Any]) -> dict[str, int]:
    section = extract_md_section(read_instruction(config), "6. Load and store")
    expected: dict[str, int] = {}
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        name_m = _MODULE_CELL_RE.search(cells[0])
        if not name_m or name_m.group(1) == "Module":
            continue
        try:
            expected[name_m.group(1)] = int(cells[1])
        except ValueError:
            continue
    return expected


@dataclass
class StreamFamilySpec:
    name: str
    count: int
    fifo_srl: bool


def expected_stream_families(config: dict[str, Any]) -> list[StreamFamilySpec]:
    section = extract_md_section(read_instruction(config), "7. Streams")
    families: list[StreamFamilySpec] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        name_m = _MODULE_CELL_RE.search(cells[0])
        if not name_m or "FIFO" in cells[0] or name_m.group(1) == "FIFO family":
            continue
        try:
            count = int(cells[1])
        except ValueError:
            continue
        pragma = cells[4]
        fifo_srl = "FIFO_SRL" in pragma and "no `FIFO_SRL`" not in pragma
        families.append(
            StreamFamilySpec(name=name_m.group(1), count=count, fifo_srl=fifo_srl)
        )
    return families


def expected_stream_count(config: dict[str, Any]) -> int:
    return sum(fam.count for fam in expected_stream_families(config))


def _matching_brace(text: str, open_idx: int) -> int:
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        elif ch == '"':
            i += 1
            while i < n and text[i] != '"':
                if text[i] == "\\":
                    i += 1
                i += 1
        i += 1
    return -1


def _matching_paren(text: str, open_idx: int) -> int:
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        elif ch == '"':
            i += 1
            while i < n and text[i] != '"':
                if text[i] == "\\":
                    i += 1
                i += 1
        i += 1
    return -1


def _function_span(code: str, name: str) -> tuple[int, int]:
    """Return the ``{...}`` span of a definition, skipping prototypes."""
    for match in re.finditer(rf"\bvoid\s+{re.escape(name)}\s*\(", code):
        close = _matching_paren(code, match.end() - 1)
        if close < 0:
            continue
        nxt = _skip_ws_comments(code, close + 1)
        if nxt >= len(code) or code[nxt] != "{":
            continue
        end = _matching_brace(code, nxt)
        if end < 0:
            continue
        return nxt, end
    return -1, -1


def function_body(code: str, name: str) -> str:
    start, end = _function_span(code, name)
    if start < 0:
        return ""
    return code[start : end + 1]


def _skip_ws_comments(text: str, i: int) -> int:
    n = len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        if text.startswith("//", i):
            nl = text.find("\n", i)
            i = n if nl < 0 else nl + 1
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        break
    return i


def _statement_body(text: str, after_cond: int) -> tuple[int, int]:
    i = _skip_ws_comments(text, after_cond)
    if i < len(text) and text[i] == "{":
        end = _matching_brace(text, i)
        return i, (end + 1 if end >= 0 else len(text))
    end = text.find(";", i)
    return i, (end + 1 if end >= 0 else len(text))


def _for_bodies(text: str) -> list[str]:
    bodies: list[str] = []
    for match in _FOR_HEAD_RE.finditer(text):
        open_paren = match.end() - 1
        depth = 0
        j = open_paren
        while j < len(text):
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
                if depth == 0:
                    j += 1
                    break
            j += 1
        _start, end = _statement_body(text, j)
        bodies.append(text[_start:end])
    return bodies


def _arg_list(text: str, open_paren: int) -> list[str]:
    depth = 0
    buf: list[str] = []
    args: list[str] = []
    for i in range(open_paren, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
            if depth > 1:
                buf.append(ch)
        elif ch == ")":
            depth -= 1
            if depth == 0:
                if buf:
                    args.append("".join(buf).strip())
                return args
            buf.append(ch)
        elif ch == "," and depth == 1:
            args.append("".join(buf).strip())
            buf = []
        elif depth >= 1:
            if not (depth == 1 and not buf and ch.isspace()):
                buf.append(ch)
    return args


def _strip_comment(token: str) -> str:
    token = re.sub(r"/\*.*?\*/", "", token, flags=re.S)
    return token.strip()


def pe_wrapper_first_args(kernel0_body: str) -> list[str]:
    args: list[str] = []
    for match in re.finditer(r"\bPE_wrapper\s*\(", kernel0_body):
        values = _arg_list(kernel0_body, match.end() - 1)
        args.append(_strip_comment(values[0]) if values else "")
    return args


def _count_module_calls(body: str, name: str) -> int:
    return len(re.findall(rf"\b{re.escape(name)}(?:_wrapper)?\s*\(", body))


def _count_prefix_calls(body: str, prefix: str) -> int:
    return len(re.findall(rf"\b{re.escape(prefix)}\w*\s*\(", body))


def _looks_like_kernel(text: str) -> bool:
    if not (text or "").strip():
        return False
    return bool(
        re.search(r"\bvoid\s+\w+\s*\(", text)
        or re.search(r"#include\s+", text)
        or 'extern "C"' in text
    )


def resolve_kernel_source(text: str) -> tuple[str, str]:
    fenced = extract_kernel_block(text or "")
    if fenced:
        return fenced, ""
    if _looks_like_kernel(text or ""):
        return text, ""
    return "", "the last reply had no kernel fence"


def attach_local_headers(code: str, config: dict[str, Any]) -> str:
    """Prepend headers this TU includes if they sit next to the gold kernel.

    Gold ``kernel_kernel.cpp`` keeps ``typedef ap_uint<512> A_t16`` in the
    sibling ``kernel_kernel.h``. Only headers named by ``#include "..."`` are
    attached, so a flash ``kernel.h`` include does not pull in gold types.
    """
    try:
        gold = resolve_gold_kernel_path(config)
    except (KeyError, TypeError):
        return code
    search_dirs = [gold.parent]
    extra: list[str] = []
    seen: set[str] = set()
    for inc in _INCLUDE_RE.findall(code):
        key = Path(inc).name
        if key in seen:
            continue
        seen.add(key)
        for directory in search_dirs:
            path = directory / inc
            if not path.is_file():
                path = directory / key
            if path.is_file():
                extra.append(path.read_text(encoding="utf-8"))
                break
    if not extra:
        return code
    return "\n".join(extra) + "\n" + code


def read_gold_kernel(config: dict[str, Any]) -> str:
    """Gold ``.cpp`` plus the sibling header named by its ``#include``."""
    return attach_local_headers(
        resolve_gold_kernel_path(config).read_text(encoding="utf-8"),
        config,
    )


def _top_function(code: str) -> str:
    match = _TOP_EXTERN_RE.search(code)
    if match:
        return match.group(1)
    if _KERNEL0_RE.search(code):
        return "kernel0"
    match = re.search(r"\bvoid\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{", code)
    return match.group(1) if match else ""


def _normalize_sig(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_PE_SIGNATURE_RE = re.compile(r"The PE signature is `([^`]+)`")


def expected_pe_signature(config: dict[str, Any]) -> str:
    """PE port list printed for this config, including its space-time shape."""
    match = _PE_SIGNATURE_RE.search(read_instruction(config))
    if not match:
        raise ValueError(f"no PE signature in instruction for {config.get('id')}")
    return match.group(1).strip()


def _space_time_port_fix(space_time: int) -> str:
    if space_time == 0:
        return (
            "Space-time 0 has one index, `idx`. A has `fifo_A_in` only. "
            "B has `fifo_B_in` and `fifo_B_out`. C leaves as "
            "`fifo_C_drain_out` of `float`. Do not add `fifo_A_out`, "
            "`fifo_C_in`, or `fifo_C_out`."
        )
    if space_time == 3:
        return (
            "Space-time 3 has two indices, `idx` and `idy`. A has `fifo_A_in` "
            "and `fifo_A_out`. B has `fifo_B_in` and `fifo_B_out`. C leaves as "
            "`fifo_C_drain_out` of `float`. Do not add `fifo_C_in` or `fifo_C_out`."
        )
    return (
        "Space-time 4 has two indices, `idx` and `idy`. A has `fifo_A_in` only. "
        "B has `fifo_B_in` and `fifo_B_out`. C is a partial on `fifo_C_in` and "
        "`fifo_C_out`, both `float`. Do not use `fifo_C_drain_out`."
    )


def _pe_loop_maxes(code: str) -> dict[str, int]:
    body = function_body(code, "PE")
    if not body:
        return {}
    maxes: dict[str, int] = {}
    for match in _FOR_BOUND_RE.finditer(body):
        var, op, hi = match.group(1), match.group(2), match.group(3).strip()
        if not re.fullmatch(r"\d+", hi):
            continue
        n = int(hi)
        mx = n if op == "<=" else n - 1
        maxes[var] = max(maxes.get(var, mx), mx)
    return maxes


def check_kernel0_dataflow(code: str, config: dict[str, Any]) -> str:
    del config
    has_kernel0 = bool(_KERNEL0_RE.search(code))
    body = function_body(code, "kernel0") if has_kernel0 else ""
    has_dataflow = bool(body and _DATAFLOW_RE.search(body))
    if has_kernel0 and has_dataflow:
        sig = re.search(
            r"void\s+kernel0\s*\(\s*A_t16\s*\*\s*A\s*,\s*B_t16\s*\*\s*B\s*,"
            r"\s*C_t16\s*\*\s*C\s*\)",
            code,
        )
        if not sig:
            return "kernel0 signature is not kernel0(A_t16 *A, B_t16 *B, C_t16 *C)"
        return ""
    parts: list[str] = []
    top = _top_function(code)
    if top and top != "kernel0":
        parts.append(f"top function is {top}")
    if not has_kernel0:
        parts.append("kernel0 is absent")
    if not has_dataflow:
        parts.append("DATAFLOW pragma is absent")
    return "; ".join(parts) or "kernel0 DATAFLOW is absent"


def check_pe_calls(code: str, config: dict[str, Any]) -> str:
    del config
    body = function_body(code, "kernel0")
    if not body:
        return "kernel0 is absent; no PE_wrapper calls"
    for for_body in _for_bodies(body):
        if re.search(r"\bPE_wrapper\s*\(", for_body) or re.search(
            r"\bPE\s*\(", for_body
        ):
            return "a C for over PEs calls PE_wrapper or PE("
    args = pe_wrapper_first_args(body)
    if not args:
        return "PE_wrapper is absent"
    non_lit = [arg for arg in args if not re.fullmatch(r"\d+", arg)]
    if non_lit:
        return f"PE_wrapper first arguments are not literals: {non_lit[:4]}"
    return ""


def _family_name(stream_name: str, families: list[StreamFamilySpec]) -> Optional[str]:
    matches = [
        fam.name
        for fam in families
        if stream_name == fam.name or stream_name.startswith(fam.name + "_")
    ]
    if not matches:
        return None
    return max(matches, key=len)


def check_streams(code: str, config: dict[str, Any]) -> str:
    body = function_body(code, "kernel0")
    if not body:
        return "kernel0 is absent; no hls::stream declarations"
    decls = list(_STREAM_DECL_RE.finditer(body))
    families = expected_stream_families(config)
    want_n = expected_stream_count(config)
    if len(decls) != want_n:
        return f"hls::stream count is {len(decls)}; expected {want_n}"
    errors: list[str] = []
    seen: dict[str, int] = {fam.name: 0 for fam in families}
    pragma_after: list[tuple[str, str]] = []
    for i, match in enumerate(decls):
        name = match.group(2)
        nxt = decls[i + 1].start() if i + 1 < len(decls) else len(body)
        window = body[match.end() : nxt]
        pragma_after.append((name, window))
        fam = _family_name(name, families)
        if fam:
            seen[fam] = seen.get(fam, 0) + 1
        if "depth=2" not in window:
            errors.append(f"{name} is missing depth=2")
        want_srl = True
        for spec in families:
            if name == spec.name or name.startswith(spec.name + "_"):
                if spec.name == _family_name(name, families):
                    want_srl = spec.fifo_srl
                    break
        has_srl = "FIFO_SRL" in window
        if want_srl and not has_srl:
            errors.append(f"{name} is missing core=FIFO_SRL")
        if not want_srl and has_srl:
            errors.append(f"{name} must not have FIFO_SRL")
    for spec in families:
        got = seen.get(spec.name, 0)
        if got != spec.count:
            errors.append(f"{spec.name} count is {got}; expected {spec.count}")
    return "; ".join(errors[:8])


def check_dram_pack(code: str, config: dict[str, Any]) -> str:
    del config
    found = set(_TYPEDEF512_RE.findall(code))
    missing = [name for name in ("A_t16", "B_t16", "C_t16") if name not in found]
    if missing:
        return "missing typedef ap_uint<512> " + ", ".join(missing)
    return ""


def check_simd_port(code: str, config: dict[str, Any]) -> str:
    want = _normalize_sig(expected_pe_signature(config))
    match = _PE_DEF_RE.search(code)
    if not match:
        return f"PE definition is absent; expected {want}"
    got = _normalize_sig("void PE(" + match.group(1) + ")")
    if got != want:
        return f"PE signature is {got}; expected {want}"
    return ""


def check_pe_count(code: str, config: dict[str, Any]) -> str:
    body = function_body(code, "kernel0")
    got = len(pe_wrapper_first_args(body))
    want = pe_count(config)
    if got != want:
        return f"PE_wrapper count is {got}; expected {want}"
    return ""


def check_modules(code: str, config: dict[str, Any]) -> str:
    body = function_body(code, "kernel0")
    if not body:
        return "kernel0 is absent; no module calls"
    errors: list[str] = []
    for name, want in expected_module_calls(config).items():
        got = _count_module_calls(body, name)
        if got != want:
            errors.append(f"{name} count is {got}; expected {want}")
    space_time = int(config.get("space_time", 0))
    forbidden = FORBIDDEN_ST4 if space_time == 4 else FORBIDDEN_ST0_ST3
    for prefix in forbidden:
        got = _count_prefix_calls(body, prefix)
        if got:
            errors.append(f"{prefix} count is {got}; expected 0")
    return "; ".join(errors[:8])


def _k_name_maxes(code: str, config: dict[str, Any], names: set[str]) -> tuple[dict[str, int], list[str]]:
    """Loop maxes for ``k``. ``p0`` and ``p1`` are PE ids, not ``for`` loops.

    Space-time 4 puts ``p1`` in the ``k`` expression. That index is the
    module id ``idy``. The time loops inside ``PE`` still have to match the
    instruction bounds.
    """
    doc = instruction_loop_maxes(config)
    loops = _pe_loop_maxes(code)
    body = function_body(code, "PE")
    values: dict[str, int] = {}
    missing: list[str] = []
    for name in sorted(names):
        if name in {"p0", "p1"}:
            if name not in doc or not body or not re.search(rf"\b{name}\s*=", body):
                missing.append(name)
                continue
            values[name] = doc[name]
            continue
        if name not in loops or loops[name] != doc.get(name):
            missing.append(name)
            continue
        values[name] = loops[name]
    return values, missing


def check_full_k(code: str, config: dict[str, Any]) -> str:
    expr = s0_k_expression(config)
    want = expected_k_max(config)
    if want != REQUIRED_K_MAX:
        return f"config S_0 {expr} max is {want}; expected {REQUIRED_K_MAX}"
    names = {
        node.id
        for node in ast.walk(ast.parse(expr, mode="eval"))
        if isinstance(node, ast.Name)
    }
    values, missing = _k_name_maxes(code, config, names)
    if missing:
        return (
            f"PE loops do not bind {', '.join(missing)} for {expr}; "
            f"maximum must be {REQUIRED_K_MAX}"
        )
    got = _eval_affine(expr, values)
    if got != REQUIRED_K_MAX:
        detail = ", ".join(f"{name}<={values[name]}" for name in sorted(values))
        return f"k = {expr} with {detail} max is {got}; expected {REQUIRED_K_MAX}"
    return ""


STRUCTURAL_CHECKERS: dict[str, ValidateFn] = {
    "kernel0_dataflow": check_kernel0_dataflow,
    "pe_calls": check_pe_calls,
    "streams": check_streams,
    "dram_pack": check_dram_pack,
    "simd_port": check_simd_port,
    "pe_count": check_pe_count,
    "modules": check_modules,
    "full_k": check_full_k,
}


@dataclass(frozen=True)
class CheckFailure:
    check_id: str
    error: str


def first_failure(
    source: str,
    config: dict[str, Any],
    *,
    compile_fn: Optional[ValidateFn] = None,
    csynth_fn: Optional[ValidateFn] = None,
    csim_fn: Optional[ValidateFn] = None,
) -> Optional[CheckFailure]:
    """Return the first failing check. Vitis helpers run only after 2-9 pass."""
    kernel, extract_error = resolve_kernel_source(source)
    if extract_error:
        return CheckFailure("extract", extract_error)
    kernel = attach_local_headers(kernel, config)
    for check_id in STRUCTURAL_CHECK_IDS:
        error = STRUCTURAL_CHECKERS[check_id](kernel, config)
        if error:
            return CheckFailure(check_id, error)
    vitis = (
        ("compile", compile_fn),
        ("csynth", csynth_fn),
        ("csim", csim_fn),
    )
    for check_id, helper in vitis:
        if helper is None:
            continue
        error = helper(kernel, config)
        if error:
            return CheckFailure(check_id, error)
    return None


def v4_csynth_kwargs() -> dict[str, Any]:
    return {
        "allow_compile_jobs": False,
        "clock_period": CLOCK_NS,
        "part": FPGA_PART,
        "cosim": False,
    }


def _gemm_math_block() -> str:
    factors = read_factors_doc()
    code = re.search(r"```c\n(.*?)```", factors, flags=re.S)
    stmt = (code.group(1).strip() if code else "C[i][j] = C[i][j] + A[i][k] * B[j][k];")
    return (
        "The kernel statement is:\n\n```c\n"
        f"{stmt}\n```\n\n"
        "The printed statement is `S_0(i, j, k)` with `I = J = K = 1024`.\n"
    )


def _part1_math(config: dict[str, Any]) -> str:
    instruction = read_instruction(config)
    task = (
        "Emit `kernel0` for this config's "
        f"`space_time`={config.get('space_time')}, "
        f"`array_part`={config.get('array_part')}, "
        f"`latency`={config.get('latency')}, "
        f"`simd`={config.get('simd')}, and PE shape {config.get('pe')}. "
        "The MAC inside `PE` must use that `S_0` map. Return one fenced kernel."
    )
    return (
        "## Part 1, math\n\n"
        f"{task}\n\n"
        f"{_gemm_math_block()}\n"
        f"{extract_md_section(instruction, '2. Loop order')}\n"
        f"{extract_md_section(instruction, '3. What each factor does')}\n"
    )


def _filter_st_tables(section: str, space_time: int) -> str:
    out = section
    for table in _md_tables(section):
        if not any(_space_time_row(line, space_time) for line in table.splitlines()):
            continue
        filtered = _keep_table_rows(
            table, lambda line, st=space_time: _space_time_row(line, st)
        )
        out = out.replace(table, filtered)
    return out


def _part2_factors(config: dict[str, Any]) -> str:
    factors = read_factors_doc()
    space_time = int(config.get("space_time", 0))
    simd = first_simd(config)
    sec1 = extract_md_section(factors, "1. What a config is")
    # Keep the space-time table and the PE-array formula table; drop later ST prose.
    sec1_keep: list[str] = []
    for table in _md_tables(sec1):
        if any(_space_time_row(line, space_time) for line in table.splitlines()):
            sec1_keep.append(
                _keep_table_rows(
                    table, lambda line, st=space_time: _space_time_row(line, st)
                )
            )
    pe_array = extract_md_section(factors, "### PE array")
    load_store = extract_md_section(factors, "Load and store modules")
    streams = extract_md_section(factors, "Streams and which array moves")
    buffers = extract_md_section(factors, "On-chip buffers")
    wide = extract_md_section(factors, "512-bit accesses")
    pipe = extract_md_section(factors, "Pipeline and unroll")
    fixed = extract_md_section(factors, "4. What is not a free knob")

    local_c_lines = [
        line
        for line in buffers.splitlines()
        if line.startswith("- ")
        and (
            "ST0" in line
            or "output-stationary" in line
            or (space_time == 0 and "local_C[1][128]" in line)
            or "does not run" in line
            or "space axis contributes" in line
            or "time axis that is an output" in line
        )
    ]
    if space_time == 0:
        local_c_lines = [
            line
            for line in buffers.splitlines()
            if line.startswith("- ")
            and (
                "contraction check does not run" in line
                or "Each space axis contributes" in line
                or "time axis that is an output" in line
                or "ST0 point" in line
            )
        ]

    fifo_rows = []
    for line in wide.splitlines():
        if line.startswith("|") and (
            f"simd[{simd}]" in line.replace(" ", "")
            or f"`simd[{simd}]`" in line
            or (simd == 8 and "ST0 `simd[8]`" in line)
        ):
            fifo_rows.append(line)
    fifo_header = []
    for line in wide.splitlines():
        if line.startswith("|") and (
            "DRAM type" in line or re.match(r"^\|[\s:|-]+\|$", line.strip())
        ):
            fifo_header.append(line)
        elif fifo_header and not line.startswith("|"):
            break

    typedef_block = ""
    typed = re.search(
        r"```c\n(typedef ap_uint<512> A_t16;.*?C_t16;\n)```",
        wide,
        flags=re.S,
    )
    if typed:
        typedef_block = "```c\n" + typed.group(1) + "```\n"

    st4_reduce = ""
    if space_time == 4:
        st4_reduce = extract_md_section(factors, "Local reduce (ST4 only)")

    fixed_keep = []
    take = False
    for line in fixed.splitlines():
        if line.startswith("Fixed once"):
            take = True
        if take:
            fixed_keep.append(line)
            if line.startswith("Changed by") or line.startswith("Left off"):
                break
    # Keep the DRAM sentence: it sits in the fixed list.
    if "ap_uint<512>" not in "\n".join(fixed_keep):
        for line in fixed.splitlines():
            if "ap_uint<512>" in line:
                fixed_keep.append(line)

    chunks = [
        "## Part 2, factors for this space time\n",
        *sec1_keep,
        _filter_st_tables(pe_array, space_time),
        _filter_st_tables(load_store, space_time),
        streams,
        "\n".join(local_c_lines) + "\n",
        typedef_block,
        "\n".join(fifo_header + fifo_rows) + "\n",
        pipe,
        st4_reduce,
        "\n".join(fixed_keep) + "\n",
    ]
    return "\n".join(chunk for chunk in chunks if chunk and chunk.strip()) + "\n"


def _part3_wiring(config: dict[str, Any]) -> str:
    stitch = read_stream_stitching()
    space_time = int(config.get("space_time", 0))
    what = extract_md_section(stitch, "What `kernel0` is")
    directions = extract_md_section(stitch, "The three directions")
    st = extract_md_section(stitch, f"Space-time {space_time}")
    produce = extract_md_section(stitch, "Produce, consume, and the depth")
    return (
        "## Part 3, wiring\n\n"
        f"{what}\n{directions}\n{st}\n{produce}\n"
    )


def _part4_numbers(config: dict[str, Any]) -> str:
    instruction = read_instruction(config)
    parts = [
        extract_md_section(instruction, "1. Identity"),
        extract_md_section(instruction, "4. PE array"),
        extract_md_section(instruction, "5. Local buffers"),
        extract_md_section(instruction, "6. Load and store"),
        extract_md_section(instruction, "7. Streams"),
        extract_md_section(instruction, "8. Pipeline and unroll"),
    ]
    return "## Part 4, this design's numbers\n\n" + "\n".join(parts)


def format_v4_initial_user(config: Optional[dict[str, Any]] = None) -> str:
    cfg = config or load_v4_config()
    prompt = (
        _part1_math(cfg)
        + "\n"
        + _part2_factors(cfg)
        + "\n"
        + _part3_wiring(cfg)
        + "\n"
        + _part4_numbers(cfg)
    )
    # Guard against accidental paste of the flash nest or v3's kernel0 ban.
    if "autosa_mm_final.cpp" in prompt:
        raise RuntimeError("v4 prompt must not mention autosa_mm_final.cpp")
    if "Do not emit AutoSA kernel0" in prompt:
        raise RuntimeError("v4 prompt must not contain the v3 kernel0 ban")
    return prompt


def _mismatch_block(check_id: str, error: str) -> str:
    if not error:
        return ""
    return f"## Measured mismatch\n\n{check_id}: {error}\n"


def _previous_kernel_block(kernel_code: str) -> str:
    return (
        "## Previous kernel\n\n```cpp\n"
        f"{kernel_code.rstrip()}\n"
        "```\n"
    )


def _no_for_over_pes_sentence() -> str:
    return "There is no C `for` over PEs in these kernels."


def repair_user(
    check_id: str,
    *,
    config: Optional[dict[str, Any]] = None,
    kernel_code: str,
    error: str = "",
) -> str:
    cfg = config or load_v4_config()
    n = pe_count(cfg)
    space_time = int(cfg.get("space_time", 0))
    simd = first_simd(cfg)
    instruction = read_instruction(cfg)
    stitch = read_stream_stitching()
    factors = read_factors_doc()

    if check_id == "extract":
        body = (
            "The last reply had no kernel fence. Put the full translation unit "
            "inside one ```kernel fence. Do not add an explanation outside the fence."
        )
    elif check_id == "kernel0_dataflow":
        body = (
            extract_md_section(stitch, "What `kernel0` is")
            + "\nAdd a function `kernel0(A_t16 *A, B_t16 *B, C_t16 *C)`. "
            "Its first pragma is `#pragma HLS DATAFLOW`. Its body is only module "
            "calls. Move any load, MAC, or store nest that currently sits in the "
            "top function into `PE` or the IO module that owns it. Mark those "
            "module bodies `#pragma HLS INLINE OFF`."
        )
    elif check_id == "pe_calls":
        ids = (
            f"Space-time 0 uses `PE_wrapper(0, ...)` through `PE_wrapper({n - 1}, ...)`."
            if space_time == 0
            else "Space-time 3 and 4 use `PE_wrapper(p0, p1, ...)` for every pair "
            "in the PE-array section."
        )
        body = (
            f"{_no_for_over_pes_sentence()}\n\n"
            "Delete the `for` whose body calls `PE` or `PE_wrapper`. Write one "
            "call per PE id, with the index as a literal. "
            f"{ids}"
        )
    elif check_id == "streams":
        body = (
            extract_md_section(stitch, "Produce, consume, and the depth")
            + "\n"
            + extract_md_section(stitch, "How many FIFOs")
            + "\nDeclare every link in `kernel0` as `hls::stream<T>`. On each one "
            "write `#pragma HLS STREAM variable=... depth=2`. On interior PE and "
            "IO streams also write `#pragma HLS RESOURCE variable=... core=FIFO_SRL`. "
            "The three serialize-edge streams stay depth 2 and do not get "
            "`FIFO_SRL`. Do not raise the depth when a tile is long."
        )
    elif check_id == "dram_pack":
        typed = extract_md_section(factors, "512-bit accesses")
        body = (
            typed
            + "\nAdd `typedef ap_uint<512> A_t16`, and the same for `B_t16` and "
            "`C_t16`. Change the `kernel0` arguments to those pointer types. The "
            "`m_axi` pragmas stay `offset=slave bundle=gmem_A` (and `gmem_B`, "
            "`gmem_C`) with no width field."
        )
    elif check_id == "simd_port":
        wide = extract_md_section(factors, "512-bit accesses")
        rows = [
            line
            for line in wide.splitlines()
            if line.startswith("|")
            and (
                f"simd[{simd}]" in line.replace(" ", "")
                or (simd == 8 and "ST0 `simd[8]`" in line)
                or "DRAM type" in line
                or re.match(r"^\|[\s:|-]+\|$", line.strip())
            )
        ]
        shift = {8: 256, 4: 128, 1: 32}.get(simd, 256)
        want = expected_pe_signature(cfg)
        body = (
            "\n".join(rows)
            + f"\n\nChange the PE definition to `{want}`.\n"
            + _space_time_port_fix(space_time)
            + "\nA and B element types follow the first SIMD: `A_t8` / `B_t8` "
            "(`ap_uint<256>`) at 8, `A_t4` / `B_t4` (`ap_uint<128>`) at 4, and "
            "`hls::stream<float>` at 1. In the serialize function, read one "
            f"`A_t16` and write narrower words with shifts (`>> {shift}` for "
            f"SIMD {simd}). Leave the `kernel0` argument types at `ap_uint<512>`."
        )
    elif check_id == "pe_count":
        body = (
            extract_md_section(instruction, "4. PE array")
            + "\nThe call list must contain exactly the ids that section lists. "
            "If the count is short, add the missing literal calls and the FIFOs "
            "those calls take. If it is long, delete the extra calls. Do not "
            "change `array_part` or `latency` to force the count."
        )
    elif check_id == "modules":
        body = (
            extract_md_section(instruction, "6. Load and store")
            + "\nPaste this config's module table, the \"Calls in kernel0\" column. "
            "Add a function and a `kernel0` call for each name whose count is "
            "short. Delete calls that belong to the other space time (`C_IO_` and "
            "`C_PE_dummy_out` on space-time 0 and 3; `C_drain_` on space-time 4). "
            "A `_wrapper` suffix counts as the same module."
        )
    elif check_id == "full_k":
        expr = s0_k_expression(cfg)
        maxes = instruction_loop_maxes(cfg)
        c2 = maxes.get("c2", 127)
        c7 = maxes.get("c7", 7)
        space_note = ""
        if space_time == 4:
            space_note = (
                " On space-time 4, `p1` is the PE id `idy` along `k`, not a "
                "`for` loop. Keep `int p1 = idy`. The `for` loops inside `PE` "
                "still use the trip counts in the loop table."
            )
        body = (
            f"This config's `k` expression is `k = {expr}`, with `c2 <= {c2}` and "
            f"`c7 <= {c7}`, maximum {REQUIRED_K_MAX}.{space_note} Set the PE "
            "loops to those bounds. Do not stop at `TK`.\n\n"
            + extract_md_section(instruction, "2. Loop order")
            + extract_md_section(instruction, "3. What each factor does")
        )
    elif check_id == "compile":
        body = (
            "Paste the compiler diagnostic and the source line it names. Change "
            "that line so the diagnostic goes away. Do not change the PE call "
            "count, the stream depth, the typedef widths, or the `k` trip count "
            "in the same edit."
        )
    elif check_id == "csynth":
        body = (
            "Paste the first HLS error. Change the line it names. If the error is "
            "a pipeline failure, keep `#pragma HLS PIPELINE II=1` on the loop the "
            "config's \"8. Pipeline and unroll\" section names, and "
            "`#pragma HLS UNROLL` on the SIMD loop. Do not remove `DATAFLOW` or "
            "the streams to silence the error.\n\n"
            + extract_md_section(instruction, "8. Pipeline and unroll")
        )
    elif check_id == "csim":
        body = (
            "Paste the `Failed with N errors!` line. The testbench calls "
            "`kernel0(A_t16 *A, B_t16 *B, C_t16 *C)`. Before that call it runs "
            "this config's `host_serialize_A` and `host_serialize_B` on the "
            "original matrices. After the call it runs `host_deserialize_C` and "
            "compares every `C[i][j]` with `sum_{k=0}^{1023} A[i][k] * B[j][k]`. "
            "The `kernel0` pointers are those packed buffers, not `A[i][k]`. "
            "Do not add `autosa_mm`. Recompute `i`, `j`, and `k` from the `S_0` "
            "line in section 3 of this config file, which is pasted again here. "
            "A PE that accumulates only one `k` tile, or only `i < PE`, fails "
            "this check. Do not shrink the `k` loop so the process can exit 0.\n\n"
            + extract_md_section(instruction, "3. What each factor does")
        )
    else:
        body = f"Fix check `{check_id}`."

    return (
        f"## Repair `{check_id}`\n\n"
        f"{body.rstrip()}\n\n"
        f"{_mismatch_block(check_id, error)}"
        f"\n{_REPAIR_TAIL}\n\n"
        f"{_previous_kernel_block(kernel_code)}"
    )


V4_SYSTEM = (
    "You write one Vitis HLS C++ kernel for one AutoSA matrix-multiply config. "
    "Follow the user message's S_0 index map, module list, and stream wiring. "
    "Return the translation unit inside one ```kernel fence."
)

_CSIM_FAIL_RE = re.compile(r"Failed with ([1-9]\d*) errors!")


def _csim_error_text(result: dict[str, Any]) -> str:
    log = str(result.get("log") or "")
    err = str(result.get("error") or "")
    match = _CSIM_FAIL_RE.search(log) or _CSIM_FAIL_RE.search(err)
    if match:
        return match.group(0)
    if result.get("passed") or result.get("success"):
        return ""
    return err or "csim failed"


def run_v4_config(
    *,
    config: dict[str, Any],
    out_dir: Path,
    orchestrator: Any,
    header_code: str,
    header_name: str = "kernel.h",
    testbench_code: str,
    part: str = FPGA_PART,
    clock_ns: float = CLOCK_NS,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """One config: initial write, then one repair per failing check.

    Structural checks do not call Vitis. Csynthes and csim run only after
    checks 2-9 pass. Cosim stays off, and synthesis does not pass ``-jobs``.
    """
    from datetime import datetime, timezone

    from hls_eval import run_csim, run_hls_synthesis
    from post_flash_dse import call_dse_llm, dse_max_tokens

    if "Passed!" not in testbench_code or "Failed with %d errors!" not in testbench_code:
        raise ValueError("n1024 bench lost its Passed! / Failed line")
    apply_v4_timeouts()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "dse_v4_summary.json"
    if skip_existing and summary_path.is_file():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
        if isinstance(existing, dict) and existing.get("legal") is True:
            return existing

    token_floor = dse_max_tokens()
    current = getattr(orchestrator, "max_completion_tokens", 0) or 0
    if current < token_floor:
        orchestrator.max_completion_tokens = token_floor

    system = V4_SYSTEM
    user = format_v4_initial_user(config)
    (out_dir / "prompt_system.txt").write_text(system, encoding="utf-8")
    (out_dir / "prompt_user.txt").write_text(user, encoding="utf-8")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    reply = call_dse_llm(
        orchestrator, messages, purpose=f"dse_v4_{config.get('id')}"
    )
    kernel_code, _extract_err = resolve_kernel_source(reply)
    attempts: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    legal = False
    repairs_left = repair_round_limit()

    for attempt in range(repairs_left + 1):
        def compile_fn(code: str, _cfg: dict[str, Any]) -> str:
            from c2hls import compile_check_cpp

            ok, err = compile_check_cpp(
                code, header_code, header_name=header_name
            )
            return "" if ok else (err or "compile failed")[:4000]

        def csynth_fn(code: str, _cfg: dict[str, Any]) -> str:
            synth = run_hls_synthesis(
                code,
                header_code,
                header_name=header_name,
                top_function="kernel0",
                part=part,
                clock_ns=clock_ns,
                allow_compile_jobs=False,
            )
            if not synth.get("success"):
                return str(synth.get("error") or "csynth failed")[:4000]
            report.clear()
            report.update(dict(synth.get("report") or {}))
            return ""

        def csim_fn(code: str, _cfg: dict[str, Any]) -> str:
            apply_v4_timeouts()
            result = run_csim(
                code,
                n1024_kernel0_testbench(config),
                header_code,
                header_name=header_name,
                top_function="kernel0",
                part=part,
                clock_ns=clock_ns,
            )
            return _csim_error_text(result)[:4000]

        fail = first_failure(
            kernel_code or reply,
            config,
            compile_fn=compile_fn,
            csynth_fn=csynth_fn,
            csim_fn=csim_fn,
        )
        if fail is None:
            attempts.append({"attempt": attempt, "check_id": "csim", "error": ""})
            legal = True
            break
        attempts.append(
            {
                "attempt": attempt,
                "check_id": fail.check_id,
                "error": fail.error[:4000],
            }
        )
        if attempt >= repairs_left:
            break
        repair = repair_user(
            fail.check_id,
            config=config,
            kernel_code=kernel_code,
            error=fail.error,
        )
        (out_dir / f"repair_{attempt}.txt").write_text(repair, encoding="utf-8")
        repair_reply = call_dse_llm(
            orchestrator,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": repair},
            ],
            purpose=f"dse_v4_repair_{config.get('id')}_{fail.check_id}",
        )
        extracted, _ = resolve_kernel_source(repair_reply)
        if extracted:
            kernel_code = extracted
            reply = repair_reply
        (out_dir / f"attempt_{attempt}.cpp").write_text(
            kernel_code, encoding="utf-8"
        )

    if kernel_code:
        (out_dir / "autosa_mm.cpp").write_text(kernel_code, encoding="utf-8")
    payload = {
        "schema": "dse_v4_config_v1",
        "config_id": config.get("id"),
        "legal": legal,
        "latency_cycles": report.get("latency_cycles"),
        "dsp": report.get("dsp"),
        "attempts": attempts,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "CHECK_IDS",
    "CLOCK_NS",
    "DEFAULT_REPAIR_ROUNDS",
    "FPGA_PART",
    "STRUCTURAL_CHECK_IDS",
    "CheckFailure",
    "attach_local_headers",
    "dse_v4_enabled",
    "expected_k_max",
    "first_failure",
    "format_v4_initial_user",
    "load_v4_config",
    "load_v4_configs",
    "pe_count",
    "read_gold_kernel",
    "repair_round_limit",
    "repair_user",
    "run_v4_config",
    "resolve_autosa_docs_dir",
    "resolve_gold_kernel_path",
    "s0_k_expression",
    "v4_csynth_kwargs",
]
