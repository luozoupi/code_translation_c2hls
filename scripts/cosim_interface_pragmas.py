"""Generate explicit Vitis m_axi / s_axilite pragmas for naive PolyBench kernels."""

from __future__ import annotations

import re
from typing import Optional


_ARRAY_RANK_RE = re.compile(r"\[[^\]]*\]")


def _array_rank(param_decl: str) -> int:
    return len(_ARRAY_RANK_RE.findall(param_decl))


def _param_name(decl: str) -> str:
    decl = decl.strip()
    name = re.sub(r"\s*\[[^\]]*\]", "", decl)
    name = name.split()[-1]
    return name.strip("*")


def parse_top_function(code: str) -> tuple[Optional[str], list[tuple[str, str, int]]]:
    """Return (function_name, [(param_name, decl_fragment, array_rank), ...])."""
    match = re.search(
        r"(?:void|extern\s+\"C\"\s+void)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*\{",
        code,
        re.DOTALL,
    )
    if not match:
        return None, []
    name = match.group(1)
    raw_params = match.group(2)
    params: list[tuple[str, str, int]] = []
    for chunk in _split_params(raw_params):
        chunk = chunk.strip()
        if not chunk:
            continue
        pname = _param_name(chunk)
        rank = _array_rank(chunk)
        params.append((pname, chunk, rank))
    return name, params


def _split_params(raw: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in raw:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def default_port_depth(rank: int, n: int = 120) -> int:
    if rank <= 0:
        return 0
    if rank == 1:
        return n
    if rank == 2:
        return n * n
    return n * n * n


def build_interface_pragmas(
    params: list[tuple[str, str, int]],
    *,
    n: int = 120,
    port_depths: Optional[dict[str, int]] = None,
    bundle_per_port: bool = True,
) -> list[str]:
    """Emit m_axi + s_axilite lines for each kernel argument."""
    depths = port_depths or {}
    lines: list[str] = []
    bundle_idx = 0
    for pname, _decl, rank in params:
        lines.append(f"#pragma HLS INTERFACE s_axilite port={pname} bundle=control")
        if rank > 0:
            bundle = f"gmem{bundle_idx}" if bundle_per_port else "gmem"
            bundle_idx += 1
            depth = depths.get(pname) or default_port_depth(rank, n)
            lines.append(
                f"#pragma HLS INTERFACE m_axi port={pname} offset=slave "
                f"bundle={bundle} depth={depth}"
            )
    lines.append("#pragma HLS INTERFACE s_axilite port=return bundle=control")
    return lines


def inject_cosim_interfaces(
    code: str,
    *,
    n: int = 120,
    port_depths: Optional[dict[str, int]] = None,
    bundle_per_port: bool = True,
) -> str:
    """Insert interface pragmas immediately after `#pragma HLS top`."""
    _name, params = parse_top_function(code)
    if not params:
        return code
    pragma_lines = build_interface_pragmas(
        params,
        n=n,
        port_depths=port_depths,
        bundle_per_port=bundle_per_port,
    )
    block = "\n".join(pragma_lines)

    def _repl(match: re.Match[str]) -> str:
        return match.group(0) + "\n\n" + block + "\n"

    updated, count = re.subn(
        r"(#pragma\s+HLS\s+top[^\n]*\n)",
        _repl,
        code,
        count=1,
    )
    if count:
        return updated

    # Fallback: after opening brace of top function.
    return re.sub(
        r"(void\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^{]*\{\s*\n)",
        r"\1" + block + "\n",
        code,
        count=1,
    )


DOITGEN_BODY_OVERRIDE = """\
#include "doitgen.h"


void kernel_doitgen(
		    double A[ NR + 0][NQ + 0][NP + 0],
		    double C4[ NP + 0][NP + 0],
		    double sum[ NP + 0])
{
  #pragma HLS top name=kernel_doitgen

#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=15000
#pragma HLS INTERFACE s_axilite port=C4 bundle=control
#pragma HLS INTERFACE m_axi port=C4 offset=slave bundle=gmem1 depth=900
#pragma HLS INTERFACE s_axilite port=sum bundle=control
#pragma HLS INTERFACE m_axi port=sum offset=slave bundle=gmem2 depth=30
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int nr = NR;
    const int nq = NQ;
    const int np = NP;

  int i, j, r, q, p, s;
  double lC4[NP + 0][NP + 0];
  double row[NP + 0];

  load_c4:
  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++)
      lC4[i][j] = C4[i][j];

  for (r = 0; r < nr; r++)
    for (q = 0; q < nq; q++)  {
      load_row:
      for (p = 0; p < np; p++)
	row[p] = A[r][q][p];

      compute:
      for (p = 0; p < np; p++)  {
	sum[p] = 0.0;
	for (s = 0; s < np; s++)
	  sum[p] += row[s] * lC4[s][p];
      }

      store_row:
      for (p = 0; p < np; p++)
	A[r][q][p] = sum[p];
    }

}
"""


TWO_MM_BODY_OVERRIDE = """\
#include "2mm.h"


void kernel_2mm(   
		double alpha,
		double beta,
		double tmp[ NI + 0][NJ + 0],
		double A[ NI + 0][NK + 0],
		double B[ NK + 0][NJ + 0],
		double C[ NJ + 0][NL + 0],
		double D[ NI + 0][NL + 0])
{
  #pragma HLS top name=kernel_2mm

#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=tmp bundle=control
#pragma HLS INTERFACE m_axi port=tmp offset=slave bundle=gmem0 depth=14400
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1 depth=14400
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2 depth=14400
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem3 depth=14400
#pragma HLS INTERFACE s_axilite port=D bundle=control
#pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem4 depth=14400
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int ni = NI;
    const int nj = NJ;
    const int nk = NK;
    const int nl = NL;

  int i, j, k;
  double ltmp[NI + 0][NJ + 0];

  gemm_ab:
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)  {
      ltmp[i][j] = 0.0;
      for (k = 0; k < nk; ++k)
	ltmp[i][j] += alpha * A[i][k] * B[k][j];
    }

  write_tmp:
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      tmp[i][j] = ltmp[i][j];

  gemm_cd:
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)  {
      double acc = D[i][j] * beta;
      for (k = 0; k < nj; ++k)
	acc += ltmp[i][k] * C[k][j];
      D[i][j] = acc;
    }

}
"""


_HLS_PRAGMA_RE = re.compile(r"\s*(?://+\s*)?#pragma\s+HLS\b", re.IGNORECASE)


def strip_hls_pragmas_only(text: str) -> tuple[str, dict]:
    """Return source with only #pragma HLS lines removed (no other transforms)."""
    report = {"removed_hls_pragmas": 0}
    lines: list[str] = []
    for line in text.splitlines():
        if _HLS_PRAGMA_RE.match(line):
            report["removed_hls_pragmas"] += 1
            continue
        lines.append(line)
    stripped = "\n".join(lines)
    report["plain_contains_hls_pragmas"] = bool(re.search(r"#pragma\s+HLS\b", stripped, re.IGNORECASE))
    return stripped, report
