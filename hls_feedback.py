"""Fine-grained HLS feedback parser (Pillar 1).

Extracts per-scope (per-loop / per-module) records, scheduler-blame messages,
and a typed bottleneck list from a Vitis HLS run. The flat top-level report
parsed by `hls_eval.parse_synthesis_xml` / `parse_synthesis_report` is the
"design summary"; what this module produces is the "diagnostics" half — the
HLS analogue of register-level slack used by Dr. RTL on the RTL side.

Wire-in:
    from hls_feedback import attach_feedback
    report = parse_synthesis_xml(xml_path)
    report = attach_feedback(report, xml_path=xml_path,
                             rpt_path=rpt_path, log_path=log_path)

The attached fields all live under the `feedback` key so the rest of the
codebase keeps working unchanged:
    report["feedback"] = {
        "scopes":      [ScopeRecord, ...],
        "scheduler_blame": [BlameRecord, ...],
        "bottlenecks": [BottleneckRecord, ...],
        "summary":     {...},      # quick rollups for prompts / scoring
    }
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

# Public schema version embedded in the feedback dict so downstream consumers
# can guard against future shape changes.
FEEDBACK_SCHEMA_VERSION = "1.0"


# === ScopeRecord ============================================================
#
# A flat dict (not a dataclass) so it serializes to JSON without effort.
# Keys may be None if Vitis didn't report that field for this scope.
#
# scope_id      stable hierarchical id ("workload/KERNEL_OUTER/KERNEL_INNER")
# kind          "module" | "loop"
# name          short name as printed in the report
# parent        scope_id of the enclosing scope, or None for top-level
# depth         0 = top-level, increments per nesting level
# issue         e.g. "Timing Violation", "II Violation", or None
# violation     ditto, the ViolationType column
# slack_ns      timing slack relative to the requested clock (negative = bad)
# latency_cycles / latency_ns / interval
# iteration_latency  per-iteration latency (cycles)
# trip_count    loop trip count, or None
# pipelined     "yes" | "no" | "rewind" | None
# pipeline_ii   achieved II in the pipeline, or None
# pipeline_depth pipeline stages, or None
# bram / dsp / ff / lut / uram  per-scope resource estimates
# source_location  file:line emitted by Vitis (when available)
# ============================================================================


# --- helpers ---------------------------------------------------------------

_DASH = {"", "-", "undef", "?", "n/a", "na", "none"}


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    if not s or s.lower() in _DASH:
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    if not s or s.lower() in _DASH:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _parse_ns(value: Any) -> Optional[float]:
    """Parse a duration that may carry a Vitis unit suffix (sec/ms/us/ns)."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in _DASH:
        return None
    multipliers = {"sec": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            num = s[: -len(suffix)].strip()
            try:
                return float(num) * mult
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None


def _strip_resource(value: Any) -> Optional[int]:
    """Parse a resource cell like '17 (~0%)' or '1399 (1%)' → 17 / 1399."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in _DASH:
        return None
    m = re.match(r"(\d+)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


# --- text-report parser ----------------------------------------------------

# The csynth.rpt loop table is a tree drawn with prefix glyphs:
#   + workload                          top-level module
#    + workload_Pipeline_1              child module
#     o Loop 1                          loop inside child module
#    o KERNEL_OUTER                     top-level loop
#     + workload_Pipeline_KERNEL_INNER  module nested inside that loop
#      o KERNEL_INNER                   loop inside that module
#
# We rebuild the nesting from indentation depth + glyph kind. Vitis emits
# leading spaces inside the table cell that encode the nesting level, so we
# count those spaces (after a leading "|") to recover depth.

_TABLE_LINE = re.compile(r"^\|\s*([+o*])\s+(.+?)\s*\|")

# Column layout (matches existing _extract_max_loop_latency):
#   Name(0) | Issue(1) | Violation(2) | IterLat(3) | Interval(4) | Trip(5)
#   | Pipelined(6) | Lat_cycles(7) | Lat_ns(8) | Slack(9) | BRAM(10)
#   | DSP(11) | FF(12) | LUT(13) | URAM(14)
#
# Note the report column order in 2023.2 differs slightly: the table produced
# by Vitis 2023.2 puts Slack right after Issue (column 1), then Latency
# (cycles, col 3), Latency (ns, col 4), Iteration Latency (col 5),
# Interval (col 6), Trip (col 7), Pipelined (col 8), BRAM (col 9), DSP (col 10),
# FF (col 11), LUT (col 12), URAM (col 13). We support both layouts by
# heuristically matching column headers from the table header line.


_PERF_SECTION_RE = re.compile(r"==\s+Performance\s+&\s+Resource\s+Estimates", re.IGNORECASE)
_NEXT_SECTION_RE = re.compile(r"^=+\s*$|==\s+(HW Interfaces|SW I/O|M_AXI|Bind Op|Storage|Pragma)", re.IGNORECASE)


def parse_synthesis_report_per_scope(report_text: str) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Parse the csynth.rpt 'Performance & Resource Estimates' table into
    a flat list of ScopeRecord dicts. Hierarchy is preserved via parent
    fields.

    Returns (scopes, header_map) where header_map maps lowercase column name
    → column index for downstream cross-checking. Strictly section-scoped:
    only consumes table rows inside the 'Performance & Resource Estimates'
    section, ignoring lookalikes (Bind Op, Storage, Pragma reports).
    """
    scopes: List[Dict[str, Any]] = []
    headers: List[str] = []
    header_map: Dict[str, int] = {}

    in_perf_section = False
    in_table = False
    parents: List[Tuple[int, str]] = []  # (indent_level, scope_id)

    lines = report_text.split("\n")
    for raw in lines:
        # Section gating ----------------------------------------------------
        if _PERF_SECTION_RE.search(raw):
            in_perf_section = True
            in_table = False
            headers = []
            header_map = {}
            parents = []
            continue
        if in_perf_section and _NEXT_SECTION_RE.search(raw):
            in_perf_section = False
            in_table = False
            continue
        if not in_perf_section:
            continue

        # Allow leading whitespace inside the section.
        stripped = raw.lstrip()
        if not stripped.startswith("|"):
            in_table = False
            continue

        # Header detection: collect from non-glyph rows that include the
        # column headings. The Performance table prints a two-row header.
        if "Modules" in stripped and "Loops" not in stripped and not headers:
            in_table = True

        if in_table and "|" in stripped and "+--" not in stripped:
            if all(g not in stripped for g in ("+ ", "o ", "* ")) and ("modules" in stripped.lower() or headers):
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                if not headers:
                    headers = [""] * len(cells)
                for i, c in enumerate(cells):
                    if c and i < len(headers):
                        headers[i] = (headers[i] + " " + c).strip()

        if "+--" in stripped:
            if headers and not header_map:
                for i, h in enumerate(headers):
                    h_low = h.lower()
                    if h_low:
                        header_map.setdefault(h_low, i)
            continue

        m = _TABLE_LINE.match(stripped)
        if not m:
            continue

        glyph = m.group(1)
        # Compute indentation by counting leading spaces in the *first cell*
        # of the (already-lstripped) row (between the leading "|" and the
        # glyph).
        first_cell_prefix = stripped.split("|", 2)[1]
        indent = len(first_cell_prefix) - len(first_cell_prefix.lstrip())
        # Vitis indents in 1-space increments; round to a logical level.
        depth = max(0, indent - 1)

        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 5:
            continue

        # Strip the glyph from the first cell to get the bare name.
        name_cell = cells[0]
        name = re.sub(r"^[+o*]\s+", "", name_cell).strip()

        kind = {"+": "module", "o": "loop", "*": "dataflow"}.get(glyph, "unknown")

        # Map by header position when available, else fall back to the
        # documented 2025/2023 order.
        def _cell(*candidate_headers: str, fallback_idx: Optional[int] = None) -> Optional[str]:
            for h in candidate_headers:
                idx = header_map.get(h.lower())
                if idx is not None and 0 <= idx < len(cells):
                    return cells[idx]
            if fallback_idx is not None and 0 <= fallback_idx < len(cells):
                return cells[fallback_idx]
            return None

        # Pop ancestors deeper than us.
        while parents and parents[-1][0] >= depth:
            parents.pop()

        parent_id = parents[-1][1] if parents else None
        scope_id = name if parent_id is None else f"{parent_id}/{name}"

        # Issue/Violation column variations: 2023.2 uses "Issue Type" and
        # there is no separate Violation column in the loop table — the issue
        # column carries values like "Timing" or "-".
        issue_raw = _cell("issue type", "issue", fallback_idx=1)
        violation_raw = _cell("violation type", "violation")

        slack = _coerce_float(_cell("slack", fallback_idx=2))
        # Latency cycles / ns: 2023.2 = cols 3 / 4; older = 7 / 8. Try both.
        lat_cyc_raw = _cell("latency (cycles)", "latency cycles", "lat_cycles", "lat", fallback_idx=3)
        lat_ns_raw = _cell("latency (ns)", "latency ns", "lat_ns", fallback_idx=4)
        iter_lat_raw = _cell("iteration latency", "iter latency", "iterlat", fallback_idx=5)
        interval_raw = _cell("interval", "ii", fallback_idx=6)
        trip_raw = _cell("trip count", "trip", fallback_idx=7)
        pipelined_raw = _cell("pipelined", fallback_idx=8)
        bram_raw = _cell("bram", "bram_18k", fallback_idx=9)
        dsp_raw = _cell("dsp", fallback_idx=10)
        ff_raw = _cell("ff", fallback_idx=11)
        lut_raw = _cell("lut", fallback_idx=12)
        uram_raw = _cell("uram", fallback_idx=13)

        scope: Dict[str, Any] = {
            "scope_id": scope_id,
            "kind": kind,
            "name": name,
            "parent": parent_id,
            "depth": depth,
            "issue": (issue_raw or "").strip() if issue_raw and issue_raw.strip() not in _DASH else None,
            "violation": (violation_raw or "").strip() if violation_raw and violation_raw.strip() not in _DASH else None,
            "slack_ns": slack,
            "latency_cycles": _coerce_int(lat_cyc_raw),
            "latency_ns": _parse_ns(lat_ns_raw),
            "iteration_latency": _coerce_int(iter_lat_raw),
            "interval": _coerce_int(interval_raw),
            "trip_count": _coerce_int(trip_raw),
            "pipelined": (pipelined_raw or "").strip() if pipelined_raw else None,
            "bram": _strip_resource(bram_raw),
            "dsp": _strip_resource(dsp_raw),
            "ff": _strip_resource(ff_raw),
            "lut": _strip_resource(lut_raw),
            "uram": _strip_resource(uram_raw),
            "source_location": None,
            "pipeline_ii": None,
            "pipeline_depth": None,
        }
        scopes.append(scope)
        if scope_id:
            parents.append((depth, scope_id))

    return scopes, header_map


# --- XML parser ------------------------------------------------------------


def parse_synthesis_xml_per_scope(xml_path: str) -> List[Dict[str, Any]]:
    """Parse `csynth.xml` `ModuleInformation` blocks into ScopeRecords. The
    XML carries per-loop `PipelineII` / `PipelineDepth` / `Slack` /
    `IterationLatency` directly, which the text report omits when the loop
    is fully pipelined."""
    scopes: List[Dict[str, Any]] = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except (ET.ParseError, FileNotFoundError, OSError):
        return scopes

    # Top-level module summary (no Module element) — derive from the global
    # Performance/Area sections so the top-level scope is always present.
    top_name = _xml_text(root, ".//UserAssignments/TopModelName") or "workload"
    top_resources = root.find(".//AreaEstimates/Resources")
    top_scope = {
        "scope_id": top_name,
        "kind": "module",
        "name": top_name,
        "parent": None,
        "depth": 0,
        "issue": _normalize_issue(
            _xml_text(root, ".//PerformanceEstimates/SummaryOfViolations/IssueType")
        ),
        "violation": _normalize_issue(
            _xml_text(root, ".//PerformanceEstimates/SummaryOfViolations/ViolationType")
        ),
        "slack_ns": None,
        "latency_cycles": _coerce_int(
            _xml_text(root, ".//PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency")
        ),
        "latency_ns": _parse_ns(
            _xml_text(root, ".//PerformanceEstimates/SummaryOfOverallLatency/Worst-caseRealTimeLatency")
        ),
        "iteration_latency": None,
        "interval": _coerce_int(
            _xml_text(root, ".//PerformanceEstimates/SummaryOfOverallLatency/Interval-max")
        ),
        "trip_count": None,
        "pipelined": _xml_text(root, ".//PerformanceEstimates/PipelineType"),
        "bram": _coerce_int(_xml_text(top_resources, "BRAM_18K")) if top_resources is not None else None,
        "dsp": _coerce_int(_xml_text(top_resources, "DSP")) if top_resources is not None else None,
        "ff": _coerce_int(_xml_text(top_resources, "FF")) if top_resources is not None else None,
        "lut": _coerce_int(_xml_text(top_resources, "LUT")) if top_resources is not None else None,
        "uram": _coerce_int(_xml_text(top_resources, "URAM")) if top_resources is not None else None,
        "source_location": _xml_text(root, ".//PerformanceEstimates/SummaryOfViolations/SourceLocation"),
        "pipeline_ii": None,
        "pipeline_depth": None,
    }
    scopes.append(top_scope)

    # Top-level loop summaries (the SummaryOfLoopLatency block right under
    # PerformanceEstimates, before any ModuleInformation).
    perf = root.find("PerformanceEstimates")
    if perf is not None:
        sol = perf.find("SummaryOfLoopLatency")
        if sol is not None:
            for child in list(sol):
                _emit_xml_loop_scope(child, parent_id=top_name, depth=1, scopes=scopes,
                                     violations_root=perf.find("SummaryOfViolations"))

    # Per-module (sub-component) info.
    module_info = root.find("ModuleInformation")
    if module_info is not None:
        for module in module_info.findall("Module"):
            mname_el = module.find("Name")
            mname = mname_el.text.strip() if mname_el is not None and mname_el.text else "module"
            mod_perf = module.find("PerformanceEstimates")
            mod_resources = module.find("AreaEstimates/Resources")
            mod_violations = mod_perf.find("SummaryOfViolations") if mod_perf is not None else None

            # Skip the top-level module entry (some Vitis versions place a
            # <Module><Name>workload</Name></Module> inside ModuleInformation
            # with the same data as the global SummaryOf*); merge any extra
            # detail it carries into the existing top_scope rather than
            # emitting a duplicate "workload/workload" scope.
            if mname == top_name:
                if mod_resources is not None:
                    for k_xml, k_out in (
                        ("BRAM_18K", "bram"), ("DSP", "dsp"),
                        ("FF", "ff"), ("LUT", "lut"), ("URAM", "uram"),
                    ):
                        v = _coerce_int(_xml_text(mod_resources, k_xml))
                        if v is not None:
                            top_scope[k_out] = v
                # Promote per-loop entries inside the top module's loop block
                # as direct children of the top scope.
                top_loops = mod_perf.find("SummaryOfLoopLatency") if mod_perf is not None else None
                if top_loops is not None:
                    for child in list(top_loops):
                        _emit_xml_loop_scope(
                            child, parent_id=top_name, depth=1, scopes=scopes,
                            violations_root=mod_violations,
                        )
                continue

            mod_scope = {
                "scope_id": f"{top_name}/{mname}",
                "kind": "module",
                "name": mname,
                "parent": top_name,
                "depth": 1,
                "issue": _normalize_issue(_xml_text(mod_violations, "IssueType")) if mod_violations is not None else None,
                "violation": _normalize_issue(_xml_text(mod_violations, "ViolationType")) if mod_violations is not None else None,
                "slack_ns": None,
                "latency_cycles": _coerce_int(_xml_text(mod_perf, "SummaryOfOverallLatency/Worst-caseLatency")) if mod_perf is not None else None,
                "latency_ns": _parse_ns(_xml_text(mod_perf, "SummaryOfOverallLatency/Worst-caseRealTimeLatency")) if mod_perf is not None else None,
                "iteration_latency": None,
                "interval": _coerce_int(_xml_text(mod_perf, "SummaryOfOverallLatency/PipelineInitiationInterval")) if mod_perf is not None else None,
                "trip_count": None,
                "pipelined": _xml_text(mod_perf, "SummaryOfOverallLatency/PipelineType") if mod_perf is not None else None,
                "bram": _coerce_int(_xml_text(mod_resources, "BRAM_18K")) if mod_resources is not None else None,
                "dsp": _coerce_int(_xml_text(mod_resources, "DSP")) if mod_resources is not None else None,
                "ff": _coerce_int(_xml_text(mod_resources, "FF")) if mod_resources is not None else None,
                "lut": _coerce_int(_xml_text(mod_resources, "LUT")) if mod_resources is not None else None,
                "uram": _coerce_int(_xml_text(mod_resources, "URAM")) if mod_resources is not None else None,
                "source_location": _xml_text(mod_violations, "SourceLocation") if mod_violations is not None else None,
                "pipeline_ii": None,
                "pipeline_depth": None,
            }
            scopes.append(mod_scope)

            mod_loops = mod_perf.find("SummaryOfLoopLatency") if mod_perf is not None else None
            if mod_loops is not None:
                for child in list(mod_loops):
                    _emit_xml_loop_scope(
                        child,
                        parent_id=mod_scope["scope_id"],
                        depth=2,
                        scopes=scopes,
                        violations_root=mod_violations,
                    )

    # Dedupe within XML output by scope_id (the global SummaryOfLoopLatency
    # and the top-Module's loop summary often emit the same scope twice;
    # merge into a single record, preferring whichever copy carries more
    # populated fields).
    by_id: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for sc in scopes:
        sid = sc["scope_id"]
        cur = by_id.get(sid)
        if cur is None:
            by_id[sid] = dict(sc)
            order.append(sid)
            continue
        for k, v in sc.items():
            if v is None:
                continue
            if cur.get(k) is None:
                cur[k] = v
    return [by_id[sid] for sid in order]


def _emit_xml_loop_scope(loop_el, *, parent_id: str, depth: int,
                         scopes: List[Dict[str, Any]],
                         violations_root) -> None:
    """Convert one XML loop element into a ScopeRecord and append."""
    raw_tag = loop_el.tag
    name = _xml_text(loop_el, "Name") or raw_tag
    scope_id = f"{parent_id}/{name}" if parent_id else name

    issue = None
    violation = None
    src_loc = None
    if violations_root is not None:
        loop_v = violations_root.find(f"SummaryOfLoopViolations/{raw_tag}")
        if loop_v is not None:
            issue = _normalize_issue(_xml_text(loop_v, "IssueType"))
            violation = _normalize_issue(_xml_text(loop_v, "ViolationType"))
            src_loc = _xml_text(loop_v, "SourceLocation")

    pipeline_ii = _coerce_int(_xml_text(loop_el, "PipelineII"))
    pipeline_depth = _coerce_int(_xml_text(loop_el, "PipelineDepth"))
    pipelined_str = _xml_text(loop_el, "PipelineType")
    iter_lat = _coerce_int(_xml_text(loop_el, "IterationLatency"))
    latency_cyc = _coerce_int(_xml_text(loop_el, "Latency"))
    abs_lat = _parse_ns(_xml_text(loop_el, "AbsoluteTimeLatency"))
    trip_count = _coerce_int(_xml_text(loop_el, "TripCount"))
    slack = _coerce_float(_xml_text(loop_el, "Slack"))

    scopes.append({
        "scope_id": scope_id,
        "kind": "loop",
        "name": name,
        "parent": parent_id,
        "depth": depth,
        "issue": issue,
        "violation": violation,
        "slack_ns": slack,
        "latency_cycles": latency_cyc,
        "latency_ns": abs_lat,
        "iteration_latency": iter_lat,
        "interval": pipeline_ii,  # for a pipelined loop, II == interval per cycle
        "trip_count": trip_count,
        "pipelined": pipelined_str,
        "pipeline_ii": pipeline_ii,
        "pipeline_depth": pipeline_depth,
        "bram": None,
        "dsp": None,
        "ff": None,
        "lut": None,
        "uram": None,
        "source_location": src_loc,
    })


def _xml_text(parent, path, default=None):
    if parent is None:
        return default
    el = parent.find(path)
    if el is not None and el.text:
        return el.text.strip()
    return default


def _normalize_issue(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    t = text.strip()
    if not t or t.lower() in _DASH:
        return None
    return t


# --- vitis_hls.log scheduler-blame parser ---------------------------------

# Patterns that name a real obstacle for the scheduler. Conservative — we
# return only lines tagged with one of these phrases.
_BLAME_PATTERNS = [
    re.compile(r"Unable to enforce.*?II\s*=\s*\d+", re.IGNORECASE),
    re.compile(r"unable to schedule.*?due to", re.IGNORECASE),
    re.compile(r"Cannot pipeline.*?because", re.IGNORECASE),
    re.compile(r"Cannot pipeline.*?with II\s*=\s*\d+", re.IGNORECASE),
    re.compile(r"loop\s+\S+.*?cannot be pipelined", re.IGNORECASE),
    re.compile(r"recurrence\s+(?:cycle|on)", re.IGNORECASE),
    re.compile(r"loop[- ]carried\s+dependence", re.IGNORECASE),
    re.compile(r"Memory\s+(?:port|access)\s+conflict", re.IGNORECASE),
    re.compile(r"port\s+limit\s+exceeded", re.IGNORECASE),
    re.compile(r"Throughput\s+\(II\)\s+is\s+\d+", re.IGNORECASE),
    re.compile(r"Cannot enforce dataflow", re.IGNORECASE),
    re.compile(r"WARNING:\s*\[(?:HLS|XFORM|SYN|SCHED)\s+\d+", re.IGNORECASE),
    re.compile(r"ERROR:\s*\[(?:HLS|XFORM|SYN|SCHED)\s+\d+", re.IGNORECASE),
]

# A sourcefile:line tag often appears at the start; capture it when present.
_LOC_RE = re.compile(r"([\w./_\-+]+\.(?:cpp|c|cc|h|hpp|hh)):(\d+)")


def parse_vitis_hls_log(log_text: str, *, max_records: int = 200) -> List[Dict[str, Any]]:
    """Pull scheduler-blame lines out of vitis_hls.log."""
    blame: List[Dict[str, Any]] = []
    if not log_text:
        return blame
    for line in log_text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if not any(p.search(line) for p in _BLAME_PATTERNS):
            continue
        loc = None
        m = _LOC_RE.search(line)
        if m:
            loc = f"{m.group(1)}:{m.group(2)}"
        blame.append({
            "kind": _classify_blame(line),
            "message": line.strip(),
            "source_location": loc,
        })
        if len(blame) >= max_records:
            break
    return blame


def _classify_blame(line: str) -> str:
    low = line.lower()
    if "recurrence" in low or "loop-carried" in low or "loop carried" in low:
        return "loop_carried_dep"
    if "port" in low and ("limit" in low or "conflict" in low):
        return "port_conflict"
    if "memory" in low and ("port" in low or "conflict" in low):
        return "port_conflict"
    if "cannot pipeline" in low or "unable to enforce" in low and "ii" in low:
        return "pipeline_blocked"
    if "dataflow" in low:
        return "dataflow_blocked"
    if "throughput" in low and "ii" in low:
        return "ii_target_miss"
    if "error" in low:
        return "error"
    return "warning"


# --- bottleneck derivation -------------------------------------------------


def derive_bottleneck_records(
    scopes: List[Dict[str, Any]],
    blame: List[Dict[str, Any]],
    *,
    requested_clock_ns: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Convert raw per-scope + log signals into typed bottleneck records.

    Each record names the scope, the bottleneck kind, the evidence we used,
    and a severity tier so a downstream agent can prioritize.
    """
    bottlenecks: List[Dict[str, Any]] = []

    # --- timing slack (per-scope) ---
    for sc in scopes:
        slack = sc.get("slack_ns")
        if slack is None:
            continue
        if slack < 0:
            severity = "high" if slack < -0.2 else "medium"
            bottlenecks.append({
                "scope_id": sc["scope_id"],
                "kind": "timing_violation",
                "evidence": f"slack {slack:.3f} ns < 0 on {sc['kind']} '{sc['name']}'",
                "severity": severity,
                "metric": {"slack_ns": slack},
                "source_location": sc.get("source_location"),
            })

    # --- pipeline II target miss (loop is pipelined but II>1) ---
    for sc in scopes:
        if sc.get("kind") != "loop":
            continue
        ii = sc.get("pipeline_ii") or sc.get("interval")
        if ii is None:
            continue
        # Heuristic threshold: II>1 on a clearly-pipelined loop deserves a
        # bottleneck record since the user almost certainly asked for II=1.
        pipelined = (sc.get("pipelined") or "").lower()
        if pipelined.startswith("yes") and ii > 1:
            bottlenecks.append({
                "scope_id": sc["scope_id"],
                "kind": "ii_target_miss",
                "evidence": f"pipelined loop achieved II={ii} (>1)",
                "severity": "high" if ii >= 4 else "medium",
                "metric": {"pipeline_ii": ii, "trip_count": sc.get("trip_count")},
                "source_location": sc.get("source_location"),
            })

    # --- non-pipelined hot loop ---
    # If a loop has high trip count and is not pipelined, that's typically
    # the dominant latency contributor.
    for sc in scopes:
        if sc.get("kind") != "loop":
            continue
        pipelined = (sc.get("pipelined") or "").lower()
        trip = sc.get("trip_count") or 0
        lat = sc.get("latency_cycles") or 0
        if pipelined.startswith("no") and trip >= 64 and lat >= 1024:
            bottlenecks.append({
                "scope_id": sc["scope_id"],
                "kind": "non_pipelined_hot_loop",
                "evidence": f"loop trip={trip}, latency={lat} cycles, not pipelined",
                "severity": "high" if lat >= 100_000 else "medium",
                "metric": {"trip_count": trip, "latency_cycles": lat},
                "source_location": sc.get("source_location"),
            })

    # --- interval > latency (dataflow / throughput regression) ---
    # On a top-level scope this means the kernel cannot start a new transaction
    # until later than the previous one finishes — a real-world "hidden
    # throughput regression" that latency-only rubrics miss (pathfinder
    # doublebuffer in our smoke test had latency=342676 but interval=680597).
    for sc in scopes:
        lat = sc.get("latency_cycles")
        ii = sc.get("interval")
        if lat is None or ii is None or ii <= 1 or lat <= 0:
            continue
        if ii > lat:
            bottlenecks.append({
                "scope_id": sc["scope_id"],
                "kind": "interval_exceeds_latency",
                "evidence": f"interval {ii} > latency {lat} cycles on {sc['kind']} '{sc['name']}'",
                "severity": "high",
                "metric": {"interval": ii, "latency_cycles": lat,
                           "interval_over_latency": round(ii / lat, 3)},
                "source_location": sc.get("source_location"),
            })

    # --- scheduler-blame promotion ---
    # Pull blame messages into bottleneck records when they classify as
    # something the agent can act on.
    for b in blame:
        if b["kind"] in {"warning"}:
            continue  # too noisy for a typed bottleneck
        bottlenecks.append({
            "scope_id": None,  # not always tied to a specific scope
            "kind": b["kind"],
            "evidence": b["message"][:240],
            "severity": "high" if b["kind"] in {"loop_carried_dep", "pipeline_blocked",
                                                "port_conflict", "dataflow_blocked"} else "medium",
            "metric": {},
            "source_location": b.get("source_location"),
        })

    # Sort: highest severity first, then by metric magnitude when comparable.
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    bottlenecks.sort(key=lambda b: (severity_rank.get(b.get("severity"), 3), b.get("kind") or ""))
    return bottlenecks


# --- summary roll-ups (cheap signal for prompts) ---------------------------


def summarize_feedback(scopes: List[Dict[str, Any]],
                       bottlenecks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compact rollup so prompt builders don't have to walk the full list."""
    total = len(scopes)
    loops = [s for s in scopes if s.get("kind") == "loop"]
    pipelined_yes = [l for l in loops if (l.get("pipelined") or "").lower().startswith("yes")]
    pipelined_no = [l for l in loops if (l.get("pipelined") or "").lower().startswith("no")]
    timing_viol = [s for s in scopes if (s.get("issue") or "").lower().find("timing") >= 0]
    neg_slack = [s for s in scopes if s.get("slack_ns") is not None and s["slack_ns"] < 0]
    high_b = [b for b in bottlenecks if b.get("severity") == "high"]
    return {
        "schema": FEEDBACK_SCHEMA_VERSION,
        "scope_count": total,
        "loop_count": len(loops),
        "pipelined_loops": len(pipelined_yes),
        "non_pipelined_loops": len(pipelined_no),
        "scopes_with_timing_violation": len(timing_viol),
        "scopes_with_negative_slack": len(neg_slack),
        "bottleneck_count": len(bottlenecks),
        "high_severity_bottlenecks": len(high_b),
        "top_bottleneck_kinds": _top_kinds(bottlenecks),
    }


def _top_kinds(bottlenecks: List[Dict[str, Any]], n: int = 5) -> List[Tuple[str, int]]:
    counts: Dict[str, int] = {}
    for b in bottlenecks:
        counts[b.get("kind") or "unknown"] = counts.get(b.get("kind") or "unknown", 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])[:n]


# --- top-level entry point -------------------------------------------------


# === Phase 7a — static report harvest =====================================
#
# Vitis HLS writes a wealth of diagnostics under
# ``<work_dir>/hls_proj/sol1/`` that go beyond the top-level csynth.rpt /
# csynth.xml we already parse:
#
# - ``.autopilot/db/burst.xml``         AXI burst inference (passed /
#                                       widened / failed) per access
# - ``.autopilot/db/fe_messages.xml``   front-end (clang) messages
# - ``.autopilot/db/be_messages.xml``   back-end (scheduler) messages
# - ``syn/report/csynth_design_size.rpt`` per-phase instruction counts
#
# These give us the smoking-gun signals for "why didn't this step help?":
# * ``msg_groups="PRAGMA_INVALID"``  → pragma was silently rejected
# * ``BURST_VERBOSE_FAILED``         → AXI access wasn't burst-inferred
# * design_size growth >> input      → unroll inflated the design

def _autopilot_db_dir(work_dir: str) -> Optional[str]:
    """Return the absolute path to ``<work_dir>/hls_proj/sol1/.autopilot/db``
    if it exists, else None. Tolerates either a synth work_dir or the
    inner project dir."""
    candidates = [
        os.path.join(work_dir, "hls_proj", "sol1", ".autopilot", "db"),
        os.path.join(work_dir, "sol1", ".autopilot", "db"),
        os.path.join(work_dir, ".autopilot", "db"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


def _syn_report_dir(work_dir: str) -> Optional[str]:
    """Return ``<work_dir>/hls_proj/sol1/syn/report`` if it exists."""
    candidates = [
        os.path.join(work_dir, "hls_proj", "sol1", "syn", "report"),
        os.path.join(work_dir, "sol1", "syn", "report"),
        os.path.join(work_dir, "syn", "report"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


# --- burst.xml parser ---

# burst groups that matter to the agent. Each ``BurstInfo/burst`` element
# has the schema documented above the file at .autopilot/db/burst.xml.
_BURST_PASSED = {"BURST_VERBOSE_PASSED", "BURST_VERBOSE_WIDEN_PASSED"}
_BURST_WIDENED = {"BURST_VERBOSE_WIDEN_PASSED"}
_BURST_FAILED = {"BURST_VERBOSE_FAILED", "BURST_FAILED"}
_BURST_SUMMARY = {"BURST_SUMMARY", "BURST_SUMMARY_FAILED"}


def parse_burst_info(work_dir: str) -> Dict[str, Any]:
    """Parse the AXI burst inference results.

    Returns:
        {
          "passed": List[burst record],     # bursts the tool inferred
          "widened": List[burst record],    # bursts where width was widened
          "failed": List[burst record],     # accesses that COULD'VE been
                                            # bursts but weren't
          "summary": List[burst record],    # bundle-level rollup the tool
                                            # already computed
          "counts": {"passed": N, "widened": N, "failed": N, "summary": N},
          "schema": "1.0",
        }

    A burst record has: kind / msg_severity / src_info / bundle / var /
    direction / length / width / loop_name / parent_func / orig_id.
    """
    db = _autopilot_db_dir(work_dir or "")
    if db is None:
        return {"schema": "1.0", "passed": [], "widened": [],
                "failed": [], "summary": [],
                "counts": {"passed": 0, "widened": 0, "failed": 0, "summary": 0}}

    path = os.path.join(db, "burst.xml")
    if not os.path.isfile(path):
        return {"schema": "1.0", "passed": [], "widened": [],
                "failed": [], "summary": [],
                "counts": {"passed": 0, "widened": 0, "failed": 0, "summary": 0}}

    # The Vitis-emitted XML uses an unbound `VitisHLS:` element prefix
    # (no `xmlns:VitisHLS=...` declaration), which ElementTree refuses
    # to parse. Strip the prefix before parsing — the data we want is
    # in the attributes anyway, the tag is just a container.
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        sanitized = re.sub(r"</?VitisHLS:", lambda m: m.group(0).replace("VitisHLS:", ""), raw)
        root = ET.fromstring(sanitized)
    except (ET.ParseError, OSError):
        return {"schema": "1.0", "passed": [], "widened": [],
                "failed": [], "summary": [],
                "counts": {"passed": 0, "widened": 0, "failed": 0, "summary": 0}}

    passed: List[Dict[str, Any]] = []
    widened: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    # The XML uses an `xmlns="VitisHLS"`-style root namespace, so
    # plain `iter("burst")` won't match. Iterate over all descendants
    # and check the local name (drop any `{namespace}` prefix).
    def _local(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag
    for el in root.iter():
        if _local(el.tag) != "burst":
            continue
        rec = {
            "group": el.attrib.get("group"),
            "severity": el.attrib.get("msg_severity"),
            "msg": el.attrib.get("msg_body"),
            "src": el.attrib.get("src_info"),
            "bundle": el.attrib.get("BundleName"),
            "var": el.attrib.get("VarName"),
            "direction": el.attrib.get("Direction"),
            "length": _coerce_int(el.attrib.get("Length")),
            "width": _coerce_int(el.attrib.get("Width")),
            "loop_name": el.attrib.get("LoopName"),
            "parent_func": el.attrib.get("ParentFunc"),
        }
        g = rec["group"]
        if g in _BURST_SUMMARY:
            summary.append(rec)
        elif g in _BURST_FAILED:
            failed.append(rec)
        else:
            passed.append(rec)
            if g in _BURST_WIDENED:
                widened.append(rec)

    return {
        "schema": "1.0",
        "passed": passed,
        "widened": widened,
        "failed": failed,
        "summary": summary,
        "counts": {
            "passed": len(passed),
            "widened": len(widened),
            "failed": len(failed),
            "summary": len(summary),
        },
    }


# --- fe_messages / be_messages parser ---

# msg_groups to surface. PRAGMA_INVALID is the single most useful one —
# tells us when a pragma the LLM emitted was silently rejected.
_INTERESTING_MSG_GROUPS = {
    "PRAGMA_INVALID",
    "PRAGMA_IGNORED",
    "PRAGMA_REJECTED",
    "DATAFLOW_INVALID",
    "DEPENDENCE_PRAGMA",
}


def parse_diagnostic_messages(work_dir: str) -> Dict[str, Any]:
    """Parse the front-end (clang) and back-end (scheduler) message
    streams. Returns a dict bucketed by severity + a focused list of
    rejected-pragma records that the FeedbackAgent should surface
    in retry prompts.

    Returns:
        {
          "warnings": int,
          "errors": int,
          "info": int,
          "rejected_pragmas": List[record],  # PRAGMA_INVALID etc.
          "examples": List[record],          # interesting non-pragma
          "schema": "1.0",
        }
    """
    db = _autopilot_db_dir(work_dir or "")
    out = {
        "schema": "1.0",
        "warnings": 0, "errors": 0, "info": 0,
        "rejected_pragmas": [],
        "examples": [],
    }
    if db is None:
        return out

    for fname in ("fe_messages.xml", "be_messages.xml"):
        path = os.path.join(db, fname)
        if not os.path.isfile(path):
            continue
        # Same `xilinx:` unbound-prefix workaround as burst.xml: read,
        # strip the prefix from element tags, then parse.
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            sanitized = re.sub(r"</?xilinx:", lambda m: m.group(0).replace("xilinx:", ""), raw)
            root = ET.fromstring(sanitized)
        except (ET.ParseError, OSError):
            continue
        def _local(tag: str) -> str:
            return tag.split("}", 1)[-1] if "}" in tag else tag
        for msg in root.iter():
            if _local(msg.tag) != "msg":
                continue
            sev = (msg.attrib.get("msg_severity") or "").upper()
            if sev == "WARNING":
                out["warnings"] += 1
            elif sev in ("ERROR", "FATAL"):
                out["errors"] += 1
            elif sev == "INFO":
                out["info"] += 1
            groups = msg.attrib.get("msg_groups", "") or ""
            group_list = [g.strip() for g in groups.split() if g.strip()]
            rec = {
                "source": fname.split("_")[0],  # "fe" or "be"
                "id": msg.attrib.get("msg_id"),
                "severity": sev,
                "groups": group_list,
                "loc": msg.attrib.get("msg_loc"),
                "body": (msg.attrib.get("msg_body") or "")[:300],
            }
            if any(g in _INTERESTING_MSG_GROUPS for g in group_list):
                out["rejected_pragmas"].append(rec)
            elif sev in ("ERROR", "FATAL"):
                out["examples"].append(rec)
            elif sev == "WARNING" and len(out["examples"]) < 8:
                out["examples"].append(rec)
    return out


# --- csynth_design_size.rpt parser ---

# Match a row with non-empty instruction count (the rows that carry data).
_PHASE_LINE_RE = re.compile(
    r"^\|\s*([\w/\\\- ]+?)?\s*\|\s*([\w() ,/+\\\-]+?)?\s*\|\s*(\d+)\s*\|"
)
# Match a phase-header row: phase name, empty step, empty instructions.
_PHASE_HEADER_RE = re.compile(
    r"^\|\s*([A-Za-z][\w/\\\- ]+?)\s*\|\s*\|\s*\|\s*"
)


def parse_design_size_report(work_dir: str) -> Dict[str, Any]:
    """Parse ``syn/report/csynth_design_size.rpt`` into per-phase
    instruction counts. Massive design-size growth across the
    Performance phase is a strong signal that the LLM's unroll/parallel
    pragmas inflated the kernel beyond what the synthesizer can
    reasonably schedule.

    Returns:
        {
          "phases": {<phase>: {<step>: instructions, ...}, ...},
          "compile_to_hw_growth": float,   # last_count / first_count
          "max_phase_growth": float,       # largest single-phase ratio
          "schema": "1.0",
        }
    """
    syn_dir = _syn_report_dir(work_dir or "")
    out = {"schema": "1.0", "phases": {},
           "compile_to_hw_growth": None, "max_phase_growth": None}
    if syn_dir is None:
        return out
    path = os.path.join(syn_dir, "csynth_design_size.rpt")
    if not os.path.isfile(path):
        return out
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError:
        return out

    phases: Dict[str, Dict[str, int]] = {}
    counts_in_order: List[int] = []
    in_table = False
    cur_phase = ""
    for line in text.splitlines():
        # Detect "* Total Instructions per Compilation Phase" or similar;
        # rows of the form "| <Phase> | <Step> | <N> | <desc> |"
        if "Total Instructions per Compilation Phase" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("* Instructions per Function"):
            break
        # First, check whether the row is a phase header (has phase name
        # but empty step + empty instruction columns). These rows update
        # cur_phase but don't carry their own row-level data.
        ph_header = _PHASE_HEADER_RE.match(line)
        if ph_header:
            candidate = ph_header.group(1).strip()
            # Skip the table header literally named "Phase".
            if candidate and candidate.lower() != "phase":
                cur_phase = candidate
            continue
        # Otherwise, look for a data row with an instruction count.
        m = _PHASE_LINE_RE.match(line)
        if not m:
            continue
        ph_raw = (m.group(1) or "").strip()
        step = (m.group(2) or "").strip()
        n_raw = m.group(3)
        try:
            n = int(n_raw)
        except ValueError:
            continue
        # If the row has both a phase name AND no step, it's the phase's
        # _total row (the rare case where a phase has no sub-steps).
        # If the row has a phase name AND a step, ph_raw and the cur_phase
        # may both be valid; prefer the explicit ph_raw if present.
        ph = ph_raw or cur_phase
        if ph_raw and ph_raw.lower() != "phase":
            cur_phase = ph_raw
        bucket = phases.setdefault(ph or "_unnamed", {})
        if step:
            bucket[step] = n
        else:
            bucket["_total"] = n
        counts_in_order.append(n)

    if counts_in_order:
        first, last = counts_in_order[0], counts_in_order[-1]
        out["compile_to_hw_growth"] = (last / first) if first > 0 else None
        max_ratio = 1.0
        for i in range(1, len(counts_in_order)):
            prev = counts_in_order[i - 1]
            cur = counts_in_order[i]
            if prev > 0:
                ratio = cur / prev
                if ratio > max_ratio:
                    max_ratio = ratio
        out["max_phase_growth"] = max_ratio
    out["phases"] = phases
    return out


# --- Phase 7a summary roll-up (used in the prompt-side renderer) ---


def summarize_static_extras(extras: Dict[str, Any]) -> Dict[str, Any]:
    """Compact rollup of the Phase 7a extras for prompt inclusion."""
    if not extras:
        return {}
    burst = extras.get("bursts") or {}
    burst_counts = burst.get("counts") or {}
    diag = extras.get("diagnostic") or {}
    ds = extras.get("design_size") or {}
    return {
        "bursts_passed": burst_counts.get("passed", 0),
        "bursts_widened": burst_counts.get("widened", 0),
        "bursts_failed": burst_counts.get("failed", 0),
        "rejected_pragmas": len(diag.get("rejected_pragmas") or []),
        "warnings": diag.get("warnings", 0),
        "errors": diag.get("errors", 0),
        "compile_to_hw_growth": ds.get("compile_to_hw_growth"),
        "max_phase_growth": ds.get("max_phase_growth"),
    }


def derive_static_bottleneck_records(static_extras: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert static report harvest into routeable bottleneck records."""
    if not static_extras:
        return []
    bursts = static_extras.get("bursts") or {}
    counts = bursts.get("counts") or {}
    failed = int(counts.get("failed") or 0)
    widened = int(counts.get("widened") or 0)
    passed = int(counts.get("passed") or 0)
    summary = int(counts.get("summary") or 0)
    records: List[Dict[str, Any]] = []
    if failed:
        rec = (bursts.get("failed") or [{}])[0]
        records.append({
            "scope_id": None,
            "kind": "axi_burst_failed",
            "evidence": (
                f"burst.xml reports {failed} failed AXI burst inference record(s); "
                f"first={rec.get('var') or '?'} {rec.get('direction') or ''} "
                f"{rec.get('src_info') or ''}".strip()
            ),
            "severity": "high",
            "metric": {"bursts_failed": failed, "bursts_widened": widened},
            "source_location": rec.get("src_info"),
        })
    elif (passed or summary) and widened == 0:
        records.append({
            "scope_id": None,
            "kind": "memory_bandwidth",
            "evidence": (
                "burst.xml has AXI burst records but no widened 512-bit transfers; "
                "latency may be bandwidth-limited"
            ),
            "severity": "medium",
            "metric": {"bursts_passed": passed, "bursts_summary": summary},
            "source_location": None,
        })
    return records


def render_static_extras_for_prompt(extras: Dict[str, Any], *,
                                     max_records: int = 4) -> str:
    """Compact human-readable static-extras block for the LLM prompt.
    Empty when nothing actionable was found."""
    if not extras:
        return ""
    lines: List[str] = []
    burst = extras.get("bursts") or {}
    counts = burst.get("counts") or {}
    if counts.get("widened") or counts.get("failed"):
        lines.append(
            f"AXI burst inference: passed={counts.get('passed', 0)}, "
            f"widened={counts.get('widened', 0)}, "
            f"failed={counts.get('failed', 0)}"
        )
        for rec in (burst.get("widened") or [])[:max_records]:
            lines.append(
                f"  WIDENED: {rec.get('var')} on bundle {rec.get('bundle')} "
                f"({rec.get('direction')}) — {rec.get('msg', '')[:120]}"
            )
        for rec in (burst.get("failed") or [])[:max_records]:
            lines.append(
                f"  FAILED:  {rec.get('var')} on bundle {rec.get('bundle')} "
                f"({rec.get('direction')}) at {rec.get('src')} — "
                f"{rec.get('msg', '')[:160]}"
            )

    diag = extras.get("diagnostic") or {}
    rejected = diag.get("rejected_pragmas") or []
    if rejected:
        lines.append(f"Pragmas silently rejected by Vitis: {len(rejected)}")
        for rec in rejected[:max_records]:
            lines.append(
                f"  {rec.get('id')} [{rec.get('severity')}] at {rec.get('loc')} — "
                f"{rec.get('body', '')[:160]}"
            )

    ds = extras.get("design_size") or {}
    if ds.get("compile_to_hw_growth"):
        lines.append(
            f"Design size growth across compilation phases: "
            f"{ds.get('compile_to_hw_growth'):.2f}x end-to-end "
            f"(max single-phase ratio {ds.get('max_phase_growth') or 0:.2f}x)"
        )

    return "\n".join(lines)


# === build_feedback / attach_feedback are extended below to wire 7a ====

def build_feedback(*, xml_path: Optional[str] = None,
                   rpt_path: Optional[str] = None,
                   log_path: Optional[str] = None,
                   log_text: Optional[str] = None,
                   rpt_text: Optional[str] = None,
                   work_dir: Optional[str] = None,
                   requested_clock_ns: Optional[float] = None) -> Dict[str, Any]:
    """Read whichever of (xml, rpt, log) are present and assemble a
    feedback dict. Resilient to missing files. `log_text` / `rpt_text` let
    callers pass in already-loaded content without writing it to disk.

    Phase 7a: pass ``work_dir`` to also harvest the static reports under
    ``<work_dir>/hls_proj/sol1/`` (burst.xml, fe/be_messages.xml,
    csynth_design_size.rpt). Returns under ``feedback["static_extras"]``."""
    scopes_xml: List[Dict[str, Any]] = []
    if xml_path and os.path.exists(xml_path):
        scopes_xml = parse_synthesis_xml_per_scope(xml_path)

    scopes_rpt: List[Dict[str, Any]] = []
    if rpt_text is None and rpt_path and os.path.exists(rpt_path):
        try:
            with open(rpt_path, "r", encoding="utf-8", errors="ignore") as f:
                rpt_text = f.read()
        except OSError:
            rpt_text = None
    if rpt_text:
        scopes_rpt, _ = parse_synthesis_report_per_scope(rpt_text)

    # Merge: prefer XML for the structural data (it carries PipelineII /
    # PipelineDepth / Slack on every loop), fall back to text for per-scope
    # resources (which the XML places on the parent module).
    scopes = _merge_scope_lists(scopes_xml, scopes_rpt)

    blame: List[Dict[str, Any]] = []
    if log_text is None and log_path and os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                log_text = f.read()
        except OSError:
            log_text = None
    if log_text:
        blame = parse_vitis_hls_log(log_text)

    bottlenecks = derive_bottleneck_records(scopes, blame,
                                            requested_clock_ns=requested_clock_ns)
    summary = summarize_feedback(scopes, bottlenecks)

    # Phase 7a: harvest static reports if work_dir is provided. All
    # parsers tolerate missing files; "static_extras" stays empty if
    # nothing useful is on disk.
    static_extras: Dict[str, Any] = {}
    if work_dir:
        bursts = parse_burst_info(work_dir)
        diagnostic = parse_diagnostic_messages(work_dir)
        design_size = parse_design_size_report(work_dir)
        # Only attach when at least one parser found something — keeps
        # the feedback dict clean for unit tests / unrelated synth runs.
        any_data = (
            (bursts.get("counts", {}).get("passed", 0)
             + bursts.get("counts", {}).get("failed", 0)
             + bursts.get("counts", {}).get("summary", 0)) > 0
            or diagnostic.get("warnings", 0) > 0
            or diagnostic.get("rejected_pragmas")
            or design_size.get("phases")
        )
        if any_data:
            static_extras = {
                "schema": "1.0",
                "bursts": bursts,
                "diagnostic": diagnostic,
                "design_size": design_size,
            }
            static_extras["summary"] = summarize_static_extras(static_extras)
            bottlenecks.extend(derive_static_bottleneck_records(static_extras))
            severity_rank = {"high": 0, "medium": 1, "low": 2}
            bottlenecks.sort(key=lambda b: (severity_rank.get(b.get("severity"), 3), b.get("kind") or ""))
            summary = summarize_feedback(scopes, bottlenecks)

    out: Dict[str, Any] = {
        "schema": FEEDBACK_SCHEMA_VERSION,
        "scopes": scopes,
        "scheduler_blame": blame,
        "bottlenecks": bottlenecks,
        "summary": summary,
    }
    if static_extras:
        out["static_extras"] = static_extras
    return out


def attach_feedback(report: Dict[str, Any], *,
                    xml_path: Optional[str] = None,
                    rpt_path: Optional[str] = None,
                    log_path: Optional[str] = None,
                    log_text: Optional[str] = None,
                    rpt_text: Optional[str] = None,
                    work_dir: Optional[str] = None) -> Dict[str, Any]:
    """Convenience wrapper: returns `report` with `report["feedback"]` filled
    in. Mutates `report` in place AND returns it.

    Phase 7a: pass ``work_dir`` to harvest the static-report extras
    (burst inference, pragma diagnostics, design-size growth)."""
    fb = build_feedback(
        xml_path=xml_path, rpt_path=rpt_path, log_path=log_path,
        log_text=log_text, rpt_text=rpt_text,
        work_dir=work_dir,
        requested_clock_ns=report.get("requested_clock_period_ns"),
    )
    report["feedback"] = fb
    return report


def _merge_scope_lists(primary: List[Dict[str, Any]],
                       secondary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge two ScopeRecord lists by scope_id, preferring primary fields
    when both have a value."""
    if not primary:
        return list(secondary)
    if not secondary:
        return list(primary)
    by_id: Dict[str, Dict[str, Any]] = {}
    for sc in primary:
        by_id[sc["scope_id"]] = dict(sc)
    for sc in secondary:
        cur = by_id.get(sc["scope_id"])
        if cur is None:
            # Best-effort name/parent matching for cases where text and XML
            # disagree on the scope_id (e.g., XML uses "Loop1" tag, text uses
            # "Loop 1"). Match by (parent, normalized name).
            normalized = _normalize_scope_match(sc)
            for cand_id, cand in by_id.items():
                if _normalize_scope_match(cand) == normalized:
                    cur = cand
                    break
        if cur is None:
            by_id[sc["scope_id"]] = dict(sc)
            continue
        for k, v in sc.items():
            if v is None:
                continue
            if cur.get(k) is None:
                cur[k] = v
    return list(by_id.values())


def _normalize_scope_match(sc: Dict[str, Any]) -> Tuple[Optional[str], str]:
    name = (sc.get("name") or "").lower().replace(" ", "")
    return (sc.get("parent"), name)


# --- prompt-side helpers ---------------------------------------------------


def render_feedback_for_prompt(feedback: Dict[str, Any], *, max_scopes: int = 12,
                               max_bottlenecks: int = 6) -> str:
    """Compact human-readable render suitable for inclusion in an LLM prompt.
    Bottlenecks first (action items), then top-N scopes."""
    if not feedback:
        return ""
    lines: List[str] = []
    summary = feedback.get("summary") or {}
    lines.append(
        f"HLS feedback summary: {summary.get('loop_count', 0)} loops, "
        f"{summary.get('pipelined_loops', 0)} pipelined, "
        f"{summary.get('scopes_with_negative_slack', 0)} with negative slack, "
        f"{summary.get('bottleneck_count', 0)} bottlenecks "
        f"({summary.get('high_severity_bottlenecks', 0)} high-severity)."
    )
    bottlenecks = feedback.get("bottlenecks") or []
    if bottlenecks:
        lines.append("Top bottlenecks:")
        for b in bottlenecks[:max_bottlenecks]:
            sid = b.get("scope_id") or "(global)"
            sev = b.get("severity") or "?"
            lines.append(f"  - [{sev}] {b.get('kind')}: {sid} :: {b.get('evidence')}")
    scopes = feedback.get("scopes") or []
    # Prioritize scopes that look hot / problematic.
    def _scope_priority(s: Dict[str, Any]) -> Tuple[int, int]:
        slack = s.get("slack_ns")
        bad_slack = 1 if (slack is not None and slack < 0) else 0
        lat = s.get("latency_cycles") or 0
        return (-bad_slack, -lat)
    for s in sorted(scopes, key=_scope_priority)[:max_scopes]:
        lines.append(
            f"  scope {s.get('scope_id')} ({s.get('kind')}) "
            f"lat={s.get('latency_cycles')} ii={s.get('interval')} "
            f"trip={s.get('trip_count')} pipelined={s.get('pipelined')} "
            f"slack={s.get('slack_ns')}"
        )
    return "\n".join(lines)


def render_diagnostic_for_prompt(feedback: Dict[str, Any], *, max_examples: int = 12) -> str:
    """Compact render of HLS diagnostic warnings/errors for curation prompts."""
    if not feedback:
        return ""
    extras = feedback.get("static_extras") or {}
    diag = extras.get("diagnostic") or {}
    if not diag:
        return ""

    lines: List[str] = []
    lines.append(
        f"HLS diagnostics: {diag.get('warnings', 0)} warnings, "
        f"{diag.get('errors', 0)} errors, "
        f"{len(diag.get('rejected_pragmas') or [])} rejected pragmas."
    )
    rejected = diag.get("rejected_pragmas") or []
    if rejected:
        lines.append("Rejected / invalid pragmas:")
        for rec in rejected[:max_examples]:
            loc = rec.get("loc") or "?"
            body = (rec.get("body") or "").strip()
            lines.append(f"  - [{rec.get('id')}] {loc}: {body[:240]}")
    examples = diag.get("examples") or []
    if examples:
        lines.append("Other warnings/errors:")
        shown = 0
        for rec in examples:
            if shown >= max_examples:
                break
            sev = rec.get("severity") or "?"
            loc = rec.get("loc") or "?"
            body = (rec.get("body") or "").strip()
            lines.append(f"  - [{sev}] {rec.get('id')} {loc}: {body[:240]}")
            shown += 1
    return "\n".join(lines)
