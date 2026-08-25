"""Deterministic, reference-blind QoR design-space utilities.

This module discovers numeric HLS design knobs that can be changed without an
LLM rewriting the surrounding algorithm.  It builds frozen-parent candidates,
extracts comparable CSynth observations, computes local trends, and identifies
the feasible Pareto frontier.  Tool execution remains owned by c2hls.py.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional, Sequence


RESOURCE_KEYS = ("dsp", "bram", "lut", "ff", "uram")
PARETO_KEYS = (
    "latency_cycles_worst",
    "dsp",
    "bram",
    "lut",
    "ff",
    "uram",
    "estimated_clock_period_ns",
)

STEP_PREFERRED_KINDS = {
    "pipeline": ("pipeline_ii",),
    "unroll": ("unroll_factor",),
    "tiling": ("tile_size", "partition_factor", "reshape_factor"),
    "doublebuffer": (
        "dataflow_enabled",
        "stream_depth",
        "tile_size",
        "partition_enabled",
        "partition_factor",
        "reshape_enabled",
        "reshape_factor",
    ),
    "coalescing": (
        "interface_max_widen_bitwidth",
        "interface_num_read_outstanding",
        "interface_num_write_outstanding",
        "interface_max_read_burst_length",
        "interface_max_write_burst_length",
    ),
    "resource": (
        "allocation_limit",
        "bind_op_latency",
        "bind_storage_latency",
        "resource_latency",
        "bind_op_enabled",
        "bind_storage_enabled",
        "resource_enabled",
        "partition_enabled",
        "reshape_enabled",
    ),
}


@dataclass(frozen=True)
class QorKnob:
    """One source-level parameter with a stable replacement span."""

    knob_id: str
    kind: str
    name: str
    line: int
    current_value: Optional[int]
    current_label: str
    candidate_values: tuple[int, ...]
    start: int
    end: int
    replacement_template: str
    source_context: str
    scope: str

    def public(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("start", None)
        value.pop("end", None)
        value.pop("replacement_template", None)
        value["candidate_values"] = list(self.candidate_values)
        return value


def code_sha256(code: str) -> str:
    return hashlib.sha256((code or "").encode("utf-8")).hexdigest()


def _stable_knob_id(kind: str, name: str, line: int, context: str) -> str:
    payload = f"{kind}|{name}|{line}|{context.strip()}".encode("utf-8")
    return f"{kind}:{hashlib.sha256(payload).hexdigest()[:12]}"


def _normalized_values(values: Iterable[int], current: Optional[int]) -> tuple[int, ...]:
    normalized = sorted({int(value) for value in values if int(value) > 0})
    return tuple(value for value in normalized if current is None or value != current)


def _line_number(code: str, offset: int) -> int:
    return code.count("\n", 0, offset) + 1


def _make_numeric_knob(
    code: str,
    match: re.Match[str],
    *,
    kind: str,
    name: str,
    values: Sequence[int],
    scope: str,
) -> QorKnob:
    start, end = match.span("value")
    current = int(match.group("value"))
    line = _line_number(code, start)
    context_start = code.rfind("\n", 0, start) + 1
    context_end = code.find("\n", end)
    if context_end < 0:
        context_end = len(code)
    context = code[context_start:context_end].strip()
    return QorKnob(
        knob_id=_stable_knob_id(kind, name, line, context),
        kind=kind,
        name=name,
        line=line,
        current_value=current,
        current_label=str(current),
        candidate_values=_normalized_values(values, current),
        start=start,
        end=end,
        replacement_template="{value}",
        source_context=context,
        scope=scope,
    )


def _make_disable_toggle(
    code: str,
    match: re.Match[str],
    *,
    kind: str,
    name: str,
    scope: str,
) -> QorKnob:
    original = match.group(0)
    indent_match = re.match(r"\s*", original)
    indent = indent_match.group(0) if indent_match else ""
    line = _line_number(code, match.start())
    return QorKnob(
        knob_id=_stable_knob_id(kind, name, line, original),
        kind=kind,
        name=name,
        line=line,
        current_value=1,
        current_label="enabled",
        candidate_values=(0,),
        start=match.start(),
        end=match.end(),
        replacement_template=(
            f"{indent}#if {{value}}\n{original}\n{indent}#endif"
        ),
        source_context=original.strip(),
        scope=scope,
    )


def discover_qor_knobs(
    code: str,
    *,
    factor_values: Sequence[int] = (1, 2, 4, 8, 16),
    ii_values: Sequence[int] = (1, 2, 4, 8),
    tile_values: Sequence[int] = (4, 8, 16, 32, 64),
    stream_depth_values: Sequence[int] = (2, 4, 8, 16, 32, 64),
    widen_values: Sequence[int] = (64, 128, 256, 512, 1024),
    max_knobs: Optional[int] = None,
    preferred_kinds: Sequence[str] = (),
) -> list[QorKnob]:
    """Discover safely replaceable pragma and named-constant knobs.

    Discovery is intentionally conservative.  It does not infer loop bounds or
    rewrite arbitrary expressions.  Every returned knob changes one integer
    token, except unqualified PIPELINE/UNROLL pragmas where a parameter is
    appended.
    """

    knobs: list[QorKnob] = []
    occupied: set[tuple[int, int, str]] = set()

    pattern_specs = [
        (
            "pipeline_ii",
            re.compile(
                r"(?im)^\s*#\s*pragma\s+HLS\s+PIPELINE\b[^\n]*?\bII\s*=\s*(?P<value>\d+)"
            ),
            ii_values,
            lambda match: f"pipeline_ii@L{_line_number(code, match.start('value'))}",
            "loop",
        ),
        (
            "unroll_factor",
            re.compile(
                r"(?im)^\s*#\s*pragma\s+HLS\s+UNROLL\b[^\n]*?\bfactor\s*=\s*(?P<value>\d+)"
            ),
            factor_values,
            lambda match: f"unroll@L{_line_number(code, match.start('value'))}",
            "loop",
        ),
        (
            "partition_factor",
            re.compile(
                r"(?im)^\s*#\s*pragma\s+HLS\s+ARRAY_PARTITION\b[^\n]*?"
                r"\bvariable\s*=\s*(?P<variable>[A-Za-z_]\w*)[^\n]*?"
                r"\bfactor\s*=\s*(?P<value>\d+)"
            ),
            factor_values,
            lambda match: f"partition:{match.group('variable')}",
            "array_dimension",
        ),
        (
            "reshape_factor",
            re.compile(
                r"(?im)^\s*#\s*pragma\s+HLS\s+ARRAY_RESHAPE\b[^\n]*?"
                r"\bvariable\s*=\s*(?P<variable>[A-Za-z_]\w*)[^\n]*?"
                r"\bfactor\s*=\s*(?P<value>\d+)"
            ),
            factor_values,
            lambda match: f"reshape:{match.group('variable')}",
            "array_dimension",
        ),
        (
            "stream_depth",
            re.compile(
                r"(?im)^\s*#\s*pragma\s+HLS\s+STREAM\b[^\n]*?"
                r"\bvariable\s*=\s*(?P<variable>[A-Za-z_]\w*)[^\n]*?"
                r"\bdepth\s*=\s*(?P<value>\d+)"
            ),
            stream_depth_values,
            lambda match: f"stream:{match.group('variable')}",
            "stream",
        ),
        (
            "allocation_limit",
            re.compile(
                r"(?im)^\s*#\s*pragma\s+HLS\s+ALLOCATION\b[^\n]*?"
                r"\blimit\s*=\s*(?P<value>\d+)"
            ),
            factor_values,
            lambda match: f"allocation@L{_line_number(code, match.start('value'))}",
            "operation",
        ),
        (
            "bind_op_latency",
            re.compile(
                r"(?im)^\s*#\s*pragma\s+HLS\s+BIND_OP\b[^\n]*?"
                r"\blatency\s*=\s*(?P<value>\d+)"
            ),
            factor_values,
            lambda match: f"bind_op_latency@L{_line_number(code, match.start('value'))}",
            "operation_binding",
        ),
        (
            "bind_storage_latency",
            re.compile(
                r"(?im)^\s*#\s*pragma\s+HLS\s+BIND_STORAGE\b[^\n]*?"
                r"\blatency\s*=\s*(?P<value>\d+)"
            ),
            factor_values,
            lambda match: f"bind_storage_latency@L{_line_number(code, match.start('value'))}",
            "storage_binding",
        ),
        (
            "resource_latency",
            re.compile(
                r"(?im)^\s*#\s*pragma\s+HLS\s+RESOURCE\b[^\n]*?"
                r"\blatency\s*=\s*(?P<value>\d+)"
            ),
            factor_values,
            lambda match: f"resource_latency@L{_line_number(code, match.start('value'))}",
            "resource_binding",
        ),
    ]
    for kind, pattern, values, name_fn, scope in pattern_specs:
        for match in pattern.finditer(code):
            span_key = (*match.span("value"), kind)
            if span_key in occupied:
                continue
            occupied.add(span_key)
            knobs.append(
                _make_numeric_knob(
                    code,
                    match,
                    kind=kind,
                    name=name_fn(match),
                    values=values,
                    scope=scope,
                )
            )

    interface_options = (
        "max_widen_bitwidth",
        "num_read_outstanding",
        "num_write_outstanding",
        "max_read_burst_length",
        "max_write_burst_length",
    )
    for option in interface_options:
        interface_pattern = re.compile(
            rf"(?im)^\s*#\s*pragma\s+HLS\s+INTERFACE\b[^\n]*?"
            rf"\b{option}\s*=\s*(?P<value>\d+)"
        )
        for match in interface_pattern.finditer(code):
            values = (
                widen_values
                if option == "max_widen_bitwidth"
                else stream_depth_values
            )
            knobs.append(
                _make_numeric_knob(
                    code,
                    match,
                    kind=f"interface_{option}",
                    name=f"{option}@L{_line_number(code, match.start('value'))}",
                    values=values,
                    scope="interface",
                )
            )

    # A coalescing rewrite often leaves a bare m_axi directive.  Appending one
    # explicit width cap is source-local and preserves the interface contract;
    # CSim/CSynth still decide whether a value is legal and useful.
    m_axi_line_pattern = re.compile(
        r"(?im)^\s*#\s*pragma\s+HLS\s+INTERFACE\b[^\n]*\bm_axi\b[^\n]*$"
    )
    for match in m_axi_line_pattern.finditer(code):
        original = match.group(0)
        if re.search(r"(?i)\bmax_widen_bitwidth\s*=", original):
            continue
        body, separator, comment = original.partition("//")
        port_match = re.search(r"(?i)\bport\s*=\s*([A-Za-z_]\w*)", body)
        port = port_match.group(1) if port_match else "unknown"
        line = _line_number(code, match.start())
        rendered_comment = f" //{comment}" if separator else ""
        context = original.strip()
        knobs.append(
            QorKnob(
                knob_id=_stable_knob_id(
                    "interface_max_widen_bitwidth",
                    f"max_widen_bitwidth:{port}",
                    line,
                    context,
                ),
                kind="interface_max_widen_bitwidth",
                name=f"max_widen_bitwidth:{port}@L{line}",
                line=line,
                current_value=None,
                current_label="auto",
                candidate_values=_normalized_values(widen_values, None),
                start=match.start(),
                end=match.end(),
                replacement_template=(
                    body.rstrip() + " max_widen_bitwidth={value}" + rendered_comment
                ),
                source_context=context,
                scope="interface",
            )
        )

    named_constant_pattern = re.compile(
        r"(?im)^(?:\s*#\s*define\s+|\s*(?:static\s+)?(?:const|constexpr)\s+"
        r"(?:unsigned\s+)?(?:int|size_t)\s+)"
        r"(?P<name>[A-Za-z_]\w*)\s*"
        r"(?:=\s*)?(?P<value>\d+)"
    )
    for match in named_constant_pattern.finditer(code):
        name = match.group("name")
        line_start = code.rfind("\n", 0, match.start()) + 1
        line_end = code.find("\n", match.end())
        if line_end < 0:
            line_end = len(code)
        line_context = code[line_start:line_end]
        explicit_name = any(
            token in name.upper() for token in ("TILE", "BLOCK")
        )
        conventional_name = name.lower() in {
            "ti", "tj", "tk", "tm", "tn", "tx", "ty", "tz"
        }
        if not explicit_name and not (
            conventional_name
            and re.search(r"(?i)\b(tile|block)\b", line_context)
        ):
            continue
        knobs.append(
            _make_numeric_knob(
                code,
                match,
                kind="tile_size",
                name=name,
                values=tile_values,
                scope="loop_nest",
            )
        )

    explicit_pipeline_lines = {
        knob.line for knob in knobs if knob.kind == "pipeline_ii"
    }
    automatic_pipeline_pattern = re.compile(
        r"(?im)^(?P<body>\s*#\s*pragma\s+HLS\s+PIPELINE\b(?:(?!//)[^\n])*)"
        r"(?P<comment>//[^\n]*)?$"
    )
    for match in automatic_pipeline_pattern.finditer(code):
        line = _line_number(code, match.start())
        body = match.group("body").rstrip()
        if (
            line in explicit_pipeline_lines
            or re.search(r"(?i)\bII\s*=", body)
            or re.search(r"(?i)\boff\b", body)
        ):
            continue
        comment = match.group("comment") or ""
        separator = " " if comment else ""
        context = match.group(0).strip()
        name = f"pipeline_ii@L{line}"
        knobs.append(
            QorKnob(
                knob_id=_stable_knob_id("pipeline_ii", name, line, context),
                kind="pipeline_ii",
                name=name,
                line=line,
                current_value=None,
                current_label="auto",
                candidate_values=_normalized_values(ii_values, None),
                start=match.start(),
                end=match.end(),
                replacement_template=(
                    body + " II={value}" + separator + comment
                ),
                source_context=context,
                scope="loop",
            )
        )

    explicit_unroll_spans = {
        (knob.line, knob.source_context)
        for knob in knobs
        if knob.kind == "unroll_factor"
    }
    full_unroll_pattern = re.compile(
        r"(?im)^(?P<prefix>\s*#\s*pragma\s+HLS\s+UNROLL\b)"
        r"(?P<trailing>\s*(?://[^\n]*)?)$"
    )
    for match in full_unroll_pattern.finditer(code):
        line = _line_number(code, match.start())
        context = match.group(0).strip()
        if (line, context) in explicit_unroll_spans or "factor" in context.lower():
            continue
        name = f"unroll@L{line}"
        knobs.append(
            QorKnob(
                knob_id=_stable_knob_id("unroll_factor", name, line, context),
                kind="unroll_factor",
                name=name,
                line=line,
                current_value=None,
                current_label="full",
                candidate_values=_normalized_values(factor_values, None),
                start=match.start(),
                end=match.end(),
                replacement_template=(
                    match.group("prefix") + " factor={value}" + match.group("trailing")
                ),
                source_context=context,
                scope="loop",
            )
        )

    toggle_specs = (
        (
            "dataflow_enabled",
            re.compile(r"(?im)^\s*#\s*pragma\s+HLS\s+DATAFLOW\b[^\n]*$"),
            "dataflow",
            lambda line: True,
        ),
        (
            "partition_enabled",
            re.compile(
                r"(?im)^\s*#\s*pragma\s+HLS\s+ARRAY_PARTITION\b[^\n]*$"
            ),
            "array_partition",
            lambda line: "complete" in line.lower() and "factor" not in line.lower(),
        ),
        (
            "reshape_enabled",
            re.compile(
                r"(?im)^\s*#\s*pragma\s+HLS\s+ARRAY_RESHAPE\b[^\n]*$"
            ),
            "array_reshape",
            lambda line: "complete" in line.lower() and "factor" not in line.lower(),
        ),
        (
            "bind_op_enabled",
            re.compile(r"(?im)^\s*#\s*pragma\s+HLS\s+BIND_OP\b[^\n]*$"),
            "operation_binding",
            lambda line: "latency" not in line.lower(),
        ),
        (
            "bind_storage_enabled",
            re.compile(r"(?im)^\s*#\s*pragma\s+HLS\s+BIND_STORAGE\b[^\n]*$"),
            "storage_binding",
            lambda line: "latency" not in line.lower(),
        ),
        (
            "resource_enabled",
            re.compile(r"(?im)^\s*#\s*pragma\s+HLS\s+RESOURCE\b[^\n]*$"),
            "resource_binding",
            lambda line: "latency" not in line.lower(),
        ),
    )
    for kind, pattern, scope, eligible in toggle_specs:
        for match in pattern.finditer(code):
            if not eligible(match.group(0)):
                continue
            line = _line_number(code, match.start())
            knobs.append(
                _make_disable_toggle(
                    code,
                    match,
                    kind=kind,
                    name=f"{kind}@L{line}",
                    scope=scope,
                )
            )

    priority = {
        "pipeline_ii": 0,
        "unroll_factor": 1,
        "partition_factor": 2,
        "reshape_factor": 3,
        "tile_size": 4,
        "stream_depth": 5,
        "allocation_limit": 6,
        "dataflow_enabled": 7,
        "partition_enabled": 8,
        "reshape_enabled": 9,
        "bind_op_latency": 10,
        "bind_storage_latency": 11,
        "resource_latency": 12,
        "bind_op_enabled": 13,
        "bind_storage_enabled": 14,
        "resource_enabled": 15,
    }
    preferred_rank = {
        kind: index for index, kind in enumerate(dict.fromkeys(preferred_kinds))
    }

    def kind_order(kind: str) -> tuple[int, int, str]:
        if kind in preferred_rank:
            return (0, preferred_rank[kind], kind)
        return (1, priority.get(kind, 100), kind)

    knobs.sort(key=lambda item: (*kind_order(item.kind), item.line, item.knob_id))
    if max_knobs is not None:
        # Preserve design-space diversity when a kernel has many instances of
        # one pragma type.  Taking the first N would otherwise commonly spend
        # the whole budget on PIPELINE II and hide partition/tile controls.
        grouped: dict[str, list[QorKnob]] = {}
        for knob in knobs:
            grouped.setdefault(knob.kind, []).append(knob)
        limited: list[QorKnob] = []
        limit = max(0, int(max_knobs))
        depth = 0
        ordered_kinds = sorted(grouped, key=kind_order)
        kind_pools = [ordered_kinds]
        if preferred_rank:
            # A winning step's controls are the first experiment family.  Fill
            # the bounded knob budget from those controls before unrelated
            # inherited pragmas, while retaining round-robin diversity within
            # the preferred family.
            preferred = [kind for kind in ordered_kinds if kind in preferred_rank]
            fallback = [kind for kind in ordered_kinds if kind not in preferred_rank]
            kind_pools = [preferred, fallback]
        for pool in kind_pools:
            depth = 0
            while pool and len(limited) < limit:
                added = False
                for kind in pool:
                    if depth >= len(grouped[kind]):
                        continue
                    limited.append(grouped[kind][depth])
                    added = True
                    if len(limited) >= limit:
                        break
                if not added:
                    break
                depth += 1
        knobs = limited
    return knobs


def apply_knob_values(code: str, assignments: Sequence[tuple[QorKnob, int]]) -> str:
    """Apply non-overlapping replacements against one frozen parent."""

    replacements = sorted(assignments, key=lambda item: item[0].start, reverse=True)
    result = code
    previous_start = len(code) + 1
    for knob, raw_value in replacements:
        value = int(raw_value)
        zero_is_valid = knob.kind.endswith("_enabled")
        if value < 0 or (value == 0 and not zero_is_valid) or value not in knob.candidate_values:
            raise ValueError(f"illegal value {value} for {knob.knob_id}")
        if knob.end > previous_start:
            raise ValueError("overlapping QoR knob replacements")
        replacement = knob.replacement_template.format(value=value)
        result = result[: knob.start] + replacement + result[knob.end :]
        previous_start = knob.start
    return result


def _diff_sha256(parent: str, candidate: str) -> str:
    diff = "".join(
        difflib.unified_diff(
            parent.splitlines(keepends=True),
            candidate.splitlines(keepends=True),
            fromfile="parent.cpp",
            tofile="candidate.cpp",
        )
    )
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def _candidate_payload(
    parent: str,
    assignments: Sequence[tuple[QorKnob, int]],
    *,
    stage: str,
) -> dict[str, Any]:
    candidate_code = apply_knob_values(parent, assignments)
    changed = [
        {
            "knob_id": knob.knob_id,
            "kind": knob.kind,
            "name": knob.name,
            "line": knob.line,
            "from": knob.current_value if knob.current_value is not None else knob.current_label,
            "to": int(value),
        }
        for knob, value in assignments
    ]
    identity = json.dumps(changed, sort_keys=True, separators=(",", ":"))
    candidate_id = f"qor-{stage}-{hashlib.sha256(identity.encode()).hexdigest()[:12]}"
    return {
        "candidate_id": candidate_id,
        "stage": stage,
        "changed_knobs": changed,
        "code": candidate_code,
        "code_sha256": code_sha256(candidate_code),
        "source_diff_sha256": _diff_sha256(parent, candidate_code),
    }


def build_ofat_candidates(
    parent: str,
    knobs: Sequence[QorKnob],
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Build a fair, round-robin one-factor-at-a-time candidate sequence."""

    limit = max(0, int(max_candidates))
    candidates: list[dict[str, Any]] = []
    seen_hashes = {code_sha256(parent)}
    depth = 0
    while len(candidates) < limit:
        added = False
        for knob in knobs:
            if depth >= len(knob.candidate_values):
                continue
            payload = _candidate_payload(
                parent,
                [(knob, knob.candidate_values[depth])],
                stage="ofat",
            )
            if payload["code_sha256"] not in seen_hashes:
                candidates.append(payload)
                seen_hashes.add(payload["code_sha256"])
                added = True
            if len(candidates) >= limit:
                break
        if not added:
            break
        depth += 1
    return candidates


def build_interaction_candidates(
    parent: str,
    knobs: Sequence[QorKnob],
    preferred_values: dict[str, int],
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Combine individually preferred values in stable pair order."""

    selected = [
        (knob, preferred_values[knob.knob_id])
        for knob in knobs
        if knob.knob_id in preferred_values
        and preferred_values[knob.knob_id] in knob.candidate_values
    ]
    candidates: list[dict[str, Any]] = []
    seen = {code_sha256(parent)}
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            payload = _candidate_payload(parent, [left, right], stage="interaction")
            if payload["code_sha256"] in seen:
                continue
            candidates.append(payload)
            seen.add(payload["code_sha256"])
            if len(candidates) >= max(0, int(max_candidates)):
                return candidates
    return candidates


def _number(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def extract_qor_metrics(report: Optional[dict[str, Any]]) -> dict[str, Any]:
    report = report or {}
    worst = _number(report.get("latency_cycles_worst"))
    nominal = _number(report.get("latency_cycles"))
    feedback = report.get("feedback") if isinstance(report.get("feedback"), dict) else {}
    scopes = feedback.get("scopes") if isinstance(feedback.get("scopes"), list) else []
    achieved_ii = []
    for scope in scopes:
        if not isinstance(scope, dict):
            continue
        pipeline_ii = _number(scope.get("pipeline_ii"))
        if pipeline_ii is None:
            continue
        achieved_ii.append({
            "scope_id": scope.get("scope_id") or scope.get("name"),
            "pipeline_ii": pipeline_ii,
            "trip_count": _number(scope.get("trip_count")),
            "violation": scope.get("violation"),
        })
    metrics = {
        "latency_cycles": nominal,
        "latency_cycles_worst": worst or nominal,
        "interval": _number(report.get("interval")),
        "slack_ns": _number(report.get("slack_ns")),
        "estimated_clock_period_ns": _number(report.get("estimated_clock_period_ns")),
        "requested_clock_period_ns": _number(report.get("requested_clock_period_ns")),
        "fmax_mhz": _number(report.get("fmax_mhz")),
        "achieved_pipeline_ii": achieved_ii,
        "achieved_pipeline_ii_max": (
            max(item["pipeline_ii"] for item in achieved_ii)
            if achieved_ii
            else None
        ),
    }
    for key in RESOURCE_KEYS:
        metrics[key] = _number(report.get(key))
    return metrics


def _pareto_value(record: dict[str, Any], key: str) -> float:
    value = _number((record.get("metrics") or {}).get(key))
    return value if value is not None else float("inf")


def selection_rank(record: dict[str, Any]) -> tuple[float, float, str]:
    """Return the deterministic rank used after feasibility is established.

    Worst-case latency is the primary objective. Aggregate resource use only
    breaks exact latency ties; missing resource evidence cannot outrank a
    candidate with a complete CSynth report.
    """
    resource_values = [
        _number((record.get("metrics") or {}).get(key))
        for key in RESOURCE_KEYS
    ]
    resource_sum = (
        sum(value for value in resource_values if value is not None)
        if all(value is not None for value in resource_values)
        else float("inf")
    )
    return (
        _pareto_value(record, "latency_cycles_worst"),
        resource_sum,
        str(record.get("candidate_id") or ""),
    )


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_values = [_pareto_value(left, key) for key in PARETO_KEYS]
    right_values = [_pareto_value(right, key) for key in PARETO_KEYS]
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def pareto_candidate_ids(records: Sequence[dict[str, Any]]) -> list[str]:
    feasible = [
        record
        for record in records
        if record.get("feasible") is True
        and _pareto_value(record, "latency_cycles_worst") < float("inf")
    ]
    frontier = []
    for candidate in feasible:
        if any(
            other is not candidate and _dominates(other, candidate)
            for other in feasible
        ):
            continue
        frontier.append(str(candidate.get("candidate_id")))
    return sorted(frontier)


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for index in order[cursor:end]:
            ranks[index] = average
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def spearman(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    return _pearson(_average_ranks(left), _average_ranks(right))


def _expectation_for_kind(kind: str) -> str:
    if kind.endswith("_enabled"):
        return "ablation_no_monotonic_prior"
    if kind == "pipeline_ii":
        return "cycles_nondecreasing_as_value_increases"
    if kind == "tile_size":
        return "nonmonotone_knee_expected"
    return "cycles_nonincreasing_as_value_increases"


def summarize_knob_trends(
    records: Sequence[dict[str, Any]],
    knobs: Sequence[QorKnob],
    *,
    parent: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    summaries = []
    for knob in knobs:
        observations = []
        if (
            parent is not None
            and parent.get("feasible") is True
            and knob.current_value is not None
        ):
            parent_cycles = _number(
                (parent.get("metrics") or {}).get("latency_cycles_worst")
            )
            if parent_cycles is not None:
                observations.append(
                    (float(knob.current_value), parent_cycles, parent, True)
                )
        for record in records:
            changed = record.get("changed_knobs") or []
            if len(changed) != 1 or changed[0].get("knob_id") != knob.knob_id:
                continue
            cycles = _number((record.get("metrics") or {}).get("latency_cycles_worst"))
            if record.get("feasible") is not True or cycles is None:
                continue
            observations.append((float(changed[0]["to"]), cycles, record, False))
        observations.sort(key=lambda item: item[0])
        values = [item[0] for item in observations]
        cycles = [item[1] for item in observations]
        expectation = _expectation_for_kind(knob.kind)
        violations = 0
        if expectation == "cycles_nonincreasing_as_value_increases":
            violations = sum(
                1 for previous, current in zip(cycles, cycles[1:])
                if current > previous
            )
        elif expectation == "cycles_nondecreasing_as_value_increases":
            violations = sum(
                1 for previous, current in zip(cycles, cycles[1:])
                if current < previous
            )
        resource_correlations = {}
        for resource in RESOURCE_KEYS:
            pairs = [
                (item[0], _number((item[2].get("metrics") or {}).get(resource)))
                for item in observations
            ]
            valid_pairs = [(value, metric) for value, metric in pairs if metric is not None]
            resource_correlations[resource] = (
                spearman(
                    [value for value, _ in valid_pairs],
                    [metric for _, metric in valid_pairs],
                )
                if len(valid_pairs) >= 2
                else None
            )
        ii_pairs = [
            (
                item[0],
                _number(
                    (item[2].get("metrics") or {}).get(
                        "achieved_pipeline_ii_max"
                    )
                ),
            )
            for item in observations
        ]
        valid_ii_pairs = [
            (value, metric) for value, metric in ii_pairs if metric is not None
        ]
        summaries.append({
            "knob_id": knob.knob_id,
            "kind": knob.kind,
            "name": knob.name,
            "valid_observations": len(observations),
            "tested_values": values,
            "worst_cycles": cycles,
            "expected_direction": expectation,
            "spearman_value_vs_worst_cycles": spearman(values, cycles),
            "monotonicity_violations": violations,
            "spearman_value_vs_resources": resource_correlations,
            "spearman_value_vs_achieved_pipeline_ii": (
                spearman(
                    [value for value, _ in valid_ii_pairs],
                    [metric for _, metric in valid_ii_pairs],
                )
                if len(valid_ii_pairs) >= 2
                else None
            ),
            "observations": [
                {
                    "candidate_id": item[2].get("candidate_id"),
                    "value": item[0],
                    "worst_cycles": item[1],
                    "is_parent": item[3],
                    "achieved_pipeline_ii_max": (
                        item[2].get("metrics") or {}
                    ).get("achieved_pipeline_ii_max"),
                    "resources": {
                        resource: (item[2].get("metrics") or {}).get(resource)
                        for resource in RESOURCE_KEYS
                    },
                }
                for item in observations
            ],
        })
    return summaries


def winner_explanation(parent: dict[str, Any], winner: dict[str, Any]) -> str:
    parent_cycles = _pareto_value(parent, "latency_cycles_worst")
    winner_cycles = _pareto_value(winner, "latency_cycles_worst")
    if winner.get("candidate_id") == parent.get("candidate_id"):
        return (
            "Retained the frozen parent because no feasible tested variant "
            "reduced worst-case CSynth latency."
        )
    improvement = (
        100.0 * (parent_cycles - winner_cycles) / parent_cycles
        if math.isfinite(parent_cycles) and parent_cycles > 0
        else None
    )
    changed = ", ".join(
        f"{item.get('name')}={item.get('to')}"
        for item in winner.get("changed_knobs") or []
    )
    resource_delta = []
    for key in RESOURCE_KEYS:
        parent_value = _number((parent.get("metrics") or {}).get(key))
        winner_value = _number((winner.get("metrics") or {}).get(key))
        if parent_value is not None and winner_value is not None:
            resource_delta.append(f"{key.upper()} {winner_value - parent_value:+.0f}")
    if winner_cycles == parent_cycles:
        latency_reason = (
            f"tied the lowest tested worst-case latency ({winner_cycles:.0f} "
            "cycles) and won the deterministic aggregate-resource tie-break"
        )
    else:
        latency_reason = (
            f"had the lowest tested worst-case latency ({winner_cycles:.0f} cycles"
            + (
                f", {improvement:.2f}% below the parent"
                if improvement is not None
                else ""
            )
            + ")"
        )
    return (
        f"Selected {winner.get('candidate_id')} ({changed}) because it passed "
        f"CSim/CSynth, timing, and resource-fit gates and {latency_reason}. "
        "Resource deltas versus the parent: "
        + (", ".join(resource_delta) if resource_delta else "unavailable")
        + "."
    )
