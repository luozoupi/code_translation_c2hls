"""Post-pass constrained latency optimization with trajectory tracking."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from post_flash_dataflow import extract_kernel_block
from post_flash_mem_parallel import discover_matrix_cells

_LOG = logging.getLogger(__name__)

DEFAULT_ROUNDS = 3
DEFAULT_REPAIR_ROUNDS = 3
DEFAULT_BUDGET_PCT = 100.0
STEP_TAG = "latency_opt"
DATAFLOW_STEP_TAG = "dataflow"

# Canonical post-flash roles plus multistep_<phase> seeds (e.g. multistep_tiling).
SourceRole = str
SOURCE_ROLES: tuple[str, ...] = ("flash_final", "dataflow")
TRAJECTORY_SCHEMA = "post_flash_latency_opt_trajectory_v1"


def is_multistep_source_role(source_role: str) -> bool:
    return str(source_role or "").startswith("multistep_")


def multistep_phase_from_role(source_role: str) -> str:
    raw = str(source_role or "")
    if not raw.startswith("multistep_"):
        return ""
    return raw[len("multistep_") :]


def _truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def latency_opt_enabled() -> bool:
    return _truthy("C2HLS_POST_FLASH_LATENCY_OPT")


def chain_after_flash() -> bool:
    raw = os.getenv("C2HLS_LATENCY_OPT_CHAIN_FLASH", "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    return latency_opt_enabled()


def chain_after_dataflow() -> bool:
    raw = os.getenv("C2HLS_LATENCY_OPT_CHAIN_DATAFLOW", "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    return latency_opt_enabled()


def latency_round_limit() -> int:
    try:
        return max(1, int(os.getenv("C2HLS_LATENCY_OPT_ROUNDS", str(DEFAULT_ROUNDS))))
    except ValueError:
        return DEFAULT_ROUNDS


def repair_round_limit() -> int:
    try:
        return max(1, int(os.getenv("C2HLS_LATENCY_OPT_REPAIR_ROUNDS", str(DEFAULT_REPAIR_ROUNDS))))
    except ValueError:
        return DEFAULT_REPAIR_ROUNDS


def budget_pct() -> float:
    try:
        return float(os.getenv("C2HLS_LATENCY_OPT_BUDGET_PCT", str(DEFAULT_BUDGET_PCT)))
    except ValueError:
        return DEFAULT_BUDGET_PCT


def _device_capacity(part: str) -> dict[str, float]:
    from rubric import _device_limits_for_part

    caps = dict(_device_limits_for_part(part) or {})
    caps.pop("_fallback_reason", None)
    return {k: float(v) for k, v in caps.items() if isinstance(v, (int, float))}


def under_device_budget(
    report: Optional[dict[str, Any]],
    part: str,
    *,
    budget_pct: float = 100.0,
) -> bool:
    if not report:
        return False
    caps = _device_capacity(part)
    limit = budget_pct / 100.0
    for key in ("lut", "dsp", "ff", "bram", "uram"):
        used = report.get(key)
        if used is None:
            used = 0
        try:
            used_f = float(used)
        except (TypeError, ValueError):
            used_f = 0.0
        cap = caps.get(key)
        if not cap or cap <= 0:
            continue
        if used_f / cap > limit + 1e-12:
            return False
    return True


def should_accept(
    candidate: dict[str, Any],
    best: Optional[dict[str, Any]],
    *,
    part: str,
    budget_pct_value: Optional[float] = None,
) -> bool:
    pct = budget_pct() if budget_pct_value is None else budget_pct_value
    report = candidate.get("report") or {}
    if not under_device_budget(report, part, budget_pct=pct):
        return False
    lat = candidate.get("latency_cycles")
    if lat is None:
        return False
    try:
        lat_i = int(lat)
    except (TypeError, ValueError):
        return False
    if best is None:
        return True  # legalization: first under-budget validated design
    try:
        best_lat = int(best["latency_cycles"])
    except (KeyError, TypeError, ValueError):
        return True
    return lat_i < best_lat


def _resource_util_lines(report: dict[str, Any], part: str) -> list[str]:
    caps = _device_capacity(part)
    lines: list[str] = []
    for key in ("lut", "dsp", "ff", "bram", "uram"):
        used = report.get(key)
        if used is None:
            used = 0
        try:
            used_f = float(used)
        except (TypeError, ValueError):
            used_f = 0.0
        cap = caps.get(key)
        if not cap or cap <= 0:
            continue
        pct = used_f / cap * 100.0
        lines.append(f"  {key.upper()}: {int(used_f)}/{int(cap)} ({pct:.1f}%)")
    return lines


def _max_resource_util_pct(report: dict[str, Any], part: str) -> float:
    caps = _device_capacity(part)
    max_pct = 0.0
    for key in ("lut", "dsp", "ff", "bram", "uram"):
        used = report.get(key)
        if used is None:
            used = 0
        try:
            used_f = float(used)
        except (TypeError, ValueError):
            used_f = 0.0
        cap = caps.get(key)
        if not cap or cap <= 0:
            continue
        max_pct = max(max_pct, used_f / cap * 100.0)
    return max_pct


def template_actions_for_report(report: dict[str, Any], part: str) -> list[str]:
    """Deterministic guided actions from bottlenecks and resource pressure."""
    actions: list[str] = []
    feedback = report.get("feedback") or {}
    seen: set[str] = set()

    for b in feedback.get("bottlenecks") or []:
        kind = b.get("kind") or ""
        scope_id = b.get("scope_id") or "(global)"
        evidence = b.get("evidence") or ""
        if kind == "non_pipelined_hot_loop":
            msg = (
                f"Add `#pragma HLS PIPELINE II=1` on loop `{scope_id}` "
                f"({evidence or 'not pipelined'})"
            )
        elif kind == "ii_target_miss":
            msg = (
                f"Reduce II on `{scope_id}`: fix dependences, partition locals, "
                f"or modest unroll ({evidence or 'II target miss'})"
            )
        elif kind == "port_conflict":
            msg = (
                f"Apply `#pragma HLS ARRAY_PARTITION` on local tiles at `{scope_id}` "
                f"plus matching `#pragma HLS UNROLL` factor ({evidence or 'port conflict'})"
            )
        else:
            continue
        if msg not in seen:
            seen.add(msg)
            actions.append(msg)

    caps = _device_capacity(part)
    for key in ("lut", "dsp", "ff", "bram", "uram"):
        used = report.get(key)
        if used is None:
            used = 0
        try:
            used_f = float(used)
        except (TypeError, ValueError):
            used_f = 0.0
        cap = caps.get(key)
        if not cap or cap <= 0:
            continue
        pct = used_f / cap * 100.0
        if pct > 100.0:
            msg = (
                f"{key.upper()} overflow ({pct:.0f}%): reduce utilization "
                f"before adding parallelism"
            )
        elif pct > 80.0:
            msg = (
                f"{key.upper()} pressure ({pct:.0f}%): prefer schedule/II fixes; "
                f"avoid large unroll/partition factors"
            )
        else:
            continue
        if msg not in seen:
            seen.add(msg)
            actions.append(msg)

    return actions


def render_latency_analysis_pack(
    report: dict[str, Any],
    part: str,
    *,
    max_scopes: int = 12,
    max_bottlenecks: int = 6,
    trajectory_summary: str = "",
) -> str:
    """Rich latency-opt analysis pack for LLM plan/modify prompts."""
    if not report:
        return ""

    lines: list[str] = []
    feedback = report.get("feedback") or {}
    summary = feedback.get("summary") or {}

    lat = report.get("latency_cycles")
    lat_s = str(lat) if lat is not None else "?"
    lines.append("=== Design PPA ===")
    lines.append(f"Latency (cycles): {lat_s}")
    interval = report.get("interval")
    if interval is not None:
        lines.append(f"Interval: {interval}")
    lines.append("Resources:")
    lines.extend(_resource_util_lines(report, part))

    scopes = feedback.get("scopes") or []
    if scopes:
        lines.append("")
        lines.append("=== Ranked scopes (hot first) ===")
        summary_bits = (
            f"{summary.get('loop_count', 0)} loops, "
            f"{summary.get('pipelined_loops', 0)} pipelined, "
            f"{summary.get('bottleneck_count', 0)} bottlenecks "
            f"({summary.get('high_severity_bottlenecks', 0)} high-severity)"
        )
        lines.append(f"Summary: {summary_bits}")
        for s in sorted(
            scopes,
            key=lambda x: -(x.get("latency_cycles") or 0),
        )[:max_scopes]:
            sid = s.get("scope_id") or "?"
            kind = s.get("kind") or "?"
            s_lat = s.get("latency_cycles")
            trip = s.get("trip_count")
            ii = s.get("interval")
            pipe_ii = s.get("pipeline_ii")
            pipelined = s.get("pipelined")
            dsp = s.get("dsp")
            lut = s.get("lut")
            parts = [
                f"  {sid} ({kind})",
                f"lat={s_lat}",
                f"trip={trip}",
                f"interval={ii}",
                f"pipelined={pipelined}",
            ]
            if pipe_ii is not None:
                parts.append(f"pipeline_ii={pipe_ii}")
            if dsp is not None:
                parts.append(f"dsp={dsp}")
            if lut is not None:
                parts.append(f"lut={lut}")
            lines.append(" ".join(parts))

    bottlenecks = feedback.get("bottlenecks") or []
    if bottlenecks:
        lines.append("")
        lines.append("=== Bottlenecks ===")
        for b in bottlenecks[:max_bottlenecks]:
            sid = b.get("scope_id") or "(global)"
            sev = b.get("severity") or "?"
            kind = b.get("kind") or "?"
            evidence = b.get("evidence") or ""
            lines.append(f"  - [{sev}] {kind}: {sid} :: {evidence}")

    actions = template_actions_for_report(report, part)
    lines.append("")
    lines.append("=== Guided actions ===")
    if actions:
        for a in actions:
            lines.append(f"  - {a}")
    else:
        lines.append("  - No specific guided actions; review hot scopes for pipeline/II opportunities.")

    pct = budget_pct()
    max_util = _max_resource_util_pct(report, part)
    lines.append("")
    lines.append("=== Budget ===")
    lines.append(
        f"Device budget: keep all resources ≤{pct:.0f}% of {part} capacity "
        f"(current max util {max_util:.1f}%)."
    )
    if max_util > 100.0:
        lines.append("  OVER BUDGET: candidate must reduce resource usage before acceptance.")
    elif max_util > 80.0:
        lines.append("  High utilization (>80%): prefer latency/II fixes over large unroll/partition.")

    if trajectory_summary:
        lines.append("")
        lines.append("=== Trajectory ===")
        lines.append(trajectory_summary.strip())

    return "\n".join(lines)


def render_budget_block(
    part: str,
    report: Optional[dict[str, Any]] = None,
    *,
    budget_pct_value: Optional[float] = None,
) -> str:
    """Short budget reminder for plan/modify/repair prompts."""
    pct = budget_pct() if budget_pct_value is None else budget_pct_value
    lines = [f"Keep all resources ≤{pct:.0f}% of {part} capacity."]
    if report:
        max_util = _max_resource_util_pct(report, part)
        lines.append(f"Current max utilization: {max_util:.1f}%.")
        if max_util > 100.0:
            lines.append("OVER BUDGET: candidate must reduce resource usage.")
        elif max_util > 80.0:
            lines.append(
                "High utilization (>80%): prefer schedule/II fixes over large unroll/partition."
            )
    return "\n".join(lines)


_PLAN_SYSTEM = """You are an expert Xilinx Vitis HLS **performance analyst**.

Given a latency analysis pack and the current kernel, produce a **concise structured plan only** — do NOT output full kernel source.

## Output format (structured text)
- **targets:** list of `scope_id`s to change (max ~3–5), cited from the analysis pack
- **actions:** pragma/transform per target, tied to guided template actions
- **avoid:** scopes/resources not to worsen
- **risk:** qualitative resource/latency trade-off notes (e.g. DSP growth)

## Rules
- Every target must cite a `scope_id` from the ranked scopes / bottlenecks table.
- Prefer pipeline/II fixes before large unroll/partition when utilization is high.
- Respect the device budget constraint.
- No full kernel rewrites in the plan — only actionable edits.
"""

_PLAN_USER = """## Latency analysis pack
{analysis_pack}

## Current kernel (reference only — do not rewrite here)
```cpp
{kernel_code}
```

## Optimization goal
Reduce **latency_cycles** below **{best_latency}** while staying within device budget.

## Budget constraint
{budget_block}

## Prior trajectory (if any)
{trajectory}

Produce a structured plan with **targets**, **actions**, **avoid**, and **risk** sections. Cite specific `scope_id`s from the analysis.
"""

_MODIFY_SYSTEM = """You are an expert Xilinx Vitis HLS **code editor**.

Apply the given latency optimization plan to the kernel exactly.

## Rules
- Preserve the exact top-level `extern "C"` signature, parameter list, array shapes, and all existing `#pragma HLS INTERFACE` lines.
- Follow the plan — do not re-diagnose from scratch or make unrelated changes.
- Label loops you touch with descriptive names.
- Stay within the device budget; avoid large unroll/partition when the plan warns of resource pressure.

## Output
Return **one** fenced block only:
```kernel
... full kernel source ...
```
"""

_MODIFY_USER = """Apply this latency optimization plan to the kernel.

## Plan
{plan_text}

## Current kernel (preserve top signature + INTERFACE pragmas)
```cpp
{kernel_code}
```

## Budget reminder
{budget_block}

## Vitis HLS 2023.2 Pragmas — Curated Usage Guide (optional reference)
{pragma_guide}

Return a single ```kernel``` block implementing the plan.
"""

_REPAIR_USER = """Repair the latency-optimized kernel after a validation failure.

## Failure
Stage: {stage}
```
{error}
```

## Last optimization plan
{plan_text}

## Current kernel
```cpp
{kernel_code}
```

## Analysis pack snapshot
{analysis_pack}

Fix the failure while staying as close as possible to the plan. Never accept over-budget resource growth. Return a single ```kernel``` block.
"""


def _load_pragma_guide_truncated(*, max_chars: int = 4000) -> str:
    from post_flash_pragma_opt import load_pragma_guide

    return load_pragma_guide(max_chars=max_chars)


def plan_mentions_scope(plan_text: str, scope_ids: list[str] | set[str]) -> bool:
    """Soft check: plan cites at least one known scope_id from the analysis table."""
    if not plan_text or not scope_ids:
        return False
    text = plan_text.lower()
    for sid in scope_ids:
        if sid and str(sid).lower() in text:
            return True
    return False


def artifact_paths(cell_dir: Path, bench: str, source_role: SourceRole) -> dict[str, Path]:
    if source_role == "dataflow":
        base = f"{bench}_{DATAFLOW_STEP_TAG}_{STEP_TAG}"
    elif is_multistep_source_role(source_role):
        phase = multistep_phase_from_role(source_role) or "step"
        base = f"{bench}_multistep_{phase}_{STEP_TAG}"
    else:
        base = f"{bench}_{STEP_TAG}"
    return {
        "kernel": cell_dir / f"{base}.cpp",
        "report": cell_dir / f"{base}_report.json",
        "result": cell_dir / f"{base}_result.json",
        "history": cell_dir / f"{base}_history.json",
        "trajectory": cell_dir / f"{base}_trajectory.json",
        "manifest": cell_dir / f"{base}_manifest.json",
    }


def new_trajectory(
    *,
    bench: str,
    source_role: SourceRole,
    part: str,
    budget_pct: float,
    N: int,
    R: int,
    seed: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": TRAJECTORY_SCHEMA,
        "benchmark": bench,
        "source_role": source_role,
        "part": part,
        "budget_pct": budget_pct,
        "N": N,
        "R": R,
        "seed": seed,
        "best_so_far": None,
        "rounds": [],
        "final": None,
    }


def append_round_event(traj: dict[str, Any], event: dict[str, Any]) -> None:
    traj.setdefault("rounds", []).append(dict(event))


def set_best_so_far(
    traj: dict[str, Any],
    *,
    round_idx: int,
    latency_cycles: int,
    resources: dict[str, Any],
    kernel_sha256: str,
) -> None:
    traj["best_so_far"] = {
        "round": round_idx,
        "latency_cycles": latency_cycles,
        "resources": resources,
        "kernel_sha256": kernel_sha256,
    }


def finalize_trajectory(
    traj: dict[str, Any],
    *,
    success: bool,
    final_latency: int,
    seed_latency: int,
) -> None:
    speedup = 1.0
    if final_latency and seed_latency:
        speedup = float(seed_latency) / float(final_latency)
    under_budget = success
    for event in reversed(traj.get("rounds") or []):
        if event.get("decision") == "accept" and event.get("under_budget") is not None:
            under_budget = bool(event["under_budget"])
            break
    traj["final"] = {
        "latency_cycles": final_latency,
        "speedup_vs_seed": speedup,
        "under_budget": under_budget,
        "success": success,
    }


def prompt_text_for_docs() -> dict[str, str]:
    """Return plan/modify/repair prompts with sample placeholders for docs/smoke."""
    part = "xcu280-fsvh2892-2L-e"
    sample_report = {
        "latency_cycles": 10000,
        "lut": 100,
        "dsp": 8,
        "ff": 200,
        "bram": 0,
        "uram": 0,
        "feedback": {
            "summary": {
                "loop_count": 2,
                "pipelined_loops": 1,
                "bottleneck_count": 1,
                "high_severity_bottlenecks": 1,
            },
            "scopes": [
                {
                    "scope_id": "k/outer",
                    "kind": "loop",
                    "latency_cycles": 9000,
                    "trip_count": 64,
                    "interval": 64,
                    "pipelined": "no",
                },
                {
                    "scope_id": "k/inner",
                    "kind": "loop",
                    "latency_cycles": 128,
                    "trip_count": 64,
                    "interval": 2,
                    "pipelined": "yes",
                    "pipeline_ii": 2,
                },
            ],
            "bottlenecks": [
                {
                    "kind": "ii_target_miss",
                    "severity": "high",
                    "scope_id": "k/inner",
                    "evidence": "II=2 target=1",
                },
            ],
        },
    }
    analysis_pack = render_latency_analysis_pack(
        sample_report,
        part,
        trajectory_summary="Round 0: seed latency 10000 cycles (accepted).",
    )
    budget_block = render_budget_block(part, sample_report)
    sample_kernel = "extern \"C\" void kernel(/* ... */) { /* sample */ }"
    sample_plan = (
        "**targets:** k/inner, k/outer\n"
        "**actions:** PIPELINE II=1 on k/outer; reduce II on k/inner\n"
        "**avoid:** large unroll on k/outer\n"
        "**risk:** modest DSP growth from inner unroll"
    )
    guide = _load_pragma_guide_truncated()
    return {
        "plan_system": _PLAN_SYSTEM,
        "plan_user": _PLAN_USER.format(
            analysis_pack=analysis_pack,
            kernel_code=sample_kernel,
            best_latency="10000",
            budget_block=budget_block,
            trajectory="Round 0: seed accepted at 10000 cycles.",
        ),
        "modify_system": _MODIFY_SYSTEM,
        "modify_user": _MODIFY_USER.format(
            plan_text=sample_plan,
            kernel_code=sample_kernel,
            budget_block=budget_block,
            pragma_guide=guide,
        ),
        "repair_user": _REPAIR_USER.format(
            stage="csynth",
            error="(validation error inserted at runtime)",
            plan_text=sample_plan,
            kernel_code=sample_kernel,
            analysis_pack=analysis_pack,
        ),
    }


_RESOURCE_KEYS = ("lut", "dsp", "ff", "bram", "uram")


def configure_post_flash_env() -> None:
    os.environ.setdefault("C2HLS_RUN_COSIM", "0")
    os.environ.setdefault("C2HLS_COSIM_REQUIRED", "0")
    os.environ.setdefault("C2HLS_REFERENCE_COSIM", "0")


def load_existing_result(result_path: Path) -> Optional[dict[str, Any]]:
    if not result_path.is_file():
        return None
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and data.get("success") is True:
        return data
    return None


def _resources_snapshot(report: Optional[dict[str, Any]]) -> dict[str, Any]:
    report = report or {}
    return {key: report.get(key) for key in _RESOURCE_KEYS}


@dataclass
class LatencyOptOutcome:
    bench: str
    source_role: SourceRole
    success: bool
    cell_dir: str
    error: str = ""
    result: Optional[dict[str, Any]] = None


def resolve_latency_source_kernel(
    cell_dir: Path,
    bench: str,
    source_role: SourceRole,
) -> tuple[Optional[Path], str, Optional[dict[str, Any]]]:
    """Return (kernel_path, role_label, prior_synth_report).

    Prefers a successful pragma-opt artifact (kept as a separate post-pass)
    for the matching source role; falls back to the flash-selected / final
    kernel (flash_final, using the base resolver with
    `include_post_passes=False`) or the DATAFLOW kernel (dataflow). This
    deliberately does NOT prefer a prior latency-opt output for itself, to
    avoid latency_opt recursively seeding from its own previous output.
    """
    import post_flash_pragma_opt as _ppo

    pragma_paths = _ppo.artifact_paths(cell_dir, bench, source_role)
    pragma_result = _ppo.load_existing_result(pragma_paths["result"])
    if pragma_result is not None and pragma_paths["kernel"].is_file():
        report = pragma_result.get("synth_report")
        if not isinstance(report, dict):
            report = None
        role = "dataflow_pragma_opt" if source_role == "dataflow" else "pragma_opt"
        return pragma_paths["kernel"], role, report

    if source_role == "dataflow":
        dataflow_kernel = cell_dir / f"{bench}_{DATAFLOW_STEP_TAG}.cpp"
        dataflow_result_path = cell_dir / f"{bench}_{DATAFLOW_STEP_TAG}_result.json"
        if not dataflow_kernel.is_file():
            return None, "", None
        if not dataflow_result_path.is_file():
            return None, "", None
        try:
            data = json.loads(dataflow_result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None, "", None
        if not (isinstance(data, dict) and data.get("success")):
            return None, "", None
        report = data.get("synth_report")
        if not isinstance(report, dict):
            report = None
        return dataflow_kernel, DATAFLOW_STEP_TAG, report

    if is_multistep_source_role(source_role):
        phase = multistep_phase_from_role(source_role) or "step"
        kernel = cell_dir / f"{bench}_multistep_{phase}.cpp"
        report_path = cell_dir / f"{bench}_multistep_{phase}_report.json"
        if not kernel.is_file():
            return None, "", None
        report = None
        if report_path.is_file():
            try:
                loaded = json.loads(report_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    report = loaded
            except json.JSONDecodeError:
                report = None
        return kernel, f"multistep_{phase}", report

    # flash_final: base flash kernel only (selected -> final -> legacy
    # final), deliberately excluding post-pass outputs (`include_post_passes
    # =False`) so latency_opt never seeds itself from its own prior output.
    from post_flash_mem_parallel import _resolve_flash_base_kernel

    path, role = _resolve_flash_base_kernel(cell_dir, bench)
    if path is None:
        return None, "", None

    report = None
    for candidate in (
        cell_dir / f"{bench}_selected_report.json",
        cell_dir / "steps" / "0_flash_report.json",
    ):
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    report = data
                    break
            except json.JSONDecodeError:
                continue
    return path, role, report


def _write_cell_manifest(
    cell_dir: Path,
    bench: str,
    source_role: SourceRole,
    kernel_path: Path,
    result_payload: dict[str, Any],
) -> None:
    paths = artifact_paths(cell_dir, bench, source_role)
    paths["manifest"].write_text(json.dumps({
        "schema": "post_flash_latency_opt_manifest_v1",
        "benchmark": bench,
        "source_role": source_role,
        "source_kernel": str(kernel_path),
        "success": result_payload.get("success"),
        "result": paths["result"].name,
    }, indent=2) + "\n", encoding="utf-8")


def _cosim_env_enabled() -> bool:
    return os.getenv("C2HLS_RUN_COSIM", "0").strip().lower() in {"1", "true", "yes", "on"}


def _cosim_required() -> bool:
    return os.getenv("C2HLS_COSIM_REQUIRED", "0").strip().lower() in {"1", "true", "yes", "on"}


def promote_latency_opt_as_selected(
    *,
    cell_dir: Path,
    bench: str,
    source_role: SourceRole,
    code: str,
    report: dict[str, Any],
    result_payload: dict[str, Any],
) -> dict[str, Any]:
    """Promote a successful latency-opt kernel to canonical selected pointers.

    - flash_final: overwrite ``{bench}_selected.cpp`` / ``_selected_report.json``
      and update ``{bench}_flow_manifest.json``.
    - dataflow: update ``{bench}_dataflow_result.json`` so top-level latency and
      selected_* point at the latency-opt best; preserve prior a0 under
      ``pre_latency_opt``.
    """
    from flash_flow_artifacts import sha256_text

    paths = artifact_paths(cell_dir, bench, source_role)
    lat = result_payload.get("latency_cycles")
    promotion: dict[str, Any] = {
        "source_role": source_role,
        "latency_cycles": lat,
        "kernel": paths["kernel"].name,
    }

    if source_role == "flash_final":
        selected_cpp = cell_dir / f"{bench}_selected.cpp"
        selected_report = cell_dir / f"{bench}_selected_report.json"
        selected_cpp.write_text(code, encoding="utf-8")
        selected_report.write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )
        promotion["selected_kernel"] = selected_cpp.name
        promotion["selected_stage"] = "latency_opt"
        promotion["selected_report"] = selected_report.name

        manifest_path = cell_dir / f"{bench}_flow_manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    manifest = loaded
            except json.JSONDecodeError:
                manifest = {}
        lat_map = manifest.get("latency_cycles")
        if not isinstance(lat_map, dict):
            lat_map = {}
        lat_map = dict(lat_map)
        if lat is not None:
            lat_map["selected"] = lat
            lat_map["latency_opt"] = lat
        manifest["latency_cycles"] = lat_map
        manifest["selected_from"] = "latency_opt"
        manifest["selected_sha256"] = sha256_text(code)
        if "files" not in manifest or not isinstance(manifest["files"], dict):
            manifest["files"] = {}
        files = dict(manifest["files"])
        files["selected"] = selected_cpp.name
        files["selected_report"] = selected_report.name
        files["latency_opt"] = paths["kernel"].name
        files["latency_opt_report"] = paths["report"].name
        manifest["files"] = files
        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
        )
        promotion["flow_manifest"] = manifest_path.name

    elif source_role == "dataflow":
        df_result_path = cell_dir / f"{bench}_{DATAFLOW_STEP_TAG}_result.json"
        data: dict[str, Any] = {}
        if df_result_path.is_file():
            try:
                loaded = json.loads(df_result_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
            except json.JSONDecodeError:
                data = {}
        if "pre_latency_opt" not in data:
            data["pre_latency_opt"] = {
                "latency_cycles": data.get("latency_cycles"),
                "synth_report": data.get("synth_report"),
                "selected_kernel": data.get("selected_kernel"),
                "selected_stage": data.get("selected_stage"),
            }
        data["latency_cycles"] = lat
        data["selected_kernel"] = paths["kernel"].name
        data["selected_stage"] = "dataflow_latency_opt"
        data["selected_report"] = paths["report"].name
        if report:
            data["latency_opt_synth_report"] = report
        chain = data.get("latency_opt_chain")
        if isinstance(chain, dict):
            chain = dict(chain)
            chain["promoted"] = True
            chain["selected_kernel"] = paths["kernel"].name
            data["latency_opt_chain"] = chain
        df_result_path.write_text(
            json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8"
        )
        promotion["selected_kernel"] = paths["kernel"].name
        promotion["selected_stage"] = "dataflow_latency_opt"
        promotion["dataflow_result"] = df_result_path.name

    elif is_multistep_source_role(source_role):
        # Mid-step multistep: keep lat-opt artifacts only; final selected
        # promotion happens after all steps complete.
        promotion["selected_stage"] = f"{source_role}_{STEP_TAG}"
        promotion["deferred_selected"] = True

    return promotion


def maybe_cosim_promoted_best(
    *,
    code: str,
    header_code: str,
    header_name: str,
    top_function: str,
    part: str,
    clock_ns: float,
    extra_files: list,
    testbench_code: str,
    cosim_depths: Optional[dict[str, Any]],
    bench: str,
    source_role: SourceRole,
) -> Optional[dict[str, Any]]:
    """Run a single cosim on the promoted best when C2HLS_RUN_COSIM is on.

    Does not un-promote a csynth-legal best unless C2HLS_COSIM_REQUIRED is set
    (caller decides); this helper only returns the cosim summary.
    """
    if not _cosim_env_enabled() or not testbench_code:
        return None
    from c2hls import _run_synth_csim_cosim
    from c2hls_temp import join_temp_tag

    outcome = _run_synth_csim_cosim(
        code,
        header_code=header_code,
        header_name=header_name,
        top_function=top_function,
        part=part,
        clock_ns=clock_ns,
        extra_files=extra_files,
        testbench_code=testbench_code,
        run_csim_check=False,
        run_cosim_check=True,
        cosim_depths=cosim_depths or {},
        cosim_requires_csim_pass=False,
        log_prefix=f"[{STEP_TAG}:promote_cosim]",
        temp_tag=join_temp_tag(bench, STEP_TAG, source_role, "promote_cosim"),
    )
    return outcome.get("cosim")


def run_latency_opt_for_cell(
    *,
    bench: str,
    bench_dir: Path,
    cell_dir: Path,
    orchestrator: Any,
    source_role: SourceRole = "flash_final",
    skip_existing: bool = True,
) -> LatencyOptOutcome:
    """Round loop: N plan->modify rounds, up to R repairs per validation
    failure, keeping the best under-budget, lower-latency design seen so far.
    """
    from c2hls import _load_benchmark_inputs, _run_synth_csim_cosim, compile_check_cpp
    from c2hls_temp import join_temp_tag
    from flash_flow_artifacts import sha256_text

    paths = artifact_paths(cell_dir, bench, source_role)
    if skip_existing:
        existing = load_existing_result(paths["result"])
        if existing is not None:
            # Re-apply promotion so older cells that succeeded before pointer
            # updates still expose latency-opt as the canonical selected.
            if paths["kernel"].is_file():
                report: dict[str, Any] = {}
                if paths["report"].is_file():
                    try:
                        loaded = json.loads(paths["report"].read_text(encoding="utf-8"))
                        if isinstance(loaded, dict):
                            report = loaded
                    except json.JSONDecodeError:
                        report = {}
                try:
                    code = paths["kernel"].read_text(encoding="utf-8")
                    promotion = promote_latency_opt_as_selected(
                        cell_dir=cell_dir,
                        bench=bench,
                        source_role=source_role,
                        code=code,
                        report=report,
                        result_payload=existing,
                    )
                    existing = dict(existing)
                    existing["promotion"] = promotion
                    paths["result"].write_text(
                        json.dumps(existing, indent=2, default=str) + "\n", encoding="utf-8"
                    )
                except OSError as exc:
                    _LOG.warning("[latency_opt] promote on skip_existing failed: %s", exc)
            return LatencyOptOutcome(bench, source_role, True, str(cell_dir), result=existing)

    kernel_path, kernel_role, _prior_report = resolve_latency_source_kernel(cell_dir, bench, source_role)
    if kernel_path is None:
        if source_role == "dataflow":
            msg = "no passing dataflow kernel"
        elif is_multistep_source_role(source_role):
            phase = multistep_phase_from_role(source_role) or "step"
            msg = f"no multistep seed kernel ({bench}_multistep_{phase}.cpp)"
        else:
            msg = "no selected/final kernel cpp"
        return LatencyOptOutcome(bench, source_role, False, str(cell_dir), msg)

    inputs = _load_benchmark_inputs(str(bench_dir))
    seed_code = kernel_path.read_text(encoding="utf-8")
    header_code = inputs.get("header_code", "")
    header_name = inputs.get("header_name") or "kernel.h"
    meta = inputs["meta"]
    top_function = (
        meta.get("translated_hls_top")
        or meta.get("hls_top")
        or meta.get("kernel_top")
        or "workload"
    )
    testbench_code = inputs.get("testbench_code", "")
    extra_files = inputs.get("extra_files", [])
    part = meta.get("part", orchestrator.part)
    clock_ns = meta.get("clock_ns", orchestrator.clock_ns)

    configure_post_flash_env()

    N = latency_round_limit()
    R = repair_round_limit()
    pct = budget_pct()

    history: list[dict[str, str]] = []
    tag_base = f"{STEP_TAG}_{source_role}"

    def _validate(code: str, tag: str) -> tuple[bool, str, dict[str, Any]]:
        ok, err = compile_check_cpp(code, header_code, header_name, extra_files=extra_files)
        if not ok:
            return False, err, {}
        if not testbench_code:
            return False, "benchmark has no testbench for csim", {}
        outcome = _run_synth_csim_cosim(
            code,
            header_code=header_code,
            header_name=header_name,
            top_function=top_function,
            part=part,
            clock_ns=clock_ns,
            extra_files=extra_files,
            testbench_code=testbench_code,
            run_csim_check=True,
            run_cosim_check=False,
            log_prefix=f"[{STEP_TAG}]",
            temp_tag=tag,
        )
        synth = outcome.get("synth") or {}
        csim_summary = outcome.get("csim")
        if not synth.get("success"):
            return False, synth.get("error") or "csynth failed", {}
        csim_pass = (
            csim_summary is None
            or csim_summary.get("passed")
            or csim_summary.get("status") == "passed"
        )
        if not csim_pass:
            return False, (csim_summary or {}).get("error") or "csim failed", synth.get("report") or {}
        return True, "", synth.get("report") or {}

    # --- Seed validation (re-validate once for honesty, even if a prior
    # report exists) ---
    seed_repair_log: list[dict[str, Any]] = []
    seed_ok, seed_err, seed_report = _validate(seed_code, join_temp_tag(bench, tag_base, "seed"))
    if not seed_ok:
        current_code = seed_code
        for r_idx in range(1, R + 1):
            repair_user = _REPAIR_USER.format(
                stage="seed_validate",
                error=seed_err[:8000],
                plan_text="(seed repair — restore validity)",
                kernel_code=current_code[:120000],
                analysis_pack="(seed repair — no analysis pack yet)",
            )
            repair_messages = [
                {"role": "system", "content": _MODIFY_SYSTEM},
                {"role": "user", "content": repair_user},
            ]
            reply = orchestrator._call_llm(repair_messages)
            history.extend([
                {"role": "user", "content": repair_user},
                {"role": "assistant", "content": reply},
            ])
            extracted = extract_kernel_block(reply)
            if extracted:
                current_code = extracted
            seed_ok, seed_err, seed_report = _validate(
                current_code, join_temp_tag(bench, tag_base, f"seed_r{r_idx}")
            )
            seed_repair_log.append({
                "round": 0,
                "phase": "repair",
                "repair_index": r_idx,
                "plan_summary": "(seed repair — restore validity)",
                "validated": seed_ok,
                "latency_cycles": (seed_report or {}).get("latency_cycles") if seed_ok else None,
                "resources": _resources_snapshot(seed_report) if seed_ok else {},
                "under_budget": None,
                "decision": "seed_repair_ok" if seed_ok else "seed_repair_retry",
                "reason": "" if seed_ok else (seed_err or "")[:500],
            })
            if seed_ok:
                break
        seed_code = current_code

    if not seed_ok:
        traj = new_trajectory(
            bench=bench,
            source_role=source_role,
            part=part,
            budget_pct=pct,
            N=N,
            R=R,
            seed={
                "latency_cycles": None,
                "resources": {},
                "under_budget": False,
                "validated": False,
                "error": seed_err[:2000],
            },
        )
        for ev in seed_repair_log:
            append_round_event(traj, ev)
        finalize_trajectory(traj, success=False, final_latency=0, seed_latency=0)
        result_payload: dict[str, Any] = {
            "schema": "post_flash_latency_opt_v1",
            "benchmark": bench,
            "source_role": source_role,
            "success": False,
            "error": f"seed kernel failed validation: {seed_err}"[:4000],
            "source_kernel": str(kernel_path.name),
            "source_kernel_role": kernel_role,
            "N": N,
            "R": R,
            "budget_pct": pct,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        paths["result"].write_text(json.dumps(result_payload, indent=2, default=str) + "\n", encoding="utf-8")
        paths["trajectory"].write_text(json.dumps(traj, indent=2, default=str) + "\n", encoding="utf-8")
        paths["history"].write_text(json.dumps({
            "model": orchestrator.gpt_model,
            "source_role": source_role,
            "messages": history,
        }, indent=2), encoding="utf-8")
        _write_cell_manifest(cell_dir, bench, source_role, kernel_path, result_payload)
        return LatencyOptOutcome(bench, source_role, False, str(cell_dir), seed_err, result_payload)

    seed_latency = seed_report.get("latency_cycles")
    seed_under_budget = under_device_budget(seed_report, part, budget_pct=pct)
    seed_snapshot = {
        "latency_cycles": seed_latency,
        "resources": _resources_snapshot(seed_report),
        "under_budget": seed_under_budget,
        "validated": True,
        "kernel_sha256": sha256_text(seed_code),
    }

    best_so_far: Optional[dict[str, Any]] = None
    if seed_under_budget and seed_latency is not None:
        best_so_far = {"latency_cycles": seed_latency, "report": seed_report, "code": seed_code}

    working_code = seed_code
    working_report = seed_report

    traj = new_trajectory(
        bench=bench,
        source_role=source_role,
        part=part,
        budget_pct=pct,
        N=N,
        R=R,
        seed=seed_snapshot,
    )
    for ev in seed_repair_log:
        append_round_event(traj, ev)
    if best_so_far is not None:
        set_best_so_far(
            traj,
            round_idx=0,
            latency_cycles=seed_latency,
            resources=seed_snapshot["resources"],
            kernel_sha256=seed_snapshot["kernel_sha256"],
        )

    trajectory_summary_lines = [
        f"Round 0: seed latency {seed_latency} cycles "
        f"({'accepted' if best_so_far is not None else 'over budget — not legal'})."
    ]

    for round_idx in range(1, N + 1):
        base_report = working_report or (best_so_far or {}).get("report") or seed_report
        analysis_pack = render_latency_analysis_pack(
            base_report,
            part,
            trajectory_summary="\n".join(trajectory_summary_lines),
        )
        budget_block = render_budget_block(part, base_report, budget_pct_value=pct)
        best_latency_val = (best_so_far or {}).get("latency_cycles")
        if best_latency_val is None:
            best_latency_val = seed_latency if seed_latency is not None else "unknown"

        plan_user = _PLAN_USER.format(
            analysis_pack=analysis_pack,
            kernel_code=working_code[:120000],
            best_latency=str(best_latency_val),
            budget_block=budget_block,
            trajectory="\n".join(trajectory_summary_lines) or "(none)",
        )
        plan_messages = [
            {"role": "system", "content": _PLAN_SYSTEM},
            {"role": "user", "content": plan_user},
        ]
        plan_reply = orchestrator._call_llm(plan_messages)
        history.extend([
            {"role": "system", "content": _PLAN_SYSTEM},
            {"role": "user", "content": plan_user},
            {"role": "assistant", "content": plan_reply},
        ])
        plan_text = (plan_reply or "").strip()

        append_round_event(traj, {
            "round": round_idx,
            "phase": "plan",
            "repair_index": None,
            "plan_summary": plan_text[:2000],
            "validated": None,
            "latency_cycles": None,
            "resources": {},
            "under_budget": None,
            "decision": None,
            "reason": None,
        })

        pragma_guide = _load_pragma_guide_truncated()
        modify_user = _MODIFY_USER.format(
            plan_text=plan_text,
            kernel_code=working_code[:120000],
            budget_block=budget_block,
            pragma_guide=pragma_guide,
        )
        modify_messages = [
            {"role": "system", "content": _MODIFY_SYSTEM},
            {"role": "user", "content": modify_user},
        ]
        modify_reply = orchestrator._call_llm(modify_messages)
        history.extend([
            {"role": "system", "content": _MODIFY_SYSTEM},
            {"role": "user", "content": modify_user},
            {"role": "assistant", "content": modify_reply},
        ])
        candidate_code = extract_kernel_block(modify_reply)

        cand_ok = False
        cand_err = ""
        cand_report: dict[str, Any] = {}
        # Initial modify validation, then up to R repair+validate cycles.
        if not candidate_code:
            cand_ok, cand_err, cand_report = False, "LLM response missing ```kernel``` fenced block", {}
        else:
            tag = join_temp_tag(bench, tag_base, f"r{round_idx}a0")
            cand_ok, cand_err, cand_report = _validate(candidate_code, tag)

        for repair_index in range(1, R + 1):
            if cand_ok:
                break
            repair_user = _REPAIR_USER.format(
                stage="csynth",
                error=cand_err[:8000],
                plan_text=plan_text,
                kernel_code=(candidate_code or working_code)[:120000],
                analysis_pack=analysis_pack,
            )
            repair_messages = [
                {"role": "system", "content": _MODIFY_SYSTEM},
                {"role": "user", "content": repair_user},
            ]
            repair_reply = orchestrator._call_llm(repair_messages)
            history.extend([
                {"role": "user", "content": repair_user},
                {"role": "assistant", "content": repair_reply},
            ])
            extracted = extract_kernel_block(repair_reply)
            if extracted:
                candidate_code = extracted
            if not candidate_code:
                cand_ok, cand_err, cand_report = False, "LLM repair missing ```kernel``` fenced block", {}
            else:
                tag = join_temp_tag(bench, tag_base, f"r{round_idx}a{repair_index}")
                cand_ok, cand_err, cand_report = _validate(candidate_code, tag)
            append_round_event(traj, {
                "round": round_idx,
                "phase": "repair",
                "repair_index": repair_index,
                "plan_summary": plan_text[:2000],
                "validated": cand_ok,
                "latency_cycles": (cand_report or {}).get("latency_cycles") if cand_ok else None,
                "resources": _resources_snapshot(cand_report) if cand_ok else {},
                "under_budget": (
                    under_device_budget(cand_report, part, budget_pct=pct) if cand_ok else None
                ),
                "decision": "repair_ok" if cand_ok else "repair_retry",
                "reason": "" if cand_ok else (cand_err or "")[:500],
            })

        if not cand_ok:
            append_round_event(traj, {
                "round": round_idx,
                "phase": "optimize",
                "repair_index": None,
                "plan_summary": plan_text[:2000],
                "validated": False,
                "latency_cycles": None,
                "resources": {},
                "under_budget": None,
                "decision": "reject_invalid",
                "reason": cand_err[:500],
            })
            trajectory_summary_lines.append(
                f"Round {round_idx}: candidate invalid ({(cand_err or '')[:120]})."
            )
            continue

        cand_latency = cand_report.get("latency_cycles")
        cand_resources = _resources_snapshot(cand_report)
        cand_under_budget = under_device_budget(cand_report, part, budget_pct=pct)
        candidate = {"latency_cycles": cand_latency, "report": cand_report}

        if should_accept(candidate, best_so_far, part=part, budget_pct_value=pct):
            best_so_far = {"latency_cycles": cand_latency, "report": cand_report, "code": candidate_code}
            working_code = candidate_code
            working_report = cand_report
            set_best_so_far(
                traj,
                round_idx=round_idx,
                latency_cycles=cand_latency,
                resources=cand_resources,
                kernel_sha256=sha256_text(candidate_code),
            )
            decision, reason = "accept", "lower latency under budget"
            trajectory_summary_lines.append(f"Round {round_idx}: accepted latency {cand_latency} cycles.")
        elif not cand_under_budget:
            decision, reason = "reject_budget", "candidate exceeds device budget"
            trajectory_summary_lines.append(f"Round {round_idx}: rejected (over budget), latency {cand_latency}.")
        else:
            decision, reason = "reject_latency", "candidate does not improve latency"
            trajectory_summary_lines.append(
                f"Round {round_idx}: rejected (no latency improvement), latency {cand_latency}."
            )

        append_round_event(traj, {
            "round": round_idx,
            "phase": "optimize",
            "repair_index": None,
            "plan_summary": plan_text[:2000],
            "validated": True,
            "latency_cycles": cand_latency,
            "resources": cand_resources,
            "under_budget": cand_under_budget,
            "decision": decision,
            "reason": reason,
        })

    success = best_so_far is not None
    final_code = (best_so_far or {}).get("code") or seed_code
    final_report = (best_so_far or {}).get("report") or {}
    final_latency = (best_so_far or {}).get("latency_cycles")

    finalize_trajectory(
        traj,
        success=success,
        final_latency=final_latency or 0,
        seed_latency=seed_latency or 0,
    )

    result_payload = {
        "schema": "post_flash_latency_opt_v1",
        "benchmark": bench,
        "source_role": source_role,
        "success": success,
        "error": "" if success else "no legal (under-budget) candidate found",
        "source_kernel": str(kernel_path.name),
        "source_kernel_role": kernel_role,
        "N": N,
        "R": R,
        "budget_pct": pct,
        "seed_latency_cycles": seed_latency,
        "latency_cycles": final_latency,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }

    artifacts: dict[str, str] = {}
    if final_code:
        paths["kernel"].write_text(final_code, encoding="utf-8")
        artifacts["kernel"] = paths["kernel"].name
        result_payload["kernel_sha256"] = sha256_text(final_code)
    if final_report:
        paths["report"].write_text(json.dumps(final_report, indent=2, default=str) + "\n", encoding="utf-8")
        artifacts["report"] = paths["report"].name
    if artifacts:
        result_payload["artifacts"] = artifacts

    if success and final_code:
        promotion = promote_latency_opt_as_selected(
            cell_dir=cell_dir,
            bench=bench,
            source_role=source_role,
            code=final_code,
            report=final_report if isinstance(final_report, dict) else {},
            result_payload=result_payload,
        )
        result_payload["promotion"] = promotion

        cosim_summary = maybe_cosim_promoted_best(
            code=final_code,
            header_code=header_code,
            header_name=header_name,
            top_function=top_function,
            part=part,
            clock_ns=clock_ns,
            extra_files=extra_files,
            testbench_code=testbench_code,
            cosim_depths=meta.get("cosim_depths") or {},
            bench=bench,
            source_role=source_role,
        )
        if cosim_summary is not None:
            result_payload["cosim"] = cosim_summary
            cosim_pass = bool(
                cosim_summary.get("passed") or cosim_summary.get("status") == "passed"
            )
            if not cosim_pass and _cosim_required():
                result_payload["success"] = False
                result_payload["error"] = (
                    cosim_summary.get("error") or "cosim failed (C2HLS_COSIM_REQUIRED)"
                )
                success = False
            # Otherwise keep promotion: csynth-legal best stays selected.

    paths["result"].write_text(json.dumps(result_payload, indent=2, default=str) + "\n", encoding="utf-8")
    paths["trajectory"].write_text(json.dumps(traj, indent=2, default=str) + "\n", encoding="utf-8")
    paths["history"].write_text(json.dumps({
        "model": orchestrator.gpt_model,
        "source_role": source_role,
        "messages": history,
    }, indent=2), encoding="utf-8")
    _write_cell_manifest(cell_dir, bench, source_role, kernel_path, result_payload)

    return LatencyOptOutcome(bench, source_role, success, str(cell_dir), result_payload.get("error", ""), result_payload)


def maybe_chain_latency_opt(
    *,
    bench: str,
    bench_dir: Path,
    cell_dir: Path,
    orchestrator: Any,
    source_role: SourceRole,
    skip_existing: bool = True,
) -> Optional[LatencyOptOutcome]:
    """Run latency-opt when enabled for the given source role; swallow errors.

    Source resolution (see `resolve_latency_source_kernel`) is intentionally
    simple for now — chaining/ordering with other post-flash steps is
    refined in a later task.
    """
    if is_multistep_source_role(source_role):
        if not latency_opt_enabled():
            return None
    elif source_role == "flash_final" and not chain_after_flash():
        return None
    elif source_role == "dataflow" and not chain_after_dataflow():
        return None
    elif not latency_opt_enabled():
        return None
    try:
        outcome = run_latency_opt_for_cell(
            bench=bench,
            bench_dir=bench_dir,
            cell_dir=cell_dir,
            orchestrator=orchestrator,
            source_role=source_role,
            skip_existing=skip_existing,
        )
        if outcome.success:
            _LOG.info("[latency_opt] %s %s passed", bench, source_role)
        else:
            _LOG.warning("[latency_opt] %s %s failed: %s", bench, source_role, outcome.error[:200])
        return outcome
    except Exception as exc:
        _LOG.exception("[latency_opt] %s %s error: %s", bench, source_role, exc)
        return None


__all__ = [
    "SOURCE_ROLES",
    "SourceRole",
    "LatencyOptOutcome",
    "TRAJECTORY_SCHEMA",
    "append_round_event",
    "artifact_paths",
    "budget_pct",
    "chain_after_dataflow",
    "chain_after_flash",
    "configure_post_flash_env",
    "discover_matrix_cells",
    "finalize_trajectory",
    "latency_opt_enabled",
    "latency_round_limit",
    "load_existing_result",
    "maybe_chain_latency_opt",
    "maybe_cosim_promoted_best",
    "new_trajectory",
    "plan_mentions_scope",
    "promote_latency_opt_as_selected",
    "prompt_text_for_docs",
    "render_budget_block",
    "render_latency_analysis_pack",
    "repair_round_limit",
    "resolve_latency_source_kernel",
    "run_latency_opt_for_cell",
    "set_best_so_far",
    "should_accept",
    "template_actions_for_report",
    "under_device_budget",
]
