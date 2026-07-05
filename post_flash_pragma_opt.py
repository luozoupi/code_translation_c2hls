"""Post-pass HLS pragma optimization (pipeline / unroll / array_partition).

Runs **after** a kernel passes csim + csynth:
- **flash_final** — flash-selected ``*_final.cpp`` / ``*_selected.cpp``
- **dataflow** — successful ``*_dataflow.cpp`` from the DATAFLOW step

The LLM adds or fixes a **small set of high-value** Vitis HLS pragmas guided by
``vitis_hls_2023_2_pragmas_curated.md``. Validation: compile + csim + csynth;
cosim off by default. Up to four repair rounds on failure.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from c2hls_paths import REPO_ROOT, VITIS_PRAGMAS_CURATED_MD
from flash_flow_artifacts import sha256_text
from post_flash_dataflow import extract_kernel_block, sanitize_kernel_source
from post_flash_mem_parallel import discover_matrix_cells, resolve_selected_kernel

SourceRole = Literal["flash_final", "dataflow"]
SOURCE_ROLES: tuple[SourceRole, ...] = ("flash_final", "dataflow")

DEFAULT_REPAIR_ROUNDS = 4
DEFAULT_PRAGMA_GUIDE_MAX_CHARS = 90_000
STEP_TAG = "pragma_opt"
DATAFLOW_STEP_TAG = "dataflow"

_LOG = logging.getLogger(__name__)

_SYSTEM = """You are an expert Xilinx Vitis HLS engineer specializing in **pragma selection** for Vitis HLS 2023.2.

Given a **working** kernel that already passes csim and csynth, add or fix HLS pragmas to improve throughput and/or latency.

## Goals (quality over quantity)
- **Correct, purposeful pragmas matter more than many pragmas.** Do not spray directives everywhere.
- **Preserve correctness and interfaces:** keep the exact top-level `extern "C"` signature, parameter list, array shapes, and all existing `#pragma HLS INTERFACE` lines unchanged.
- **Do not change the algorithm** — only scheduling/memory directives on loops and local arrays.
- **Label loops** with descriptive names (`load_A_i:`, `compute_j:`) when you touch them so reports are readable.

## Minimum expectations
1. **Pipeline:** every performance-critical inner loop that lacks pipelining should get `#pragma HLS PIPELINE II=1` (or justified `II=N` if II=1 is impossible). Outer loops only when you understand the resource impact.
2. **Unroll:** add `#pragma HLS UNROLL` (full or `factor=N`) on **small, fixed-bound** inner loops where it exposes parallelism — especially with MAC/reduction patterns. Pair with memory partitioning when needed.
3. **Array partition / reshape:** when parallel accesses need multiple ports, add `#pragma HLS array_partition` (`complete`, `cyclic`, or `block` with `factor`) or `array_reshape` on **local** arrays/tiles. Prefer `complete` only when small enough for BRAM/LUT budget.

## Use the curated pragma guide
Follow the attached **Vitis HLS 2023.2 Pragmas — Curated Usage Guide** for syntax, placement, and trade-offs. Prefer `pipeline`, `unroll`, and `array_partition` first; add others only when clearly justified (`dependence`, `loop_tripcount`, `inline off` in DATAFLOW tasks, etc.).

## Forbidden
- Changing top-level ports, bundles, or function name.
- Removing existing correct pragmas without cause.
- Partitioning top-level `m_axi` arrays directly.
- `malloc`, system calls, or non-synthesizable constructs.

## Output
Return **one** fenced block only:
```kernel
... full kernel source ...
```
"""

_INITIAL_USER = """Improve HLS pragmas on this **{source_label}** kernel.

**Source role:** `{source_role}` (already passes csim + csynth).

## Benchmark context
{benchmark_context}

## Header ({header_name})
```cpp
{header_code}
```

## Baseline kernel (preserve top signature + INTERFACE pragmas)
```cpp
{kernel_code}
```

## Baseline csynth summary (for context)
{synth_summary}

## Vitis HLS 2023.2 Pragmas — Curated Usage Guide
{pragma_guide}

Audit loops: add **pipeline** and **unroll** where missing and useful; add **array_partition** on locals when parallelism needs bandwidth. Return a single ```kernel``` block.
"""

_REPAIR_USER = """Repair the **pragma-optimized** kernel after a validation failure.

Keep the **exact** top-level signature and `#pragma HLS INTERFACE` pragmas.

## Failure
Stage: {stage}
```
{error}
```

Fix the failure while keeping intentional pipeline/unroll/partition improvements where still valid.

## Benchmark context
{benchmark_context}

## Header ({header_name})
```cpp
{header_code}
```

## Current kernel
```cpp
{kernel_code}
```

## Vitis HLS 2023.2 Pragmas — Curated Usage Guide
{pragma_guide}

Return a corrected single ```kernel``` block.
"""


def pragma_opt_enabled() -> bool:
    return os.getenv("C2HLS_POST_FLASH_PRAGMA_OPT", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def chain_after_flash() -> bool:
    raw = os.getenv("C2HLS_PRAGMA_OPT_CHAIN_FLASH", "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    return pragma_opt_enabled()


def chain_after_dataflow() -> bool:
    raw = os.getenv("C2HLS_PRAGMA_OPT_CHAIN_DATAFLOW", "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    return pragma_opt_enabled()


def repair_round_limit() -> int:
    try:
        return max(1, int(os.getenv("C2HLS_PRAGMA_OPT_REPAIR_ROUNDS", str(DEFAULT_REPAIR_ROUNDS))))
    except ValueError:
        return DEFAULT_REPAIR_ROUNDS


def pragma_guide_max_chars() -> int:
    try:
        return max(8_000, int(os.getenv("C2HLS_PRAGMA_OPT_GUIDE_MAX_CHARS", str(DEFAULT_PRAGMA_GUIDE_MAX_CHARS))))
    except ValueError:
        return DEFAULT_PRAGMA_GUIDE_MAX_CHARS


def load_pragma_guide(*, max_chars: Optional[int] = None) -> str:
    path = VITIS_PRAGMAS_CURATED_MD
    if not path.is_file():
        raise FileNotFoundError(f"pragma guide missing: {path}")
    text = path.read_text(encoding="utf-8")
    limit = max_chars if max_chars is not None else pragma_guide_max_chars()
    if len(text) > limit:
        return text[:limit] + "\n\n...(truncated — see full file in repo)...\n"
    return text


def artifact_stem(source_role: SourceRole) -> str:
    if source_role == "dataflow":
        return f"{{bench}}_{DATAFLOW_STEP_TAG}_{STEP_TAG}"
    return f"{{bench}}_{STEP_TAG}"


def artifact_paths(cell_dir: Path, bench: str, source_role: SourceRole) -> dict[str, Path]:
    if source_role == "dataflow":
        base = f"{bench}_{DATAFLOW_STEP_TAG}_{STEP_TAG}"
    else:
        base = f"{bench}_{STEP_TAG}"
    return {
        "kernel": cell_dir / f"{base}.cpp",
        "report": cell_dir / f"{base}_report.json",
        "result": cell_dir / f"{base}_result.json",
        "history": cell_dir / f"{base}_history.json",
    }


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


def resolve_source_kernel(
    cell_dir: Path,
    bench: str,
    source_role: SourceRole,
) -> tuple[Optional[Path], str, Optional[dict[str, Any]]]:
    """Return (kernel_path, role_label, prior_synth_report)."""
    if source_role == "flash_final":
        path, role = resolve_selected_kernel(cell_dir, bench)
        report: Optional[dict[str, Any]] = None
        if path is not None:
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
        return path, role or "final", report

    dataflow_kernel = cell_dir / f"{bench}_{DATAFLOW_STEP_TAG}.cpp"
    dataflow_result = cell_dir / f"{bench}_{DATAFLOW_STEP_TAG}_result.json"
    if not dataflow_kernel.is_file():
        return None, "", None
    report = None
    if dataflow_result.is_file():
        try:
            data = json.loads(dataflow_result.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("success"):
                report = data.get("synth_report") if isinstance(data.get("synth_report"), dict) else None
            else:
                return None, "", None
        except json.JSONDecodeError:
            return None, "", None
    else:
        return None, "", None
    return dataflow_kernel, DATAFLOW_STEP_TAG, report


def summarize_synth_report(report: Optional[dict[str, Any]]) -> str:
    if not report:
        return "(no prior synth report available)"
    lines = [
        f"- latency_cycles: {report.get('latency_cycles')}",
        f"- interval: {report.get('interval')}",
        f"- BRAM: {report.get('bram')}  DSP: {report.get('dsp')}  FF: {report.get('ff')}  LUT: {report.get('lut')}",
        f"- fmax_mhz: {report.get('fmax_mhz')}",
    ]
    feedback = report.get("feedback") if isinstance(report.get("feedback"), dict) else {}
    scopes = feedback.get("scopes") if isinstance(feedback.get("scopes"), list) else []
    issues = []
    for scope in scopes[:8]:
        if not isinstance(scope, dict):
            continue
        issue = scope.get("issue")
        if issue:
            name = scope.get("name") or scope.get("scope_id") or "?"
            issues.append(f"{name}: {issue}")
    if issues:
        lines.append("- report issues: " + "; ".join(issues[:5]))
    return "\n".join(lines)


def _source_label(source_role: SourceRole) -> str:
    return "flash-final" if source_role == "flash_final" else "DATAFLOW"


@dataclass
class PragmaOptOutcome:
    bench: str
    source_role: SourceRole
    success: bool
    cell_dir: str
    error: str = ""
    result: Optional[dict[str, Any]] = None


def run_pragma_opt_for_cell(
    *,
    bench: str,
    bench_dir: Path,
    cell_dir: Path,
    orchestrator: Any,
    source_role: SourceRole = "flash_final",
    skip_existing: bool = True,
) -> PragmaOptOutcome:
    from c2hls import _load_benchmark_inputs, _run_synth_csim_cosim, compile_check_cpp
    from c2hls_temp import join_temp_tag

    kernel_path, kernel_role, prior_report = resolve_source_kernel(cell_dir, bench, source_role)
    if kernel_path is None:
        msg = (
            "no passing dataflow kernel"
            if source_role == "dataflow"
            else "no selected/final kernel cpp"
        )
        return PragmaOptOutcome(bench, source_role, False, str(cell_dir), msg)

    inputs = _load_benchmark_inputs(str(bench_dir))
    source_kernel = kernel_path.read_text(encoding="utf-8")
    header_code = inputs.get("header_code", "")
    header_name = inputs.get("header_name") or "kernel.h"
    meta = inputs["meta"]
    top_function = (
        meta.get("translated_hls_top")
        or meta.get("hls_top")
        or meta.get("kernel_top")
        or "workload"
    )
    benchmark_context = inputs.get("benchmark_context", "")
    testbench_code = inputs.get("testbench_code", "")
    extra_files = inputs.get("extra_files", [])
    part = meta.get("part", orchestrator.part)
    clock_ns = meta.get("clock_ns", orchestrator.clock_ns)

    paths = artifact_paths(cell_dir, bench, source_role)
    if skip_existing:
        existing = load_existing_result(paths["result"])
        if existing is not None:
            return PragmaOptOutcome(bench, source_role, True, str(cell_dir), result=existing)

    pragma_guide = load_pragma_guide()
    synth_summary = summarize_synth_report(prior_report)

    history: list[dict[str, str]] = []
    attempts: list[dict[str, Any]] = []
    kernel_code = ""
    success = False
    last_error = ""
    synth_report: dict[str, Any] = {}
    csim_summary: Optional[dict[str, Any]] = None

    system = _SYSTEM
    user = _INITIAL_USER.format(
        source_role=source_role,
        source_label=_source_label(source_role),
        benchmark_context=benchmark_context,
        header_name=header_name,
        header_code=header_code[:12000],
        kernel_code=source_kernel[:120000],
        synth_summary=synth_summary,
        pragma_guide=pragma_guide,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    reply = orchestrator._call_llm(messages)
    history.extend([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": reply},
    ])
    kernel_code = extract_kernel_block(reply)

    tag_base = f"{STEP_TAG}_{source_role}"
    for attempt in range(repair_round_limit()):
        attempt_error = ""
        stage = "extract"
        if not kernel_code:
            attempt_error = "LLM response missing ```kernel``` fenced block"
        else:
            ok, err = compile_check_cpp(
                kernel_code,
                header_code,
                header_name,
                extra_files=extra_files,
            )
            stage = "compile_kernel"
            if not ok:
                attempt_error = err
            elif not testbench_code:
                attempt_error = "benchmark has no testbench for csim"
                stage = "testbench"
            else:
                tag = join_temp_tag(bench, tag_base, f"a{attempt}")
                outcome = _run_synth_csim_cosim(
                    kernel_code,
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
                stage = "csynth"
                if synth.get("success"):
                    csim_pass = (
                        csim_summary is None
                        or csim_summary.get("passed")
                        or csim_summary.get("status") == "passed"
                    )
                    if csim_pass:
                        success = True
                        synth_report = synth.get("report") or {}
                    else:
                        attempt_error = (csim_summary or {}).get("error") or "csim failed"
                        stage = "csim"
                else:
                    attempt_error = synth.get("error") or "csynth failed"

        attempts.append({
            "attempt": attempt,
            "stage": stage,
            "success": success,
            "error": attempt_error[:4000],
        })
        last_error = attempt_error
        if success:
            break
        if attempt >= repair_round_limit() - 1:
            break

        repair_user = _REPAIR_USER.format(
            stage=stage,
            error=attempt_error[:8000],
            benchmark_context=benchmark_context,
            header_name=header_name,
            header_code=header_code[:12000],
            kernel_code=kernel_code[:120000],
            pragma_guide=pragma_guide,
        )
        repair_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": repair_user},
        ]
        reply = orchestrator._call_llm(repair_messages)
        history.extend([
            {"role": "user", "content": repair_user},
            {"role": "assistant", "content": reply},
        ])
        extracted = extract_kernel_block(reply)
        if extracted:
            kernel_code = extracted

    kernel_code = sanitize_kernel_source(kernel_code)

    result_payload: dict[str, Any] = {
        "schema": "post_flash_pragma_opt_v1",
        "benchmark": bench,
        "source_role": source_role,
        "success": success,
        "error": last_error if not success else "",
        "source_kernel": str(kernel_path.name),
        "source_kernel_role": kernel_role,
        "testbench": "benchmark_original",
        "attempts": attempts,
        "synth_report": synth_report,
        "csim": csim_summary,
        "latency_cycles": synth_report.get("latency_cycles"),
        "baseline_latency_cycles": (prior_report or {}).get("latency_cycles"),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "pragma_guide": str(VITIS_PRAGMAS_CURATED_MD.relative_to(REPO_ROOT)),
    }

    artifacts: dict[str, str] = {}
    if kernel_code:
        paths["kernel"].write_text(kernel_code, encoding="utf-8")
        artifacts["kernel"] = paths["kernel"].name
        result_payload["kernel_sha256"] = sha256_text(kernel_code)
    if synth_report:
        paths["report"].write_text(json.dumps(synth_report, indent=2, default=str) + "\n", encoding="utf-8")
        artifacts["report"] = paths["report"].name
    if artifacts:
        result_payload["artifacts"] = artifacts

    paths["result"].write_text(json.dumps(result_payload, indent=2, default=str) + "\n", encoding="utf-8")
    paths["history"].write_text(json.dumps({
        "model": orchestrator.gpt_model,
        "source_role": source_role,
        "messages": history,
    }, indent=2), encoding="utf-8")
    _write_cell_manifest(cell_dir, bench, source_role, kernel_path, result_payload)
    return PragmaOptOutcome(bench, source_role, success, str(cell_dir), last_error, result_payload)


def _write_cell_manifest(
    cell_dir: Path,
    bench: str,
    source_role: SourceRole,
    kernel_path: Path,
    result_payload: dict[str, Any],
) -> None:
    if source_role == "dataflow":
        manifest_name = f"{bench}_post_flash_dataflow_{STEP_TAG}.json"
    else:
        manifest_name = f"{bench}_post_flash_{STEP_TAG}.json"
    manifest_path = cell_dir / manifest_name
    manifest_path.write_text(json.dumps({
        "schema": "post_flash_pragma_opt_manifest_v1",
        "benchmark": bench,
        "source_role": source_role,
        "source_kernel": str(kernel_path),
        "success": result_payload.get("success"),
        "result": artifact_paths(cell_dir, bench, source_role)["result"].name,
    }, indent=2) + "\n", encoding="utf-8")


def maybe_chain_pragma_opt(
    *,
    bench: str,
    bench_dir: Path,
    cell_dir: Path,
    orchestrator: Any,
    source_role: SourceRole,
    skip_existing: bool = True,
) -> Optional[PragmaOptOutcome]:
    """Run pragma-opt when enabled for the given source role; swallow errors."""
    if source_role == "flash_final" and not chain_after_flash():
        return None
    if source_role == "dataflow" and not chain_after_dataflow():
        return None
    if not pragma_opt_enabled():
        return None
    try:
        outcome = run_pragma_opt_for_cell(
            bench=bench,
            bench_dir=bench_dir,
            cell_dir=cell_dir,
            orchestrator=orchestrator,
            source_role=source_role,
            skip_existing=skip_existing,
        )
        if outcome.success:
            _LOG.info("[pragma_opt] %s %s passed", bench, source_role)
        else:
            _LOG.warning("[pragma_opt] %s %s failed: %s", bench, source_role, outcome.error[:200])
        return outcome
    except Exception as exc:
        _LOG.exception("[pragma_opt] %s %s error: %s", bench, source_role, exc)
        return None


def configure_post_flash_env() -> None:
    os.environ.setdefault("C2HLS_RUN_COSIM", "0")
    os.environ.setdefault("C2HLS_COSIM_REQUIRED", "0")
    os.environ.setdefault("C2HLS_REFERENCE_COSIM", "0")


def prompt_text_for_docs() -> dict[str, str]:
    guide = load_pragma_guide(max_chars=4000)
    return {
        "system": _SYSTEM,
        "initial_user": _INITIAL_USER.format(
            source_role="flash_final",
            source_label="flash-final",
            benchmark_context="...",
            header_name="kernel.h",
            header_code="...",
            kernel_code="...",
            synth_summary="...",
            pragma_guide=guide,
        ),
        "repair_user": _REPAIR_USER.format(
            stage="csynth",
            error="...",
            benchmark_context="...",
            header_name="kernel.h",
            header_code="...",
            kernel_code="...",
            pragma_guide=guide,
        ),
        "pragma_guide_path": str(VITIS_PRAGMAS_CURATED_MD),
    }


__all__ = [
    "SOURCE_ROLES",
    "PragmaOptOutcome",
    "SourceRole",
    "artifact_paths",
    "chain_after_dataflow",
    "chain_after_flash",
    "configure_post_flash_env",
    "discover_matrix_cells",
    "load_existing_result",
    "load_pragma_guide",
    "maybe_chain_pragma_opt",
    "pragma_opt_enabled",
    "prompt_text_for_docs",
    "repair_round_limit",
    "resolve_source_kernel",
    "run_pragma_opt_for_cell",
]
