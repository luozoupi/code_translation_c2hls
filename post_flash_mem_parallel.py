"""Optional post-flash memory-parallelism step (2x / 4x duplicate IO + compute).

Runs **after** flash has selected a final kernel (``*_final.cpp`` or record-flow
``*_selected.cpp``). The LLM emits an updated HLS kernel **and** a matching
``testbench.cpp``. Each factor is validated with compile + csim + csynth;
cosim is off by default. Up to four repair rounds per factor on failure.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from flash_flow_artifacts import resolve_cell_final_cpp, sha256_text

MEMORY_PARALLEL_FACTORS = (2, 4)
DEFAULT_REPAIR_ROUNDS = 4
STEP_TAG = "mem_parallel"

_SYSTEM = """You are an expert Xilinx Vitis HLS engineer specializing in memory-level parallelism.

Given a working flash-optimized HLS kernel and its C testbench, produce a **memory-parallel** variant that duplicates independent input/output memory interfaces (or equivalent m_axi / array banks) and runs the same compute on multiple lanes in parallel.

Constraints:
- Preserve algorithm correctness: the testbench must still validate the parallelized design.
- Keep exactly one testbench-visible `extern "C"` top function (named in benchmark context). Do not add a second wrapper such as `workload()` forwarding to `kernel_*`.
- Use only synthesizable Vitis HLS C/C++ (no malloc, no system calls in the kernel).
- Target parallelism factor is exactly {factor}x: duplicate IO/compute paths so up to {factor} independent instances can run in parallel (replicate ports, ping-pong banks, or dataflow PEs as appropriate for small kernels).
- Do NOT tune resource usage yet — only implement fixed 2x or 4x structural parallelism.
- Output **two** fenced blocks in this order:
  1. ```kernel ...```  — full HLS kernel source
  2. ```testbench ...``` — full C++ testbench that drives the parallel kernel and checks correctness
"""

_INITIAL_USER = """Apply **{factor}x memory parallelism** to this flash-selected kernel.

## Benchmark context
{benchmark_context}

## Header ({header_name})
```cpp
{header_code}
```

## Selected flash kernel (baseline for this step)
```cpp
{kernel_code}
```

## Original testbench (adapt for {factor}x IO / parallel lanes)
```cpp
{testbench_code}
```

Produce the ```kernel``` and ```testbench``` blocks. The testbench must compile with g++ and exercise all {factor} parallel lanes against the same numerical semantics as the original (aggregate or per-lane checks as appropriate).
"""

_REPAIR_USER = """Repair the **{factor}x memory-parallel** kernel and testbench after a validation failure.

## Failure
Stage: {stage}
```
{error}
```

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

## Current testbench
```cpp
{testbench_code}
```

Return corrected ```kernel``` and ```testbench``` blocks that fix the failure while keeping {factor}x memory parallelism.
"""


def mem_parallel_enabled() -> bool:
    return os.getenv("C2HLS_POST_FLASH_MEM_PARALLEL", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def mem_parallel_factors() -> tuple[int, ...]:
    raw = os.getenv("C2HLS_MEM_PARALLEL_FACTORS", "").strip()
    if not raw:
        return MEMORY_PARALLEL_FACTORS
    out: list[int] = []
    for part in raw.replace(",", " ").split():
        try:
            n = int(part.strip().lower().rstrip("x"))
        except ValueError:
            continue
        if n in (2, 4):
            out.append(n)
    return tuple(out) or MEMORY_PARALLEL_FACTORS


def repair_round_limit() -> int:
    try:
        return max(1, int(os.getenv("C2HLS_MEM_PARALLEL_REPAIR_ROUNDS", str(DEFAULT_REPAIR_ROUNDS))))
    except ValueError:
        return DEFAULT_REPAIR_ROUNDS


def extract_labeled_cpp_blocks(text: str) -> dict[str, str]:
    """Parse ```kernel / ```testbench (or first two generic fences)."""
    if not text:
        return {}
    out: dict[str, str] = {}
    labeled = re.findall(
        r"```\s*(kernel|testbench)\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    for label, body in labeled:
        key = label.strip().lower()
        if body.strip():
            out[key] = body.strip()
    if "kernel" in out:
        return out
    if "testbench" in out:
        return out
    generic = re.findall(r"```(?:cpp|c\+\+|c|hls)\s*(.*?)```", text, flags=re.DOTALL)
    if len(generic) >= 2:
        return {"kernel": generic[0].strip(), "testbench": generic[1].strip()}
    if len(generic) == 1:
        return {"kernel": generic[0].strip()}
    return out


def factor_suffix(factor: int) -> str:
    return f"{factor}x"


def artifact_paths(cell_dir: Path, bench: str, factor: int) -> dict[str, Path]:
    tag = factor_suffix(factor)
    base = f"{bench}_{STEP_TAG}_{tag}"
    return {
        "kernel": cell_dir / f"{base}.cpp",
        "testbench": cell_dir / f"{base}_testbench.cpp",
        "report": cell_dir / f"{base}_report.json",
        "result": cell_dir / f"{base}_result.json",
        "history": cell_dir / f"{base}_history.json",
    }


def _result_success(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return isinstance(data, dict) and data.get("success") is True


def _resolve_flash_base_kernel(cell_dir: Path, bench: str) -> tuple[Optional[Path], str]:
    """Return (path, role) for the flash-selected kernel source.

    This is the base resolver: selected -> final -> legacy final. It never
    considers post-pass outputs (pragma_opt/latency_opt), so it is safe to
    use when a post-pass itself is resolving its own seed kernel.
    """
    selected = cell_dir / f"{bench}_selected.cpp"
    if selected.is_file():
        return selected, "selected"
    resolved = resolve_cell_final_cpp(cell_dir, bench)
    if resolved is not None:
        return resolved, "final"
    legacy = cell_dir / f"{bench}_final.cpp"
    if legacy.is_file():
        return legacy, "final"
    return None, ""


def _latency_cycles_from_result(result_path: Path) -> Optional[float]:
    if not result_path.is_file():
        return None
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("success") is not True:
        return None
    lat = data.get("latency_cycles")
    if lat is None:
        return None
    try:
        return float(lat)
    except (TypeError, ValueError):
        return None


def resolve_selected_kernel(
    cell_dir: Path, bench: str, *, include_post_passes: bool = True
) -> tuple[Optional[Path], str]:
    """Return (path, role) for the flash-selected kernel source.

    When `include_post_passes` is True (the default), a successful
    latency_opt / dataflow_latency_opt output is preferred (lower
    ``latency_cycles`` wins when both exist), then a successful pragma_opt
    output, falling back to the base flash-selected/final kernel. Pass
    `include_post_passes=False` when a post-pass is resolving its own seed
    kernel, to avoid recursively preferring its own (or a later) post-pass
    output.
    """
    if include_post_passes:
        candidates: list[tuple[float, Path, str]] = []
        for role, cpp_name, res_name in (
            ("latency_opt", f"{bench}_latency_opt.cpp", f"{bench}_latency_opt_result.json"),
            (
                "dataflow_latency_opt",
                f"{bench}_dataflow_latency_opt.cpp",
                f"{bench}_dataflow_latency_opt_result.json",
            ),
        ):
            cpp = cell_dir / cpp_name
            res = cell_dir / res_name
            if not (cpp.is_file() and _result_success(res)):
                continue
            lat = _latency_cycles_from_result(res)
            # Missing latency sorts after numbered ones; stable by role order.
            key = lat if lat is not None else float("inf")
            candidates.append((key, cpp, role))
        if candidates:
            candidates.sort(key=lambda item: (item[0], 0 if item[2] == "dataflow_latency_opt" else 1))
            _lat, path, role = candidates[0]
            return path, role
        pragma_cpp = cell_dir / f"{bench}_pragma_opt.cpp"
        pragma_res = cell_dir / f"{bench}_pragma_opt_result.json"
        if pragma_cpp.is_file() and _result_success(pragma_res):
            return pragma_cpp, "pragma_opt"
    return _resolve_flash_base_kernel(cell_dir, bench)


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


def discover_matrix_cells(matrix_root: Path) -> list[dict[str, Any]]:
    matrix_file = matrix_root / "matrix.json"
    if matrix_file.is_file():
        rows = json.loads(matrix_file.read_text(encoding="utf-8"))
        cells: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            cell_dir = Path(row.get("cell_dir", ""))
            if not cell_dir.is_dir():
                continue
            cells.append({
                "bench": row.get("bench") or cell_dir.parent.name,
                "cell_dir": cell_dir,
                "status": row.get("status"),
                "model": row.get("model"),
            })
        return cells
    # Fallback: walk hlsfactory_* / devstral2__*
    cells = []
    for bench_dir in sorted(matrix_root.glob("hlsfactory_*")):
        if not bench_dir.is_dir():
            continue
        for cell_dir in sorted(bench_dir.glob("devstral2__*")):
            if cell_dir.is_dir():
                cells.append({
                    "bench": bench_dir.name,
                    "cell_dir": cell_dir,
                    "status": "unknown",
                    "model": "",
                })
    return cells


@dataclass
class MemParallelOutcome:
    bench: str
    factor: int
    success: bool
    cell_dir: str
    error: str = ""
    result: Optional[dict[str, Any]] = None


def _format_attempt_history(attempts: list[dict[str, Any]]) -> str:
    if not attempts:
        return "(no prior attempts)"
    lines = []
    for item in attempts:
        lines.append(
            f"- attempt {item.get('attempt')}: stage={item.get('stage')} "
            f"ok={item.get('success')} err={(item.get('error') or '')[:200]}"
        )
    return "\n".join(lines)


def run_memory_parallel_for_cell(
    *,
    bench: str,
    bench_dir: Path,
    cell_dir: Path,
    orchestrator: Any,
    factors: tuple[int, ...] = MEMORY_PARALLEL_FACTORS,
    skip_existing: bool = True,
) -> list[MemParallelOutcome]:
    from c2hls import (
        _load_benchmark_inputs,
        _run_synth_csim_cosim,
        compile_check_cpp,
    )
    from c2hls_temp import join_temp_tag

    inputs = _load_benchmark_inputs(str(bench_dir))
    kernel_path, kernel_role = resolve_selected_kernel(cell_dir, bench)
    if kernel_path is None:
        return [MemParallelOutcome(bench, f, False, str(cell_dir), "no selected/final kernel cpp") for f in factors]

    selected_kernel = kernel_path.read_text(encoding="utf-8")
    header_code = inputs.get("header_code", "")
    header_name = inputs.get("header_name") or "kernel.h"
    meta = inputs["meta"]
    top_function = meta.get("translated_hls_top") or meta.get("hls_top") or meta.get("kernel_top") or "workload"
    benchmark_context = inputs.get("benchmark_context", "")
    original_tb = inputs.get("testbench_code", "")
    extra_files = inputs.get("extra_files", [])
    part = meta.get("part", orchestrator.part)
    clock_ns = meta.get("clock_ns", orchestrator.clock_ns)

    outcomes: list[MemParallelOutcome] = []
    for factor in factors:
        paths = artifact_paths(cell_dir, bench, factor)
        if skip_existing:
            existing = load_existing_result(paths["result"])
            if existing is not None:
                outcomes.append(MemParallelOutcome(bench, factor, True, str(cell_dir), result=existing))
                continue

        history: list[dict[str, str]] = []
        attempts: list[dict[str, Any]] = []
        kernel_code = ""
        testbench_code = ""
        success = False
        last_error = ""
        synth_report: dict[str, Any] = {}
        csim_summary: Optional[dict[str, Any]] = None

        # Initial LLM call
        system = _SYSTEM.format(factor=factor)
        user = _INITIAL_USER.format(
            factor=factor,
            benchmark_context=benchmark_context,
            header_name=header_name,
            header_code=header_code[:12000],
            kernel_code=selected_kernel[:120000],
            testbench_code=original_tb[:24000],
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
        blocks = extract_labeled_cpp_blocks(reply)
        kernel_code = blocks.get("kernel", "")
        testbench_code = blocks.get("testbench", "")

        for attempt in range(repair_round_limit()):
            attempt_error = ""
            stage = "extract"
            if not kernel_code or not testbench_code:
                attempt_error = "LLM response missing kernel or testbench fenced block"
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
                else:
                    ok, err = compile_check_cpp(
                        testbench_code,
                        header_code,
                        header_name,
                        extra_files=extra_files,
                    )
                    stage = "compile_testbench"
                    if not ok:
                        attempt_error = f"testbench compile: {err}"
                    else:
                        tag = join_temp_tag(bench, STEP_TAG, f"{factor}x", f"a{attempt}")
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
                            log_prefix=f"[mem_parallel {factor}x]",
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
                                attempt_error = (
                                    (csim_summary or {}).get("error")
                                    or "csim failed"
                                )
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
                factor=factor,
                stage=stage,
                error=attempt_error[:8000],
                benchmark_context=benchmark_context,
                header_name=header_name,
                header_code=header_code[:12000],
                kernel_code=kernel_code[:120000],
                testbench_code=testbench_code[:24000],
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
            blocks = extract_labeled_cpp_blocks(reply)
            if blocks.get("kernel"):
                kernel_code = blocks["kernel"]
            if blocks.get("testbench"):
                testbench_code = blocks["testbench"]

        result_payload: dict[str, Any] = {
            "schema": "post_flash_mem_parallel_v1",
            "benchmark": bench,
            "factor": factor,
            "success": success,
            "error": last_error if not success else "",
            "source_kernel": str(kernel_path.name),
            "source_kernel_role": kernel_role,
            "attempts": attempts,
            "synth_report": synth_report,
            "csim": csim_summary,
            "latency_cycles": synth_report.get("latency_cycles"),
            "selected_from_flash": kernel_role,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }

        if success and kernel_code and testbench_code:
            paths["kernel"].write_text(kernel_code, encoding="utf-8")
            paths["testbench"].write_text(testbench_code, encoding="utf-8")
            paths["report"].write_text(json.dumps(synth_report, indent=2, default=str) + "\n", encoding="utf-8")
            result_payload["artifacts"] = {k: str(v.name) for k, v in paths.items() if k != "history"}
            result_payload["kernel_sha256"] = sha256_text(kernel_code)
            result_payload["testbench_sha256"] = sha256_text(testbench_code)

        paths["result"].write_text(json.dumps(result_payload, indent=2, default=str) + "\n", encoding="utf-8")
        paths["history"].write_text(json.dumps({
            "model": orchestrator.gpt_model,
            "messages": history,
        }, indent=2), encoding="utf-8")

        outcomes.append(MemParallelOutcome(bench, factor, success, str(cell_dir), last_error, result_payload))

    # Cell-level manifest
    manifest_path = cell_dir / f"{bench}_post_flash_mem_parallel.json"
    manifest_path.write_text(json.dumps({
        "schema": "post_flash_mem_parallel_manifest_v1",
        "benchmark": bench,
        "source_kernel": str(kernel_path),
        "factors": {
            str(f): {
                "success": o.success,
                "result": artifact_paths(cell_dir, bench, f)["result"].name,
            }
            for f, o in zip(factors, outcomes)
        },
    }, indent=2) + "\n", encoding="utf-8")

    return outcomes


def configure_post_flash_env() -> None:
    os.environ.setdefault("C2HLS_RUN_COSIM", "0")
    os.environ.setdefault("C2HLS_COSIM_REQUIRED", "0")
    os.environ.setdefault("C2HLS_REFERENCE_COSIM", "0")
