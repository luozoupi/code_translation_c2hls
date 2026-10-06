"""Post-parent memory self-improve loop (generic mem, no gold, bounded prompt).

Seed is the parent variant's own kernel. Each turn sees a full QoR table and
at most three kernel bodies (best / last accepted / last failed). Winner is
min(latency_cycles_worst) among legal csim-pass synths.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

_LOG = logging.getLogger(__name__)

STEP_TAG = "mem_iter"
DEFAULT_ROUNDS = 50
DEFAULT_PROMPT_TOKENS = 32768
DEFAULT_MAX_TOKENS = 65536
DEFAULT_CONTEXT_TOKENS = 131072
DEFAULT_TIMEOUT_RETRIES = 8
DEFAULT_RETRY_BACKOFF_S = 30.0
DEFAULT_RETRY_BACKOFF_CAP_S = 300.0
GOLD_BAN = ("gt_code", "gold hls", "goal code", "ground-truth hls", "ground truth hls")
_TRANSIENT_LLM_NEEDLES = (
    "timed out",
    "timeout",
    "429",
    "502",
    "503",
    "529",
    "connection reset",
    "connection aborted",
    "temporarily unavailable",
    "overloaded",
    "server disconnected",
)

_SEED_CPP_REPORTS = (
    ("autosa_mm_stream.cpp", "autosa_mm_stream_report.json"),
    ("autosa_mm.cpp", "autosa_mm_enforcement.json"),
    ("autosa_mm_selected.cpp", "autosa_mm_selected_report.json"),
    ("autosa_mm_dse.cpp", "autosa_mm_dse_report.json"),
    ("autosa_mm_flash_opt.cpp", "autosa_mm_flash_opt_report.json"),
    ("autosa_mm.cpp", "autosa_mm_flash_opt_report.json"),
)

_SEED_BY_FAMILY: dict[str, tuple[tuple[str, str], ...]] = {
    "flash": (
        ("autosa_mm_flash_opt.cpp", "autosa_mm_flash_opt_report.json"),
        ("autosa_mm_selected.cpp", "autosa_mm_selected_report.json"),
    ),
    "oneshot": (
        ("autosa_mm_flash_opt.cpp", "autosa_mm_flash_opt_report.json"),
        ("autosa_mm_selected.cpp", "autosa_mm_selected_report.json"),
    ),
    "dse_v1": (
        ("autosa_mm_selected.cpp", "autosa_mm_selected_report.json"),
        ("autosa_mm_dse.cpp", "autosa_mm_dse_report.json"),
    ),
    "dse_v2": (
        ("autosa_mm_selected.cpp", "autosa_mm_selected_report.json"),
        ("autosa_mm_dse.cpp", "autosa_mm_dse_report.json"),
    ),
    "dse": (
        ("autosa_mm_selected.cpp", "autosa_mm_selected_report.json"),
        ("autosa_mm_dse.cpp", "autosa_mm_dse_report.json"),
    ),
    "stream": (("autosa_mm_stream.cpp", "autosa_mm_stream_report.json"),),
    "enf": (("autosa_mm.cpp", "autosa_mm_enforcement.json"),),
}

_SYSTEM = """You are an expert Xilinx Vitis HLS engineer.

Improve the memory behavior, loop II, and worst-case latency of the kernel
you already produced. You will see a QoR table of prior iterations and at
most three full sources: the best legal kernel so far, the last accepted
kernel (edit this), and the last failed kernel (do not repeat that break).

Rules:
- Do not invent or request gold / goal / reference HLS. Only use the kernels
  and metrics in this prompt.
- Keep the top-level ABI and existing INTERFACE pragmas unless a change is
  required for a legal memory improvement.
- Use synthesizable Vitis HLS C/C++ only.
- Return one ```kernel``` block with the full source.
"""


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def mem_iter_rounds() -> int:
    return _env_int("C2HLS_MEM_ITER_ROUNDS", DEFAULT_ROUNDS)


def mem_iter_prompt_tokens() -> int:
    return _env_int("C2HLS_MEM_ITER_PROMPT_TOKENS", DEFAULT_PROMPT_TOKENS)


def mem_iter_max_tokens() -> int:
    for key in (
        "C2HLS_MEM_ITER_MAX_TOKENS",
        "C2HLS_FLASH_MAX_TOKENS",
        "C2HLS_LLM_MAX_TOKENS",
    ):
        raw = os.getenv(key, "").strip()
        if raw.isdigit():
            return max(int(raw), 8192)
    return DEFAULT_MAX_TOKENS


def mem_iter_context_tokens() -> int:
    return _env_int("C2HLS_MEM_ITER_CONTEXT_TOKENS", DEFAULT_CONTEXT_TOKENS)


def mem_llm_timeout_retries() -> int:
    return _env_int("C2HLS_LLM_TIMEOUT_RETRIES", DEFAULT_TIMEOUT_RETRIES)


def mem_llm_retry_backoff_s() -> float:
    raw = os.getenv("C2HLS_LLM_RETRY_BACKOFF_S", "").strip()
    if not raw:
        return DEFAULT_RETRY_BACKOFF_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_RETRY_BACKOFF_S


def is_transient_llm_error(err: str) -> bool:
    lowered = (err or "").lower()
    return any(needle in lowered for needle in _TRANSIENT_LLM_NEEDLES)


@contextmanager
def _llm_lock() -> Iterator[None]:
    """Serialize mem LLM calls across jobs so one hosted endpoint is not stampeded."""
    flag = os.getenv("C2HLS_LLM_LOCK", "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        yield
        return
    root = Path(os.getenv("C2HLS_TMP_ROOT") or os.getenv("TMPDIR") or "/tmp")
    path = Path(os.getenv("C2HLS_LLM_LOCK_PATH") or (root / "c2hls_llm.lock"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def worst_latency(row: dict[str, Any]) -> Optional[int]:
    raw = row.get("latency_cycles_worst")
    if raw is None:
        raw = row.get("latency_cycles")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def is_legal(row: dict[str, Any]) -> bool:
    if row.get("status") != "ok":
        return False
    if (row.get("csim") or "") == "fail":
        return False
    return worst_latency(row) is not None


def select_winner(rows: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    legal = [r for r in rows if is_legal(r)]
    if not legal:
        return None
    return min(legal, key=lambda r: (int(worst_latency(r) or 10**18), int(r.get("iter") or 0)))


def extract_metrics_from_report(report: Any, *, status: str = "ok") -> dict[str, Any]:
    payload = report if isinstance(report, dict) else {}
    nested = payload.get("report")
    if isinstance(nested, dict) and ("latency_cycles" in nested or "dsp" in nested):
        payload = nested
    feedback = {}
    if isinstance(report, dict):
        fb = report.get("feedback")
        if isinstance(fb, dict):
            feedback = fb
        elif isinstance(payload.get("feedback"), dict):
            feedback = payload["feedback"]
    scopes = []
    for scope in feedback.get("scopes") or []:
        if not isinstance(scope, dict):
            continue
        scopes.append(
            {
                "scope_id": scope.get("scope_id") or scope.get("name") or "",
                "kind": scope.get("kind") or "",
                "pipeline_ii": scope.get("pipeline_ii") or scope.get("interval"),
                "latency_cycles": scope.get("latency_cycles"),
                "trip_count": scope.get("trip_count"),
                "dsp": scope.get("dsp"),
                "bram": scope.get("bram"),
                "ff": scope.get("ff"),
                "lut": scope.get("lut"),
                "uram": scope.get("uram"),
            }
        )
    violations: list[str] = []
    for blame in feedback.get("scheduler_blame") or []:
        if not isinstance(blame, dict):
            continue
        kind = str(blame.get("kind") or "ii")
        msg = str(blame.get("message") or "")[:180]
        loc = str(blame.get("source_location") or "")
        violations.append(f"{kind}: {msg} {loc}".strip())
    for bn in feedback.get("bottlenecks") or []:
        if not isinstance(bn, dict):
            continue
        kind = str(bn.get("kind") or "")
        if kind and "ii" in kind.lower():
            ev = str(bn.get("evidence") or "")[:160]
            violations.append(f"{kind}: {ev}".strip())
    csim = ""
    if isinstance(report, dict):
        raw = report.get("csim")
        if isinstance(raw, dict):
            if raw.get("passed") is True or raw.get("success") is True:
                csim = "pass"
            elif raw.get("passed") is False or raw.get("success") is False:
                csim = "fail"
        elif report.get("csim_passed") is True:
            csim = "pass"
        elif report.get("csim_passed") is False:
            csim = "fail"
    return {
        "status": status,
        "latency_cycles": payload.get("latency_cycles"),
        "latency_cycles_worst": payload.get("latency_cycles_worst", payload.get("latency_cycles")),
        "dsp": payload.get("dsp"),
        "bram": payload.get("bram"),
        "ff": payload.get("ff"),
        "lut": payload.get("lut"),
        "uram": payload.get("uram"),
        "csim": csim,
        "scopes": scopes,
        "ii_violations": violations[:8],
        "error": "",
    }


def _hot_ii(row: dict[str, Any]) -> str:
    scopes = row.get("scopes") or []
    if not scopes:
        return "-"
    hot = max(
        scopes,
        key=lambda s: int(s.get("latency_cycles") or 0),
    )
    return str(hot.get("pipeline_ii") if hot.get("pipeline_ii") is not None else "-")


def _hot_trip(row: dict[str, Any]) -> str:
    scopes = row.get("scopes") or []
    if not scopes:
        return "-"
    hot = max(scopes, key=lambda s: int(s.get("latency_cycles") or 0))
    return str(hot.get("trip_count") if hot.get("trip_count") is not None else "-")


def _hot_mod_lat(row: dict[str, Any]) -> str:
    scopes = row.get("scopes") or []
    if not scopes:
        return "-"
    hot = max(scopes, key=lambda s: int(s.get("latency_cycles") or 0))
    return str(hot.get("latency_cycles") if hot.get("latency_cycles") is not None else "-")


def metric_table_lines(
    rows: list[dict[str, Any]],
    *,
    include_scope_resources: bool = True,
    include_violation_tails: bool = True,
) -> list[str]:
    header = (
        "iter status lat lat_worst dsp bram ff lut uram csim ii trip mod_lat"
        + (" viol" if include_violation_tails else "")
    )
    lines = [header]
    for row in rows:
        viol = ""
        if include_violation_tails:
            vs = row.get("ii_violations") or []
            viol = (vs[0] if vs else "")[:80]
        extra = f" {viol}" if include_violation_tails else ""
        lines.append(
            f"{row.get('iter', 0)} {row.get('status') or '-'} "
            f"{row.get('latency_cycles') if row.get('latency_cycles') is not None else '-'} "
            f"{row.get('latency_cycles_worst') if row.get('latency_cycles_worst') is not None else '-'} "
            f"{row.get('dsp') if row.get('dsp') is not None else '-'} "
            f"{row.get('bram') if row.get('bram') is not None else '-'} "
            f"{row.get('ff') if row.get('ff') is not None else '-'} "
            f"{row.get('lut') if row.get('lut') is not None else '-'} "
            f"{row.get('uram') if row.get('uram') is not None else '-'} "
            f"{row.get('csim') or '-'} {_hot_ii(row)} {_hot_trip(row)} {_hot_mod_lat(row)}"
            f"{extra}"
        )
        if include_scope_resources:
            for scope in (row.get("scopes") or [])[:4]:
                lines.append(
                    f"  scope {scope.get('scope_id') or '-'} ii={scope.get('pipeline_ii') if scope.get('pipeline_ii') is not None else '-'} "
                    f"lat={scope.get('latency_cycles') if scope.get('latency_cycles') is not None else '-'} "
                    f"trip={scope.get('trip_count') if scope.get('trip_count') is not None else '-'} "
                    f"dsp={scope.get('dsp') if scope.get('dsp') is not None else '-'} "
                    f"bram={scope.get('bram') if scope.get('bram') is not None else '-'}"
                )
    return lines


def one_line_deltas(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    last_legal_lat: Optional[int] = None
    for row in rows:
        lat = worst_latency(row)
        dlat = ""
        if lat is not None and last_legal_lat is not None:
            dlat = f" dLat={lat - last_legal_lat}"
        note = row.get("note") or row.get("status") or ""
        lines.append(
            f"iter={row.get('iter')} status={row.get('status')} "
            f"lat_worst={lat if lat is not None else '-'} "
            f"dsp={row.get('dsp') if row.get('dsp') is not None else '-'} "
            f"ii={_hot_ii(row)}{dlat} note={note}"
        )
        if is_legal(row) and lat is not None:
            last_legal_lat = lat
    return lines


def lessons_block(rows: list[dict[str, Any]]) -> str:
    winner = select_winner(rows)
    best = worst_latency(winner) if winner else None
    iis = []
    for row in rows:
        raw = _hot_ii(row)
        try:
            iis.append(int(raw))
        except (TypeError, ValueError):
            continue
    fails = [str(r.get("status")) for r in rows if r.get("status") not in {"ok", None, ""}][-3:]
    improved: list[str] = []
    prev: Optional[int] = None
    for row in rows:
        lat = worst_latency(row) if is_legal(row) else None
        if lat is None:
            continue
        if prev is not None and lat < prev:
            improved.append(f"iter {row.get('iter')}: {prev}->{lat}")
        prev = lat
    lines = [
        f"best_lat_worst={best if best is not None else '-'}",
        f"worst_ii={max(iis) if iis else '-'}",
        f"last_failures={','.join(fails) if fails else 'none'}",
        f"last_improvements={'; '.join(improved[-3:]) if improved else 'none'}",
    ]
    return "\n".join(lines)


def pick_kernel_bodies(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Return de-duplicated (label, row) for best / last accepted / last failed."""
    chosen: list[tuple[str, dict[str, Any]]] = []
    seen_iters: set[int] = set()

    def _add(label: str, row: Optional[dict[str, Any]]) -> None:
        if row is None:
            return
        it = int(row.get("iter") or 0)
        if it in seen_iters:
            return
        if not row.get("kernel"):
            return
        seen_iters.add(it)
        chosen.append((label, row))

    _add("best-so-far", select_winner(rows))
    last_ok = None
    last_fail = None
    for row in rows:
        if is_legal(row):
            last_ok = row
        elif row.get("status") and row.get("status") != "ok":
            last_fail = row
    _add("last-accepted", last_ok)
    _add("last-failed", last_fail)
    return chosen


def assert_no_gold(text: str) -> None:
    low = text.lower()
    for banned in GOLD_BAN:
        if banned in low:
            raise ValueError(f"gold/goal leak in mem-iter prompt: {banned}")


def build_mem_prompt(
    rows: list[dict[str, Any]],
    *,
    prompt_token_budget: Optional[int] = None,
) -> str:
    budget = prompt_token_budget if prompt_token_budget is not None else mem_iter_prompt_tokens()
    include_scope = True
    include_viol = True
    deltas = one_line_deltas(rows)
    bodies = pick_kernel_bodies(rows)

    def _assemble(drop_deltas: int = 0) -> str:
        table = metric_table_lines(
            rows,
            include_scope_resources=include_scope,
            include_violation_tails=include_viol,
        )
        kept_deltas = deltas[drop_deltas:] if drop_deltas else deltas
        parts = [
            "Improve memory / II / worst-case latency of your last accepted kernel.",
            "Do not ask for gold, goal, or reference HLS.",
            "",
            "## QoR table (all prior iters)",
            *table,
            "",
            "## Lessons",
            lessons_block(rows),
        ]
        if kept_deltas:
            parts.extend(["", "## Older-iter deltas (no source)", *kept_deltas])
        for label, row in bodies:
            parts.extend(
                [
                    "",
                    f"## Kernel: {label} (iter {row.get('iter')})",
                    "```kernel",
                    str(row.get("kernel") or ""),
                    "```",
                ]
            )
            err = str(row.get("error") or "")
            if err and label == "last-failed":
                parts.extend(["", "## Last failure tail", err[:800]])
        return "\n".join(parts)

    drop = 0
    text = _assemble(drop)
    while estimate_tokens(text) > budget and drop < len(deltas):
        drop += 1
        text = _assemble(drop)
    if estimate_tokens(text) > budget and include_scope:
        include_scope = False
        drop = 0
        text = _assemble(drop)
        while estimate_tokens(text) > budget and drop < len(deltas):
            drop += 1
            text = _assemble(drop)
    if estimate_tokens(text) > budget and include_viol:
        include_viol = False
        drop = 0
        text = _assemble(drop)
        while estimate_tokens(text) > budget and drop < len(deltas):
            drop += 1
            text = _assemble(drop)
    assert_no_gold(text)
    return text


def count_fenced_kernels(text: str) -> int:
    return text.count("```kernel")


def parent_family_from_cell(cell: dict[str, Any] | None) -> str:
    """Infer the parent family of a mem_* child from copied flags."""
    row = cell or {}
    if int(row.get("enf") or 0):
        return "enf"
    if int(row.get("stream") or 0):
        return "stream"
    dse = str(row.get("dse") or "").strip()
    if dse in {"v2", "2", "dse_v2"}:
        return "dse_v2"
    if dse:
        return "dse_v1"
    if str(row.get("skill") or "") == "one_shot":
        return "oneshot"
    explicit = str(row.get("parent_family") or "").strip()
    if explicit in _SEED_BY_FAMILY:
        return explicit
    return "flash"


def resolve_mem_seed(
    cell_dir: Path,
    parent_family: str | None = None,
) -> tuple[Path, Path]:
    pairs: list[tuple[str, str]] = []
    if parent_family:
        pairs.extend(_SEED_BY_FAMILY.get(parent_family, ()))
    for item in _SEED_CPP_REPORTS:
        if item not in pairs:
            pairs.append(item)
    for cpp_name, rpt_name in pairs:
        cpp = cell_dir / cpp_name
        rpt = cell_dir / rpt_name
        if cpp.is_file() and rpt.is_file():
            return cpp, rpt
    raise FileNotFoundError(f"no mem-iter seed kernel+report under {cell_dir}")


def call_mem_llm(orchestrator: Any, messages: list[dict[str, str]], max_tokens: int) -> tuple[str, str]:
    """Invoke the LLM; retry transient timeouts instead of burning a mem iter."""
    from post_flash_dse import _invoke_llm

    retries = mem_llm_timeout_retries()
    backoff = mem_llm_retry_backoff_s()
    last_err = "empty LLM reply"
    for attempt in range(max(1, retries)):
        try:
            with _llm_lock():
                raw = _invoke_llm(orchestrator, messages, max_tokens)
        except Exception as exc:
            last_err = str(exc)[:2000]
            if is_transient_llm_error(last_err) and attempt + 1 < retries:
                delay = min(DEFAULT_RETRY_BACKOFF_CAP_S, backoff * (2 ** attempt))
                _LOG.warning(
                    "[%s] transient LLM error attempt %d/%d: %s; sleep %.1fs",
                    STEP_TAG,
                    attempt + 1,
                    retries,
                    last_err[:200],
                    delay,
                )
                if delay:
                    time.sleep(delay)
                continue
            return "", last_err
        if raw is None:
            last_err = "empty LLM reply"
            if attempt + 1 < retries:
                delay = min(DEFAULT_RETRY_BACKOFF_CAP_S, backoff * (2 ** attempt))
                _LOG.warning(
                    "[%s] empty LLM reply attempt %d/%d; sleep %.1fs",
                    STEP_TAG,
                    attempt + 1,
                    retries,
                    delay,
                )
                if delay:
                    time.sleep(delay)
                continue
            return "", last_err
        if attempt:
            _LOG.info(
                "[%s] LLM reply on retry %d/%d (len=%d)",
                STEP_TAG,
                attempt + 1,
                retries,
                len(raw) if isinstance(raw, str) else len(str(raw)),
            )
        return (raw if isinstance(raw, str) else str(raw)), ""
    return "", last_err


def _write_mem_trajectory(cell_dir: Path, rows: list[dict[str, Any]], *, n_rounds: int) -> None:
    winner = select_winner(rows)
    traj = {
        "schema": "autosa_mm_mem_iter_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rounds": n_rounds,
        "winner_iter": None if winner is None else winner.get("iter"),
        "iters": [
            {k: v for k, v in row.items() if k != "kernel"}
            | {"kernel": row.get("kernel") or ""}
            for row in rows
        ],
    }
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "autosa_mm_mem_trajectory.json").write_text(
        json.dumps(traj, indent=2, default=str) + "\n", encoding="utf-8"
    )


def _load_mem_trajectory(cell_dir: Path) -> list[dict[str, Any]]:
    path = cell_dir / "autosa_mm_mem_trajectory.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = data.get("iters") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


def _csim_pass(summary: Optional[dict[str, Any]]) -> bool:
    if summary is None:
        return True
    if summary.get("passed") is True or summary.get("success") is True:
        return True
    return str(summary.get("status") or "").lower() in {"pass", "passed", "ok", "success"}


def run_mem_iter_for_cell(
    *,
    bench: str,
    bench_dir: Path,
    cell_dir: Path,
    orchestrator: Any,
    rounds: Optional[int] = None,
    parent_family: Optional[str] = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    from c2hls import _load_benchmark_inputs, _run_synth_csim_cosim, compile_check_cpp
    from c2hls_temp import join_temp_tag
    from post_flash_dataflow import extract_kernel_block

    n_rounds = rounds if rounds is not None else mem_iter_rounds()
    selected_rpt = cell_dir / "autosa_mm_mem_report.json"
    if skip_existing and selected_rpt.is_file():
        try:
            existing = json.loads(selected_rpt.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if isinstance(existing, dict) and (
            existing.get("latency_cycles") is not None or existing.get("dsp") is not None
        ):
            return {
                "success": True,
                "skipped": True,
                "iter": existing.get("iter"),
                "latency_cycles": existing.get("latency_cycles"),
                "latency_cycles_worst": existing.get("latency_cycles_worst"),
                "dsp": existing.get("dsp"),
                "selected": str(cell_dir / "autosa_mm_mem_selected.cpp"),
            }
    seed_cpp, seed_rpt = resolve_mem_seed(cell_dir, parent_family=parent_family)
    seed_code = seed_cpp.read_text(encoding="utf-8")
    seed_report = json.loads(seed_rpt.read_text(encoding="utf-8"))
    seed_metrics = extract_metrics_from_report(seed_report, status="ok")
    seed_metrics["iter"] = 0
    seed_metrics["kernel"] = seed_code
    seed_metrics["note"] = "seed"
    prior = _load_mem_trajectory(cell_dir)
    rows: list[dict[str, Any]] = prior if prior else [seed_metrics]
    start_round = max(int(r.get("iter") or 0) for r in rows) + 1 if prior else 1

    inputs = _load_benchmark_inputs(str(bench_dir))
    header_code = inputs.get("header_code", "")
    header_name = inputs.get("header_name") or "kernel.h"
    meta = inputs["meta"]
    top_function = (
        meta.get("translated_hls_top")
        or meta.get("hls_top")
        or meta.get("kernel_top")
        or "autosa_mm"
    )
    extra_files = inputs.get("extra_files", [])
    testbench_code = inputs.get("testbench_code", "")
    part = meta.get("part", getattr(orchestrator, "part", None))
    clock_ns = meta.get("clock_ns", getattr(orchestrator, "clock_ns", None))
    token_floor = mem_iter_max_tokens()
    if (getattr(orchestrator, "max_completion_tokens", 0) or 0) < token_floor:
        orchestrator.max_completion_tokens = token_floor

    for round_idx in range(start_round, n_rounds + 1):
        user = build_mem_prompt(rows)
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ]
        reply, llm_err = call_mem_llm(orchestrator, messages, token_floor)
        kernel_code = extract_kernel_block(reply) if reply else ""
        row: dict[str, Any] = {
            "iter": round_idx,
            "status": "ok",
            "kernel": kernel_code,
            "error": "",
            "note": "",
        }
        if llm_err and not kernel_code:
            row["status"] = "llm_fail"
            row["error"] = llm_err
            rows.append(row)
            _write_mem_trajectory(cell_dir, rows, n_rounds=n_rounds)
            continue
        if not kernel_code:
            row["status"] = "extract_fail"
            row["error"] = "LLM response missing ```kernel``` block"
            rows.append(row)
            _write_mem_trajectory(cell_dir, rows, n_rounds=n_rounds)
            continue
        ok, err = compile_check_cpp(
            kernel_code,
            header_code,
            header_name,
            extra_files=extra_files,
        )
        if not ok:
            row["status"] = "compile_fail"
            row["error"] = str(err or "compile failed")[:2000]
            rows.append(row)
            _write_mem_trajectory(cell_dir, rows, n_rounds=n_rounds)
            continue
        if not testbench_code:
            row["status"] = "csim_fail"
            row["error"] = "benchmark has no testbench for csim"
            rows.append(row)
            _write_mem_trajectory(cell_dir, rows, n_rounds=n_rounds)
            continue
        tag = join_temp_tag(bench, STEP_TAG, f"r{round_idx:02d}")
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
        if not synth.get("success"):
            row["status"] = "synth_fail"
            row["error"] = str(synth.get("error") or "csynth failed")[:2000]
            rows.append(row)
            _write_mem_trajectory(cell_dir, rows, n_rounds=n_rounds)
            continue
        if not _csim_pass(csim_summary):
            metrics = extract_metrics_from_report(synth, status="csim_fail")
            metrics["iter"] = round_idx
            metrics["kernel"] = kernel_code
            metrics["error"] = str((csim_summary or {}).get("error") or "csim failed")[:2000]
            metrics["csim"] = "fail"
            rows.append(metrics)
            _write_mem_trajectory(cell_dir, rows, n_rounds=n_rounds)
            continue
        metrics = extract_metrics_from_report(synth, status="ok")
        metrics["iter"] = round_idx
        metrics["kernel"] = kernel_code
        metrics["csim"] = "pass"
        metrics["note"] = "ok"
        rows.append(metrics)
        _write_mem_trajectory(cell_dir, rows, n_rounds=n_rounds)

    winner = select_winner(rows)
    cell_dir.mkdir(parents=True, exist_ok=True)
    _write_mem_trajectory(cell_dir, rows, n_rounds=n_rounds)
    if winner is None:
        return {"success": False, "error": "no legal csim-pass mem-iter kernel", "rows": len(rows)}
    selected_cpp = cell_dir / "autosa_mm_mem_selected.cpp"
    selected_rpt = cell_dir / "autosa_mm_mem_report.json"
    selected_cpp.write_text(str(winner.get("kernel") or ""), encoding="utf-8")
    report = {
        "latency_cycles": winner.get("latency_cycles"),
        "latency_cycles_worst": winner.get("latency_cycles_worst"),
        "dsp": winner.get("dsp"),
        "bram": winner.get("bram"),
        "ff": winner.get("ff"),
        "lut": winner.get("lut"),
        "uram": winner.get("uram"),
        "csim": {"passed": True},
        "success": True,
        "iter": winner.get("iter"),
        "feedback": {
            "scopes": winner.get("scopes") or [],
            "scheduler_blame": [],
        },
    }
    selected_rpt.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {
        "success": True,
        "iter": winner.get("iter"),
        "latency_cycles": winner.get("latency_cycles"),
        "latency_cycles_worst": winner.get("latency_cycles_worst"),
        "dsp": winner.get("dsp"),
        "selected": str(selected_cpp),
    }
