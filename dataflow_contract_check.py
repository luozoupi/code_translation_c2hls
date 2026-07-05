"""Hybrid DATAFLOW contract checking before csynth.

Runs deterministic structural checks plus an optional LLM auditor pass.
Breaches use schema ``dataflow_contract_breach_v1``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

CONTRACT_BREACH_SCHEMA = "dataflow_contract_breach_v1"
DEFAULT_CONTRACT_ROUNDS = 4

RULE_FIX_SKILL: dict[str, str] = {
    "dataflow-pragma-missing": "hls-dataflow-structure",
    "dataflow-min-tasks": "hls-dataflow-structure",
    "m_axi-in-compute-task": "hls-distinct-gmem-bundle-per-port",
    "tile-loop-m_axi-in-dataflow": "hls-dual-layout-fused-load-dataflow",
    "local-buffer-multi-writer": "hls-dataflow-fused-compute-phases",
    "local-buffer-fanout": "hls-dataflow-merge-parallel-consumers",
    "m_axi-bundle-multi-reader": "hls-distinct-gmem-bundle-per-port",
    "m_axi-bundle-multi-writer": "hls-distinct-gmem-bundle-per-port",
    "m_axi-port-concurrent-rw": "hls-distinct-gmem-bundle-per-port",
    "inline-copy-in-dataflow": "hls-dataflow-fused-compute-phases",
    "timestep-multi-compute": "hls-dataflow-fused-compute-phases",
    "dual-layout-unfused-load": "hls-dual-layout-fused-load-dataflow",
}

_CONTRACT_AUDIT_SYSTEM = """You are a Vitis HLS DATAFLOW **contract auditor**.

Review the kernel against the mandatory DATAFLOW rules below. Report **only** structural contract violations that would cause HLS 200-779, 200-979, 200-971, 200-1013, or 200-984 — not style or performance nits.

## Mandatory rules (audit checklist)
1. Top kernel body contains exactly one `#pragma HLS DATAFLOW` with ≥3 static task calls.
2. Each `m_axi` bundle: ≤1 concurrent reader task and ≤1 concurrent writer task among DATAFLOW processes.
3. Every on-chip local array crossing concurrent tasks: exactly **one writer** and **one reader** (no fan-out).
4. No `for (tile…)` or `for (t…)` loop **inside** `#pragma HLS DATAFLOW` whose body calls tasks that read/write `m_axi` ports.
5. No top-level `m_axi` port pointers passed into compute tasks while another concurrent task accesses that port.
6. Dual-pass matrix kernels: one fused load fills all layouts; no per-tile `load_*` on the same port inside DATAFLOW.
7. No inline copy `for` loops inside DATAFLOW to duplicate locals for another task.
8. Time-step loops must be inside a single compute task, not split across concurrent compute tasks.

## Output — JSON only
Return **one** fenced block:
```json
{
  "schema": "dataflow_contract_breach_v1",
  "passed": true,
  "breaches": []
}
```

Each breach object must include:
- `rule_id` (string, use ids from the checklist themes above)
- `severity` (`error` or `warning`)
- `symbol` (array/local/port name or null)
- `tasks` (list of task function names involved)
- `location` (short human location)
- `message` (one sentence)
- `fix_skill_id` (matching skill id or null)
- `source`: `"llm"`

If no breaches, set `"passed": true` and `"breaches": []`.
Do **not** return kernel code — audit only.
"""

_CONTRACT_AUDIT_USER = """Audit this DATAFLOW kernel for mandatory contract violations.

## Kernel
```cpp
{kernel_code}
```

Return the JSON contract report only (```json``` fence).
"""

_CONTRACT_FIX_USER_LEGACY = """Fix **DATAFLOW contract breaches** in the kernel below.

Keep the exact top-level `extern "C"` signature and INTERFACE pragmas.
**Keep `#pragma HLS DATAFLOW`** — fix task topology only.

## Contract breaches (must all be resolved)
```json
{breaches_json}
```

Re-read the mandatory DATAFLOW rules in the system message and the pre-output checklist.
Merge shared consumers into fused compute tasks; fuse tile loads; use `hls::stream` for legal fan-out.

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

Return a corrected single ```kernel``` block.
"""

_CONTRACT_FIX_USER_RICH = """## Task — fix DATAFLOW contract breaches

Keep the exact top-level `extern "C"` signature and INTERFACE pragmas.
**Keep `#pragma HLS DATAFLOW`** — fix task topology only.

Apply every matching FLASH skill below when resolving breaches.

{skills_block}

## Contract breaches (must all be resolved)
```json
{breaches_json}
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

Return a corrected single ```kernel``` block.
"""


def contract_round_limit() -> int:
    try:
        return max(
            1,
            int(os.getenv("C2HLS_DATAFLOW_CONTRACT_ROUNDS", str(DEFAULT_CONTRACT_ROUNDS))),
        )
    except ValueError:
        return DEFAULT_CONTRACT_ROUNDS


def contract_check_enabled() -> bool:
    raw = os.getenv("C2HLS_DATAFLOW_CONTRACT_CHECK", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


@dataclass
class ContractBreach:
    rule_id: str
    severity: str
    symbol: Optional[str]
    tasks: list[str]
    location: str
    message: str
    fix_skill_id: Optional[str]
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "symbol": self.symbol,
            "tasks": list(self.tasks),
            "location": self.location,
            "message": self.message,
            "fix_skill_id": self.fix_skill_id,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, default_source: str = "static") -> Optional["ContractBreach"]:
        if not isinstance(raw, dict):
            return None
        rule_id = str(raw.get("rule_id") or "").strip()
        if not rule_id:
            return None
        tasks_raw = raw.get("tasks") or []
        tasks = [str(t) for t in tasks_raw] if isinstance(tasks_raw, list) else []
        symbol = raw.get("symbol")
        if symbol is not None:
            symbol = str(symbol)
        return cls(
            rule_id=rule_id,
            severity=str(raw.get("severity") or "error"),
            symbol=symbol,
            tasks=tasks,
            location=str(raw.get("location") or ""),
            message=str(raw.get("message") or ""),
            fix_skill_id=(
                str(raw["fix_skill_id"])
                if raw.get("fix_skill_id")
                else RULE_FIX_SKILL.get(rule_id)
            ),
            source=str(raw.get("source") or default_source),
        )


@dataclass
class ContractReport:
    passed: bool
    breaches: list[ContractBreach] = field(default_factory=list)
    static_count: int = 0
    llm_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_BREACH_SCHEMA,
            "passed": self.passed,
            "breaches": [b.to_dict() for b in self.breaches],
            "static_count": self.static_count,
            "llm_count": self.llm_count,
        }


def _find_matching_brace(text: str, open_index: int) -> int:
    depth = 0
    for idx in range(open_index, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _extract_top_function_body(code: str, top_function: Optional[str] = None) -> tuple[str, str]:
    """Return (top_function_name, function_body_including_outer_braces)."""
    if top_function:
        match = re.search(
            rf"\bvoid\s+{re.escape(top_function)}\s*\([^)]*\)\s*\{{",
            code,
            flags=re.DOTALL,
        )
        if match:
            brace = match.end() - 1
            end = _find_matching_brace(code, brace)
            if end >= 0:
                return top_function, code[brace:end + 1]
    for match in re.finditer(r"\bvoid\s+(\w+)\s*\([^)]*\)\s*\{", code):
        name = match.group(1)
        if name in {"load_A_task", "load_B_task"}:
            continue
        brace = match.end() - 1
        end = _find_matching_brace(code, brace)
        if end < 0:
            continue
        body = code[brace:end + 1]
        if re.search(r"#pragma\s+HLS\s+INTERFACE", body):
            return name, body
    return "", ""


def _parse_m_axi_ports(code: str) -> dict[str, str]:
    """Map port name -> bundle name from top-level INTERFACE pragmas."""
    ports: dict[str, str] = {}
    for match in re.finditer(
        r"#pragma\s+HLS\s+INTERFACE\s+m_axi\s+port=(\w+)[^\n]*bundle=(\w+)",
        code,
    ):
        ports[match.group(1)] = match.group(2)
    return ports


def _parse_top_locals(body: str, before: str) -> set[str]:
    locals_set: set[str] = set()
    prefix = before if before else body
    for match in re.finditer(
        r"\b(?:double|float|int|ap_\w+)\s+(local_\w+)",
        prefix,
    ):
        locals_set.add(match.group(1))
    return locals_set


def _extract_static_functions(code: str) -> dict[str, str]:
    funcs: dict[str, str] = {}
    for match in re.finditer(
        r"static\s+void\s+(\w+)\s*\([^)]*\)\s*\{",
        code,
        flags=re.DOTALL,
    ):
        name = match.group(1)
        brace = match.end() - 1
        end = _find_matching_brace(code, brace)
        if end >= 0:
            funcs[name] = code[brace:end + 1]
    return funcs


def _locals_written_in_body(body: str, params: list[str]) -> set[str]:
    written: set[str] = set()
    for param in params:
        if not param.startswith("local_"):
            continue
        if re.search(
            rf"\b{re.escape(param)}\s*(?:\[[^\]]+\])+\s*[\+\-\*]?=",
            body,
        ):
            written.add(param)
        elif re.search(rf"(?<!\.)\b{re.escape(param)}\s*[\+\-\*]?=", body):
            written.add(param)
    return written


def _classify_task_io(
    task_name: str,
    func_body: str,
    call_args: list[str],
    top_locals: set[str],
    m_axi_ports: set[str],
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Return (ports_read, ports_written, locals_written, locals_read)."""
    ports_read: set[str] = set()
    ports_written: set[str] = set()
    locals_written: set[str] = set()
    locals_read: set[str] = set()

    if task_name.startswith("load_"):
        for arg in call_args:
            if arg in m_axi_ports:
                ports_read.add(arg)
            elif arg in top_locals:
                locals_written.add(arg)
    elif task_name.startswith("store_"):
        for idx, arg in enumerate(call_args):
            if idx == 0 and arg in m_axi_ports:
                ports_written.add(arg)
            elif arg in top_locals:
                locals_read.add(arg)
    elif "compute" in task_name:
        params = [a for a in call_args if a in top_locals]
        written_in_compute = _locals_written_in_body(func_body, params)
        for loc in params:
            if loc in written_in_compute:
                locals_written.add(loc)
            else:
                locals_read.add(loc)
    return ports_read, ports_written, locals_written, locals_read


def _dataflow_sections(body: str) -> list[tuple[str, bool]]:
    """Return list of (section_text, is_inside_outer_for_loop)."""
    sections: list[tuple[str, bool]] = []
    # Direct DATAFLOW in top body (not inside for)
    for match in re.finditer(r"#pragma\s+HLS\s+DATAFLOW", body):
        after = body[match.end():]
        # Grab statements until next pragma or closing brace at depth 0
        lines: list[str] = []
        depth = 0
        for line in after.splitlines():
            stripped = line.strip()
            if not lines and not stripped:
                continue
            if depth == 0 and stripped.startswith("#pragma"):
                break
            if depth == 0 and stripped == "}":
                break
            lines.append(line)
            depth += line.count("{") - line.count("}")
            if depth < 0:
                break
        sections.append(("\n".join(lines), False))

    # DATAFLOW inside for-loops (tile/timestep hazard)
    for match in re.finditer(r"\bfor\s*\([^)]*\)\s*\{", body):
        loop_brace = match.end() - 1
        loop_end = _find_matching_brace(body, loop_brace)
        if loop_end < 0:
            continue
        loop_body = body[loop_brace:loop_end + 1]
        if "#pragma HLS DATAFLOW" in loop_body or "#pragma HLS DATAFLOW" in loop_body.replace(
            " ", ""
        ):
            df_match = re.search(r"#pragma\s+HLS\s+DATAFLOW", loop_body)
            if not df_match:
                continue
            after = loop_body[df_match.end():]
            lines = []
            depth = 0
            for line in after.splitlines():
                stripped = line.strip()
                if not lines and not stripped:
                    continue
                if depth == 0 and stripped == "}":
                    break
                lines.append(line)
                depth += line.count("{") - line.count("}")
            sections.append(("\n".join(lines), True))
    return sections


def _parse_task_calls(section: str) -> list[tuple[str, list[str]]]:
    calls: list[tuple[str, list[str]]] = []
    for match in re.finditer(r"(\w+)\s*\(([^)]*)\)\s*;", section):
        name = match.group(1)
        if name in {"if", "for", "while"}:
            continue
        raw_args = match.group(2)
        args = [a.strip() for a in raw_args.split(",") if a.strip()]
        calls.append((name, args))
    return calls


def static_contract_check(
    kernel_code: str,
    *,
    top_function: Optional[str] = None,
) -> ContractReport:
    """Run deterministic DATAFLOW contract checks."""
    breaches: list[ContractBreach] = []
    code = kernel_code or ""
    top_name, body = _extract_top_function_body(code, top_function)
    if not body:
        breaches.append(
            ContractBreach(
                rule_id="dataflow-pragma-missing",
                severity="error",
                symbol=None,
                tasks=[],
                location="top kernel",
                message="Could not locate top kernel function body for contract check.",
                fix_skill_id=RULE_FIX_SKILL["dataflow-pragma-missing"],
                source="static",
            )
        )
        return ContractReport(passed=False, breaches=breaches, static_count=len(breaches))

    if not re.search(r"#pragma\s+HLS\s+DATAFLOW", body):
        breaches.append(
            ContractBreach(
                rule_id="dataflow-pragma-missing",
                severity="error",
                symbol=None,
                tasks=[],
                location=f"{top_name} body",
                message="Missing `#pragma HLS DATAFLOW` in top kernel body.",
                fix_skill_id=RULE_FIX_SKILL["dataflow-pragma-missing"],
                source="static",
            )
        )

    m_axi_map = _parse_m_axi_ports(code)
    m_axi_ports = set(m_axi_map.keys())
    df_idx = body.find("#pragma HLS DATAFLOW")
    prefix = body[:df_idx] if df_idx >= 0 else body
    top_locals = _parse_top_locals(body, prefix)
    static_funcs = _extract_static_functions(code)

    sections = _dataflow_sections(body)
    if not sections and "#pragma HLS DATAFLOW" in body:
        after = body.split("#pragma HLS DATAFLOW", 1)[1]
        sections = [(after, False)]

    all_calls: list[tuple[str, list[str], bool]] = []
    for section_text, in_outer_for in sections:
        for name, args in _parse_task_calls(section_text):
            all_calls.append((name, args, in_outer_for))

    task_calls = [(n, a) for n, a, _ in all_calls]
    if task_calls and len(task_calls) < 3:
        breaches.append(
            ContractBreach(
                rule_id="dataflow-min-tasks",
                severity="error",
                symbol=None,
                tasks=[n for n, _ in task_calls],
                location="DATAFLOW region",
                message=f"Only {len(task_calls)} task call(s) under DATAFLOW; need ≥3 (load/compute/store).",
                fix_skill_id=RULE_FIX_SKILL["dataflow-min-tasks"],
                source="static",
            )
        )

    # Inline copy loop inside DATAFLOW (heuristic: for-loop between task calls)
    for section_text, _ in sections:
        if re.search(
            r"for\s*\([^)]*\)\s*\{[^}]*\blocal_\w+",
            section_text,
            flags=re.DOTALL,
        ) and not re.search(r"\w+_task\s*\(", section_text.split("for", 1)[0]):
            # for loop with local assignment that is not itself a labeled task function call
            if re.search(r"for\s*\([^)]*\)\s*\{", section_text):
                breaches.append(
                    ContractBreach(
                        rule_id="inline-copy-in-dataflow",
                        severity="error",
                        symbol=None,
                        tasks=[],
                        location="DATAFLOW region",
                        message="Inline `for` loop copying locals inside DATAFLOW — use fused compute or streams.",
                        fix_skill_id=RULE_FIX_SKILL["inline-copy-in-dataflow"],
                        source="static",
                    )
                )
                break

    local_writers: dict[str, list[str]] = {}
    local_readers: dict[str, list[str]] = {}
    bundle_readers: dict[str, list[str]] = {}
    bundle_writers: dict[str, list[str]] = {}
    port_read_tasks: dict[str, list[str]] = {}
    port_write_tasks: dict[str, list[str]] = {}

    for task_name, args, in_outer_for in all_calls:
        func_body = static_funcs.get(task_name, "")
        ports_read, ports_written, locals_written, locals_read = _classify_task_io(
            task_name,
            func_body,
            args,
            top_locals,
            m_axi_ports,
        )

        if task_name.startswith("load_"):
            for loc in locals_written:
                local_writers.setdefault(loc, []).append(task_name)
        elif task_name.startswith("store_"):
            for loc in locals_read:
                local_readers.setdefault(loc, []).append(task_name)
        elif "compute" in task_name:
            for loc in locals_read:
                local_readers.setdefault(loc, []).append(task_name)
            for loc in locals_written:
                local_writers.setdefault(loc, []).append(task_name)

        for port in ports_read:
            bundle = m_axi_map.get(port, port)
            bundle_readers.setdefault(bundle, []).append(task_name)
            port_read_tasks.setdefault(port, []).append(task_name)
        for port in ports_written:
            bundle = m_axi_map.get(port, port)
            bundle_writers.setdefault(bundle, []).append(task_name)
            port_write_tasks.setdefault(port, []).append(task_name)

        if in_outer_for and (ports_read or ports_written):
            breaches.append(
                ContractBreach(
                    rule_id="tile-loop-m_axi-in-dataflow",
                    severity="error",
                    symbol=next(iter(ports_read | ports_written), None),
                    tasks=[task_name],
                    location="for-loop wrapping DATAFLOW",
                    message=(
                        f"Task `{task_name}` accesses `m_axi` inside a `for` loop that wraps "
                        "`#pragma HLS DATAFLOW` — fuse loads or stream tiles."
                    ),
                    fix_skill_id=RULE_FIX_SKILL["tile-loop-m_axi-in-dataflow"],
                    source="static",
                )
            )

        # m_axi port in compute task args
        for arg in args:
            if arg in m_axi_ports and "compute" in task_name:
                breaches.append(
                    ContractBreach(
                        rule_id="m_axi-in-compute-task",
                        severity="error",
                        symbol=arg,
                        tasks=[task_name],
                        location=f"{task_name} signature/call",
                        message=f"Compute task `{task_name}` takes `m_axi` port `{arg}` — use locals only.",
                        fix_skill_id=RULE_FIX_SKILL["m_axi-in-compute-task"],
                        source="static",
                    )
                )

    for loc, writers in local_writers.items():
        uniq_writers = sorted(set(writers))
        if len(uniq_writers) > 1:
            breaches.append(
                ContractBreach(
                    rule_id="local-buffer-multi-writer",
                    severity="error",
                    symbol=loc,
                    tasks=uniq_writers,
                    location="DATAFLOW task calls",
                    message=(
                        f"Local `{loc}` has multiple writer tasks {uniq_writers} — "
                        "use fused load/compute or ping-pong buffers."
                    ),
                    fix_skill_id=RULE_FIX_SKILL["local-buffer-multi-writer"],
                    source="static",
                )
            )

    for loc in top_locals:
        readers = sorted(set(local_readers.get(loc, [])))
        writers = sorted(set(local_writers.get(loc, [])))
        if len(readers) > 1:
            breaches.append(
                ContractBreach(
                    rule_id="local-buffer-fanout",
                    severity="error",
                    symbol=loc,
                    tasks=readers + writers,
                    location="DATAFLOW task calls",
                    message=(
                        f"Local `{loc}` is read by multiple concurrent tasks {readers} — "
                        "merge consumers or use `hls::stream`."
                    ),
                    fix_skill_id=RULE_FIX_SKILL["local-buffer-fanout"],
                    source="static",
                )
            )

    for bundle, readers in bundle_readers.items():
        uniq = sorted(set(readers))
        if len(uniq) > 1:
            breaches.append(
                ContractBreach(
                    rule_id="m_axi-bundle-multi-reader",
                    severity="error",
                    symbol=bundle,
                    tasks=uniq,
                    location="DATAFLOW region",
                    message=(
                        f"Bundle `{bundle}` has concurrent reader tasks {uniq} — "
                        "fuse loads or assign distinct gmemN bundles."
                    ),
                    fix_skill_id=RULE_FIX_SKILL["m_axi-bundle-multi-reader"],
                    source="static",
                )
            )

    for bundle, writers in bundle_writers.items():
        uniq = sorted(set(writers))
        if len(uniq) > 1:
            breaches.append(
                ContractBreach(
                    rule_id="m_axi-bundle-multi-writer",
                    severity="error",
                    symbol=bundle,
                    tasks=uniq,
                    location="DATAFLOW region",
                    message=(
                        f"Bundle `{bundle}` has concurrent writer tasks {uniq} — "
                        "fuse stores or use distinct gmemN bundles."
                    ),
                    fix_skill_id=RULE_FIX_SKILL["m_axi-bundle-multi-writer"],
                    source="static",
                )
            )

    for port in m_axi_ports:
        readers = sorted(set(port_read_tasks.get(port, [])))
        writers = sorted(set(port_write_tasks.get(port, [])))
        if readers and writers:
            breaches.append(
                ContractBreach(
                    rule_id="m_axi-port-concurrent-rw",
                    severity="error",
                    symbol=port,
                    tasks=readers + writers,
                    location="DATAFLOW region",
                    message=(
                        f"Port `{port}` has concurrent reader(s) {readers} and writer(s) {writers} — "
                        "split read/write phases or fuse tasks."
                    ),
                    fix_skill_id=RULE_FIX_SKILL["m_axi-port-concurrent-rw"],
                    source="static",
                )
            )

    # Deduplicate breaches by (rule_id, symbol, tuple(tasks))
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    unique: list[ContractBreach] = []
    for breach in breaches:
        key = (breach.rule_id, breach.symbol or "", tuple(sorted(breach.tasks)))
        if key in seen:
            continue
        seen.add(key)
        unique.append(breach)

    return ContractReport(
        passed=len(unique) == 0,
        breaches=unique,
        static_count=len(unique),
        llm_count=0,
    )


def extract_contract_json_block(text: str) -> dict[str, Any]:
    """Parse ```json ... ``` contract report from LLM output."""
    if not text:
        return {}
    blocks = re.findall(r"```json\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if not blocks:
        blocks = re.findall(r"```\s*(.*?)```", text, flags=re.DOTALL)
    for raw in blocks:
        raw = raw.strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("schema") == CONTRACT_BREACH_SCHEMA:
            return data
        if isinstance(data, dict) and "breaches" in data:
            data.setdefault("schema", CONTRACT_BREACH_SCHEMA)
            return data
    return {}


def parse_llm_contract_report(text: str) -> ContractReport:
    data = extract_contract_json_block(text)
    breaches: list[ContractBreach] = []
    for raw in data.get("breaches") or []:
        breach = ContractBreach.from_dict(raw, default_source="llm")
        if breach is not None:
            breaches.append(breach)
    passed = bool(data.get("passed")) if data else False
    if data and not passed and not breaches:
        passed = False
    elif data and passed:
        passed = len(breaches) == 0
    return ContractReport(
        passed=passed and len(breaches) == 0,
        breaches=breaches,
        static_count=0,
        llm_count=len(breaches),
    )


def merge_contract_reports(static: ContractReport, llm: ContractReport) -> ContractReport:
    merged: list[ContractBreach] = list(static.breaches)
    seen = {
        (b.rule_id, b.symbol or "", tuple(sorted(b.tasks)), b.message[:80])
        for b in merged
    }
    for breach in llm.breaches:
        key = (breach.rule_id, breach.symbol or "", tuple(sorted(breach.tasks)), breach.message[:80])
        if key in seen:
            continue
        seen.add(key)
        merged.append(breach)
    return ContractReport(
        passed=len(merged) == 0,
        breaches=merged,
        static_count=static.static_count,
        llm_count=llm.llm_count,
    )


def hybrid_contract_check(
    kernel_code: str,
    *,
    top_function: Optional[str] = None,
    llm_report: Optional[ContractReport] = None,
) -> ContractReport:
    static = static_contract_check(kernel_code, top_function=top_function)
    if llm_report is None:
        return static
    return merge_contract_reports(static, llm_report)


def format_contract_breaches_json(report: ContractReport) -> str:
    return json.dumps(report.to_dict(), indent=2)


def format_contract_audit_user(kernel_code: str) -> str:
    return _CONTRACT_AUDIT_USER.format(kernel_code=kernel_code[:120000])


def format_contract_fix_user(
    *,
    prompt_policy: str,
    breaches_json: str,
    benchmark_context: str,
    header_name: str,
    header_code: str,
    kernel_code: str,
    skills_block: str = "",
) -> str:
    template = (
        _CONTRACT_FIX_USER_RICH
        if prompt_policy == "user_skills"
        else _CONTRACT_FIX_USER_LEGACY
    )
    kwargs = {
        "breaches_json": breaches_json,
        "benchmark_context": benchmark_context,
        "header_name": header_name,
        "header_code": header_code[:12000],
        "kernel_code": kernel_code[:120000],
        "skills_block": skills_block,
    }
    if prompt_policy == "user_skills":
        return template.format(**kwargs)
    return template.format(**kwargs)


def contract_audit_system_prompt() -> str:
    return _CONTRACT_AUDIT_SYSTEM


def contract_failure_message(report: ContractReport) -> str:
    if report.passed:
        return ""
    lines = [f"DATAFLOW contract check failed ({len(report.breaches)} breach(es)):"]
    for breach in report.breaches[:12]:
        tasks = ", ".join(breach.tasks) if breach.tasks else "—"
        lines.append(
            f"- [{breach.rule_id}] {breach.message} "
            f"(symbol={breach.symbol or '—'}, tasks={tasks}, source={breach.source})"
        )
    if len(report.breaches) > 12:
        lines.append(f"... and {len(report.breaches) - 12} more")
    return "\n".join(lines)
