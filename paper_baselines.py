"""Matched-budget, reference-blind baselines for the HPCA 2027 evaluation.

This module implements the two deliberately simple comparison methods that do
not belong in the agentic controller:

* ``one_shot_best_of_five`` requests five independent full translations.
* ``pragma_only`` requests one full translation followed by four independent
  revisions of that same translation, accepting a revision only when its
  complete non-pragma C/C++ token stream is unchanged.

The search loop has hard local caps in addition to the controller-wide
environment caps.  CSim (including the independent golden comparator when one
is configured) runs before synthesis, every actual synthesis is counted, and
RTL cosim runs once for the selected feasible winner.  Expert/reference files
are not loaded by the search path.  They are consulted only by the post-run
transcript auditor in :func:`finalize_baseline_result`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence

from evaluation_repro import (
    FINGERPRINT_SCHEMA,
    attach_run_provenance,
    build_run_fingerprint,
    canonical_json,
    fingerprint_completeness,
    fingerprint_matches,
    sha256_bytes,
    sha256_json,
)
from reference_isolation import audit_history_file


METHOD_ONE_SHOT = "one_shot_best_of_five"
METHOD_PRAGMA_ONLY = "pragma_only"
SUPPORTED_METHODS = (METHOD_ONE_SHOT, METHOD_PRAGMA_ONLY)
MAX_LLM_CANDIDATES = 5
MAX_SYNTHESIS_EVALUATIONS = 5
BASELINE_SCHEMA = "c2hls.paper-baseline.v1"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _env_flag(environ: Mapping[str, str], name: str, default: str = "0") -> bool:
    return str(environ.get(name, default)).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _positive_budget(environ: Mapping[str, str], name: str, expected: int) -> int:
    try:
        value = int(str(environ.get(name, expected)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be integer {expected}") from exc
    if value != expected:
        raise ValueError(
            f"paper baseline requires {name}={expected}, got {value}"
        )
    return value


def enforce_baseline_contract(environ: Mapping[str, str] | None = None) -> None:
    """Reject an invocation that could violate the matched paper contract."""

    env = os.environ if environ is None else environ
    required = {
        "C2HLS_REFERENCE_BLIND": "1",
        "C2HLS_GT_COMPARISON_IN_CONTROL": "0",
        "C2HLS_REFERENCE_CODE_IN_PROMPTS": "0",
        "C2HLS_REFERENCE_METRICS_IN_PROMPTS": "0",
        "C2HLS_COSIM_SELECTED_ONLY": "1",
        "C2HLS_FORCE_SELECTED_COSIM": "1",
        "C2HLS_FEASIBILITY_SELECTION": "1",
        "C2HLS_CORRECTNESS_BEFORE_SYNTH": "1",
        "C2HLS_TRANSCRIPT_AUDIT": "1",
    }
    mismatches = [
        f"{name}={env.get(name)!r} (expected {expected!r})"
        for name, expected in required.items()
        if str(env.get(name, "")) != expected
    ]
    if mismatches:
        raise ValueError(
            "unsafe paper-baseline environment: " + "; ".join(mismatches)
        )
    _positive_budget(env, "C2HLS_LLM_CANDIDATE_BUDGET", MAX_LLM_CANDIDATES)
    _positive_budget(
        env, "C2HLS_SYNTHESIS_EVAL_BUDGET", MAX_SYNTHESIS_EVALUATIONS
    )


@dataclass(frozen=True)
class PublicBenchmarkInputs:
    benchmark_dir: Path
    benchmark: str
    c_code: str
    header_code: str
    header_name: str
    testbench_code: str
    extra_files: tuple[dict[str, str], ...]
    translated_hls_top: str
    part: str
    clock_ns: float
    cosim_depths: Mapping[str, Any]
    independent_golden_output: str = ""
    independent_golden_specs: Mapping[str, Any] | None = None
    independent_golden_provenance: Mapping[str, Any] | None = None


def load_public_benchmark_inputs(
    benchmark_dir: Path | str,
    *,
    environ: Mapping[str, str] | None = None,
) -> PublicBenchmarkInputs:
    """Load only plain input, public header/testbench, and support files.

    This intentionally does not call ``c2hls._load_benchmark_inputs`` because
    that historical helper also materializes expert variants.  Metadata is
    used only to locate public inputs and target properties; reference-related
    keys are never copied into the returned object.
    """

    env = os.environ if environ is None else environ
    root = Path(benchmark_dir).resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    plain_name = str(metadata.get("plain_c_file") or "plain.cpp")
    header_name = str(metadata.get("header_file") or "kernel.h")
    testbench_name = str(metadata.get("testbench_file") or "")
    plain_path = root / plain_name
    header_path = root / header_name
    testbench_path = root / testbench_name if testbench_name else None
    if not plain_path.is_file():
        raise FileNotFoundError(f"missing pragma-stripped input: {plain_path}")
    if not testbench_path or not testbench_path.is_file():
        raise FileNotFoundError(
            f"paper baseline requires public golden-checking testbench: {testbench_path}"
        )

    extras: list[dict[str, str]] = []
    seen: set[str] = set()
    for value in metadata.get("support_files") or []:
        relative = str(value)
        path = root / relative
        if path.is_file() and relative not in seen:
            extras.append({"path": relative, "content": path.read_text(encoding="utf-8")})
            seen.add(relative)
    support_dir = root / "support"
    if support_dir.is_dir():
        for path in sorted(support_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if relative not in seen:
                extras.append(
                    {"path": relative, "content": path.read_text(encoding="utf-8")}
                )
                seen.add(relative)

    part = str(env.get("C2HLS_PART") or metadata.get("part") or "")
    clock_ns = float(env.get("C2HLS_CLOCK_NS") or metadata.get("clock_ns") or 0)
    if not part or clock_ns <= 0:
        raise ValueError("matched target part and positive clock period are required")

    public = PublicBenchmarkInputs(
        benchmark_dir=root,
        benchmark=str(metadata.get("benchmark") or root.name),
        c_code=plain_path.read_text(encoding="utf-8"),
        header_code=(
            header_path.read_text(encoding="utf-8") if header_path.is_file() else ""
        ),
        header_name=header_name,
        testbench_code=testbench_path.read_text(encoding="utf-8"),
        extra_files=tuple(extras),
        translated_hls_top=str(metadata.get("translated_hls_top") or "workload"),
        part=part,
        clock_ns=clock_ns,
        cosim_depths=dict(metadata.get("cosim_depths") or {}),
    )

    # Reuse the controller's independent CPU-golden path for suites whose
    # public testbench prints outputs instead of checking them internally.
    from c2hls import _prepare_independent_golden

    golden = _prepare_independent_golden(
        {
            "meta": {
                "source_repo": metadata.get("source_repo"),
                "independent_golden_required": metadata.get(
                    "independent_golden_required"
                ),
                "golden_output_specs": metadata.get("golden_output_specs") or {},
                "golden_output_atol": metadata.get("golden_output_atol", 1e-6),
                "golden_output_rtol": metadata.get("golden_output_rtol", 1e-5),
            },
            "c_code": public.c_code,
            "header_code": public.header_code,
            "header_name": public.header_name,
            "testbench_code": public.testbench_code,
            "extra_files": list(public.extra_files),
        }
    )
    if not golden.get("success"):
        raise RuntimeError(
            golden.get("error") or "independent public CPU golden is invalid"
        )
    return PublicBenchmarkInputs(
        **{
            **public.__dict__,
            "independent_golden_output": str(golden.get("output") or ""),
            "independent_golden_specs": dict(golden.get("specs") or {}),
            "independent_golden_provenance": dict(golden.get("provenance") or {}),
        }
    )


_RAW_STRING_START_RE = re.compile(
    r'(?:u8|u|U|L)?R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\('
)


def _mask_comments_preserving_lines(
    source: str, *, mask_literals: bool = False
) -> str:
    """Mask comments, optionally literals, while preserving all newlines."""

    out: list[str] = []
    index = 0
    state = "code"
    quote = ""
    while index < len(source):
        char = source[index]
        nxt = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            raw_match = _RAW_STRING_START_RE.match(source, index)
            if raw_match:
                terminator = ")" + raw_match.group("delimiter") + '"'
                end = source.find(terminator, raw_match.end())
                end = len(source) if end < 0 else end + len(terminator)
                raw_literal = source[index:end]
                out.extend(
                    ("\n" if char == "\n" else " ")
                    if mask_literals
                    else char
                    for char in raw_literal
                )
                index = end
                continue
            if char == "/" and nxt == "/":
                out.extend((" ", " "))
                index += 2
                state = "line_comment"
                continue
            if char == "/" and nxt == "*":
                out.extend((" ", " "))
                index += 2
                state = "block_comment"
                continue
            if char in {'"', "'"}:
                state = "literal"
                quote = char
            out.append(" " if mask_literals and state == "literal" else char)
            index += 1
            continue
        if state == "line_comment":
            if char == "\n":
                out.append("\n")
                # Translation phase 2 removes escaped newlines before comment
                # recognition, so a trailing backslash continues // comments.
                previous = source[index - 1] if index else ""
                state = "line_comment" if previous == "\\" else "code"
            else:
                out.append(" ")
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and nxt == "/":
                out.extend((" ", " "))
                index += 2
                state = "code"
            else:
                out.append("\n" if char == "\n" else " ")
                index += 1
            continue
        # Quoted literal.
        out.append("\n" if char == "\n" else (" " if mask_literals else char))
        index += 1
        if char == "\\" and index < len(source):
            escaped = source[index]
            out.append(
                "\n" if escaped == "\n" else (" " if mask_literals else escaped)
            )
            index += 1
        elif char == quote:
            state = "code"
    return "".join(out)


_PRAGMA_DIRECTIVE_RE = re.compile(r"^[ \t]*#[ \t]*pragma(?:[ \t]|$)")


def strip_pragma_directives(source: str) -> tuple[str, tuple[str, ...]]:
    """Remove complete ``#pragma`` logical lines and return their text.

    Comments are masked only for directive recognition.  Backslash-continued
    pragma directives are removed as a unit, preventing continuation payloads
    from becoming an unguarded semantic-edit channel.
    """

    lines = source.splitlines(keepends=True)
    masked_lines = _mask_comments_preserving_lines(
        source, mask_literals=True
    ).splitlines(keepends=True)
    kept: list[str] = []
    pragmas: list[str] = []
    index = 0
    while index < len(lines):
        logical_end = index + 1
        while logical_end < len(lines) and masked_lines[logical_end - 1].rstrip(
            "\r\n"
        ).rstrip().endswith("\\"):
            logical_end += 1
        raw = "".join(lines[index:logical_end])
        masked = "".join(masked_lines[index:logical_end])
        if _PRAGMA_DIRECTIVE_RE.match(masked):
            pragmas.append(raw)
            # Preserve line count so diagnostics/hash manifests remain useful.
            kept.append("".join("\n" if ch == "\n" else " " for ch in raw))
        else:
            kept.append(raw)
        index = logical_end
    return "".join(kept), tuple(pragmas)


_TOKEN_RE = re.compile(
    r"""
    (?:u8|u|U|L)?R\"(?P<rawdelim>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=rawdelim)\"
    |(?:u8|u|U|L)?\"(?:\\.|[^\"\\])*\"
    |(?:u|U|L)?'(?:\\.|[^'\\])*'
    |[A-Za-z_$][A-Za-z0-9_$]*
    |(?:0[xX][0-9A-Fa-f](?:[0-9A-Fa-f']*[0-9A-Fa-f])?(?:\.[0-9A-Fa-f']*)?(?:[pP][+-]?[0-9']+)?|0[bB][01](?:[01']*[01])?|(?:[0-9](?:[0-9']*[0-9])?\.?[0-9']*|\.[0-9](?:[0-9']*[0-9])?)(?:[eE][+-]?[0-9']+)?)[A-Za-z0-9_]*
    |>>=|<<=|<=>|\.\.\.|->\*|::|\+\+|--|->|<<|>>|<=|>=|==|!=|&&|\|\||\*=|/=|%=|\+=|-=|&=|\^=|\|=|\#\#|\.<|\.\*
    |[^\s]
    """,
    re.VERBOSE | re.DOTALL,
)


def cpp_token_stream(source: str) -> tuple[str, ...]:
    """Return preprocessing-token-like C/C++ tokens, excluding comments."""

    masked = _mask_comments_preserving_lines(source)
    return tuple(match.group(0) for match in _TOKEN_RE.finditer(masked))


def pragma_only_guard(base_code: str, candidate_code: str) -> dict[str, Any]:
    """Prove that a candidate differs only in ``#pragma`` directives."""

    base_nonpragma, base_pragmas = strip_pragma_directives(base_code)
    candidate_nonpragma, candidate_pragmas = strip_pragma_directives(candidate_code)
    base_tokens = cpp_token_stream(base_nonpragma)
    candidate_tokens = cpp_token_stream(candidate_nonpragma)
    mismatch_index: Optional[int] = None
    for index, (left, right) in enumerate(zip(base_tokens, candidate_tokens)):
        if left != right:
            mismatch_index = index
            break
    if mismatch_index is None and len(base_tokens) != len(candidate_tokens):
        mismatch_index = min(len(base_tokens), len(candidate_tokens))
    passed = mismatch_index is None
    context: dict[str, Any] = {}
    if mismatch_index is not None:
        start = max(0, mismatch_index - 3)
        stop = mismatch_index + 4
        # Tokens are public/generated source; hashes avoid duplicating code in
        # result artifacts while still making the rejection reproducible.
        context = {
            "index": mismatch_index,
            "base_context_sha256": sha256_json(base_tokens[start:stop]),
            "candidate_context_sha256": sha256_json(
                candidate_tokens[start:stop]
            ),
        }
    return {
        "schema_version": "c2hls.pragma-only-guard.v1",
        "passed": passed,
        "nonpragma_token_count_base": len(base_tokens),
        "nonpragma_token_count_candidate": len(candidate_tokens),
        "nonpragma_token_sha256_base": sha256_json(base_tokens),
        "nonpragma_token_sha256_candidate": sha256_json(candidate_tokens),
        "pragma_count_base": len(base_pragmas),
        "pragma_count_candidate": len(candidate_pragmas),
        "pragma_sha256_base": sha256_json(base_pragmas),
        "pragma_sha256_candidate": sha256_json(candidate_pragmas),
        "pragma_changed": base_pragmas != candidate_pragmas,
        "mismatch": context or None,
    }


def _translation_messages(inputs: PublicBenchmarkInputs) -> list[dict[str, str]]:
    system = (
        "You translate plain C/C++ kernels to synthesizable Vitis HLS. Work "
        "only from the supplied public specification. Preserve observable "
        "behavior and the top-function signature. Return exactly one complete "
        "C++ source file in a fenced ```cpp block."
    )
    user = (
        f"Target top function: {inputs.translated_hls_top}\n"
        f"Target part: {inputs.part}\n"
        f"Target clock: {inputs.clock_ns} ns\n"
        f"Public header ({inputs.header_name}):\n```cpp\n{inputs.header_code}\n```\n"
        f"Pragma-stripped kernel:\n```cpp\n{inputs.c_code}\n```\n"
        "Produce one correct, optimized, self-contained HLS translation."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _pragma_revision_messages(
    inputs: PublicBenchmarkInputs, initial_translation: str
) -> list[dict[str, str]]:
    system = (
        "You optimize Vitis HLS only by editing #pragma directives. Return "
        "exactly one complete C++ source file in a fenced ```cpp block."
    )
    user = (
        f"Target part: {inputs.part}; target clock: {inputs.clock_ns} ns.\n"
        "Revise the source below using only insertion, deletion, replacement, "
        "or movement of complete #pragma directives. Apart from comments and "
        "whitespace, every non-pragma C/C++ token must remain byte-for-byte "
        "identical and in the same order. Do not alter constants, types, "
        "expressions, declarations, loops, functions, macros, or includes.\n"
        f"```cpp\n{initial_translation}\n```"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@contextmanager
def _temporary_seed(seed: Optional[int]):
    previous = os.environ.get("C2HLS_LLM_SEED")
    try:
        if seed is None:
            os.environ.pop("C2HLS_LLM_SEED", None)
        else:
            os.environ["C2HLS_LLM_SEED"] = str(seed)
        yield
    finally:
        if previous is None:
            os.environ.pop("C2HLS_LLM_SEED", None)
        else:
            os.environ["C2HLS_LLM_SEED"] = previous


LLMRequest = Callable[
    [Sequence[Mapping[str, str]], int, Optional[int], bool], Mapping[str, Any] | str
]
CSimCallback = Callable[[str], Mapping[str, Any]]
SynthCallback = Callable[[str], Mapping[str, Any]]
CosimCallback = Callable[[str], Mapping[str, Any]]
FeasibilityCallback = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


def _extract_cpp(text: str) -> Optional[str]:
    from c2hls import extract_cpp_code

    return extract_cpp_code(text)


def _latency_key(candidate: Mapping[str, Any]) -> tuple[float, int]:
    """Use the same single scalar fallback chain as core C2HLS selection."""
    report = candidate.get("report") or {}
    latency = None
    for name in (
        "latency_cycles_worst",
        "latency_cycles",
        "latency_ns_worst",
        "latency_ns",
    ):
        try:
            parsed = float(report.get(name))
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            latency = parsed
            break
    return (latency if latency is not None else float("inf"), int(candidate.get("index", 0)))


def _candidate_seed_schedule(
    base_seed: int, *, seed_supported: bool
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_index": index,
            "requested_seed": int(base_seed) + index,
            "effective_seed": int(base_seed) + index if seed_supported else None,
            "seed_supported": bool(seed_supported),
        }
        for index in range(MAX_LLM_CANDIDATES)
    ]


class PaperBaselineEngine:
    """Budget-enforcing search engine with dependency-injected tool calls."""

    def __init__(
        self,
        *,
        inputs: PublicBenchmarkInputs,
        method: str,
        model_id: str,
        base_seed: int,
        seed_supported: bool,
        llm_request: LLMRequest,
        csim: CSimCallback,
        synthesize: SynthCallback,
        cosim: CosimCallback,
        feasibility: FeasibilityCallback,
    ) -> None:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"unsupported paper baseline: {method!r}")
        self.inputs = inputs
        self.method = method
        self.model_id = model_id
        self.base_seed = int(base_seed)
        self.seed_supported = bool(seed_supported)
        self.llm_request = llm_request
        self.csim = csim
        self.synthesize = synthesize
        self.cosim = cosim
        self.feasibility = feasibility
        self.llm_count = 0
        self.synthesis_count = 0
        self.csim_count = 0
        self.selected_winner_cosim_count = 0
        self.history: list[dict[str, str]] = []
        self.llm_events: list[dict[str, Any]] = []
        self.synthesis_events: list[dict[str, Any]] = []
        self._search_started_monotonic: float | None = None
        self._last_candidate_elapsed_seconds = 0.0

    def _complete_candidate(self, record: dict[str, Any]) -> dict[str, Any]:
        """Stamp a candidate at the instant its complete outcome is known."""

        if self._search_started_monotonic is None:
            raise RuntimeError("candidate completed outside PaperBaselineEngine.run")
        elapsed = max(
            self._last_candidate_elapsed_seconds,
            time.monotonic() - self._search_started_monotonic,
        )
        # Persist the rounded value used by the paper adapter and retain it as
        # the lower bound for the next candidate.  This keeps completion times
        # nondecreasing even when two fast failure paths round to the same tick.
        elapsed = round(max(0.0, elapsed), 6)
        self._last_candidate_elapsed_seconds = elapsed
        record["cumulative_elapsed_seconds"] = elapsed
        return record

    def _request(
        self, messages: Sequence[Mapping[str, str]], candidate_index: int
    ) -> tuple[str, dict[str, Any]]:
        if self.llm_count >= MAX_LLM_CANDIDATES:
            raise RuntimeError("paper baseline LLM candidate budget exhausted")
        requested_seed = self.base_seed + candidate_index
        effective_seed = requested_seed if self.seed_supported else None
        self.llm_count += 1
        for message in messages:
            self.history.append(
                {"role": str(message["role"]), "content": str(message["content"])}
            )
        started = time.time()
        try:
            response = self.llm_request(
                messages, candidate_index, effective_seed, self.seed_supported
            )
            if isinstance(response, Mapping):
                text = str(response.get("text") or "")
                provider_event = dict(response.get("event") or {})
            else:
                text = str(response)
                provider_event = {}
            error = ""
        except Exception as exc:  # An individual request is a visible failed candidate.
            text = ""
            provider_event = {}
            error = f"{type(exc).__name__}: {exc}"
        self.history.append({"role": "assistant", "content": text})
        event = {
            "candidate_index": candidate_index,
            "prompt_sha256": _sha256_text(canonical_json(list(messages))),
            "response_sha256": _sha256_text(text),
            "response_characters": len(text),
            "requested_seed": requested_seed,
            "effective_seed": effective_seed,
            "seed_supported": self.seed_supported,
            "elapsed_seconds": round(time.time() - started, 6),
            "error": error,
            **provider_event,
        }
        # The baseline contract, not a provider claim, owns these fields.
        event.update(
            {
                "candidate_index": candidate_index,
                "requested_seed": requested_seed,
                "effective_seed": effective_seed,
                "seed_supported": self.seed_supported,
            }
        )
        self.llm_events.append(event)
        return text, event

    def _evaluate(
        self,
        *,
        index: int,
        kind: str,
        response_text: str,
        guard: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        code = _extract_cpp(response_text)
        record: dict[str, Any] = {
            "index": index,
            "kind": kind,
            "response_sha256": _sha256_text(response_text),
            "code_extracted": bool(code),
            "code_sha256": _sha256_text(code or ""),
            "guard": dict(guard or {}),
            "csim": {"status": "not_run", "ran": False, "passed": False},
            "synthesis": {"status": "not_run", "ran": False, "success": False},
            "report": {},
            "feasibility": {
                "schema_version": "c2hls.candidate-feasibility.v1",
                "feasible": False,
                "reasons": ["candidate_not_evaluated"],
            },
        }
        if not code:
            record["rejection_reason"] = "missing_complete_fenced_cpp"
            return self._complete_candidate(record)
        if guard and not guard.get("passed"):
            record["rejection_reason"] = "non_pragma_token_edit"
            return self._complete_candidate(record)

        # Correctness is intentionally before synthesis.  This guarantees
        # that a bad candidate cannot consume scarce Vitis synthesis budget.
        self.csim_count += 1
        try:
            csim = dict(self.csim(code))
        except Exception as exc:
            csim = {
                "status": "tool_error",
                "ran": True,
                "passed": False,
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        record["csim"] = csim
        if not csim.get("passed"):
            record["rejection_reason"] = "csim_or_golden_failed"
            record["feasibility"] = dict(self.feasibility({}, csim))
            return self._complete_candidate(record)
        if self.synthesis_count >= MAX_SYNTHESIS_EVALUATIONS:
            record["rejection_reason"] = "synthesis_budget_exhausted"
            return self._complete_candidate(record)

        synth_index = self.synthesis_count
        self.synthesis_count += 1
        started = time.time()
        synthesis_tool_failure = False
        try:
            synth = dict(self.synthesize(code))
        except Exception as exc:
            synthesis_tool_failure = True
            synth = {
                "success": False,
                "report": {},
                "error": f"{type(exc).__name__}: {exc}",
            }
        elapsed = time.time() - started
        report = dict(synth.get("report") or {})
        synthesis_error = str(synth.get("error") or "")
        synthesis_timed_out = bool(
            synth.get("timed_out")
            or "timed out" in synthesis_error.lower()
        )
        synthesis_status = (
            "timeout"
            if synthesis_timed_out
            else "tool_failure"
            if synthesis_tool_failure or synth.get("tool_failure")
            else "passed"
            if synth.get("success")
            else "failed"
        )
        record["synthesis"] = {
            "status": synthesis_status,
            "ran": True,
            "success": bool(synth.get("success")),
            "error": synthesis_error,
        }
        record["report"] = report
        event = {
            "index": synth_index,
            "candidate_index": index,
            "code_sha256": record["code_sha256"],
            "success": bool(synth.get("success")),
            "status": synthesis_status,
            "error": synthesis_error,
            "timed_out": synthesis_timed_out,
            "tool_failure": bool(
                synthesis_tool_failure or synth.get("tool_failure")
            ),
            "elapsed_seconds": round(elapsed, 6),
        }
        self.synthesis_events.append(event)
        if not synth.get("success"):
            record["rejection_reason"] = "synthesis_failed"
            record["feasibility"] = dict(self.feasibility({}, csim))
            return self._complete_candidate(record)
        record["feasibility"] = dict(self.feasibility(report, csim))
        if not record["feasibility"].get("feasible"):
            record["rejection_reason"] = "paper_feasibility_failed"
        # Code is retained only in memory for winner cosim and emitted to its
        # own source file by the CLI, never duplicated in result JSON.
        record["_code"] = code
        return self._complete_candidate(record)

    def run(self) -> dict[str, Any]:
        if self._search_started_monotonic is not None:
            raise RuntimeError("PaperBaselineEngine instances are single-use")
        self._search_started_monotonic = time.monotonic()
        candidates: list[dict[str, Any]] = []
        if self.method == METHOD_ONE_SHOT:
            messages = _translation_messages(self.inputs)
            for index in range(MAX_LLM_CANDIDATES):
                response, _ = self._request(messages, index)
                candidates.append(
                    self._evaluate(
                        index=index,
                        kind="independent_full_translation",
                        response_text=response,
                    )
                )
        else:
            initial_response, _ = self._request(_translation_messages(self.inputs), 0)
            initial = self._evaluate(
                index=0,
                kind="initial_full_translation",
                response_text=initial_response,
            )
            candidates.append(initial)
            initial_code = initial.get("_code") or _extract_cpp(initial_response)
            if initial_code:
                revision_messages = _pragma_revision_messages(self.inputs, initial_code)
                # Every revision sees the same initial translation and no
                # previous response, making the four trials independent.
                for index in range(1, MAX_LLM_CANDIDATES):
                    response, _ = self._request(revision_messages, index)
                    candidate_code = _extract_cpp(response) or ""
                    guard = pragma_only_guard(initial_code, candidate_code)
                    candidates.append(
                        self._evaluate(
                            index=index,
                            kind="independent_pragma_revision",
                            response_text=response,
                            guard=guard,
                        )
                    )
            else:
                # The method cannot define a pragma-only search space without
                # an initial translation; this is a transparent failed run.
                for index in range(1, MAX_LLM_CANDIDATES):
                    candidates.append(
                        self._complete_candidate({
                            "index": index,
                            "kind": "independent_pragma_revision",
                            "code_extracted": False,
                            "code_sha256": _sha256_text(""),
                            "rejection_reason": "initial_translation_missing",
                            "csim": {"status": "not_run", "ran": False, "passed": False},
                            "synthesis": {"status": "not_run", "ran": False, "success": False},
                            "report": {},
                            "feasibility": {"feasible": False, "reasons": ["initial_translation_missing"]},
                        })
                    )

        feasible = [
            candidate
            for candidate in candidates
            if (candidate.get("feasibility") or {}).get("feasible")
        ]
        winner = min(feasible, key=_latency_key) if feasible else None
        cosim = {
            "status": "not_run",
            "supported": True,
            "ran": False,
            "passed": False,
            "error": "no correct feasible candidate",
        }
        if winner is not None:
            # The selected-winner cosim flow is an additional Vitis synthesis
            # call for paper cost accounting, even if the tool later fails.
            self.selected_winner_cosim_count += 1
            try:
                cosim = dict(self.cosim(str(winner["_code"])))
            except Exception as exc:
                cosim = {
                    "status": "tool_error",
                    "supported": True,
                    "ran": True,
                    "passed": False,
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            # Bind the selected-winner tool result to the exact source passed
            # into the cosim flow. Downstream evidence normalization rejects a
            # stale/mixed cosim record whose target hash differs.
            cosim["target_code_sha256"] = winner["code_sha256"]

        public_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            clean = dict(candidate)
            clean.pop("_code", None)
            public_candidates.append(clean)

        executed_cycles = (
            cosim.get("kernel_runtime_cycles")
            if cosim.get("kernel_runtime_cycles") is not None
            else cosim.get("cycles")
        )
        try:
            executed_cycles_valid = int(executed_cycles) > 0
        except (TypeError, ValueError):
            executed_cycles_valid = False
        cosim_measurement_valid = bool(
            winner is not None
            and cosim.get("ran")
            and cosim.get("passed")
            and executed_cycles_valid
        )
        success = cosim_measurement_valid
        selected_index = int(winner["index"]) if winner is not None else None
        selected_report = dict(winner.get("report") or {}) if winner else {}
        selected_csim = dict(winner.get("csim") or {}) if winner else {}
        candidate_csim_results = [
            candidate.get("csim") or {}
            for candidate in candidates
            if isinstance(candidate.get("csim"), Mapping)
        ]
        if winner is not None and selected_csim.get("passed"):
            correctness_status = "passed"
        elif any(summary.get("passed") for summary in candidate_csim_results):
            # A correct candidate may still be excluded for timing/resources.
            correctness_status = "passed"
        elif any(summary.get("ran") for summary in candidate_csim_results):
            correctness_status = "failed"
        else:
            correctness_status = "not_run"
        usage_totals = {
            key: sum(int(event.get(key) or 0) for event in self.llm_events)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }
        result = {
            "schema_version": BASELINE_SCHEMA,
            "benchmark": self.inputs.benchmark,
            "method": self.method,
            "model": self.model_id,
            "success": success,
            "phase": "complete" if success else ("selected_cosim" if winner else "selection"),
            "error": "" if success else (
                cosim.get("error")
                or (
                    "selected winner cosim passed without an executed cycle count"
                    if winner is not None and cosim.get("passed")
                    else "no correct feasible candidate"
                )
            ),
            "candidate_count": len(candidates),
            "candidates": public_candidates,
            "selected_candidate_index": selected_index,
            "selected_code_sha256": winner.get("code_sha256") if winner else None,
            "cosim_target_code_sha256": (
                winner.get("code_sha256") if winner is not None else None
            ),
            "final_report": selected_report,
            "csim": selected_csim,
            "cosim": cosim,
            "executed_cosim_cycles": executed_cycles,
            "selected_cosim_measurement_valid": cosim_measurement_valid,
            # Functional/golden correctness is deliberately independent of
            # the selected winner's executed RTL cosim measurement status.
            "correctness_status": correctness_status,
            "candidate_feasibility": dict(winner.get("feasibility") or {}) if winner else {},
            "llm_usage": {
                "schema_version": "c2hls.paper-baseline-llm.v1",
                "calls": self.llm_count,
                "candidate_requests": self.llm_count,
                "candidate_budget": MAX_LLM_CANDIDATES,
                "base_seed": self.base_seed,
                "seed_control": (
                    "base_plus_candidate_index"
                    if self.seed_supported
                    else "unsupported_by_provider"
                ),
                "candidate_seed_schedule": _candidate_seed_schedule(
                    self.base_seed, seed_supported=self.seed_supported
                ),
                **usage_totals,
                "events": self.llm_events,
            },
            "synthesis_evaluations": {
                "schema_version": "c2hls.synthesis-evaluations.v1",
                "count": self.synthesis_count,
                "budget": MAX_SYNTHESIS_EVALUATIONS,
                "events": self.synthesis_events,
            },
            "synthesis_evaluation_count": self.synthesis_count,
            "csim_evaluation_count": self.csim_count,
            "selected_winner_cosim_count": self.selected_winner_cosim_count,
            "total_synthesis_calls": (
                self.synthesis_count + self.selected_winner_cosim_count
            ),
            "independent_golden": dict(
                self.inputs.independent_golden_provenance or {}
            ),
            "run": {
                "decoding": {
                    "effective": {
                        "temperature": os.getenv("C2HLS_LLM_TEMPERATURE"),
                        "top_p": os.getenv("C2HLS_LLM_TOP_P"),
                        "seed": (
                            [self.base_seed + index for index in range(self.llm_count)]
                            if self.seed_supported
                            else None
                        ),
                        "seed_supported": self.seed_supported,
                        "seed_policy": (
                            "base_plus_candidate_index"
                            if self.seed_supported
                            else "unsupported_by_provider"
                        ),
                    }
                },
                "synthesis_evaluations": self.synthesis_count,
                "llm_usage": {"calls": self.llm_count},
            },
            "_selected_code": str(winner.get("_code") or "") if winner else "",
        }
        return result


def production_callbacks(
    inputs: PublicBenchmarkInputs,
    *,
    model_id: str,
) -> tuple[LLMRequest, CSimCallback, SynthCallback, CosimCallback, FeasibilityCallback]:
    """Bind the pure search engine to the existing C2HLS/Vitis utilities."""

    from c2hls import (
        C2HLSOrchestrator,
        _paper_candidate_feasibility,
        _summarize_test_result,
    )
    from hls_eval import run_cosim, run_csim, run_hls_synthesis

    orchestrator = C2HLSOrchestrator(gpt_model=model_id, turns_limitation=1)

    def llm_request(
        messages: Sequence[Mapping[str, str]],
        candidate_index: int,
        effective_seed: Optional[int],
        seed_supported: bool,
    ) -> Mapping[str, Any]:
        del candidate_index
        with _temporary_seed(effective_seed if seed_supported else None):
            reply = orchestrator._call_llm_with_model(
                list(messages),
                model=model_id,
                max_tokens=int(os.getenv("C2HLS_MAX_COMPLETION_TOKENS", "8192")),
                agent_name="paper_baseline",
            )
        event = (
            dict(orchestrator.llm_usage_events[-1])
            if orchestrator.llm_usage_events
            else {}
        )
        return {"text": reply, "event": event}

    common = {
        "header_code": inputs.header_code,
        "header_name": inputs.header_name,
        "top_function": inputs.translated_hls_top,
        "part": inputs.part,
        "clock_ns": inputs.clock_ns,
        "extra_files": list(inputs.extra_files),
    }

    def csim(code: str) -> Mapping[str, Any]:
        raw = run_csim(
            code,
            inputs.testbench_code,
            **common,
            golden_output_text=inputs.independent_golden_output,
            golden_output_specs=dict(inputs.independent_golden_specs or {}),
        )
        return _summarize_test_result(raw, True)

    def synthesize(code: str) -> Mapping[str, Any]:
        return run_hls_synthesis(code, **common)

    def cosim(code: str) -> Mapping[str, Any]:
        raw = run_cosim(
            code,
            inputs.testbench_code,
            **common,
            interface_depths=dict(inputs.cosim_depths),
            golden_output_text=inputs.independent_golden_output,
            golden_output_specs=dict(inputs.independent_golden_specs or {}),
        )
        return _summarize_test_result(raw, True)

    def feasibility(
        report: Mapping[str, Any], csim_result: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return _paper_candidate_feasibility(
            dict(report),
            csim=dict(csim_result),
            correctness_required=True,
            part=inputs.part,
            clock_ns=inputs.clock_ns,
        )

    return llm_request, csim, synthesize, cosim, feasibility


def build_baseline_fingerprint(
    *,
    repo: Path,
    inputs: PublicBenchmarkInputs,
    method: str,
    model_id: str,
    model_label: str,
    base_seed: int,
    profile: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Extend the shared fingerprint with baseline implementation/contract."""

    env = dict(os.environ if environ is None else environ)
    # CLI --base-seed is authoritative.  Override any inherited environment
    # value before invoking the shared fingerprint builder so resume identity
    # cannot silently attest a different seed from the one the engine uses.
    env["C2HLS_LLM_SEED"] = str(int(base_seed))
    seed_supported = not model_id.lower().startswith("claude")
    # The shared fingerprint builder inventories every file below its input
    # directory. Give it a temporary *public-only* view so it cannot read or
    # identity-couple the baseline to expert variants stored beside plain.cpp.
    with tempfile.TemporaryDirectory(prefix="c2hls_public_fingerprint_") as tmp:
        public_root = Path(tmp)
        public_metadata = {
            "benchmark": inputs.benchmark,
            "header_file": inputs.header_name,
            "testbench_file": "testbench.cpp",
            "translated_hls_top": inputs.translated_hls_top,
            "part": inputs.part,
            "clock_ns": inputs.clock_ns,
            "cosim_depths": dict(inputs.cosim_depths),
            "independent_golden": dict(
                inputs.independent_golden_provenance or {}
            ),
        }
        (public_root / "metadata.json").write_text(
            canonical_json(public_metadata) + "\n", encoding="utf-8"
        )
        (public_root / "plain.cpp").write_text(inputs.c_code, encoding="utf-8")
        (public_root / inputs.header_name).parent.mkdir(parents=True, exist_ok=True)
        (public_root / inputs.header_name).write_text(
            inputs.header_code, encoding="utf-8"
        )
        (public_root / "testbench.cpp").write_text(
            inputs.testbench_code, encoding="utf-8"
        )
        for item in inputs.extra_files:
            relative = Path(str(item["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe public support path: {relative}")
            path = public_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(item.get("content") or ""), encoding="utf-8")
        fingerprint = build_run_fingerprint(
            repo=repo,
            benchmark_dir=public_root,
            benchmark=inputs.benchmark,
            model_id=model_id,
            model_label=model_label,
            skill_mode="skill_off",
            steps=[method],
            profile=profile,
            environ=env,
        )
    payload = dict(fingerprint["payload"])
    implementation_files = [repo / "paper_baselines.py", repo / "run_paper_baseline.py"]
    payload["paper_baseline"] = {
        "schema_version": BASELINE_SCHEMA,
        "method": method,
        "max_llm_candidates": MAX_LLM_CANDIDATES,
        "max_synthesis_evaluations": MAX_SYNTHESIS_EVALUATIONS,
        "correctness_order": "csim_golden_before_synthesis",
        "cosim_policy": "selected_winner_only",
        "base_seed": int(base_seed),
        "seed_policy": (
            "unsupported_by_provider"
            if not seed_supported
            else "base_plus_candidate_index"
        ),
        "candidate_seed_schedule": _candidate_seed_schedule(
            int(base_seed), seed_supported=seed_supported
        ),
        "prompt_protocol": {
            "translation_prompt_sha256": sha256_json(
                _translation_messages(inputs)
            ),
            "pragma_revision_template_sha256": sha256_json(
                _pragma_revision_messages(inputs, "<INITIAL_TRANSLATION>")
            ),
            "exact_request_prompt_hashes_recorded_post_run": True,
        },
        "implementation": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_bytes(path.read_bytes()),
            }
            for path in implementation_files
            if path.is_file()
        ],
    }
    return {
        "schema_version": FINGERPRINT_SCHEMA,
        "sha256": sha256_json(payload),
        "payload": payload,
    }


def load_matching_resume(result_path: Path, fingerprint: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return an exact resumable result, reject stale/corrupt results."""

    if not result_path.is_file():
        return None
    try:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"resume result is unreadable: {exc}") from exc
    recorded = existing.get("run_fingerprint") or (
        (existing.get("run") or {}).get("run_fingerprint")
    )
    if not fingerprint_matches(recorded, fingerprint):
        raise RuntimeError("resume rejected: full run fingerprint mismatch")
    return existing


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def finalize_baseline_result(
    result: MutableMapping[str, Any],
    *,
    fingerprint: Mapping[str, Any],
    profile: Mapping[str, Any],
    benchmark_dir: Path,
    output_dir: Path,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Persist transcript, audit it, and attach shared provenance/status."""

    history = list(result.pop("_history", []))
    selected_code = str(result.pop("_selected_code", ""))
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"{result['benchmark']}_baseline_history.json"
    history_bytes = (
        json.dumps({"messages": history}, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(history_path, history_bytes)
    expected_transcript_sha256 = sha256_bytes(history_bytes)
    audit = audit_history_file(history_path, benchmark_dir=benchmark_dir)
    if audit.get("transcript_sha256") != expected_transcript_sha256:
        audit = {
            **dict(audit),
            "passed": False,
            "error": "persisted transcript changed before reference-isolation audit",
        }
    audit_path = output_dir / f"{result['benchmark']}_reference_isolation_audit.json"
    _atomic_write_bytes(
        audit_path,
        (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    if selected_code:
        selected_path = output_dir / f"{result['benchmark']}_selected.cpp"
        _atomic_write_bytes(selected_path, selected_code.encode("utf-8"))
        result["selected_code_file"] = selected_path.name

    attach_run_provenance(
        result,
        fingerprint=fingerprint,
        profile=profile,
        elapsed_seconds=elapsed_seconds,
        history_path=history_path,
        reference_audit=audit,
    )
    result.setdefault("run", {}).update({
        "search_elapsed_seconds": float(elapsed_seconds),
        "preflight_elapsed_seconds": 0.0,
        "post_route_elapsed_seconds": 0.0,
        "total_elapsed_seconds": float(elapsed_seconds),
        "paper_method_wall_time_field": "search_elapsed_seconds",
    })
    result.setdefault("run", {})["transcript_file"] = history_path.name
    result.setdefault("run", {})[
        "reference_isolation_audit_path"
    ] = audit_path.name
    if not audit.get("passed"):
        result["controller_success_before_isolation_audit"] = bool(
            result.get("success")
        )
        result["success"] = False
        result["phase"] = "reference_isolation"
        result["error"] = "reference-isolation transcript audit failed"
    return dict(result)


def run_baseline_case(
    *,
    repo: Path,
    inputs: PublicBenchmarkInputs,
    method: str,
    model_id: str,
    model_label: str,
    base_seed: int,
    profile: Mapping[str, Any],
    output_dir: Path,
    result_path: Path,
    resume: bool,
    callbacks: tuple[
        LLMRequest,
        CSimCallback,
        SynthCallback,
        CosimCallback,
        FeasibilityCallback,
    ] | None = None,
) -> dict[str, Any]:
    """Execute or exactly resume one matrix cell."""

    enforce_baseline_contract()
    fingerprint = build_baseline_fingerprint(
        repo=repo,
        inputs=inputs,
        method=method,
        model_id=model_id,
        model_label=model_label,
        base_seed=base_seed,
        profile=profile,
    )
    completeness = fingerprint_completeness(fingerprint)
    if not completeness.get("complete"):
        raise RuntimeError(
            "paper baseline fingerprint incomplete: "
            + ", ".join(completeness.get("issues") or [])
        )
    if resume:
        resumed = load_matching_resume(result_path, fingerprint)
        if resumed is not None:
            return resumed
    elif result_path.is_file():
        raise RuntimeError(
            "result already exists and resume is disabled; refusing overwrite"
        )

    runtime = callbacks or production_callbacks(inputs, model_id=model_id)
    seed_supported = not model_id.lower().startswith("claude")
    engine = PaperBaselineEngine(
        inputs=inputs,
        method=method,
        model_id=model_id,
        base_seed=base_seed,
        seed_supported=seed_supported,
        llm_request=runtime[0],
        csim=runtime[1],
        synthesize=runtime[2],
        cosim=runtime[3],
        feasibility=runtime[4],
    )
    started = time.time()
    result = engine.run()
    result["_history"] = engine.history
    finalized = finalize_baseline_result(
        result,
        fingerprint=fingerprint,
        profile=profile,
        benchmark_dir=inputs.benchmark_dir,
        output_dir=output_dir,
        elapsed_seconds=time.time() - started,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_name(f".{result_path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(finalized, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, result_path)
    return finalized
