"""Independent, typed comparison of CPU-golden and candidate kernel outputs.

The HLSFactory PolyBench testbenches emit output arrays between ``begin dump``
and ``end dump`` markers, but they do not decide whether the values are correct.
This module parses those dumps and compares them without depending on Vitis or
the C2HLS controller.  It also accepts nested Python sequences, which makes the
same comparison policy usable by unit tests and non-PolyBench benchmarks.

The public comparison functions never raise for bad candidate output.  Instead,
they return :class:`ComparisonResult`, whose ``to_dict`` representation is safe
to place directly in a run-result JSON document.  ``parse_hlsfactory_dumps`` is
the lower-level exception-raising API for callers that need the parsed arrays.
"""

from __future__ import annotations

import math
import numbers
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from operator import mul
from typing import Any, Mapping, Optional, Sequence, Tuple, Union


Number = Union[int, float]
Shape = Tuple[int, ...]


class CorrectnessStatus(str, Enum):
    """Top-level correctness classification for a candidate output."""

    PASSED = "passed"
    FAILED = "failed"
    INVALID_OUTPUT = "invalid_output"


class ComparisonReason(str, Enum):
    """Machine-readable explanation for a comparison result."""

    MATCH = "match"
    NO_OUTPUT = "no_output"
    MALFORMED_OUTPUT = "malformed_output"
    OUTPUT_SET_MISMATCH = "output_set_mismatch"
    COUNT_MISMATCH = "count_mismatch"
    SHAPE_MISMATCH = "shape_mismatch"
    TYPE_MISMATCH = "type_mismatch"
    INTEGER_MISMATCH = "integer_mismatch"
    FLOAT_MISMATCH = "float_mismatch"
    NAN_MISMATCH = "nan_mismatch"
    INFINITY_MISMATCH = "infinity_mismatch"


class NumericKind(str, Enum):
    """Comparison arithmetic to use for one output array."""

    AUTO = "auto"
    INTEGER = "integer"
    FLOAT = "float"


@dataclass(frozen=True)
class OutputSpec:
    """Expected representation and comparison policy for one named output.

    ``shape`` is especially useful for text dumps, which otherwise contain only
    a flat stream.  Integer outputs are always compared exactly; ``atol`` and
    ``rtol`` apply only to floating-point outputs.  NaN and infinity are rejected
    by default, even when they occur in both outputs, so a broken golden run
    cannot silently validate a broken candidate.
    """

    shape: Optional[Shape] = None
    kind: NumericKind = NumericKind.AUTO
    atol: Optional[float] = None
    rtol: Optional[float] = None
    allow_nan: bool = False
    allow_infinity: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, NumericKind):
            object.__setattr__(self, "kind", NumericKind(self.kind))
        if self.shape is not None:
            shape = tuple(self.shape)
            if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape):
                raise ValueError("shape dimensions must be non-negative integers")
            object.__setattr__(self, "shape", shape)
        for name, value in (("atol", self.atol), ("rtol", self.rtol)):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite non-negative number")


@dataclass(frozen=True)
class ParsedOutput:
    """One named, flattened output parsed from a dump block."""

    name: str
    values: Tuple[Number, ...]
    shape: Shape
    integer_tokens: bool


@dataclass(frozen=True)
class ComparisonResult:
    """Typed result with a stable JSON representation."""

    correctness_status: CorrectnessStatus
    reason: ComparisonReason
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.correctness_status is CorrectnessStatus.PASSED

    @property
    def status(self) -> CorrectnessStatus:
        """Compatibility alias for code that uses a generic ``status`` field."""

        return self.correctness_status

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "correctness_status": self.correctness_status.value,
            "reason": self.reason.value,
            "passed": self.passed,
            "details": _json_safe(self.details),
        }


class OutputParseError(ValueError):
    """Raised by the low-level dump parser with a typed failure reason."""

    def __init__(
        self,
        reason: ComparisonReason,
        message: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.details = {"message": message, **dict(details or {})}


@dataclass(frozen=True)
class _NormalizedOutput:
    values: Tuple[Number, ...]
    shape: Shape
    integer_values: bool
    source: str


_START_DUMP_RE = re.compile(
    r"^[ \t]*begin[ \t]+dump[ \t]*:[ \t]*(?P<payload>[^\r\n]*)",
    re.IGNORECASE | re.MULTILINE,
)
_END_DUMP_RE = re.compile(
    r"^[ \t]*end[ \t]+dump[ \t]*:[ \t]*(?P<name>[^\r\n]*?)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_INTEGER_TOKEN_RE = re.compile(r"[+-]?\d+")
_FLOAT_TOKEN_RE = re.compile(
    r"[+-]?(?:"
    r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    r"|inf(?:inity)?"
    r"|nan(?:\([^)]*\))?"
    r")",
    re.IGNORECASE,
)


def parse_hlsfactory_dumps(text: str) -> dict[str, ParsedOutput]:
    """Parse HLSFactory/PolyBench ``begin dump`` blocks from process output.

    Tool chatter outside dump blocks is ignored.  Inside a block every token
    must be numeric.  The parser deliberately derives the output name from the
    corresponding ``end dump`` marker: some upstream testbenches print the first
    value immediately after the name with no intervening whitespace.

    Raises:
        OutputParseError: if no output exists or the marker/value stream is bad.
    """

    if not isinstance(text, str):
        raise OutputParseError(
            ComparisonReason.MALFORMED_OUTPUT,
            "dump output must be text",
            {"received_type": type(text).__name__},
        )

    starts = list(_START_DUMP_RE.finditer(text))
    ends = list(_END_DUMP_RE.finditer(text))
    if not starts and not ends:
        raise OutputParseError(
            ComparisonReason.NO_OUTPUT,
            "no begin/end dump blocks were found",
        )
    if len(starts) != len(ends):
        raise OutputParseError(
            ComparisonReason.MALFORMED_OUTPUT,
            "unbalanced begin/end dump markers",
            {"begin_markers": len(starts), "end_markers": len(ends)},
        )

    parsed: dict[str, ParsedOutput] = {}
    previous_end = -1
    for block_index, (start, end) in enumerate(zip(starts, ends)):
        next_start = starts[block_index + 1].start() if block_index + 1 < len(starts) else None
        if start.start() < previous_end or end.start() <= start.end():
            raise OutputParseError(
                ComparisonReason.MALFORMED_OUTPUT,
                "dump markers are out of order",
                {"block_index": block_index},
            )
        if next_start is not None and next_start < end.start():
            raise OutputParseError(
                ComparisonReason.MALFORMED_OUTPUT,
                "nested or unterminated dump block",
                {"block_index": block_index},
            )

        name = end.group("name").strip()
        payload = start.group("payload").lstrip()
        if not name:
            raise OutputParseError(
                ComparisonReason.MALFORMED_OUTPUT,
                "empty output name in end marker",
                {"block_index": block_index},
            )
        if not payload.startswith(name):
            raise OutputParseError(
                ComparisonReason.MALFORMED_OUTPUT,
                "begin/end dump names do not match",
                {
                    "block_index": block_index,
                    "end_name": name,
                    "begin_payload": payload[:80],
                },
            )
        if name in parsed:
            raise OutputParseError(
                ComparisonReason.MALFORMED_OUTPUT,
                "duplicate output dump name",
                {"output": name},
            )

        # ``payload`` can be ``x0.125`` rather than ``x 0.125``.  Stripping the
        # end-marker name exactly handles both forms without guessing where the
        # C identifier ends.
        same_line_values = payload[len(name) :]
        value_text = same_line_values + text[start.end() : end.start()]
        values, integer_tokens = _parse_numeric_stream(value_text, name)
        parsed[name] = ParsedOutput(
            name=name,
            values=values,
            shape=(len(values),),
            integer_tokens=integer_tokens,
        )
        previous_end = end.end()

    return parsed


def compare_hlsfactory_dumps(
    golden_text: str,
    candidate_text: str,
    specs: Optional[Mapping[str, Union[OutputSpec, Mapping[str, Any]]]] = None,
    *,
    default_atol: float = 1e-6,
    default_rtol: float = 1e-5,
    max_mismatches: int = 8,
) -> ComparisonResult:
    """Compare CPU-golden and candidate HLSFactory dump text.

    ``specs`` may declare original array shapes and per-output numeric policies.
    When a declared shape is present, its element count is checked against both
    flattened dumps before values are compared.
    """

    normalized_specs = _normalize_specs(specs)
    try:
        golden = parse_hlsfactory_dumps(golden_text)
    except OutputParseError as exc:
        return _parse_failure(exc, source="golden")
    try:
        candidate = parse_hlsfactory_dumps(candidate_text)
    except OutputParseError as exc:
        return _parse_failure(exc, source="candidate")

    golden_outputs = _from_parsed_outputs(golden, normalized_specs)
    candidate_outputs = _from_parsed_outputs(candidate, normalized_specs)
    return _compare_normalized_outputs(
        golden_outputs,
        candidate_outputs,
        normalized_specs,
        default_atol=default_atol,
        default_rtol=default_rtol,
        max_mismatches=max_mismatches,
    )


def compare_structured_outputs(
    golden: Any,
    candidate: Any,
    specs: Optional[Mapping[str, Union[OutputSpec, Mapping[str, Any]]]] = None,
    *,
    default_atol: float = 1e-6,
    default_rtol: float = 1e-5,
    max_mismatches: int = 8,
) -> ComparisonResult:
    """Compare mappings of named outputs or a single nested numeric sequence.

    A non-mapping input is assigned the stable name ``"output"``.  Nested
    sequence shapes are inferred recursively; ragged sequences are invalid.
    """

    normalized_specs = _normalize_specs(specs)
    try:
        golden_outputs = _normalize_structured_outputs(golden)
    except OutputParseError as exc:
        return _parse_failure(exc, source="golden")
    try:
        candidate_outputs = _normalize_structured_outputs(candidate)
    except OutputParseError as exc:
        return _parse_failure(exc, source="candidate")

    return _compare_normalized_outputs(
        golden_outputs,
        candidate_outputs,
        normalized_specs,
        default_atol=default_atol,
        default_rtol=default_rtol,
        max_mismatches=max_mismatches,
    )


# Concise alias for controller integrations that already know their source type.
compare_outputs = compare_structured_outputs


def _parse_numeric_stream(value_text: str, output_name: str) -> tuple[Tuple[Number, ...], bool]:
    tokens = [token for token in re.split(r"[\s,]+", value_text.strip()) if token]
    if not tokens:
        raise OutputParseError(
            ComparisonReason.MALFORMED_OUTPUT,
            "output dump contains no numeric values",
            {"output": output_name},
        )

    values: list[Number] = []
    integer_tokens = True
    for index, token in enumerate(tokens):
        if _INTEGER_TOKEN_RE.fullmatch(token):
            values.append(int(token))
            continue
        integer_tokens = False
        if not _FLOAT_TOKEN_RE.fullmatch(token):
            raise OutputParseError(
                ComparisonReason.MALFORMED_OUTPUT,
                "non-numeric token inside output dump",
                {"output": output_name, "token_index": index, "token": token[:120]},
            )
        lower = token.lower().lstrip("+")
        if "nan" in lower:
            values.append(math.nan)
        elif "inf" in lower:
            values.append(-math.inf if token.startswith("-") else math.inf)
        else:
            values.append(float(token))
    return tuple(values), integer_tokens


def _normalize_specs(
    specs: Optional[Mapping[str, Union[OutputSpec, Mapping[str, Any]]]],
) -> dict[str, OutputSpec]:
    normalized: dict[str, OutputSpec] = {}
    for name, value in (specs or {}).items():
        if isinstance(value, OutputSpec):
            spec = value
        elif isinstance(value, Mapping):
            spec = OutputSpec(**dict(value))
        else:
            raise TypeError(f"output spec for {name!r} must be OutputSpec or a mapping")
        normalized[str(name)] = spec
    return normalized


def _from_parsed_outputs(
    outputs: Mapping[str, ParsedOutput],
    specs: Mapping[str, OutputSpec],
) -> dict[str, _NormalizedOutput]:
    normalized: dict[str, _NormalizedOutput] = {}
    for name, output in outputs.items():
        spec = specs.get(name, OutputSpec())
        shape = output.shape
        if spec.shape is not None and _shape_size(spec.shape) == len(output.values):
            shape = spec.shape
        normalized[name] = _NormalizedOutput(
            values=output.values,
            shape=shape,
            integer_values=output.integer_tokens,
            source="dump",
        )
    return normalized


def _normalize_structured_outputs(value: Any) -> dict[str, _NormalizedOutput]:
    named = value if isinstance(value, Mapping) else {"output": value}
    if not named:
        raise OutputParseError(ComparisonReason.NO_OUTPUT, "structured output mapping is empty")

    outputs: dict[str, _NormalizedOutput] = {}
    for raw_name, raw_output in named.items():
        name = str(raw_name)
        if not name:
            raise OutputParseError(
                ComparisonReason.MALFORMED_OUTPUT,
                "structured output has an empty name",
            )
        shape, values = _flatten_structured(raw_output, name)
        outputs[name] = _NormalizedOutput(
            values=tuple(values),
            shape=shape,
            integer_values=all(_is_integer_value(item) for item in values),
            source="structured",
        )
    return outputs


def _flatten_structured(value: Any, output_name: str) -> tuple[Shape, list[Number]]:
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        value = value.tolist()

    if isinstance(value, numbers.Integral):
        return (), [int(value)]
    if isinstance(value, numbers.Real):
        return (), [float(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) == 0:
            return (0,), []
        child_shapes: list[Shape] = []
        flattened: list[Number] = []
        for child in value:
            child_shape, child_values = _flatten_structured(child, output_name)
            child_shapes.append(child_shape)
            flattened.extend(child_values)
        if any(shape != child_shapes[0] for shape in child_shapes[1:]):
            raise OutputParseError(
                ComparisonReason.MALFORMED_OUTPUT,
                "ragged nested sequence",
                {"output": output_name, "child_shapes": [list(shape) for shape in child_shapes]},
            )
        return (len(value),) + child_shapes[0], flattened

    raise OutputParseError(
        ComparisonReason.MALFORMED_OUTPUT,
        "structured output contains a non-numeric value",
        {"output": output_name, "received_type": type(value).__name__},
    )


def _compare_normalized_outputs(
    golden: Mapping[str, _NormalizedOutput],
    candidate: Mapping[str, _NormalizedOutput],
    specs: Mapping[str, OutputSpec],
    *,
    default_atol: float,
    default_rtol: float,
    max_mismatches: int,
) -> ComparisonResult:
    _validate_comparison_options(default_atol, default_rtol, max_mismatches)

    golden_names = set(golden)
    candidate_names = set(candidate)
    if golden_names != candidate_names:
        return ComparisonResult(
            CorrectnessStatus.FAILED,
            ComparisonReason.OUTPUT_SET_MISMATCH,
            {
                "missing_outputs": sorted(golden_names - candidate_names),
                "unexpected_outputs": sorted(candidate_names - golden_names),
            },
        )

    total_values = 0
    summaries: dict[str, Any] = {}
    for name in golden:
        expected = golden[name]
        actual = candidate[name]
        spec = specs.get(name, OutputSpec())
        expected_count = len(expected.values)
        actual_count = len(actual.values)

        if expected_count != actual_count:
            return ComparisonResult(
                CorrectnessStatus.FAILED,
                ComparisonReason.COUNT_MISMATCH,
                {
                    "output": name,
                    "expected_count": expected_count,
                    "actual_count": actual_count,
                    "expected_shape": list(expected.shape),
                    "actual_shape": list(actual.shape),
                },
            )

        if spec.shape is not None:
            declared_count = _shape_size(spec.shape)
            if expected_count != declared_count or actual_count != declared_count:
                return ComparisonResult(
                    CorrectnessStatus.FAILED,
                    ComparisonReason.COUNT_MISMATCH,
                    {
                        "output": name,
                        "declared_shape": list(spec.shape),
                        "declared_count": declared_count,
                        "expected_count": expected_count,
                        "actual_count": actual_count,
                    },
                )
            if expected.shape != spec.shape or actual.shape != spec.shape:
                return ComparisonResult(
                    CorrectnessStatus.FAILED,
                    ComparisonReason.SHAPE_MISMATCH,
                    {
                        "output": name,
                        "declared_shape": list(spec.shape),
                        "expected_shape": list(expected.shape),
                        "actual_shape": list(actual.shape),
                    },
                )
        elif expected.shape != actual.shape:
            return ComparisonResult(
                CorrectnessStatus.FAILED,
                ComparisonReason.SHAPE_MISMATCH,
                {
                    "output": name,
                    "expected_shape": list(expected.shape),
                    "actual_shape": list(actual.shape),
                    "count": expected_count,
                },
            )

        kind = spec.kind
        if kind is NumericKind.AUTO:
            kind = NumericKind.INTEGER if expected.integer_values else NumericKind.FLOAT
        atol = default_atol if spec.atol is None else spec.atol
        rtol = default_rtol if spec.rtol is None else spec.rtol

        if kind is NumericKind.INTEGER:
            mismatch = _compare_integer_values(
                name,
                expected,
                actual,
                max_mismatches=max_mismatches,
            )
        else:
            mismatch = _compare_float_values(
                name,
                expected,
                actual,
                atol=atol,
                rtol=rtol,
                allow_nan=spec.allow_nan,
                allow_infinity=spec.allow_infinity,
                max_mismatches=max_mismatches,
            )
        if mismatch is not None:
            return mismatch

        total_values += expected_count
        summaries[name] = {
            "shape": list(expected.shape),
            "count": expected_count,
            "kind": kind.value,
            **({"atol": atol, "rtol": rtol} if kind is NumericKind.FLOAT else {}),
        }

    return ComparisonResult(
        CorrectnessStatus.PASSED,
        ComparisonReason.MATCH,
        {
            "outputs_compared": len(golden),
            "values_compared": total_values,
            "outputs": summaries,
        },
    )


def _compare_integer_values(
    name: str,
    expected: _NormalizedOutput,
    actual: _NormalizedOutput,
    *,
    max_mismatches: int,
) -> Optional[ComparisonResult]:
    mismatches: list[dict[str, Any]] = []
    mismatch_count = 0
    type_error = False
    for flat_index, (expected_value, actual_value) in enumerate(zip(expected.values, actual.values)):
        expected_integer = _exact_integer(expected_value)
        actual_integer = _exact_integer(actual_value)
        if expected_integer is None or actual_integer is None:
            type_error = True
            differs = True
        else:
            differs = expected_integer != actual_integer
        if differs:
            mismatch_count += 1
            if len(mismatches) < max_mismatches:
                mismatches.append(
                    _mismatch_detail(
                        flat_index,
                        expected.shape,
                        expected_value,
                        actual_value,
                    )
                )

    if mismatch_count:
        return ComparisonResult(
            CorrectnessStatus.FAILED,
            ComparisonReason.TYPE_MISMATCH if type_error else ComparisonReason.INTEGER_MISMATCH,
            {
                "output": name,
                "comparison": "exact_integer",
                "mismatch_count": mismatch_count,
                "reported_mismatches": mismatches,
            },
        )
    return None


def _compare_float_values(
    name: str,
    expected: _NormalizedOutput,
    actual: _NormalizedOutput,
    *,
    atol: float,
    rtol: float,
    allow_nan: bool,
    allow_infinity: bool,
    max_mismatches: int,
) -> Optional[ComparisonResult]:
    mismatches: list[dict[str, Any]] = []
    mismatch_count = 0
    first_nonfinite_reason: Optional[ComparisonReason] = None
    max_abs_error = 0.0
    max_rel_error = 0.0

    for flat_index, (expected_value, actual_value) in enumerate(zip(expected.values, actual.values)):
        expected_float = float(expected_value)
        actual_float = float(actual_value)
        reason: Optional[ComparisonReason] = None
        differs = False

        if math.isnan(expected_float) or math.isnan(actual_float):
            equal_nan = math.isnan(expected_float) and math.isnan(actual_float) and allow_nan
            differs = not equal_nan
            if differs:
                reason = ComparisonReason.NAN_MISMATCH
        elif math.isinf(expected_float) or math.isinf(actual_float):
            equal_infinity = (
                math.isinf(expected_float)
                and math.isinf(actual_float)
                and expected_float == actual_float
                and allow_infinity
            )
            differs = not equal_infinity
            if differs:
                reason = ComparisonReason.INFINITY_MISMATCH
        else:
            abs_error = abs(actual_float - expected_float)
            rel_error = abs_error / abs(expected_float) if expected_float != 0 else abs_error
            max_abs_error = max(max_abs_error, abs_error)
            max_rel_error = max(max_rel_error, rel_error)
            differs = abs_error > (atol + rtol * abs(expected_float))

        if differs:
            mismatch_count += 1
            if reason is not None and first_nonfinite_reason is None:
                first_nonfinite_reason = reason
            if len(mismatches) < max_mismatches:
                detail = _mismatch_detail(
                    flat_index,
                    expected.shape,
                    expected_value,
                    actual_value,
                )
                detail["tolerance"] = atol + rtol * abs(expected_float) if math.isfinite(expected_float) else None
                if reason is not None:
                    detail["nonfinite_reason"] = reason.value
                mismatches.append(detail)

    if mismatch_count:
        return ComparisonResult(
            CorrectnessStatus.FAILED,
            first_nonfinite_reason or ComparisonReason.FLOAT_MISMATCH,
            {
                "output": name,
                "comparison": "tolerant_float",
                "atol": atol,
                "rtol": rtol,
                "allow_nan": allow_nan,
                "allow_infinity": allow_infinity,
                "mismatch_count": mismatch_count,
                "max_abs_error": max_abs_error,
                "max_rel_error": max_rel_error,
                "reported_mismatches": mismatches,
            },
        )
    return None


def _mismatch_detail(
    flat_index: int,
    shape: Shape,
    expected: Number,
    actual: Number,
) -> dict[str, Any]:
    return {
        "flat_index": flat_index,
        "index": list(_unravel_index(flat_index, shape)),
        "expected": _json_number(expected),
        "actual": _json_number(actual),
    }


def _unravel_index(flat_index: int, shape: Shape) -> Shape:
    if not shape:
        return ()
    indices: list[int] = []
    remaining = flat_index
    for dim_index, dim in enumerate(shape):
        stride = _shape_size(shape[dim_index + 1 :])
        if stride == 0 or dim == 0:
            indices.append(0)
        else:
            indices.append(remaining // stride)
            remaining %= stride
    return tuple(indices)


def _shape_size(shape: Shape) -> int:
    return reduce(mul, shape, 1)


def _exact_integer(value: Number) -> Optional[int]:
    if isinstance(value, numbers.Integral):
        return int(value)
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _is_integer_value(value: Number) -> bool:
    return isinstance(value, numbers.Integral)


def _validate_comparison_options(atol: float, rtol: float, max_mismatches: int) -> None:
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("default_atol must be finite and non-negative")
    if not math.isfinite(rtol) or rtol < 0:
        raise ValueError("default_rtol must be finite and non-negative")
    if not isinstance(max_mismatches, int) or isinstance(max_mismatches, bool) or max_mismatches < 1:
        raise ValueError("max_mismatches must be a positive integer")


def _parse_failure(exc: OutputParseError, *, source: str) -> ComparisonResult:
    return ComparisonResult(
        CorrectnessStatus.INVALID_OUTPUT,
        exc.reason,
        {"source": source, **exc.details},
    )


def _json_number(value: Number) -> Union[int, float, str]:
    if isinstance(value, numbers.Integral):
        return int(value)
    numeric = float(value)
    if math.isnan(numeric):
        return "NaN"
    if math.isinf(numeric):
        return "+Infinity" if numeric > 0 else "-Infinity"
    return numeric


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, numbers.Real):
        return _json_number(value)
    return value


__all__ = [
    "ComparisonReason",
    "ComparisonResult",
    "CorrectnessStatus",
    "NumericKind",
    "OutputParseError",
    "OutputSpec",
    "ParsedOutput",
    "compare_hlsfactory_dumps",
    "compare_outputs",
    "compare_structured_outputs",
    "parse_hlsfactory_dumps",
]
