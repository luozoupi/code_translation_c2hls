"""Source transformations for coupled QoR architecture controls."""

from __future__ import annotations

import hashlib
import re
from typing import Any


GEMM_STEP_CONTROL_SPECS = {
    "pipeline": {
        "control_name": "GEMM_COMPUTE_II",
        "knob_kind": "pipeline_ii",
        "semantic_role": "compute_reduction_pipeline_target_ii",
    },
    "tiling": {
        "control_name": "GEMM_K_TILE",
        "knob_kind": "tile_size",
        "semantic_role": "compute_k_strip_size",
    },
    "doublebuffer": {
        "control_name": "GEMM_BUFFER_DEPTH",
        "knob_kind": "buffer_tile_depth",
        "semantic_role": "ping_pong_k_tile_buffer_depth",
    },
    "coalescing": {
        "control_name": "GEMM_AXI_BITS",
        "knob_kind": "interface_max_widen_bitwidth",
        "semantic_role": "shared_m_axi_max_widen_bitwidth",
    },
    "resource": {
        "control_name": "GEMM_MUL_IMPL",
        "knob_kind": "bind_op_implementation",
        "semantic_role": "double_multiplier_implementation_family",
        "value_labels": {
            1: "fabric",
            2: "meddsp",
            3: "fulldsp",
            4: "maxdsp",
            5: "primitivedsp",
        },
    },
}

TWO_MM_STEP_CONTROL_SPECS = {
    "pipeline": {
        "control_name": "MM2_COMPUTE_II",
        "knob_kind": "pipeline_ii",
        "semantic_role": "both_reduction_pipeline_targets",
    },
    "tiling": {
        "control_name": "MM2_K_TILE",
        "knob_kind": "tile_size",
        "semantic_role": "phase1_k_strip_size",
    },
    "doublebuffer": {
        "control_name": "MM2_BUFFER_DEPTH",
        "knob_kind": "buffer_tile_depth",
        "semantic_role": "phase1_alternating_b_tile_depth",
    },
    "coalescing": {
        "control_name": "MM2_AXI_BITS",
        "knob_kind": "interface_max_widen_bitwidth",
        "semantic_role": "shared_five_port_m_axi_max_widen_bitwidth",
    },
    "resource": {
        "control_name": "MM2_MUL_IMPL",
        "knob_kind": "bind_op_implementation",
        "semantic_role": "both_reduction_double_multiplier_implementation",
        "value_labels": {
            1: "fabric",
            2: "meddsp",
            3: "fulldsp",
            4: "maxdsp",
            5: "primitivedsp",
        },
    },
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _replace_once(
    source: str,
    pattern: str,
    replacement: str,
    *,
    label: str,
) -> str:
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"expected exactly one {label}; found {count}")
    return updated


def _resolve_2mm_dimensions(header_code: str) -> dict[str, int]:
    dimensions: dict[str, int] = {}
    for name in ("NI", "NJ", "NK", "NL"):
        match = re.search(
            rf"(?m)^\s*#\s*define\s+{name}\s+(?P<value>\d+)\b",
            header_code or "",
        )
        if match is not None:
            dimensions[name.lower()] = int(match.group("value"))
    if len(dimensions) == 4:
        return dimensions

    def array_shape(name: str) -> tuple[int, int] | None:
        match = re.search(
            rf"\bdouble\s+{re.escape(name)}\s*"
            r"\[\s*(?P<rows>\d+)(?:\s*\+\s*0)?\s*\]\s*"
            r"\[\s*(?P<cols>\d+)(?:\s*\+\s*0)?\s*\]",
            header_code or "",
        )
        return (
            (int(match.group("rows")), int(match.group("cols")))
            if match is not None
            else None
        )

    a_shape = array_shape("A")
    b_shape = array_shape("B")
    c_shape = array_shape("C")
    d_shape = array_shape("D")
    if not all((a_shape, b_shape, c_shape, d_shape)):
        raise ValueError("could not resolve compile-time 2MM dimensions")
    assert a_shape is not None
    assert b_shape is not None
    assert c_shape is not None
    assert d_shape is not None
    ni, nk = a_shape
    b_nk, nj = b_shape
    c_nj, nl = c_shape
    d_ni, d_nl = d_shape
    if (nk, nj, ni, nl) != (b_nk, c_nj, d_ni, d_nl):
        raise ValueError("inconsistent 2MM array dimensions in public header")
    return {"ni": ni, "nj": nj, "nk": nk, "nl": nl}


def validate_2mm_lane_values(
    header_code: str,
    values: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    """Validate boundary-guarded lane values shared by both 2MM reductions."""

    dimensions = _resolve_2mm_dimensions(header_code)
    normalized = sorted({int(value) for value in values})
    maximum = min(dimensions["nk"], dimensions["nj"])
    if not normalized or any(value <= 0 for value in normalized):
        raise ValueError("2MM lane values must be positive")
    oversized = [value for value in normalized if value > maximum]
    if oversized:
        raise ValueError(
            f"2MM lane values {oversized} exceed min(NK, NJ)={maximum}"
        )
    return {
        "schema_version": "c2hls.2mm-lane-domain.v1",
        **dimensions,
        "validated_values": normalized,
        "boundary_guarded_remainders": True,
        "maximum_lane_count": maximum,
    }


def validate_2mm_step_control_values(
    header_code: str,
    step: str,
    values: list[int] | tuple[int, ...],
    *,
    fixed_lanes: int = 4,
) -> dict[str, Any]:
    """Validate one step-native control domain for the 2MM campaign."""

    normalized_step = str(step).strip().lower().replace(
        "double_buffer", "doublebuffer"
    )
    if normalized_step not in TWO_MM_STEP_CONTROL_SPECS:
        raise ValueError(f"unsupported 2MM step control: {step!r}")
    normalized = sorted({int(value) for value in values})
    if not normalized or any(value <= 0 for value in normalized):
        raise ValueError("2MM step-control values must be positive")
    dimensions = _resolve_2mm_dimensions(header_code)
    precondition = "positive_integer"
    incompatible: list[int] = []
    if normalized_step in {"tiling", "doublebuffer"}:
        incompatible = [
            value
            for value in normalized
            if dimensions["nk"] % value != 0
            or value % int(fixed_lanes) != 0
        ]
        precondition = "NK % value == 0 and value % fixed_lanes == 0"
        if incompatible:
            raise ValueError(
                f"{normalized_step} values {incompatible} must divide "
                f"NK={dimensions['nk']} and be multiples of "
                f"fixed_lanes={int(fixed_lanes)}"
            )
    elif normalized_step == "coalescing":
        incompatible = [
            value
            for value in normalized
            if value < 64 or value > 1024 or value & (value - 1)
        ]
        precondition = "power_of_two_between_64_and_1024"
        if incompatible:
            raise ValueError(
                f"AXI widths {incompatible} must be powers of two in [64, 1024]"
            )
    elif normalized_step == "resource":
        incompatible = [value for value in normalized if value not in range(1, 6)]
        precondition = (
            "1=fabric, 2=meddsp, 3=fulldsp, 4=maxdsp, 5=primitivedsp"
        )
        if incompatible:
            raise ValueError(
                f"multiplier implementation selectors {incompatible} must be in [1, 5]"
            )
    return {
        "schema_version": "c2hls.2mm-step-control-domain.v1",
        "step": normalized_step,
        **dimensions,
        "fixed_lanes": int(fixed_lanes),
        "validated_values": normalized,
        "precondition": precondition,
    }


def localize_2mm_dimensions(
    code: str,
    header_code: str,
) -> tuple[str, dict[str, Any]]:
    """Replace generated 2MM dimension symbols with public-header literals."""

    dimensions = _resolve_2mm_dimensions(header_code)
    transformed = code
    replacement_counts: dict[str, int] = {}
    for name in ("NI", "NJ", "NK", "NL"):
        transformed, count = re.subn(
            rf"\b{name}\b",
            str(dimensions[name.lower()]),
            transformed,
        )
        replacement_counts[name] = count
    missing = [name for name, count in replacement_counts.items() if count == 0]
    if missing:
        raise ValueError(
            "2MM generated source is missing dimension symbols: "
            + ", ".join(missing)
        )
    return transformed, {
        "schema_version": "c2hls.2mm-dimension-localization.v1",
        "dimensions": dimensions,
        "replacement_counts": replacement_counts,
        "source_code_sha256": _sha256(code),
        "localized_code_sha256": _sha256(transformed),
        "source": "public_header_array_extents",
    }


def _resolve_gemm_nk(header_code: str) -> int:
    match = re.search(
        r"(?m)^\s*#\s*define\s+NK\s+(?P<value>\d+)\b",
        header_code or "",
    )
    if match is None:
        match = re.search(
            r"(?m)^\s*(?:static\s+)?(?:const|constexpr)\s+"
            r"(?:unsigned\s+)?(?:int|size_t)\s+NK\s*=\s*(?P<value>\d+)\b",
            header_code or "",
        )
    if match is None:
        raise ValueError("could not resolve compile-time GEMM NK")
    return int(match.group("value"))


def validate_gemm_lane_values(
    header_code: str,
    values: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    """Require every striped lane count to divide GEMM's K dimension."""

    nk = _resolve_gemm_nk(header_code)
    normalized = sorted({int(value) for value in values})
    if not normalized or any(value <= 0 for value in normalized):
        raise ValueError("GEMM lane values must be positive")
    incompatible = [value for value in normalized if nk % value != 0]
    if incompatible:
        raise ValueError(
            f"GEMM lane values {incompatible} do not divide NK={nk}"
        )
    return {
        "schema_version": "c2hls.gemm-lane-domain.v1",
        "nk": nk,
        "validated_values": normalized,
        "all_values_divide_nk": True,
    }


def validate_gemm_step_control_values(
    header_code: str,
    step: str,
    values: list[int] | tuple[int, ...],
    *,
    fixed_lanes: int = 8,
) -> dict[str, Any]:
    """Validate the legal domain for one linked GEMM step control."""

    normalized_step = str(step).strip().lower().replace("double_buffer", "doublebuffer")
    if normalized_step not in GEMM_STEP_CONTROL_SPECS:
        raise ValueError(f"unsupported GEMM step control: {step!r}")
    normalized = sorted({int(value) for value in values})
    if not normalized or any(value <= 0 for value in normalized):
        raise ValueError("GEMM step-control values must be positive")

    nk = _resolve_gemm_nk(header_code)
    errors: list[str] = []
    precondition = "positive_integer"
    if normalized_step in {"tiling", "doublebuffer"}:
        incompatible = [
            value
            for value in normalized
            if value > nk or nk % value != 0 or value % int(fixed_lanes) != 0
        ]
        if incompatible:
            errors.append(
                f"{normalized_step} values {incompatible} must divide NK={nk} and be multiples "
                f"of fixed_lanes={int(fixed_lanes)}"
            )
        precondition = "NK % tile == 0 and tile % fixed_lanes == 0"
    elif normalized_step == "coalescing":
        incompatible = [
            value
            for value in normalized
            if value < 64 or value > 1024 or value & (value - 1)
        ]
        if incompatible:
            errors.append(
                f"AXI widths {incompatible} must be powers of two in [64, 1024]"
            )
        precondition = "power_of_two_between_64_and_1024"
    elif normalized_step == "resource":
        incompatible = [value for value in normalized if value not in range(1, 6)]
        if incompatible:
            errors.append(
                f"multiplier implementation selectors {incompatible} must be in [1, 5]"
            )
        precondition = (
            "1=fabric, 2=meddsp, 3=fulldsp, 4=maxdsp, 5=primitivedsp"
        )
    if errors:
        raise ValueError("; ".join(errors))
    contract = {
        "schema_version": "c2hls.gemm-step-control-domain.v1",
        "step": normalized_step,
        "nk": nk,
        "fixed_lanes": int(fixed_lanes),
        "validated_values": normalized,
        "precondition": precondition,
    }
    return contract


def _insert_control_define(
    code: str,
    *,
    control_name: str,
    parent_value: int,
) -> str:
    lane_define = re.compile(r"(?m)^\s*#\s*define\s+GEMM_LANES\s+\d+\b[^\n]*$")
    match = lane_define.search(code)
    if match is None:
        raise ValueError("linked GEMM parent is missing GEMM_LANES")
    return code[:match.end()] + f"\n#define {control_name} {int(parent_value)}" + code[match.end():]


def _linked_control_contract(
    source_code: str,
    linked_code: str,
    *,
    step: str,
    parent_value: int,
    fixed_lanes: int,
    base_lane_contract: dict[str, Any],
) -> dict[str, Any]:
    spec = GEMM_STEP_CONTROL_SPECS[step]
    control_name = str(spec["control_name"])
    dependencies = []
    source_dependencies = []
    definition_pattern = re.compile(
        rf"^\s*#\s*define\s+{re.escape(control_name)}\b"
    )
    for line_number, line in enumerate(linked_code.splitlines(), start=1):
        if control_name not in line or definition_pattern.search(line):
            continue
        pragma = re.search(
            r"#\s*pragma\s+HLS\s+(?P<control>[A-Za-z_]+)", line, re.IGNORECASE
        )
        if pragma:
            role = "hls_directive"
            dependencies.append({
                "control": pragma.group("control").lower(),
                "line": line_number,
                "source": line.strip(),
            })
        elif re.search(rf"\[\s*{re.escape(control_name)}\s*\]", line):
            role = "array_extent"
        elif re.search(rf"\+=\s*{re.escape(control_name)}\b", line):
            role = "loop_stride"
        elif re.search(rf"<[^\n;]*\b{re.escape(control_name)}\b", line):
            role = "loop_bound"
        else:
            role = "source_expression"
        source_dependencies.append({
            "role": role,
            "line": line_number,
            "source": line.strip(),
        })
    if not dependencies and step not in {"tiling", "doublebuffer"}:
        raise ValueError(f"{control_name} is not linked to an HLS directive")
    if not source_dependencies:
        raise ValueError(f"{control_name} has no linked source dependencies")
    contract = {
        "schema_version": "c2hls.linked-qor-control.v2",
        "benchmark": "gemm",
        "step": step,
        "architecture": "striped_step_native",
        "control_name": control_name,
        "knob_kind": spec["knob_kind"],
        "parent_value": int(parent_value),
        "fixed_lanes": int(fixed_lanes),
        "semantic_role": spec["semantic_role"],
        "dependencies": dependencies,
        "dependency_count": len(dependencies),
        "source_dependencies": source_dependencies,
        "source_dependency_count": len(source_dependencies),
        "base_lane_control": base_lane_contract,
        "source_code_sha256": _sha256(source_code),
        "linked_parent_code_sha256": _sha256(linked_code),
    }
    if spec.get("value_labels"):
        contract["value_labels"] = {
            str(key): value for key, value in spec["value_labels"].items()
        }
    return contract


def build_gemm_step_control_parent(
    code: str,
    *,
    step: str,
    parent_value: int,
    fixed_lanes: int = 8,
) -> tuple[str, dict[str, Any]]:
    """Build a frozen GEMM parent with one step-native linked control."""

    normalized_step = str(step).strip().lower().replace("double_buffer", "doublebuffer")
    if normalized_step not in GEMM_STEP_CONTROL_SPECS:
        raise ValueError(f"unsupported GEMM step control: {step!r}")
    if int(parent_value) <= 0 or int(fixed_lanes) <= 0:
        raise ValueError("parent_value and fixed_lanes must be positive")

    transformed, lane_contract = build_gemm_linked_lanes_parent(
        code,
        parent_lanes=int(fixed_lanes),
        architecture="striped_accumulators",
    )
    spec = GEMM_STEP_CONTROL_SPECS[normalized_step]
    control_name = str(spec["control_name"])
    transformed = _insert_control_define(
        transformed,
        control_name=control_name,
        parent_value=int(parent_value),
    )

    compute_marker = "// Compute with independent reduction lanes."
    compute_offset = transformed.find(compute_marker)
    store_offset = transformed.find("// Store C", compute_offset)
    if compute_offset < 0 or store_offset < 0:
        raise ValueError("striped GEMM compute region is missing")

    if normalized_step == "pipeline":
        prefix = transformed[:compute_offset]
        compute = transformed[compute_offset:store_offset]
        compute = _replace_once(
            compute,
            r"(?m)^(\s*#\s*pragma\s+HLS\s+PIPELINE\s+II\s*=\s*)1\s*$",
            rf"\g<1>{control_name}",
            label="striped compute pipeline target",
        )
        transformed = prefix + compute + transformed[store_offset:]
    elif normalized_step == "tiling":
        old_loop = """\
            for (k = 0; k < nk; k += GEMM_LANES) {
#pragma HLS PIPELINE II=1
                for (int lane = 0; lane < GEMM_LANES; lane++) {
#pragma HLS UNROLL factor=GEMM_LANES
                    partial_sums[lane] +=
                        local_A[i][k + lane] * local_B[k + lane][j];
                }
            }
"""
        new_loop = f"""\
            for (int tile_index = 0; tile_index < NK / {control_name}; tile_index++) {{
                for (int tile_k = 0; tile_k < {control_name}; tile_k += GEMM_LANES) {{
#pragma HLS PIPELINE II=1
                    int reduction_k = tile_index * {control_name} + tile_k;
                    for (int lane = 0; lane < GEMM_LANES; lane++) {{
#pragma HLS UNROLL factor=GEMM_LANES
                        partial_sums[lane] +=
                            local_A[i][reduction_k + lane]
                            * local_B[reduction_k + lane][j];
                    }}
                }}
            }}
"""
        if transformed.count(old_loop) != 1:
            raise ValueError("expected one striped GEMM reduction loop")
        transformed = transformed.replace(old_loop, new_loop, 1)
    elif normalized_step == "doublebuffer":
        transformed = _replace_once(
            transformed,
            r"(?m)^\s*double\s+local_B\s*\[NK\]\[NJ\]\s*;\s*$",
            (
                f"    double b_ping[{control_name}][NJ];\n"
                f"    double b_pong[{control_name}][NJ];"
            ),
            label="GEMM local B buffer declaration",
        )
        transformed = _replace_once(
            transformed,
            r"(?m)^\s*#\s*pragma\s+HLS\s+ARRAY_PARTITION\s+"
            r"variable\s*=\s*local_B\s+cyclic\s+factor\s*=\s*GEMM_LANES\s+"
            r"dim\s*=\s*1\s*$",
            (
                "#pragma HLS ARRAY_PARTITION variable=b_ping cyclic "
                "factor=GEMM_LANES dim=1\n"
                "#pragma HLS ARRAY_PARTITION variable=b_pong cyclic "
                "factor=GEMM_LANES dim=1"
            ),
            label="GEMM local B partition",
        )
        load_b_start = transformed.find("    // Load B")
        load_c_start = transformed.find("    // Load C", load_b_start)
        if load_b_start < 0 or load_c_start < 0:
            raise ValueError("GEMM B-load region is missing")
        transformed = transformed[:load_b_start] + transformed[load_c_start:]
        compute_block = f"""\
    // Scale C once before accumulating buffered K tiles.
    for (i = 0; i < ni; i++) {{
        for (j = 0; j < nj; j++) {{
#pragma HLS PIPELINE II=1
            local_C[i][j] *= beta;
        }}
    }}

    // Alternate two K-tile buffers. The depth controls storage and tile work.
    for (int k_base = 0; k_base < NK; k_base += {control_name}) {{
        bool use_ping = ((k_base / {control_name}) & 1) == 0;
        if (use_ping) {{
            for (int kk = 0; kk < {control_name}; kk++) {{
                for (j = 0; j < nj; j++) {{
#pragma HLS PIPELINE II=1
                    b_ping[kk][j] = B[k_base + kk][j];
                }}
            }}
        }} else {{
            for (int kk = 0; kk < {control_name}; kk++) {{
                for (j = 0; j < nj; j++) {{
#pragma HLS PIPELINE II=1
                    b_pong[kk][j] = B[k_base + kk][j];
                }}
            }}
        }}

        for (i = 0; i < ni; i++) {{
            for (j = 0; j < nj; j++) {{
                double partial_sums[GEMM_LANES];
#pragma HLS ARRAY_PARTITION variable=partial_sums complete dim=1
                for (int lane = 0; lane < GEMM_LANES; lane++) {{
#pragma HLS UNROLL factor=GEMM_LANES
                    partial_sums[lane] = 0.0;
                }}
                for (int kk = 0; kk < {control_name}; kk += GEMM_LANES) {{
#pragma HLS PIPELINE II=1
                    for (int lane = 0; lane < GEMM_LANES; lane++) {{
#pragma HLS UNROLL factor=GEMM_LANES
                        double b_value = use_ping
                            ? b_ping[kk + lane][j]
                            : b_pong[kk + lane][j];
                        partial_sums[lane] +=
                            local_A[i][k_base + kk + lane] * b_value;
                    }}
                }}
                double tile_sum = 0.0;
                for (int lane = 0; lane < GEMM_LANES; lane++) {{
#pragma HLS UNROLL factor=GEMM_LANES
                    tile_sum += partial_sums[lane];
                }}
                local_C[i][j] += alpha * tile_sum;
            }}
        }}
    }}

"""
        compute_offset = transformed.find(compute_marker)
        store_offset = transformed.find("// Store C", compute_offset)
        compute_line_start = transformed.rfind("\n", 0, compute_offset) + 1
        store_line_start = transformed.rfind("\n", 0, store_offset) + 1
        transformed = (
            transformed[:compute_line_start]
            + compute_block
            + transformed[store_line_start:]
        )
    elif normalized_step == "coalescing":
        pattern = re.compile(
            r"(?im)^(?P<body>\s*#\s*pragma\s+HLS\s+INTERFACE\s+m_axi\b[^\n]*)$"
        )
        transformed, count = pattern.subn(
            rf"\g<body> max_widen_bitwidth={control_name}", transformed
        )
        if count != 3:
            raise ValueError(f"expected three GEMM m_axi interfaces; found {count}")
    else:
        product_expression = re.compile(
            r"(?m)^(\s*)partial_sums\[lane\]\s*\+=\s*\n"
            r"\s*local_A\[i\]\[k \+ lane\]\s*\*\s*local_B\[k \+ lane\]\[j\];\s*$"
        )
        replacement = rf"""\1double lane_product;
\1#if {control_name} == 1
\1#pragma HLS BIND_OP variable=lane_product op=dmul impl=fabric // {control_name}=1
\1#elif {control_name} == 2
\1#pragma HLS BIND_OP variable=lane_product op=dmul impl=meddsp // {control_name}=2
\1#elif {control_name} == 3
\1#pragma HLS BIND_OP variable=lane_product op=dmul impl=fulldsp // {control_name}=3
\1#elif {control_name} == 4
\1#pragma HLS BIND_OP variable=lane_product op=dmul impl=maxdsp // {control_name}=4
\1#elif {control_name} == 5
\1#pragma HLS BIND_OP variable=lane_product op=dmul impl=primitivedsp // {control_name}=5
\1#else
\1#error Unsupported {control_name}
\1#endif
\1lane_product = local_A[i][k + lane] * local_B[k + lane][j];
\1partial_sums[lane] += lane_product;"""
        transformed, count = product_expression.subn(
            replacement, transformed, count=1
        )
        if count != 1:
            raise ValueError("expected one GEMM lane multiplication")

    contract = _linked_control_contract(
        code,
        transformed,
        step=normalized_step,
        parent_value=int(parent_value),
        fixed_lanes=int(fixed_lanes),
        base_lane_contract=lane_contract,
    )
    return transformed, contract


def build_gemm_linked_lanes_parent(
    code: str,
    *,
    parent_lanes: int = 2,
    control_name: str = "GEMM_LANES",
    architecture: str = "direct_reduction",
) -> tuple[str, dict[str, Any]]:
    """Build a GEMM parent whose lane count controls compute and memory ports.

    The original generated GEMM pipelines the enclosing output-column loop.
    Vitis consequently completes the inner reduction unroll, so changing its
    explicit factor does not alter hardware.  This transformation removes that
    conflicting pipeline and links one macro to the reduction unroll plus the
    two local-memory dimensions read by every reduction lane.
    """

    if not code.strip():
        raise ValueError("GEMM source is empty")
    if int(parent_lanes) <= 0:
        raise ValueError("parent_lanes must be positive")
    if architecture not in {"direct_reduction", "striped_accumulators"}:
        raise ValueError(f"unsupported GEMM lane architecture: {architecture!r}")
    if not re.fullmatch(r"[A-Za-z_]\w*", control_name):
        raise ValueError(f"invalid control name: {control_name!r}")
    if re.search(rf"(?m)^\s*#\s*define\s+{re.escape(control_name)}\b", code):
        raise ValueError(f"source already defines {control_name}")

    transformed = _replace_once(
        code,
        r"(?m)^(\s*#\s*include[^\n]*\n)",
        rf"\1\n#define {control_name} {int(parent_lanes)}\n",
        label="include insertion point",
    )
    transformed = _replace_once(
        transformed,
        r"(?m)^(\s*#\s*pragma\s+HLS\s+ARRAY_PARTITION\s+"
        r"variable\s*=\s*local_A\s+cyclic\s+factor\s*=\s*)\d+"
        r"(\s+dim\s*=\s*2\b[^\n]*)$",
        rf"\g<1>{control_name}\2",
        label="local_A reduction-dimension partition",
    )
    transformed = _replace_once(
        transformed,
        r"(?m)^(\s*#\s*pragma\s+HLS\s+ARRAY_PARTITION\s+"
        r"variable\s*=\s*local_B\s+cyclic\s+factor\s*=\s*)\d+"
        r"(\s+dim\s*=\s*1\b[^\n]*)$",
        rf"\g<1>{control_name}\2",
        label="local_B reduction-dimension partition",
    )

    marker = "// Compute: C = alpha * A * B + beta * C"
    marker_offset = transformed.find(marker)
    if marker_offset < 0:
        raise ValueError("GEMM compute marker is missing")
    if architecture == "direct_reduction":
        compute_prefix = transformed[:marker_offset]
        compute_region = transformed[marker_offset:]
        unroll_match = re.search(
            r"(?m)^(\s*#\s*pragma\s+HLS\s+UNROLL\s+factor\s*=\s*)\d+([^\n]*)$",
            compute_region,
        )
        if unroll_match is None:
            raise ValueError("GEMM reduction unroll pragma is missing")
        before_unroll = compute_region[: unroll_match.start()]
        pipeline_matches = list(re.finditer(
            r"(?m)^\s*#\s*pragma\s+HLS\s+PIPELINE\b[^\n]*\n?",
            before_unroll,
        ))
        if len(pipeline_matches) != 1:
            raise ValueError(
                "expected one enclosing compute pipeline before the reduction; "
                f"found {len(pipeline_matches)}"
            )
        pipeline = pipeline_matches[0]
        compute_region = (
            compute_region[: pipeline.start()]
            + compute_region[pipeline.end() :]
        )
        compute_region, unroll_count = re.subn(
            r"(?m)^(\s*#\s*pragma\s+HLS\s+UNROLL\s+factor\s*=\s*)\d+([^\n]*)$",
            rf"\g<1>{control_name}\2",
            compute_region,
            count=1,
        )
        if unroll_count != 1:
            raise ValueError(
                f"expected one GEMM reduction unroll; found {unroll_count}"
            )
        transformed = compute_prefix + compute_region
    else:
        compute_line_start = transformed.rfind("\n", 0, marker_offset) + 1
        store_offset = transformed.find("// Store C", marker_offset)
        if store_offset < 0:
            raise ValueError("GEMM store marker is missing")
        store_line_start = transformed.rfind("\n", 0, store_offset) + 1
        compute_block = f"""\
    // Compute with independent reduction lanes.
    for (i = 0; i < ni; i++) {{
        for (j = 0; j < nj; j++) {{
            double partial_sums[{control_name}];
#pragma HLS ARRAY_PARTITION variable=partial_sums complete dim=1
            for (int lane = 0; lane < {control_name}; lane++) {{
#pragma HLS UNROLL factor={control_name}
                partial_sums[lane] = 0.0;
            }}
            for (k = 0; k < nk; k += {control_name}) {{
#pragma HLS PIPELINE II=1
                for (int lane = 0; lane < {control_name}; lane++) {{
#pragma HLS UNROLL factor={control_name}
                    partial_sums[lane] +=
                        local_A[i][k + lane] * local_B[k + lane][j];
                }}
            }}
            double sum = 0.0;
            for (int lane = 0; lane < {control_name}; lane++) {{
#pragma HLS UNROLL factor={control_name}
                sum += partial_sums[lane];
            }}
            local_C[i][j] = alpha * sum + beta * local_C[i][j];
        }}
    }}

"""
        transformed = (
            transformed[:compute_line_start]
            + compute_block
            + transformed[store_line_start:]
        )

    dependency_pattern = re.compile(
        rf"(?im)^\s*#\s*pragma\s+HLS\s+"
        rf"(?P<control>UNROLL|ARRAY_PARTITION)\b[^\n]*\b"
        rf"{re.escape(control_name)}\b[^\n]*$"
    )
    dependencies = [
        {
            "control": match.group("control").lower(),
            "line": transformed.count("\n", 0, match.start()) + 1,
            "source": match.group(0).strip(),
        }
        for match in dependency_pattern.finditer(transformed)
    ]
    expected_controls = ["array_partition", "array_partition", "unroll"]
    if architecture == "striped_accumulators":
        expected_controls.extend(["unroll", "unroll"])
    if sorted(item["control"] for item in dependencies) != sorted(expected_controls):
        raise ValueError(
            f"linked lane contract expected {expected_controls}, got "
            f"{[item['control'] for item in dependencies]}"
        )

    source_dependencies = []
    for line_number, line in enumerate(transformed.splitlines(), start=1):
        if control_name not in line or re.search(
            rf"^\s*#\s*define\s+{re.escape(control_name)}\b", line
        ):
            continue
        if "#pragma" in line:
            role = "hls_directive"
        elif re.search(rf"\[\s*{re.escape(control_name)}\s*\]", line):
            role = "array_extent"
        elif re.search(rf"\+=\s*{re.escape(control_name)}\b", line):
            role = "loop_stride"
        elif re.search(rf"<\s*{re.escape(control_name)}\b", line):
            role = "loop_bound"
        else:
            role = "source_expression"
        source_dependencies.append({
            "role": role,
            "line": line_number,
            "source": line.strip(),
        })

    derived_dependencies = []
    if architecture == "striped_accumulators":
        complete_partition = re.search(
            r"(?im)^\s*#\s*pragma\s+HLS\s+ARRAY_PARTITION\s+"
            r"variable\s*=\s*partial_sums\s+complete\s+dim\s*=\s*1\b[^\n]*$",
            transformed,
        )
        if complete_partition is None:
            raise ValueError("striped partial-sum complete partition is missing")
        derived_dependencies.append({
            "control": "array_partition_complete",
            "line": transformed.count("\n", 0, complete_partition.start()) + 1,
            "source": complete_partition.group(0).strip(),
            "derivation": (
                "Complete partition tracks the GEMM_LANES-sized partial_sums "
                "array extent."
            ),
        })

    contract = {
        "schema_version": "c2hls.linked-qor-control.v1",
        "benchmark": "gemm",
        "architecture": architecture,
        "control_name": control_name,
        "parent_value": int(parent_lanes),
        "semantic_role": "reduction_compute_lanes_and_k_dimension_memory_banks",
        "structural_precondition": (
            "scalar_reduction_replaced_by_independent_striped_accumulators"
            if architecture == "striped_accumulators"
            else "enclosing_compute_j_pipeline_removed"
        ),
        "dependencies": dependencies,
        "dependency_count": len(dependencies),
        "source_dependencies": source_dependencies,
        "source_dependency_count": len(source_dependencies),
        "derived_dependencies": derived_dependencies,
        "derived_dependency_count": len(derived_dependencies),
        "divisibility_precondition": (
            "NK % GEMM_LANES == 0"
            if architecture == "striped_accumulators"
            else None
        ),
        "source_code_sha256": _sha256(code),
        "linked_parent_code_sha256": _sha256(transformed),
    }
    return transformed, contract


def build_2mm_linked_lanes_parent(
    code: str,
    *,
    parent_lanes: int = 4,
    control_name: str = "MM2_LANES",
) -> tuple[str, dict[str, Any]]:
    """Couple both 2MM reductions to one boundary-safe lane control."""

    if not code.strip():
        raise ValueError("2MM source is empty")
    if int(parent_lanes) <= 0:
        raise ValueError("parent_lanes must be positive")
    if not re.fullmatch(r"[A-Za-z_]\w*", control_name):
        raise ValueError(f"invalid control name: {control_name!r}")
    if re.search(rf"(?m)^\s*#\s*define\s+{re.escape(control_name)}\b", code):
        raise ValueError(f"source already defines {control_name}")

    transformed = _replace_once(
        code,
        r"(?m)^(\s*#\s*include[^\n]*\n)",
        rf"\1\n#define {control_name} {int(parent_lanes)}\n",
        label="2MM include insertion point",
    )
    partition_specs = (
        ("local_A", 2, 2),
        ("local_B", 1, 1),
        ("local_C", 2, 1),
        ("local_tmp", 2, 2),
    )
    for variable, source_dim, target_dim in partition_specs:
        transformed = _replace_once(
            transformed,
            rf"(?m)^\s*#\s*pragma\s+HLS\s+ARRAY_PARTITION\s+"
            rf"variable\s*=\s*{variable}\s+cyclic\s+factor\s*=\s*\d+\s+"
            rf"dim\s*=\s*{source_dim}\s*$",
            (
                f"#pragma HLS ARRAY_PARTITION variable={variable} cyclic "
                f"factor={control_name} dim={target_dim}"
            ),
            label=f"2MM {variable} reduction-dimension partition",
        )

    phase1_offset = transformed.find("// ---- Phase 1:")
    store_offset = transformed.find("// ---- Store phase:", phase1_offset)
    if phase1_offset < 0 or store_offset < 0:
        raise ValueError("2MM compute phase markers are missing")
    prefix = transformed[:phase1_offset]
    compute = transformed[phase1_offset:store_offset]

    compute, partial_count = re.subn(
        r"double\s+partial\s*\[\s*4\s*\]",
        f"double partial[{control_name}]",
        compute,
    )
    if partial_count != 2:
        raise ValueError(f"expected two 2MM partial arrays; found {partial_count}")
    compute, bound_count = re.subn(
        r"for\s*\(\s*int\s+p\s*=\s*0\s*;\s*p\s*<\s*4\s*;\s*p\+\+\s*\)",
        f"for (int p = 0; p < {control_name}; p++)",
        compute,
    )
    if bound_count != 6:
        raise ValueError(f"expected six 2MM lane loops; found {bound_count}")
    compute, stride_k_count = re.subn(
        r"for\s*\(\s*int\s+k\s*=\s*0\s*;\s*k\s*<\s*nk\s*;\s*"
        r"k\s*\+=\s*4\s*\)",
        f"for (int k = 0; k < nk; k += {control_name})",
        compute,
    )
    compute, stride_j_count = re.subn(
        r"for\s*\(\s*int\s+j\s*=\s*0\s*;\s*j\s*<\s*nj\s*;\s*"
        r"j\s*\+=\s*4\s*\)",
        f"for (int j = 0; j < nj; j += {control_name})",
        compute,
    )
    if (stride_k_count, stride_j_count) != (1, 1):
        raise ValueError(
            "expected one stride-controlled reduction in each 2MM phase"
        )
    compute, outer_pipeline_count = re.subn(
        rf"(?m)^\s*#\s*pragma\s+HLS\s+PIPELINE\s+II\s*=\s*1\s*\n"
        rf"(?=\s*double\s+partial\s*\[\s*{re.escape(control_name)}\s*\])",
        "",
        compute,
    )
    if outer_pipeline_count != 2:
        raise ValueError(
            f"expected two enclosing 2MM compute pipelines; found {outer_pipeline_count}"
        )
    compute, unroll_count = re.subn(
        r"(?m)^(\s*#\s*pragma\s+HLS\s+UNROLL)\s*$",
        rf"\1 factor={control_name}",
        compute,
    )
    if unroll_count != 6:
        raise ValueError(f"expected six 2MM lane unrolls; found {unroll_count}")
    transformed = prefix + compute + transformed[store_offset:]

    dependency_pattern = re.compile(
        rf"(?im)^\s*#\s*pragma\s+HLS\s+"
        rf"(?P<control>UNROLL|ARRAY_PARTITION)\b[^\n]*\b"
        rf"{re.escape(control_name)}\b[^\n]*$"
    )
    dependencies = [
        {
            "control": match.group("control").lower(),
            "line": transformed.count("\n", 0, match.start()) + 1,
            "source": match.group(0).strip(),
        }
        for match in dependency_pattern.finditer(transformed)
    ]
    if len(dependencies) != 10:
        raise ValueError(
            f"2MM lane contract expected ten HLS dependencies; found {len(dependencies)}"
        )
    source_dependencies = []
    for line_number, line in enumerate(transformed.splitlines(), start=1):
        if control_name not in line or re.search(
            rf"^\s*#\s*define\s+{re.escape(control_name)}\b", line
        ):
            continue
        if "#pragma" in line:
            role = "hls_directive"
        elif re.search(rf"\[\s*{re.escape(control_name)}\s*\]", line):
            role = "array_extent"
        elif re.search(rf"\+=\s*{re.escape(control_name)}\b", line):
            role = "loop_stride"
        elif re.search(rf"<\s*{re.escape(control_name)}\b", line):
            role = "loop_bound"
        else:
            role = "source_expression"
        source_dependencies.append({
            "role": role,
            "line": line_number,
            "source": line.strip(),
        })
    complete_partitions = list(re.finditer(
        r"(?im)^\s*#\s*pragma\s+HLS\s+ARRAY_PARTITION\s+"
        r"variable\s*=\s*partial\s+complete\s+dim\s*=\s*1\b[^\n]*$",
        transformed,
    ))
    if len(complete_partitions) != 2:
        raise ValueError("2MM linked parent requires two complete partial partitions")
    derived_dependencies = [
        {
            "control": "array_partition_complete",
            "line": transformed.count("\n", 0, match.start()) + 1,
            "source": match.group(0).strip(),
            "derivation": (
                f"Complete partition tracks the {control_name}-sized partial array."
            ),
        }
        for match in complete_partitions
    ]
    contract = {
        "schema_version": "c2hls.linked-qor-control.v2",
        "benchmark": "2mm",
        "step": "unroll",
        "architecture": "two_boundary_guarded_striped_reductions",
        "control_name": control_name,
        "knob_kind": "compute_lanes",
        "parent_value": int(parent_lanes),
        "semantic_role": (
            "both_reduction_lane_counts_and_four_reduction_memory_bank_factors"
        ),
        "structural_precondition": (
            "enclosing_output_pipelines_removed_and_local_C_banked_on_reduction_dim"
        ),
        "dependencies": dependencies,
        "dependency_count": len(dependencies),
        "source_dependencies": source_dependencies,
        "source_dependency_count": len(source_dependencies),
        "derived_dependencies": derived_dependencies,
        "derived_dependency_count": len(derived_dependencies),
        "boundary_guards": ("k + p < nk", "j + p < nj"),
        "source_code_sha256": _sha256(code),
        "linked_parent_code_sha256": _sha256(transformed),
    }
    return transformed, contract


def _insert_2mm_control_define(
    code: str,
    *,
    control_name: str,
    parent_value: int,
) -> str:
    lane_define = re.compile(r"(?m)^\s*#\s*define\s+MM2_LANES\s+\d+\b[^\n]*$")
    match = lane_define.search(code)
    if match is None:
        raise ValueError("linked 2MM parent is missing MM2_LANES")
    return (
        code[:match.end()]
        + f"\n#define {control_name} {int(parent_value)}"
        + code[match.end():]
    )


def _two_mm_step_contract(
    source_code: str,
    linked_code: str,
    *,
    step: str,
    parent_value: int,
    fixed_lanes: int,
    base_lane_contract: dict[str, Any],
) -> dict[str, Any]:
    spec = TWO_MM_STEP_CONTROL_SPECS[step]
    control_name = str(spec["control_name"])
    dependencies = []
    source_dependencies = []
    definition_pattern = re.compile(
        rf"^\s*#\s*define\s+{re.escape(control_name)}\b"
    )
    for line_number, line in enumerate(linked_code.splitlines(), start=1):
        if control_name not in line or definition_pattern.search(line):
            continue
        pragma = re.search(
            r"#\s*pragma\s+HLS\s+(?P<control>[A-Za-z_]+)", line, re.IGNORECASE
        )
        if pragma:
            role = "hls_directive"
            dependencies.append({
                "control": pragma.group("control").lower(),
                "line": line_number,
                "source": line.strip(),
            })
        elif re.search(rf"\[\s*{re.escape(control_name)}\s*\]", line):
            role = "array_extent"
        elif re.search(rf"\+=\s*{re.escape(control_name)}\b", line):
            role = "loop_stride"
        elif re.search(rf"<[^\n;]*\b{re.escape(control_name)}\b", line):
            role = "loop_bound"
        else:
            role = "source_expression"
        source_dependencies.append({
            "role": role,
            "line": line_number,
            "source": line.strip(),
        })
    if not dependencies and step not in {"tiling", "doublebuffer"}:
        raise ValueError(f"{control_name} is not linked to an HLS directive")
    if not source_dependencies:
        raise ValueError(f"{control_name} has no linked source dependencies")
    contract = {
        "schema_version": "c2hls.linked-qor-control.v2",
        "benchmark": "2mm",
        "step": step,
        "architecture": "two_reduction_step_native",
        "control_name": control_name,
        "knob_kind": spec["knob_kind"],
        "parent_value": int(parent_value),
        "fixed_lanes": int(fixed_lanes),
        "semantic_role": spec["semantic_role"],
        "dependencies": dependencies,
        "dependency_count": len(dependencies),
        "source_dependencies": source_dependencies,
        "source_dependency_count": len(source_dependencies),
        "base_lane_control": base_lane_contract,
        "source_code_sha256": _sha256(source_code),
        "linked_parent_code_sha256": _sha256(linked_code),
    }
    if spec.get("value_labels"):
        contract["value_labels"] = {
            str(key): value for key, value in spec["value_labels"].items()
        }
    return contract


def _render_2mm_bound_product(
    *,
    indent: str,
    control_name: str,
    expression: str,
) -> str:
    lines = [
        f"{indent}double lane_product;",
        f"{indent}#if {control_name} == 1",
        f"{indent}#pragma HLS BIND_OP variable=lane_product op=dmul impl=fabric // {control_name}=1",
        f"{indent}#elif {control_name} == 2",
        f"{indent}#pragma HLS BIND_OP variable=lane_product op=dmul impl=meddsp // {control_name}=2",
        f"{indent}#elif {control_name} == 3",
        f"{indent}#pragma HLS BIND_OP variable=lane_product op=dmul impl=fulldsp // {control_name}=3",
        f"{indent}#elif {control_name} == 4",
        f"{indent}#pragma HLS BIND_OP variable=lane_product op=dmul impl=maxdsp // {control_name}=4",
        f"{indent}#elif {control_name} == 5",
        f"{indent}#pragma HLS BIND_OP variable=lane_product op=dmul impl=primitivedsp // {control_name}=5",
        f"{indent}#else",
        f"{indent}#error Unsupported {control_name}",
        f"{indent}#endif",
        f"{indent}lane_product = {expression};",
        f"{indent}partial[p] += lane_product;",
    ]
    return "\n".join(lines)


def build_2mm_step_control_parent(
    code: str,
    *,
    step: str,
    parent_value: int,
    fixed_lanes: int = 4,
) -> tuple[str, dict[str, Any]]:
    """Build a frozen 2MM parent with one step-native linked control."""

    normalized_step = str(step).strip().lower().replace(
        "double_buffer", "doublebuffer"
    )
    if normalized_step not in TWO_MM_STEP_CONTROL_SPECS:
        raise ValueError(f"unsupported 2MM step control: {step!r}")
    if int(parent_value) <= 0 or int(fixed_lanes) <= 0:
        raise ValueError("parent_value and fixed_lanes must be positive")
    transformed, lane_contract = build_2mm_linked_lanes_parent(
        code,
        parent_lanes=int(fixed_lanes),
    )
    spec = TWO_MM_STEP_CONTROL_SPECS[normalized_step]
    control_name = str(spec["control_name"])
    transformed = _insert_2mm_control_define(
        transformed,
        control_name=control_name,
        parent_value=int(parent_value),
    )
    phase1_marker = "// ---- Phase 1: tmp = alpha * A * B ----"
    phase2_marker = "// ---- Phase 2: D = beta * D + tmp * C ----"
    phase1_offset = transformed.find(phase1_marker)
    phase2_offset = transformed.find(phase2_marker, phase1_offset)
    store_offset = transformed.find("// ---- Store phase:", phase2_offset)
    if phase1_offset < 0 or phase2_offset < 0 or store_offset < 0:
        raise ValueError("2MM compute phase markers are missing")

    if normalized_step == "pipeline":
        prefix = transformed[:phase1_offset]
        compute = transformed[phase1_offset:store_offset]
        compute, count = re.subn(
            r"(?m)^(\s*#\s*pragma\s+HLS\s+PIPELINE\s+II\s*=\s*)1\s*$",
            rf"\g<1>{control_name}",
            compute,
        )
        if count != 2:
            raise ValueError(f"expected two 2MM reduction pipelines; found {count}")
        transformed = prefix + compute + transformed[store_offset:]
    elif normalized_step == "tiling":
        old_loop = """\
            for (int k = 0; k < nk; k += MM2_LANES) {
#pragma HLS PIPELINE II=1
                for (int p = 0; p < MM2_LANES; p++) {
#pragma HLS UNROLL factor=MM2_LANES
                    if (k + p < nk) {
                        partial[p] += local_A[i][k + p] * local_B[k + p][j];
                    }
                }
            }
"""
        new_loop = f"""\
            for (int k_tile = 0; k_tile < nk; k_tile += {control_name}) {{
                for (int tile_k = 0; tile_k < {control_name}; tile_k += MM2_LANES) {{
#pragma HLS PIPELINE II=1
                    int k = k_tile + tile_k;
                    for (int p = 0; p < MM2_LANES; p++) {{
#pragma HLS UNROLL factor=MM2_LANES
                        if (k + p < nk) {{
                            partial[p] += local_A[i][k + p] * local_B[k + p][j];
                        }}
                    }}
                }}
            }}
"""
        if transformed.count(old_loop) != 1:
            raise ValueError("expected one 2MM phase-1 reduction loop")
        transformed = transformed.replace(old_loop, new_loop, 1)
    elif normalized_step == "doublebuffer":
        transformed = _replace_once(
            transformed,
            r"(?m)^\s*double\s+local_B\s*\[NK\]\[NJ\]\s*;\s*$",
            (
                f"    double b_ping[{control_name}][NJ];\n"
                f"    double b_pong[{control_name}][NJ];"
            ),
            label="2MM local B buffer declaration",
        )
        transformed = _replace_once(
            transformed,
            r"(?m)^\s*#\s*pragma\s+HLS\s+ARRAY_PARTITION\s+"
            r"variable\s*=\s*local_B\s+cyclic\s+factor\s*=\s*MM2_LANES\s+"
            r"dim\s*=\s*1\s*$",
            (
                "#pragma HLS ARRAY_PARTITION variable=b_ping cyclic "
                "factor=MM2_LANES dim=1\n"
                "#pragma HLS ARRAY_PARTITION variable=b_pong cyclic "
                "factor=MM2_LANES dim=1"
            ),
            label="2MM local B partition",
        )
        load_b_offset = transformed.find("load_B:")
        load_c_offset = transformed.find("load_C:", load_b_offset)
        if load_b_offset < 0 or load_c_offset < 0:
            raise ValueError("2MM B-load region is missing")
        transformed = transformed[:load_b_offset] + transformed[load_c_offset:]
        phase1_offset = transformed.find(phase1_marker)
        phase2_offset = transformed.find(phase2_marker, phase1_offset)
        phase1_line_start = transformed.rfind("\n", 0, phase1_offset) + 1
        phase2_line_start = transformed.rfind("\n", 0, phase2_offset) + 1
        phase1_block = f"""\
    // ---- Phase 1: alternating B tiles feed tmp = alpha * A * B ----
    for (int i = 0; i < ni; i++) {{
        for (int j = 0; j < nj; j++) {{
#pragma HLS PIPELINE II=1
            local_tmp[i][j] = 0.0;
        }}
    }}

    for (int k_base = 0; k_base < nk; k_base += {control_name}) {{
        bool use_ping = ((k_base / {control_name}) & 1) == 0;
        for (int kk = 0; kk < {control_name}; kk++) {{
            for (int j = 0; j < nj; j++) {{
#pragma HLS PIPELINE II=1
                if (use_ping) {{
                    b_ping[kk][j] = B[k_base + kk][j];
                }} else {{
                    b_pong[kk][j] = B[k_base + kk][j];
                }}
            }}
        }}
        for (int i = 0; i < ni; i++) {{
            for (int j = 0; j < nj; j++) {{
                double partial[MM2_LANES];
#pragma HLS ARRAY_PARTITION variable=partial complete dim=1
                for (int p = 0; p < MM2_LANES; p++) {{
#pragma HLS UNROLL factor=MM2_LANES
                    partial[p] = 0.0;
                }}
                for (int kk = 0; kk < {control_name}; kk += MM2_LANES) {{
#pragma HLS PIPELINE II=1
                    for (int p = 0; p < MM2_LANES; p++) {{
#pragma HLS UNROLL factor=MM2_LANES
                        double b_value = use_ping
                            ? b_ping[kk + p][j]
                            : b_pong[kk + p][j];
                        partial[p] += local_A[i][k_base + kk + p] * b_value;
                    }}
                }}
                double tile_sum = 0.0;
                for (int p = 0; p < MM2_LANES; p++) {{
#pragma HLS UNROLL factor=MM2_LANES
                    tile_sum += partial[p];
                }}
                local_tmp[i][j] += alpha * tile_sum;
            }}
        }}
    }}

"""
        transformed = (
            transformed[:phase1_line_start]
            + phase1_block
            + transformed[phase2_line_start:]
        )
    elif normalized_step == "coalescing":
        pattern = re.compile(
            r"(?im)^(?P<body>\s*#\s*pragma\s+HLS\s+INTERFACE\s+m_axi\b[^\n]*)$"
        )
        transformed, count = pattern.subn(
            rf"\g<body> max_widen_bitwidth={control_name}", transformed
        )
        if count != 5:
            raise ValueError(f"expected five 2MM m_axi interfaces; found {count}")
    else:
        phase1_product = re.compile(
            r"(?m)^(?P<indent>\s*)partial\[p\]\s*\+=\s*"
            r"local_A\[i\]\[k \+ p\]\s*\*\s*local_B\[k \+ p\]\[j\];\s*$"
        )
        phase2_product = re.compile(
            r"(?m)^(?P<indent>\s*)partial\[p\]\s*\+=\s*"
            r"local_tmp\[i\]\[j \+ p\]\s*\*\s*local_C\[j \+ p\]\[l\];\s*$"
        )
        first = phase1_product.search(transformed)
        second = phase2_product.search(transformed)
        if first is None or second is None:
            raise ValueError("expected one multiplication in each 2MM reduction")
        transformed = phase1_product.sub(
            _render_2mm_bound_product(
                indent=first.group("indent"),
                control_name=control_name,
                expression="local_A[i][k + p] * local_B[k + p][j]",
            ),
            transformed,
            count=1,
        )
        second = phase2_product.search(transformed)
        if second is None:
            raise ValueError("2MM phase-2 multiplication disappeared during binding")
        transformed = phase2_product.sub(
            _render_2mm_bound_product(
                indent=second.group("indent"),
                control_name=control_name,
                expression="local_tmp[i][j + p] * local_C[j + p][l]",
            ),
            transformed,
            count=1,
        )

    contract = _two_mm_step_contract(
        code,
        transformed,
        step=normalized_step,
        parent_value=int(parent_value),
        fixed_lanes=int(fixed_lanes),
        base_lane_contract=lane_contract,
    )
    return transformed, contract
