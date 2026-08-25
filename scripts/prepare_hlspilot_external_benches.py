#!/usr/bin/env python3
"""Materialize a conservative, testable HLSPilot subset for c2hls sweeps."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dataset_pipeline.external_adapter import adapt_external_kernel  # noqa: E402


HLSPILOT = REPO / "external_datasets" / "HLSPilot" / "benchmark"
DEFAULT_OUTPUT_ROOT = REPO / "benchmarks_external" / "HLSPilot" / "simple"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    bench_name: str
    kernel: str
    header: str
    top: str
    cosim_depths: dict[str, int]
    testbench: Callable[[], str]
    disabled_reason: str = ""


def _matrix_tb() -> str:
    return r'''#include <stdio.h>
#include "matrix_multiplication.h"

int main() {
    BaseType A[N][M];
    BaseType B[M][P];
    BaseType AB[N][P];
    BaseType ref[N][P];
    int errors = 0;

    for (int i = 0; i < N; i++)
        for (int k = 0; k < M; k++)
            A[i][k] = (i * 3 + k * 5 + 1) % 17 - 8;
    for (int k = 0; k < M; k++)
        for (int j = 0; j < P; j++)
            B[k][j] = (k * 7 + j * 2 + 3) % 19 - 9;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            BaseType sum = 0;
            AB[i][j] = 0;
            for (int k = 0; k < M; k++)
                sum += A[i][k] * B[k][j];
            ref[i][j] = sum;
        }
    }

    matrix_mul(A, B, AB);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < P; j++)
            if (AB[i][j] != ref[i][j]) {
                if (errors < 8)
                    printf("FAIL AB[%d][%d]=%d expected %d\n", i, j, AB[i][j], ref[i][j]);
                errors++;
            }
    printf(errors ? "FAIL matrix_mul mismatches=%d\n" : "PASS matrix_mul\n", errors);
    return errors ? 1 : 0;
}
'''


def _histogram_tb() -> str:
    return r'''#include <stdio.h>
#include "histogram.h"

int main() {
    int in[INPUT_SIZE];
    int hist[VALUE_SIZE];
    int ref[VALUE_SIZE];
    int errors = 0;

    for (int i = 0; i < INPUT_SIZE; i++)
        in[i] = (i * 37 + 5) % VALUE_SIZE;
    for (int i = 0; i < VALUE_SIZE; i++) {
        hist[i] = 0;
        ref[i] = 0;
    }
    for (int i = 0; i < INPUT_SIZE; i++)
        ref[in[i]]++;

    histogram(in, hist);

    for (int i = 0; i < VALUE_SIZE; i++)
        if (hist[i] != ref[i]) {
            if (errors < 8)
                printf("FAIL hist[%d]=%d expected %d\n", i, hist[i], ref[i]);
            errors++;
        }
    printf(errors ? "FAIL histogram mismatches=%d\n" : "PASS histogram\n", errors);
    return errors ? 1 : 0;
}
'''


def _fir_tb() -> str:
    return r'''#include <stdio.h>
#include "fir.h"

int main() {
    int taps[NUM_TAPS] = {1, -2, 3, 4};
    int samples[8] = {2, -1, 3, 0, 4, 5, -2, 1};
    int delay[NUM_TAPS] = {0};
    int errors = 0;

    for (int n = 0; n < 8; n++) {
        int out = 0;
        for (int i = NUM_TAPS - 1; i > 0; i--)
            delay[i] = delay[i - 1];
        delay[0] = samples[n];
        int ref = 0;
        for (int i = 0; i < NUM_TAPS; i++)
            ref += delay[i] * taps[i];

        fir(samples[n], &out, taps);
        if (out != ref) {
            printf("FAIL fir sample %d got %d expected %d\n", n, out, ref);
            errors++;
        }
    }
    printf(errors ? "FAIL fir mismatches=%d\n" : "PASS fir\n", errors);
    return errors ? 1 : 0;
}
'''


def _spmv_tb() -> str:
    return r'''#include <math.h>
#include <stdio.h>
#include "spmv.h"

int main() {
    int rowPtr[NUM_ROWS + 1] = {0, 2, 4, 7, 9};
    int columnIdx[NNZ] = {0, 2, 1, 3, 0, 2, 3, 1, 2};
    DTYPE values[NNZ] = {1.0, 2.0, -1.0, 3.0, 4.0, 1.5, -2.0, 2.5, 1.0};
    DTYPE x[SIZE] = {1.0, -2.0, 3.0, 4.0};
    DTYPE y[SIZE] = {0};
    DTYPE ref[SIZE] = {0};
    int errors = 0;

    for (int i = 0; i < NUM_ROWS; i++)
        for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++)
            ref[i] += values[k] * x[columnIdx[k]];

    spmv(rowPtr, columnIdx, values, y, x);

    for (int i = 0; i < SIZE; i++)
        if (fabs(y[i] - ref[i]) > 1.0e-4) {
            printf("FAIL y[%d]=%.6f expected %.6f\n", i, y[i], ref[i]);
            errors++;
        }
    printf(errors ? "FAIL spmv mismatches=%d\n" : "PASS spmv\n", errors);
    return errors ? 1 : 0;
}
'''


def _merge_sort_tb() -> str:
    return r'''#include <math.h>
#include <stdio.h>
#include "merge_sort.h"

static void ref_sort(DTYPE a[SIZE]) {
    for (int i = 1; i < SIZE; i++) {
        DTYPE item = a[i];
        int j = i;
        while (j > 0 && a[j - 1] > item) {
            a[j] = a[j - 1];
            j--;
        }
        a[j] = item;
    }
}

int main() {
    DTYPE data[SIZE];
    DTYPE ref[SIZE];
    int errors = 0;

    for (int i = 0; i < SIZE; i++) {
        data[i] = (DTYPE)((SIZE * 13 - i * 7 + (i % 3) * 5) % 31);
        ref[i] = data[i];
    }
    ref_sort(ref);
    merge_sort(data);

    for (int i = 0; i < SIZE; i++)
        if (fabs(data[i] - ref[i]) > 1.0e-4) {
            printf("FAIL A[%d]=%.6f expected %.6f\n", i, data[i], ref[i]);
            errors++;
        }
    printf(errors ? "FAIL merge_sort mismatches=%d\n" : "PASS merge_sort\n", errors);
    return errors ? 1 : 0;
}
'''


CASES: dict[str, CaseSpec] = {
    "matrix_multiplication": CaseSpec(
        name="matrix_multiplication",
        bench_name="hlspilot_matrix_multiplication",
        kernel="matrix_multiplication/matrix_multiplication.cpp",
        header="matrix_multiplication/matrix_multiplication.h",
        top="matrix_mul",
        cosim_depths={"A": 1024, "B": 1024, "AB": 1024},
        testbench=_matrix_tb,
    ),
    "histogram": CaseSpec(
        name="histogram",
        bench_name="hlspilot_histogram",
        kernel="histogram/histogram.cpp",
        header="histogram/histogram.h",
        top="histogram",
        cosim_depths={"in": 8, "hist": 256},
        testbench=_histogram_tb,
    ),
    "fir": CaseSpec(
        name="fir",
        bench_name="hlspilot_fir",
        kernel="fir/fir.cpp",
        header="fir/fir.h",
        top="fir",
        cosim_depths={"output": 1, "taps": 4},
        testbench=_fir_tb,
    ),
    "spmv": CaseSpec(
        name="spmv",
        bench_name="hlspilot_spmv",
        kernel="spmv/spmv.cpp",
        header="spmv/spmv.h",
        top="spmv",
        cosim_depths={"rowPtr": 5, "columnIdx": 9, "values": 9, "y": 4, "x": 4},
        testbench=_spmv_tb,
    ),
}


KNOWN_FAILING_CASES: dict[str, CaseSpec] = {
    "merge_sort": CaseSpec(
        name="merge_sort",
        bench_name="hlspilot_merge_sort",
        kernel="merge_sort/merge_sort.cpp",
        header="merge_sort/merge_sort.h",
        top="merge_sort",
        cosim_depths={"A": 16},
        testbench=_merge_sort_tb,
        disabled_reason=(
            "HLSPilot merge_sort baseline failed independent generated csim "
            "reference; excluded from default testable subset"
        ),
    ),
}


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _plain_has_strip_leak(info: dict) -> bool:
    strip = info.get("strip_report") or {}
    return bool(
        strip.get("plain_contains_hls_pragmas")
        or strip.get("plain_contains_accel_pragmas")
        or strip.get("plain_contains_ap_uint")
    )


def _normalize_header_prototypes(header_path: Path | None) -> None:
    if not header_path or not header_path.exists():
        return
    text = header_path.read_text(encoding="utf-8")
    # `extern` on ordinary function prototypes is semantically redundant, but
    # c2hls' signature compatibility check treats it as a different testbench
    # contract. Normalize generated benchmark headers so validation follows the
    # actual callable signature.
    text = re.sub(r"(?m)^(\s*)extern\s+(?=[A-Za-z_][\w:<>,\s*&]*\s+[A-Za-z_]\w*\s*\()", r"\1", text)
    header_path.write_text(text, encoding="utf-8")


def _materialize_case(spec: CaseSpec, output_root: Path) -> dict:
    out_dir = output_root / spec.bench_name
    info = adapt_external_kernel(
        kernel_path=HLSPILOT / spec.kernel,
        header_path=HLSPILOT / spec.header,
        bench_name=spec.bench_name,
        output_dir=out_dir,
        source_repo="HLSPilot",
        top_function=spec.top,
    )
    header_name = info.get("header_copied")
    _normalize_header_prototypes(out_dir / header_name if header_name else None)
    (out_dir / "testbench.cpp").write_text(spec.testbench(), encoding="utf-8")
    meta_path = out_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta.update(
        {
            "testbench_file": "testbench.cpp",
            "supports_csim": True,
            "supports_cosim": False,
            "cosim_depths": spec.cosim_depths,
            "cosim_harness": "vitis_hls_c_rtl",
            "cosim_disabled_reason": (
                "HLSPilot adapter validated csynth/csim only; reference cosim "
                "hit Vitis 2023.2 XSIM runtime failure after C TB pass"
            ),
            "hlspilot_adapter": {
                "case": spec.name,
                "testbench_policy": "deterministic_small_input_against_independent_c_reference",
            },
        }
    )
    if spec.disabled_reason:
        meta["status"] = "disabled"
        meta["disabled_reason"] = spec.disabled_reason
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {
        "case": spec.name,
        "bench_name": spec.bench_name,
        "status": "skip" if _plain_has_strip_leak(info) else "ok",
        "output_dir": str(out_dir),
        **info,
        "testbench_copied": "generated",
        "cosim_depths": spec.cosim_depths,
    }


def _disable_stale_known_failing(output_root: Path, selected_names: set[str]) -> list[dict]:
    rows: list[dict] = []
    for name, spec in KNOWN_FAILING_CASES.items():
        if name in selected_names:
            continue
        meta_path = output_root / spec.bench_name / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        meta["status"] = "disabled"
        meta["disabled_reason"] = spec.disabled_reason
        meta["supports_cosim"] = False
        meta["cosim_disabled_reason"] = spec.disabled_reason
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        rows.append(
            {
                "case": name,
                "bench_name": spec.bench_name,
                "status": "disabled",
                "output_dir": str(meta_path.parent),
                "reason": spec.disabled_reason,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--benches", default="matrix_multiplication,histogram,fir,spmv")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--exclude", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-known-failing", action="store_true",
                        help="Also materialize cases whose source baseline failed independent csim.")
    args = parser.parse_args()

    available = dict(CASES)
    if args.include_known_failing:
        available.update(KNOWN_FAILING_CASES)
    names = sorted(available) if args.all else _split_csv(args.benches)
    excluded = set(_split_csv(args.exclude))
    names = [name for name in names if name not in excluded]
    if args.limit > 0:
        names = names[: args.limit]

    unknown = [name for name in names if name not in available]
    if unknown:
        raise SystemExit(f"unknown HLSPilot case(s): {unknown}; valid={sorted(available)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    selected_names = set(names)
    rows = [_materialize_case(available[name], args.output_root) for name in names]
    rows.extend(_disable_stale_known_failing(args.output_root, selected_names))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = REPO / "artifacts" / f"hlspilot_materialized_{stamp}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "source_root": str(HLSPILOT),
                "output_root": str(args.output_root),
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "ok": sum(1 for row in rows if row.get("status") == "ok"),
                "skip": sum(1 for row in rows if row.get("status") != "ok"),
                "output_root": str(args.output_root),
            },
            indent=2,
        )
    )
    return 0 if any(row.get("status") == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
