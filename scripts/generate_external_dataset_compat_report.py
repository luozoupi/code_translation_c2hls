"""Generate `artifacts/external_dataset_compat_<timestamp>.md`.

Walks every cloned external HLS dataset under external_datasets/, classifies
each C/C++ source file by compatibility with the c2hls plain.cpp input
shape, and demonstrates end-to-end adaptation on one external kernel
(CollectiveHLS knn). Produces a single markdown artifact + leaves the
adapted bench dir under C2HLS_TMP_ROOT for inspection.

Run:
    cd /home/luo00466/code_translation-c2hls
    python scripts/generate_external_dataset_compat_report.py
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from c2hls_temp import configure_temp_env  # noqa: E402
from dataset_pipeline.external_adapter import (  # noqa: E402
    adapt_external_kernel,
    render_survey_markdown,
    survey_dataset,
)


EXT = REPO / "external_datasets"
DATASETS = ("HLSPilot", "HLSyn", "HLSFactory", "CollectiveHLS", "hls-eval")
DEMO_OUT = configure_temp_env(create=True) / "c2hls_external_adapter_demo"


def main() -> int:
    reports = []
    for name in DATASETS:
        print(f"... surveying {name}")
        reports.append(survey_dataset(EXT / name, name))

    if DEMO_OUT.exists():
        shutil.rmtree(DEMO_OUT)
    print()
    print("=== adapt CollectiveHLS knn ===")
    knn = EXT / "CollectiveHLS/Applications/RodiniaHLS-KNN-Pipeline"
    adapt = adapt_external_kernel(
        kernel_path=knn / "knn.cpp",
        bench_name="external_knn",
        output_dir=DEMO_OUT / "external_knn",
        header_path=knn / "knn.h",
        source_repo="CollectiveHLS",
        top_function="workload",
    )
    print(json.dumps(adapt, indent=2))

    plain = (DEMO_OUT / "external_knn/plain.cpp").read_text()
    has_pragma = bool(re.search(r"#pragma\s+HLS", plain))
    has_extern_c = 'extern "C"' in plain

    gpp = subprocess.run(
        [
            "g++", "-fsyntax-only", "-x", "c++", "-std=c++17",
            str(DEMO_OUT / "external_knn/plain.cpp"),
            "-I", str(DEMO_OUT / "external_knn"),
        ],
        capture_output=True, text=True,
    )
    print()
    print(f'plain.cpp contains "#pragma HLS": {has_pragma} (expect False)')
    print(f'plain.cpp contains \'extern "C"\': {has_extern_c} (expect False)')
    print(f"g++ -fsyntax-only exit: {gpp.returncode}")
    if gpp.returncode != 0:
        print("g++ stderr (first 600):", gpp.stderr[:600])

    body = render_survey_markdown(reports)

    demo_md = []
    demo_md.append("\n## Adaptation Demo: CollectiveHLS knn → c2hls bench dir\n")
    demo_md.append(
        "Reused `_strip_hls_constructs()` from "
        "[prepare_benchmarks.py](../prepare_benchmarks.py) via "
        "`dataset_pipeline.external_adapter.adapt_external_kernel()`. The "
        f"adapted bench dir lives at `{DEMO_OUT / 'external_knn'}` "
        "and follows the same shape as `benchmarks/<bench>/` "
        "(`plain.cpp` + `hls_baseline.cpp` + `<bench>.h` + `metadata.json`).\n"
    )
    demo_md.append("**Strip report**:")
    demo_md.append("")
    demo_md.append("```json")
    demo_md.append(json.dumps(adapt, indent=2))
    demo_md.append("```")
    demo_md.append("")
    demo_md.append("**Smoke checks on adapted `plain.cpp`**:")
    demo_md.append("")
    demo_md.append(f"- contains `#pragma HLS`: **{has_pragma}** (expect False)")
    demo_md.append(f'- contains `extern "C"`: **{has_extern_c}** (expect False)')
    demo_md.append(f"- `g++ -fsyntax-only` exit code: **{gpp.returncode}**")
    demo_md.append("")
    demo_md.append(
        "Note: `g++` syntax-check exits non-zero on some external kernels when "
        "they reference helpers not present in the c2hls support tree (e.g. "
        "`MARS_WIDE_BUS_TYPE`, `mc.h`). The adapted bench dir is still usable "
        "by Vitis HLS (which has its own header search path) — `g++` is just a "
        "cheap pre-flight, not a hard gate."
    )

    body += "\n".join(demo_md)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPO / "artifacts" / f"external_dataset_compat_{ts}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    print()
    print(f"survey artifact written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
