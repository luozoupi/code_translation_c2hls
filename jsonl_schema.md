# JSON Lines Result Schema Contract

This document defines the JSON Lines contract for extracted Vitis result records.

Each JSONL file contains one JSON object per line. The current schema version is `1.0`.

Each record contains a common envelope plus exactly one report-specific payload. The payload key must match `report_type`.


## Core Principles

The goal of this schema is to provide a unified format to compare HLS implementations, specifically among AI generated implementations and baseline or human-optimized ground-truths.

As such, the schema should be:
- Universal
  - Data recorded should be complete enough for a robust comparison between any HLS implementations
  - Format should support both benchmark results and AI or custom implementation results
  - Collected data should not depend on a specific implementation's architecture or implementation details
- Easy to generate
  - Required data should only need parsing of a minimal number of build files
  - Data should ideally only come from structured/parsable files (ie. XML, JSON, CSV, etc)
  - Outputting a valid run result should be independent of other runs or the rest of a suite


## Common Record

Every result record has this shape:

```json
{
  "schema_version": "1.0",
  "report_type": "rtl_sim",
  "run": {
    "target": "vitis.hw_emu",
    "device": "xilinx_u280_gen3x16_xdma_1_202211_1",
    "vitis_version": "2023.2",
    "runtime_seconds": 60.25
  },
  "problem": {
    "suite": "rodinia_hls",
    "group_path": ["StreamCluster"]
  },
  "implementation": {
    "origin": "rodinia_hls_benchmark",
    "origin_version": null,
    "origin_meta": null,
    "variant": {
      "index": 4,
      "name": "coalescing"
    }
  }
}
```

## Common Fields

```text
schema_version: "1.0"
report_type: "sw_run" | "hls_synth" | "rtl_sim"

run:
    target: string
    device: string
    vitis_version: string
    runtime_seconds: number | null

problem:
    suite: string
    group_path: string[]

implementation:
    origin: string
    origin_version: string | null
    origin_meta: object | null
    variant:
        index: integer
        name: string
```

The `report_type` specifies the payload shape. The `run.target` specifies the concrete flow used to generate the data for the record. The available combinations are:
```text
sw_run run.target:    vitis.sw_emu | vitis.csim
hls_synth run.target: vitis.csynth | vitis.hw_emu | vitis.cosim
rtl_sim run.target:   vitis.hw_emu | vitis.cosim
hw_build run.target:  vitis.hw (TBD)
hw_run run.target:    vitis.hw (TBD)
```

While runs such as `vitis.sw_emu` and `vitis.csim` (and `vitis.hw_emu` and `vitis.cosim`) are not identical in implementation, they serve the same objective and have the same key outputs. As such, they share the same report schema.

Similarly, the `hls_synth` report can be generated from `vitis.hw_emu`, `vitis.cosim`, or a dedicated `vitis.csynth` target, but all produce the same information.

`origin` and `origin_version` must uniquely identify a "run producer". For AI harnesses it is suggested `origin` be the harness name and `origin_version` be either a unique string to distinguish between multiple runs with different behaviour. For mature harnesses, it could be a final version number or for in development harnesses it could be a git commit hash.

`run.runtime_seconds` is wall time of the run. Even if a single run generates multiple reports (ie `hw_emu` produces both `hls_synth` and `rtl_sim`), it is the runtime of the entire job.

`implementation.variant` is required and must not be `null`.

For origins with only one implementation variant and no meaningful variant name, use this explicit default:

```json
{
  "index": 0,
  "name": "implementation"
}
```

Consumers must not infer this default from a missing or `null` value. A missing or `null` variant is a schema violation.

## Payload Rules

Each record must include exactly one payload whose key matches `report_type`. The payloads currently defined here are:

```text
sw_run     -> sw_run
hls_synth  -> hls_synth
rtl_sim    -> rtl_sim
```

Required payload fields should be present even when a value was not observed. Use `null` for unavailable measurements.

### `sw_run` Payload

```json
{
  "sw_run": {
    "status": "pass"
  }
}
```

```text
status: "pass" | "fail" | "timeout"
```

### `hls_synth` Payload

`hls_synth` is a structured subset of `csynth.xml`. XML section names are preserved so extracted records remain easy to trace back to the source report.

Scalar values copied from `csynth.xml` remain strings, including numeric-looking values and values with units. Consumers that need numeric values should parse them into derived fields outside this source-report payload.

```json
{
  "hls_synth": {
    "status": "pass",
    "ReportVersion": {
      "Version": "2023.2"
    },
    "UserAssignments": {
      "unit": "ns",
      "ProductFamily": "virtexuplusHBM",
      "Part": "xcu50-fsvh2104-2-e",
      "TopModelName": "workload",
      "TargetClockPeriod": "3.33",
      "ClockUncertainty": "0.90",
      "FlowTarget": "vitis"
    },
    "PerformanceEstimates": {
      "SummaryOfTimingAnalysis": {
        "unit": "ns",
        "EstimatedClockPeriod": "2.433"
      },
      "SummaryOfOverallLatency": {
        "unit": "clock cycles",
        "Best-caseLatency": "1755138",
        "Average-caseLatency": "1795074",
        "Worst-caseLatency": "1832962",
        "Best-caseRealTimeLatency": "5.850 ms",
        "Average-caseRealTimeLatency": "5.983 ms",
        "Worst-caseRealTimeLatency": "6.109 ms",
        "Interval-min": "1755139",
        "Interval-max": "1832963"
      }
    },
    "AreaEstimates": {
      "Resources": {
        "BRAM_18K": "62",
        "DSP": "5",
        "FF": "25381",
        "LUT": "22917",
        "URAM": "0"
      },
      "AvailableResources": {
        "BRAM_18K": "2688",
        "DSP": "5952",
        "FF": "1743360",
        "LUT": "871680",
        "URAM": "640"
      }
    }
  }
}
```

```text
status: "pass" | "fail" | "timeout"

ReportVersion: object | null
    Version: string | null

UserAssignments: object | null
    unit: string | null
    ProductFamily: string | null
    Part: string | null
    TopModelName: string | null
    TargetClockPeriod: string | null
    ClockUncertainty: string | null
    FlowTarget: string | null

PerformanceEstimates: object | null
    SummaryOfTimingAnalysis:
        unit: string | null
        EstimatedClockPeriod: string | null
    SummaryOfOverallLatency:
        unit: string | null
        Best-caseLatency: string | null
        Average-caseLatency: string | null
        Worst-caseLatency: string | null
        Best-caseRealTimeLatency: string | null
        Average-caseRealTimeLatency: string | null
        Worst-caseRealTimeLatency: string | null
        Interval-min: string | null
        Interval-max: string | null

AreaEstimates: object | null
    Resources:
        BRAM_18K: string | null
        DSP: string | null
        FF: string | null
        LUT: string | null
        URAM: string | null
    AvailableResources:
        BRAM_18K: string | null
        DSP: string | null
        FF: string | null
        LUT: string | null
        URAM: string | null
```

When `status` is `pass`, `ReportVersion`, `UserAssignments`, `PerformanceEstimates`, and `AreaEstimates` must be objects with the fields shown above. When `status` is `fail` or `timeout`, those keys must still be present but their values may be `null`.

For example, a timed-out synthesis payload may use this shape:

```json
{
  "hls_synth": {
    "status": "timeout",
    "ReportVersion": null,
    "UserAssignments": null,
    "PerformanceEstimates": null,
    "AreaEstimates": null
  }
}
```

### `rtl_sim` Payload

```json
{
  "rtl_sim": {
    "status": "timeout",
    "kernel_runtime_cycles": null,
    "kernel_runtime_us": null,
    "kernel_clock_freq_mhz": null
  }
}
```

```text
status: "pass" | "fail" | "timeout"
kernel_runtime_cycles: integer | null
kernel_runtime_us: number | null
kernel_clock_freq_mhz: number | null
```

For target `vitis.hw_emu`:
- must be compiled and linked with the `-g` flag to ensure reliable profiling.
- `kernel_runtime_us` should be collected from `profile_kernels.csv` and `kernel_clock_freq_mhz` should be collected from `systemDiagramModel.json` as the clock with id 0.
- `kernel_runtime_cycles` must be derived from `kernel_runtime_us` and `kernel_clock_freq_mhz` and explicitly included, rounded to the nearest integer.

For target `vitis.cosim`:
- `kernel_runtime_cycles` should be collected from `lat.rpt`
- `kernel_runtime_us` and `kernel_clock_freq_mhz` should be left `null`


## Collecting Data From Vitis Build Files

Collectors should generate each JSONL row from the smallest stable set of structured build artifacts. Runtime duration and pass/fail inference may come from the runner, but source-report fields should come from Vitis output files.

### Common Build Metadata

Prefer explicit runner metadata when available, for example:

```text
report_type=rtl_sim
target=vitis.hw_emu
device=xilinx_u280_gen3x16_xdma_1_202211_1
vitis_version=2023.2
```

When metadata is missing, these fields can often be recovered from `v++` logs:

```text
INFO: [v++ 60-585] Compiling for hardware emulation target
INFO: [v++ 60-423]   Target device: xilinx_u280_gen3x16_xdma_1_202211_1
INFO: [v++ 74-78] Compiler Version string: 2023.2
```

The `problem` and `implementation` fields usually come from the project path. For example:

```text
Benchmarks/cfd/cfd_step_factor/cfd_step_factor_2_pipeline
```

maps to:

```json
{
  "problem": {
    "suite": "rodinia_hls",
    "group_path": ["cfd", "cfd_step_factor"]
  },
  "implementation": {
    "origin": "rodinia_hls_benchmark",
    "origin_version": null,
    "origin_meta": null,
    "variant": {
      "index": 2,
      "name": "pipeline"
    }
  }
}
```

Variants are multiple implementations of the same design and can be used to record step-by-step optimizations/changes.

### Parsing `csynth.xml`

An `hls_synth` record should be generated for every discovered project. If `csynth.xml` exists, parse the canonical HLS report and set `hls_synth.status` to `pass`. If it does not exist, still emit a row with `hls_synth.status` set by the runner and all top-level report sections set to `null`.

Prefer the canonical report path ending in:

```text
solution/syn/report/csynth.xml
```

Avoid duplicate packaged copies under paths such as `impl`, `ip`, `ip_repo`, `link`, `int`, `temp`, and `sys_link` when a direct HLS report exists.

Example source:

```xml
<profile>
  <ReportVersion>
    <Version>2023.2</Version>
  </ReportVersion>
  <UserAssignments>
    <TargetClockPeriod>3.33</TargetClockPeriod>
  </UserAssignments>
  <PerformanceEstimates>
    <SummaryOfTimingAnalysis>
      <EstimatedClockPeriod>2.433</EstimatedClockPeriod>
    </SummaryOfTimingAnalysis>
  </PerformanceEstimates>
  <AreaEstimates>
    <Resources>
      <LUT>22917</LUT>
    </Resources>
  </AreaEstimates>
</profile>
```

Example Python:

```python
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

CANONICAL_PATH_PENALTIES = {
    "impl", "ip", "ip_repo", "hls_files", "link", "int", "temp", "sys_link"
}

def choose_csynth(paths: list[Path]) -> Path:
    def score(path: Path) -> tuple[int, int, int, str]:
        parts = path.parts
        penalty_count = sum(1 for part in parts if part in CANONICAL_PATH_PENALTIES)
        suffix_penalty = 0 if parts[-4:] == ("solution", "syn", "report", "csynth.xml") else 1
        return (penalty_count, suffix_penalty, len(parts), path.as_posix())

    return min(paths, key=score)

def xml_node_to_dict(node: ET.Element | None) -> Any:
    if node is None:
        return None
    children = list(node)
    if not children:
        return (node.text or "").strip()
    return {child.tag: xml_node_to_dict(child) for child in children}

def parse_csynth(csynth_xml: Path | None, status: str) -> dict[str, Any]:
    if csynth_xml is None:
        return {
            "status": status,
            "ReportVersion": None,
            "UserAssignments": None,
            "PerformanceEstimates": None,
            "AreaEstimates": None,
        }

    root = ET.parse(csynth_xml).getroot()
    performance = root.find("PerformanceEstimates")
    return {
        "status": "pass",
        "ReportVersion": xml_node_to_dict(root.find("ReportVersion")),
        "UserAssignments": xml_node_to_dict(root.find("UserAssignments")),
        "PerformanceEstimates": {
            "SummaryOfTimingAnalysis": xml_node_to_dict(
                None if performance is None else performance.find("SummaryOfTimingAnalysis")
            ),
            "SummaryOfOverallLatency": xml_node_to_dict(
                None if performance is None else performance.find("SummaryOfOverallLatency")
            ),
        },
        "AreaEstimates": xml_node_to_dict(root.find("AreaEstimates")),
    }
```

### For `vitis.cosim` Target

Parse the latency `$TOTAL_EXECUTE_TIME` from the `lat.rpt` file, likely located under `sim/report/verilog/lat.rpt` in the solution directory, and use it as `kernel_runtime_cycles`. Leave `kernel_runtime_us` and `kernel_clock_freq_mhz` as `null`.

```text
$MAX_LATENCY = "9"
$MIN_LATENCY = "5"
$AVER_LATENCY = "7"
$MAX_THROUGHPUT = "11"
$MIN_THROUGHPUT = "8"
$AVER_THROUGHPUT = "9"
$TOTAL_EXECUTE_TIME = "44"
```

Example Python:

```python
import re
from pathlib import Path

def parse_cosim_kernel_runtime_cycles(lat_rpt: Path | None) -> int | None:
    if lat_rpt is None or not lat_rpt.is_file():
        return None

    text = lat_rpt.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r'\$TOTAL_EXECUTE_TIME\s*=\s*"([^"]+)"', text)
    if match is None:
        return None

    try:
        return round(float(match.group(1)))
    except ValueError:
        return None
```

### For `vitis.hw_emu` Target

#### Parsing `profile_kernels.csv`

Parse `kernel_runtime_us` from the `Compute Units: Running Time and Stalls` section of `profile_kernels.csv`.

Example source:

```csv
Compute Units: Running Time and Stalls
Compute Unit, Running Time (us), Intra-Kernel Dataflow Stalls (%), External Memory Stalls (%), External Stream Stalls (%)
workload_1,14216.453,0.000,0.000,0.000

Functions: Running Time and Stalls
...
```

If multiple compute-unit rows are present, sum their runtime values.

```python
import csv
from pathlib import Path

def parse_kernel_runtime_us(profile_csv: Path | None) -> float | None:
    if profile_csv is None or not profile_csv.is_file():
        return None

    lines = profile_csv.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_section = False
    header = None
    total = 0.0
    rows = 0

    for raw in lines:
        line = raw.strip()
        if line == "Compute Units: Running Time and Stalls":
            in_section = True
            header = None
            continue
        if not in_section:
            continue
        if not line: # exit on empty line
            if header is not None:
                break
            continue

        row = next(csv.reader([raw]))
        if header is None:
            header = row
            continue

        try:
            time_index = header.index("Time (us)")
        except ValueError:
            time_index = 1

        try:
            total += float(row[time_index])
            rows += 1
        except (IndexError, ValueError):
            pass

    return total if rows else None
```

#### Parsing `systemDiagramModel.json`

For `rtl_sim` records from Vitis targets such as `vitis.hw_emu`, parse `kernel_clock_freq_mhz` from `systemDiagramModel.json`. Use the clock entry whose `id` is `0`, and read `spec_frequency`.

Example source:

```json
{
  "system_diagram_metadata": {
    "xsa": {
      "clocks": [
        { "orig_name": "ulp_ucs_aclk_kernel_00", "id": 0, "spec_frequency": 300.0 },
        { "orig_name": "ulp_ucs_aclk_data_00", "id": 1, "spec_frequency": 500.0 }
      ]
    }
  }
}
```

Example Python:

```python
import json
from pathlib import Path
from typing import Any

def parse_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def parse_kernel_clock_freq_mhz(system_diagram_model: Path | None) -> float | None:
    if system_diagram_model is None or not system_diagram_model.is_file():
        return None

    data = json.loads(system_diagram_model.read_text(encoding="utf-8"))
    clocks = (
        data.get("system_diagram_metadata", {})
            .get("xsa", {})
            .get("clocks", [])
    )

    for clock in clocks:
        if isinstance(clock, dict) and parse_number(clock.get("id")) == 0:
            return parse_number(clock.get("spec_frequency"))

    return None

def derive_kernel_runtime_cycles(
    kernel_runtime_us: float | None,
    kernel_clock_freq_mhz: float | None,
) -> int | None:
    if kernel_runtime_us is None or kernel_clock_freq_mhz is None:
        return None
    return round(kernel_runtime_us * kernel_clock_freq_mhz)
```

## Complete Examples

### `sw_run`

```json
{
  "schema_version": "1.0",
  "report_type": "sw_run",
  "run": {
    "target": "vitis.sw_emu",
    "device": "xilinx_u280_gen3x16_xdma_1_202211_1",
    "vitis_version": "2023.2",
    "runtime_seconds": 12.345
  },
  "problem": {
    "suite": "rodinia_hls",
    "group_path": ["hotspot"]
  },
  "implementation": {
    "origin": "rodinia_hls_benchmark",
    "origin_version": null,
    "origin_meta": null,
    "variant": {
      "index": 6,
      "name": "multiddr"
    }
  },
  "sw_run": {
    "status": "pass"
  }
}
```

### `hls_synth`

```json
{
  "schema_version": "1.0",
  "report_type": "hls_synth",
  "run": {
    "target": "vitis.hw_emu",
    "device": "xilinx_u50_gen3x16_xdma_5_202210_1",
    "vitis_version": "2023.2",
    "runtime_seconds": null
  },
  "problem": {
    "suite": "rodinia_hls",
    "group_path": ["StreamCluster"]
  },
  "implementation": {
    "origin": "rodinia_hls_benchmark",
    "origin_version": null,
    "origin_meta": null,
    "variant": {
      "index": 0,
      "name": "baseline"
    }
  },
  "hls_synth": {
    "status": "pass",
    "ReportVersion": {
      "Version": "2023.2"
    },
    "UserAssignments": {
      "unit": "ns",
      "ProductFamily": "virtexuplusHBM",
      "Part": "xcu50-fsvh2104-2-e",
      "TopModelName": "workload",
      "TargetClockPeriod": "3.33",
      "ClockUncertainty": "0.90",
      "FlowTarget": "vitis"
    },
    "PerformanceEstimates": {
      "SummaryOfTimingAnalysis": {
        "unit": "ns",
        "EstimatedClockPeriod": "2.433"
      },
      "SummaryOfOverallLatency": {
        "unit": "clock cycles",
        "Best-caseLatency": "1755138",
        "Average-caseLatency": "1795074",
        "Worst-caseLatency": "1832962",
        "Best-caseRealTimeLatency": "5.850 ms",
        "Average-caseRealTimeLatency": "5.983 ms",
        "Worst-caseRealTimeLatency": "6.109 ms",
        "Interval-min": "1755139",
        "Interval-max": "1832963"
      }
    },
    "AreaEstimates": {
      "Resources": {
        "BRAM_18K": "62",
        "DSP": "5",
        "FF": "25381",
        "LUT": "22917",
        "URAM": "0"
      },
      "AvailableResources": {
        "BRAM_18K": "2688",
        "DSP": "5952",
        "FF": "1743360",
        "LUT": "871680",
        "URAM": "640"
      }
    }
  }
}
```

### `rtl_sim`

```json
{
  "schema_version": "1.0",
  "report_type": "rtl_sim",
  "run": {
    "target": "vitis.hw_emu",
    "device": "xilinx_u280_gen3x16_xdma_1_202211_1",
    "vitis_version": "2023.2",
    "runtime_seconds": 72000.0
  },
  "problem": {
    "suite": "rodinia_hls",
    "group_path": ["StreamCluster"]
  },
  "implementation": {
    "origin": "rodinia_hls_benchmark",
    "origin_version": null,
    "origin_meta": null,
    "variant": {
      "index": 4,
      "name": "coalescing"
    }
  },
  "rtl_sim": {
    "status": "pass",
    "kernel_runtime_cycles": 4264936,
    "kernel_runtime_us": 14216.453,
    "kernel_clock_freq_mhz": 300.0
  }
}
```
