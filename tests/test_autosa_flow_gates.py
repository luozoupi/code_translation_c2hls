# tests/test_autosa_flow_gates.py
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import autosa_flow_gates as g


def test_compute_pass_dsp_even_if_kernel_still_serial_io():
    report = {"dsp": 352, "latency_cycles": 13160, "interval": 13161}
    v = g.compute_architecture_ok(report, min_dsp=200)
    assert v.ok is True
    assert "workers" in v.reason.lower() or "dsp" in v.reason.lower()


def test_compute_fail_flash_leftover_dsp():
    report = {"dsp": 6, "latency_cycles": 149082, "interval": 149083}
    v = g.compute_architecture_ok(report, min_dsp=200)
    assert v.ok is False


def test_overlap_fail_enforcement_lcst():
    # 12893 vs 4553: second launch could start, this C is not done
    report = {"dsp": 320, "latency_cycles": 12893, "interval": 4553}
    v = g.io_overlap_ok(report, max_latency_over_interval=1.15)
    assert v.ok is False
    assert "overlap" in v.reason.lower() or "chapter" in v.reason.lower() or "interval" in v.reason.lower()


def test_overlap_pass_rank1_like():
    report = {"dsp": 320, "latency_cycles": 4228, "interval": 4133}
    v = g.io_overlap_ok(report, max_latency_over_interval=1.15)
    assert v.ok is True


def test_overlap_pass_locked_stream():
    report = {"dsp": 320, "latency_cycles": 4285, "interval": 4173}
    v = g.io_overlap_ok(report, max_latency_over_interval=1.15)
    assert v.ok is True


def test_rank1_success_gate():
    assert g.within_rank1(4285, rank1=4228, tol=1.02) is True
    assert g.within_rank1(4553, rank1=4228, tol=1.02) is False


_WRAP_101836 = """
#define NT 2
#define LANES 16
static void load_B(float B[64][64], float B_loc[64][64]) {
#pragma HLS INLINE off
  for (int j = 0; j < 64; ++j)
    for (int k0 = 0; k0 < 64; k0 += LANES) {
#pragma HLS PIPELINE II=1
      for (int u = 0; u < LANES; ++u) {
#pragma HLS UNROLL
        B_loc[j][k0 + u] = B[j][k0 + u];
      }
    }
}
static void load_A_tile(float A[64][64], float A_loc[32][64], int t) {
#pragma HLS INLINE off
}
static void compute_tile(float A_loc[32][64], float B_loc[64][64], float C_loc[32][64]) {
#pragma HLS INLINE off
}
static void store_C_tile(float C_loc[32][64], float C[64][64], int t) {
#pragma HLS INLINE off
}
extern "C" void autosa_mm(float A[64][64], float B[64][64], float C[64][64]) {
  tile_loop: for (int t = 0; t < NT; ++t) {
#pragma HLS DATAFLOW
    float B_loc[64][64];
    float A_loc[32][64];
    float C_loc[32][64];
    load_B(B, B_loc);
    load_A_tile(A, A_loc, t);
    compute_tile(A_loc, B_loc, C_loc);
    store_C_tile(C_loc, C, t);
  }
}
"""

_LEGAL_PP = """
#define NT 2
#define LANES 16
static void load_B(float B[64][64], float B_local[64][64]) {
#pragma HLS INLINE off
  for (int j = 0; j < 64; ++j)
    for (int k0 = 0; k0 < 64; k0 += LANES) {
#pragma HLS PIPELINE II=1
      for (int u = 0; u < LANES; ++u) {
#pragma HLS UNROLL
        B_local[j][k0 + u] = B[j][k0 + u];
      }
    }
}
static void load_A_tile(float A[64][64], float A_buf[32][64], int t) {
#pragma HLS INLINE off
}
static void compute_tile(float A_buf[32][64], float B_local[64][64], float C_buf[32][64]) {
#pragma HLS INLINE off
}
static void store_C_tile(float C_buf[32][64], float C[64][64], int t) {
#pragma HLS INLINE off
}
extern "C" void autosa_mm(float A[64][64], float B[64][64], float C[64][64]) {
  float B_local[64][64];
  float A_buf[2][32][64];
  float C_buf[2][32][64];
  load_B(B, B_local);
  for (int t = 0; t < NT; ++t) {
#pragma HLS DATAFLOW
    load_A_tile(A, A_buf[(t + 1) & 1], t + 1);
    compute_tile(A_buf[t & 1], B_local, C_buf[t & 1]);
    store_C_tile(C_buf[(t - 1) & 1], C, t - 1);
  }
}
"""

_BULK_12893 = """
extern "C" void autosa_mm(float A[64][64], float B[64][64], float C[64][64]) {
  float B_local[64][64];
  float A_tile[2][16][64];
  float C_tile[2][16][64];
#pragma HLS DATAFLOW
  load_B(&B[0][0], B_local);
  load_tiles(&A[0][0], A_tile);
  compute_tiles(A_tile, B_local, C_tile);
  store_tiles(&C[0][0], C_tile);
}
"""

_OUTSIDE_214_397 = """
extern "C" void autosa_mm(float A[64][64], float B[64][64], float C[64][64]) {
  float A_buf[2][16][64];
  float C_buf[2][16][64];
  for (int t = 0; t < 4; ++t) {
#pragma HLS DATAFLOW
    load_A_tile(A, A_buf[t & 1], t);
    compute_tile(A_buf[t & 1], B, C_buf[t & 1]);
    store_C_tile(C_buf[t & 1], C, t);
  }
}
"""

_RPT_12893 = """
== Vitis HLS Report for 'autosa_mm'
+ Latency:
    * Summary:
    |   min   |   max   |    min    |    max    |  min |  max |   Type   |
    |    12893|    12893|  42.934 us|  42.934 us|  4553|  4553|  dataflow|
    + Detail:
        * Instance:
        |     Instance     |     Module    |   min   |   max   |    min    |    max    |  min |  max |   Type  |
        |load_B_U0         |load_B         |     4170|     4170|  13.886 us|  13.886 us|  4170|  4170|       no|
        |load_tiles_U0     |load_tiles     |     4170|     4170|  13.886 us|  13.886 us|  4170|  4170|       no|
        |compute_tiles_U0  |compute_tiles  |     4552|     4552|  15.158 us|  15.158 us|  4552|  4552|       no|
        |store_tiles_U0    |store_tiles    |     4169|     4169|  13.883 us|  13.883 us|  4169|  4169|       no|
        * Loop:
        N/A
== Utilization Estimates
"""

_PE_PP_PLUS_REPORT = {
    "latency_cycles": 4688,
    "interval": 4689,
    "dsp": 352,
    "modules": [
        {"name": "load_B", "latency_cycles": 299},
        {"name": "dataflow_parent_loop_proc", "latency_cycles": 4385},
    ],
    "dataflow_regions": [
        {
            "name": "dataflow_in_loop_tile_pp_1",
            "latency_cycles": 1175,
            "interval": 1069,
            "trip_count": 4,
            "modules": [
                {"name": "load_A_tile", "latency_cycles": 106},
                {"name": "compute_tile", "latency_cycles": 1068},
                {"name": "store_C_tile", "latency_cycles": 104},
            ],
        }
    ],
}

_PE_PP_REPORT = {
    "latency_cycles": 9640,
    "interval": 9641,
    "dsp": 352,
    "modules": [
        {"name": "load_B", "latency_cycles": 4171},
        {"name": "dataflow_parent_loop_proc", "latency_cycles": 5465},
    ],
    "dataflow_regions": [
        {
            "name": "dataflow_in_loop_tile_pp_1",
            "latency_cycles": 2165,
            "interval": 1099,
            "trip_count": 4,
            "modules": [
                {"name": "load_A_tile", "latency_cycles": 1098},
                {"name": "compute_tile", "latency_cycles": 1066},
                {"name": "store_C_tile", "latency_cycles": 1096},
            ],
        }
    ],
}


_LEGAL_PP_B_IN_DF = """
#define NT 2
#define LANES 16
static void load_B(float B[64][64], float B_local[64][64]) {
#pragma HLS INLINE off
}
static void load_A_tile(float A[64][64], float A_buf[2][32][64], int slot, int t) {
#pragma HLS INLINE off
}
static void compute_tile(float A_buf[2][32][64], int sa, float B_local[64][64],
                         float C_buf[2][32][64], int sc) {
#pragma HLS INLINE off
}
static void store_C_tile(float C_buf[2][32][64], int slot, float C[64][64], int t) {
#pragma HLS INLINE off
}
extern "C" void autosa_mm(float A[64][64], float B[64][64], float C[64][64]) {
  for (int t = 0; t < NT; ++t) {
#pragma HLS DATAFLOW
    float B_local[64][64];
    float A_buf[2][32][64];
    float C_buf[2][32][64];
    load_B(B, B_local);
    load_A_tile(A, A_buf, t & 1, t);
    compute_tile(A_buf, t & 1, B_local, C_buf, t & 1);
    store_C_tile(C_buf, t & 1, C, t);
  }
}
"""


def test_explicit_pingpong_required_not_arrays_inside():
    ok = g.pingpong_dataflow_code_ok(_LEGAL_PP)
    assert ok.ok is True, ok.reason
    intent = g.pingpong_dataflow_code_intent(_LEGAL_PP)
    assert intent["ping_pong"] is True
    wrap = g.pingpong_dataflow_code_ok(_WRAP_101836)
    assert wrap.ok is False
    wrap_i = g.pingpong_dataflow_code_intent(_WRAP_101836)
    assert wrap_i["arrays_inside"] is True
    assert wrap_i["ping_pong"] is False
    bulk = g.pingpong_dataflow_code_ok(_BULK_12893)
    assert bulk.ok is False
    assert "tile" in bulk.reason.lower() or "one-shot" in bulk.reason.lower()
    # [2] outside without INLINE-off load/compute/store is still not a pass
    outside = g.pingpong_dataflow_code_ok(_OUTSIDE_214_397)
    assert outside.ok is False


def test_load_b_inside_dataflow_fails_by_default_passes_with_opt_in(monkeypatch):
    monkeypatch.delenv("C2HLS_PP_LOAD_B_IN_DF", raising=False)
    denied = g.pingpong_dataflow_code_ok(_LEGAL_PP_B_IN_DF)
    assert denied.ok is False
    assert "load_b" in denied.reason.lower()
    intent = g.pingpong_dataflow_code_intent(_LEGAL_PP_B_IN_DF)
    assert intent["load_b_inside"] is True
    assert intent["ping_pong"] is False
    monkeypatch.setenv("C2HLS_PP_LOAD_B_IN_DF", "1")
    allowed = g.pingpong_dataflow_code_ok(_LEGAL_PP_B_IN_DF)
    assert allowed.ok is True, allowed.reason
    assert "C2HLS_PP_LOAD_B_IN_DF" in allowed.reason
    intent_on = g.pingpong_dataflow_code_intent(_LEGAL_PP_B_IN_DF)
    assert intent_on["ping_pong"] is True
    wrap = g.pingpong_dataflow_code_ok(_WRAP_101836)
    assert wrap.ok is False
    assert "buf[2]" in wrap.reason.lower() or "explicit" in wrap.reason.lower()


def test_flash_serial_lcst_still_fails_overlap():
    serial = """
extern "C" void autosa_mm(float A[64][64], float B[64][64], float C[64][64]) {
  float A_local[64][64];
  load: for (int i = 0; i < 64; ++i)
    for (int k = 0; k < 64; ++k) A_local[i][k] = A[i][k];
}
"""
    intent = g.pingpong_dataflow_code_intent(serial)
    assert intent["ok"] is False
    assert intent["ping_pong"] is False
    v = g.pingpong_dataflow_ok(serial, {"latency_cycles": 5214, "interval": 5215, "dsp": 320})
    assert v.ok is False


def test_parse_csynth_12893_instance_sum():
    summary = g.parse_csynth_latency_summary(_RPT_12893)
    assert summary["latency_cycles"] == 12893
    assert summary["interval"] == 4553
    assert summary["pipeline_type"] == "dataflow"
    inst = g.parse_csynth_instance_latencies(_RPT_12893)
    names = {row["name"] for row in inst}
    assert names >= {"load_B", "load_tiles", "compute_tiles", "store_tiles"}
    serial, max_p = g.lcst_serial_sum(inst)
    assert serial == 4170 + 4552 + 4169
    assert max_p == 4552
    assert g._near(12893, serial, 0.10)


def test_csynth_rejects_12893_sum_not_overlap():
    report = {"csynth_rpt": _RPT_12893, "latency_cycles": 12893, "interval": 4553}
    v = g.pingpong_dataflow_csynth_ok(report, tile_loop=False, code=_BULK_12893)
    assert v.ok is False
    assert "sum" in v.reason.lower() or "one-shot" in v.reason.lower()
    combo = g.pingpong_dataflow_ok(_BULK_12893, report)
    assert combo.ok is False


def test_csynth_accepts_pe_pp_plus_overlap():
    v = g.pingpong_dataflow_ok(_LEGAL_PP, _PE_PP_PLUS_REPORT)
    assert v.ok is True
    assert "overlap" in v.reason.lower() or "max" in v.reason.lower()


def test_csynth_pe_pp_9640_still_overlaps_tiles():
    v = g.pingpong_dataflow_csynth_ok(
        _PE_PP_REPORT, tile_loop=True, code=_LEGAL_PP.replace("NT 2", "NT 4")
    )
    assert v.ok is True


def test_interval_below_latency_is_not_enough():
    v = g.pingpong_dataflow_csynth_ok(
        {"latency_cycles": 12893, "interval": 4553, "dsp": 320},
        tile_loop=False,
    )
    assert v.ok is False


def test_real_enforcement_selected_cpp_fails_new_judge():
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260829_123043"
        / "variants/autosa_aav_n_gf/autosa_mm"
        / "deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp"
    )
    if not path.is_file():
        return
    code = path.read_text(encoding="utf-8")
    rpt = (
        Path(__file__).resolve().parents[1]
        / "c2hls_tmp/batch_parallel_autosa_mm_enf_aav_n_gf_20260829_123043"
        / "autosa_mm/hls_synth__step_synth/hls_proj/sol1/syn/report/autosa_mm_csynth.rpt"
    )
    report = {"latency_cycles": 12893, "interval": 4553, "dsp": 320}
    if rpt.is_file():
        report["csynth_rpt"] = rpt.read_text(encoding="utf-8", errors="replace")
    v = g.pingpong_dataflow_ok(code, report)
    assert v.ok is False


def test_tile_pp_guidance_mentions_explicit_buf2():
    text = g.flash_tile_pp_initial_guidance()
    low = text.lower()
    assert "buf[2]" in text or "t & 1" in text
    assert "arrays inside" in low or "not ping-pong" in low or "explicit" in low


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def test_real_pe_pp_plus_channel_only_is_not_explicit_ping_pong():
    cpp = _repo() / "artifacts/pc2/manual_mmflow_pe_pp/autosa_mm_pe_pp_plus.cpp"
    if not cpp.is_file():
        return
    code = cpp.read_text(encoding="utf-8")
    intent = g.pingpong_dataflow_code_intent(code)
    assert intent["tile_loop"] is True
    assert intent["ping_pong"] is False
    assert "buf[2]" not in code.replace(" ", "")


def test_real_tile_pp2_channel_only_is_not_explicit_ping_pong():
    cpp = _repo() / "artifacts/pc2/manual_mm_lcst_tile_pp2/autosa_mm_tile_pp2.cpp"
    if not cpp.is_file():
        return
    code = cpp.read_text(encoding="utf-8")
    intent = g.pingpong_dataflow_code_intent(code)
    assert intent["tile_loop"] is True
    assert intent["ping_pong"] is False


_FLASH_WIDE_LOAD = """
extern "C" void autosa_mm(float A[64][64], float B[64][64], float C[64][64]) {
  const int LANES = 16;
  float A_local[64][64];
  float B_local[64][64];
  load_a_rows: for (int i = 0; i < 64; ++i)
    load_a_chunks: for (int k0 = 0; k0 < 64; k0 += LANES) {
#pragma HLS PIPELINE II=1
      load_a_lanes: for (int u = 0; u < LANES; ++u) {
#pragma HLS UNROLL
        A_local[i][k0 + u] = A[i][k0 + u];
      }
    }
  load_b_rows: for (int j = 0; j < 64; ++j)
    load_b_chunks: for (int k0 = 0; k0 < 64; k0 += LANES) {
#pragma HLS PIPELINE II=1
      load_b_lanes: for (int u = 0; u < LANES; ++u) {
#pragma HLS UNROLL
        B_local[j][k0 + u] = B[j][k0 + u];
      }
    }
}
"""

_SCALAR_TILE_LOAD = """
#define NT 2
extern "C" void autosa_mm(float A[64][64], float B[64][64], float C[64][64]) {
  float B_local[64][64];
  for (int j = 0; j < 64; ++j)
    for (int k = 0; k < 64; ++k) {
#pragma HLS PIPELINE II=1
      B_local[j][k] = B[j][k];
    }
  for (int t = 0; t < NT; ++t) {
#pragma HLS DATAFLOW
    float A_buf[32][64];
    float C_buf[32][64];
    load_A_tile(A, A_buf, t);
    compute_tile(A_buf, B_local, C_buf);
    store_C_tile(C_buf, C, t);
  }
}
"""

_FLASH_4808_REPORT = {
    "latency_cycles": 4808,
    "interval": 4809,
    "dsp": 318,
    "modules": [
        {"name": "autosa_mm_Pipeline_load_a_rows_load_a_chunks", "latency_cycles": 259},
        {"name": "autosa_mm_Pipeline_load_b_rows_load_b_chunks", "latency_cycles": 259},
        {"name": "autosa_mm_Pipeline_compute_i_compute_j", "latency_cycles": 4145},
        {"name": "autosa_mm_Pipeline_store_c_rows_store_c_chunks", "latency_cycles": 259},
    ],
}

_ENF_12642_REPORT = {
    "latency_cycles": 12642,
    "interval": 12643,
    "dsp": 318,
    "modules": [
        {"name": "load_B", "latency_cycles": 4171},
        {"name": "dataflow_parent_loop_proc", "latency_cycles": 8467},
    ],
    "dataflow_regions": [
        {
            "name": "dataflow_in_loop_tile_loop_1",
            "latency_cycles": 6341,
            "interval": 2123,
            "trip_count": 2,
            "pipeline_type": "dataflow",
            "modules": [
                {"name": "load_A_tile", "latency_cycles": 2122},
                {"name": "compute_tile", "latency_cycles": 2097},
                {"name": "store_C_tile", "latency_cycles": 2120},
            ],
        }
    ],
}


def test_wide_axi_copy_detects_lanes_step():
    assert g.wide_axi_copy_in_code(_FLASH_WIDE_LOAD) is True
    assert g.wide_axi_copy_in_code(_SCALAR_TILE_LOAD) is False
    assert g.wide_axi_copy_in_code(_LEGAL_PP) is True
    assert g.wide_axi_copy_in_code(_BULK_12893) is False


def test_keep_flash_rejects_dropped_lanes_and_latency_regression():
    v = g.enforcement_keep_flash_ok(
        code=_SCALAR_TILE_LOAD,
        report=_ENF_12642_REPORT,
        baseline_code=_FLASH_WIDE_LOAD,
        baseline_report=_FLASH_4808_REPORT,
    )
    assert v.ok is False
    low = v.reason.lower()
    assert "lanes" in low or "512" in low or "regress" in low or "259" in low or "4171" in low


def test_keep_flash_accepts_wide_pingpong_not_worse_than_flash():
    wide_pp = """
#define NT 2
extern "C" void autosa_mm(float A[64][64], float B[64][64], float C[64][64]) {
  const int LANES = 16;
  float B_local[64][64];
  load_b_rows: for (int j = 0; j < 64; ++j)
    load_b_chunks: for (int k0 = 0; k0 < 64; k0 += LANES) {
#pragma HLS PIPELINE II=1
      for (int u = 0; u < LANES; ++u) {
#pragma HLS UNROLL
        B_local[j][k0 + u] = B[j][k0 + u];
      }
    }
  for (int t = 0; t < NT; ++t) {
#pragma HLS DATAFLOW
    load_A_tile(A, A_buf[(t + 1) & 1], t + 1);
    compute_tile(A_buf[t & 1], B_local, C_buf[t & 1]);
    store_C_tile(C_buf[(t - 1) & 1], C, t - 1);
  }
}
"""
    v = g.enforcement_keep_flash_ok(
        code=wide_pp,
        report=_PE_PP_PLUS_REPORT,
        baseline_code=_FLASH_WIDE_LOAD,
        baseline_report=_FLASH_4808_REPORT,
    )
    assert v.ok is True, v.reason


def test_keep_flash_skips_when_still_the_flash_kernel():
    v = g.enforcement_keep_flash_ok(
        code=_FLASH_WIDE_LOAD,
        report=_FLASH_4808_REPORT,
        baseline_code=_FLASH_WIDE_LOAD,
        baseline_report=_FLASH_4808_REPORT,
    )
    assert v.ok is True


_FLASH_139484_UF8 = """
#define UF 8
extern "C" void autosa_mm(float A[64][64], float B[64][64], float C[64][64]) {
  float A_local[64][64];
  float B_local[64][64];
  float acc[UF];
  for (int i = 0; i < 64; i++)
    for (int k = 0; k < 64; k++) {
#pragma HLS PIPELINE II=1
      A_local[i][k] = A[i][k];
    }
  for (int j = 0; j < 64; j++)
    for (int k = 0; k < 64; k++) {
#pragma HLS PIPELINE II=1
      B_local[j][k] = B[j][k];
    }
}
"""

_WRAP_139K_NO_LANES = """
#define NT 2
#define UF 8
static void load_B(float B[64][64], float B_local[64][64]) {
#pragma HLS INLINE off
  for (int j = 0; j < 64; j++)
    for (int k = 0; k < 64; k++) {
#pragma HLS PIPELINE II=1
      B_local[j][k] = B[j][k];
    }
}
static void load_A_tile(float A[64][64], float A_buf[32][64], int t) {
#pragma HLS INLINE off
}
static void compute_tile(float A_buf[32][64], float B_local[64][64], float C_buf[32][64]) {
#pragma HLS INLINE off
}
static void store_C_tile(float C_buf[32][64], float C[64][64], int t) {
#pragma HLS INLINE off
}
extern "C" void autosa_mm(float A[64][64], float B[64][64], float C[64][64]) {
  float B_local[64][64];
  float A_buf[2][32][64];
  float C_buf[2][32][64];
  load_B(B, B_local);
  tile_loop: for (int t = 0; t < NT; ++t) {
#pragma HLS DATAFLOW
    load_A_tile(A, A_buf[(t + 1) & 1], t + 1);
    compute_tile(A_buf[t & 1], B_local, C_buf[t & 1]);
    store_C_tile(C_buf[(t - 1) & 1], C, t - 1);
  }
}
"""


def test_keep_flash_139k_does_not_demand_lanes():
    """mmflow flash 139484/10 has UF=8, no LANES=16. Gate must not require them."""
    assert g.wide_axi_copy_in_code(_FLASH_139484_UF8) is False
    worse = g.enforcement_keep_flash_ok(
        code=_WRAP_139K_NO_LANES,
        report={"latency_cycles": 200000, "dsp": 10},
        baseline_code=_FLASH_139484_UF8,
        baseline_report={"latency_cycles": 139484, "dsp": 10},
    )
    assert worse.ok is False
    low = worse.reason.lower()
    assert "lanes" not in low
    assert "139484" in worse.reason or "1.10" in low or "regress" in low
    kept = g.enforcement_keep_flash_ok(
        code=_WRAP_139K_NO_LANES,
        report={"latency_cycles": 130000, "dsp": 10},
        baseline_code=_FLASH_139484_UF8,
        baseline_report={"latency_cycles": 139484, "dsp": 10},
    )
    assert kept.ok is True, kept.reason
    still_flash = g.enforcement_keep_flash_ok(
        code=_FLASH_139484_UF8,
        report={"latency_cycles": 139484, "dsp": 10},
        baseline_code=_FLASH_139484_UF8,
        baseline_report={"latency_cycles": 139484, "dsp": 10},
    )
    assert still_flash.ok is True


_DSE_PE16 = """
#define PE 16
#define SIMD 4
extern "C" void autosa_mm(float A[64][64], float B[64][64], float C[64][64]) {
  float Crow[PE][64];
  for (int i0 = 0; i0 < 64; i0 += PE) {
    for (int k0 = 0; k0 < 64; k0 += SIMD) {
      for (int j = 0; j < 64; j++) {
#pragma HLS PIPELINE II=1
        pe_mac: for (int p = 0; p < PE; p++) {
#pragma HLS UNROLL
          Crow[p][j] += A[i0 + p][k0] * B[j][k0];
        }
      }
    }
  }
}
"""

_PE_WRAP = """
#define NT 4
#define PE 16
#define SIMD 4
static void load_B(float B[64][64], float B_local[64][64]) {
#pragma HLS INLINE off
}
static void load_A_tile(float A[64][64], float A_buf[PE][64], int t) {
#pragma HLS INLINE off
}
static void compute_tile(float A_buf[PE][64], float B_local[64][64], float C_buf[PE][64]) {
#pragma HLS INLINE off
  float Crow[PE][64];
  for (int k0 = 0; k0 < 64; k0 += SIMD) {
    for (int j = 0; j < 64; j++) {
#pragma HLS PIPELINE II=1
      pe_mac: for (int p = 0; p < PE; p++) {
#pragma HLS UNROLL
        Crow[p][j] += A_buf[p][k0] * B_local[j][k0];
      }
    }
  }
}
static void store_C_tile(float C_buf[PE][64], float C[64][64], int t) {
#pragma HLS INLINE off
}
extern "C" void autosa_mm(float A[64][64], float B[64][64], float C[64][64]) {
  float B_local[64][64];
  float A_buf[2][PE][64];
  float C_buf[2][PE][64];
  load_B(B, B_local);
  tile_loop: for (int t = 0; t < NT; ++t) {
#pragma HLS DATAFLOW
    load_A_tile(A, A_buf[(t + 1) & 1], t + 1);
    compute_tile(A_buf[t & 1], B_local, C_buf[t & 1]);
    store_C_tile(C_buf[(t - 1) & 1], C, t - 1);
  }
}
"""


def test_keep_flash_dse_requires_pe_nest_not_unroll_k():
    """mmflow DSE 13160/352 is PE 16 × SIMD 4. Do not accept 123043-class unroll-K."""
    dse_report = {"latency_cycles": 13160, "dsp": 352}
    dropped = g.enforcement_keep_flash_ok(
        code=_WRAP_139K_NO_LANES,
        report={"latency_cycles": 12893, "dsp": 320},
        baseline_code=_DSE_PE16,
        baseline_report=dse_report,
    )
    assert dropped.ok is False
    low = dropped.reason.lower()
    assert "pe" in low and "simd" in low
    kept = g.enforcement_keep_flash_ok(
        code=_PE_WRAP,
        report={"latency_cycles": 4688, "dsp": 352},
        baseline_code=_DSE_PE16,
        baseline_report=dse_report,
    )
    assert kept.ok is True, kept.reason
    still_dse = g.enforcement_keep_flash_ok(
        code=_DSE_PE16,
        report=dse_report,
        baseline_code=_DSE_PE16,
        baseline_report=dse_report,
    )
    assert still_dse.ok is True
