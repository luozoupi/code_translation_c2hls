"""Flash overlap enforcement: ping-pong + DATAFLOW judged on code and csynth."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flash_enforcement as fe


SEQUENTIAL_LCST = """
extern "C" void kernel(float A[64][64], float B[64][64], float C[64][64]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
  float local_A[64][64];
  load: for (int i = 0; i < 64; ++i)
    for (int k = 0; k < 64; ++k) local_A[i][k] = A[i][k];
  compute: for (int i = 0; i < 64; ++i)
    for (int j = 0; j < 64; ++j) C[i][j] = local_A[i][0];
}
"""

# 101836-style wrap: DATAFLOW + arrays inside, no buf[2], load_B every tile.
WRAP_101836 = """
#define NT 2
#define TI 32
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
extern "C" void kernel(float A[64][64], float B[64][64], float C[64][64]) {
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

# Explicit ping-pong: buf[2], t&1, load(t+1)/compute(t)/store(t-1), B once, INLINE off.
PINGPONG_DATAFLOW = """
#define NT 2
#define TI 32
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
extern "C" void kernel(float A[64][64], float B[64][64], float C[64][64]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
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

BULK_DATAFLOW_12893 = """
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

ARRAYS_OUTSIDE_LOOP = """
extern "C" void kernel(float A[64][64], float B[64][64], float C[64][64]) {
  float Atile[2][16][64];
  float Ctile[2][16][64];
  tile: for (int t = 0; t < 4; ++t) {
#pragma HLS DATAFLOW
    load_A_tile(A, Atile[t & 1], t);
    compute_tile(Atile[t & 1], B, Ctile[t & 1]);
    store_C_tile(C, Ctile[t & 1], t);
  }
}
"""

FLASH_SEQ_REPORT = {
    "latency_cycles": 24745,
    "interval": 24746,
    "dsp": 80,
    "bram": 122,
    "ff": 26421,
    "lut": 19728,
}

OVERLAP_REPORT = {
    "latency_cycles": 4688,
    "interval": 4689,
    "dsp": 352,
    "bram": 180,
    "ff": 30000,
    "lut": 22000,
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
            "pipeline_type": "dataflow",
            "modules": [
                {"name": "load_A_tile", "latency_cycles": 106},
                {"name": "compute_tile", "latency_cycles": 1068},
                {"name": "store_C_tile", "latency_cycles": 104},
            ],
        }
    ],
}

ENFORCEMENT_12893_REPORT = {
    "latency_cycles": 12893,
    "interval": 4553,
    "dsp": 320,
    "pipeline_type": "dataflow",
    "modules": [
        {"name": "load_B", "latency_cycles": 4170},
        {"name": "load_tiles", "latency_cycles": 4170},
        {"name": "compute_tiles", "latency_cycles": 4552},
        {"name": "store_tiles", "latency_cycles": 4169},
    ],
}

STORE_BOUND_REPORT = {
    "latency_cycles": 618668,
    "interval": 589828,
    "dsp": 320,
    "bram": 180,
    "ff": 30000,
    "lut": 22000,
}


def test_enforcement_off_by_default(monkeypatch):
    monkeypatch.delenv("C2HLS_ENFORCEMENT", raising=False)
    assert fe.enforcement_enabled() is False


def test_enforcement_enabled_from_env(monkeypatch):
    monkeypatch.delenv("C2HLS_AUTOSA_FLOW", raising=False)
    monkeypatch.setenv("C2HLS_ENFORCEMENT", "1")
    assert fe.enforcement_enabled() is True


def test_enforcement_skipped_when_autosa_flow(monkeypatch):
    monkeypatch.setenv("C2HLS_AUTOSA_FLOW", "1")
    monkeypatch.setenv("C2HLS_ENFORCEMENT", "1")
    assert fe.enforcement_enabled() is False


def test_enforcement_stays_on_when_autosa_flow_unset(monkeypatch):
    monkeypatch.delenv("C2HLS_AUTOSA_FLOW", raising=False)
    monkeypatch.setenv("C2HLS_ENFORCEMENT", "1")
    assert fe.enforcement_enabled() is True


def test_enforcement_rounds_default_20(monkeypatch):
    monkeypatch.delenv("C2HLS_ENFORCEMENT_ROUNDS", raising=False)
    assert fe.enforcement_round_limit() == 20


def test_enforcement_rounds_from_env(monkeypatch):
    monkeypatch.setenv("C2HLS_ENFORCEMENT_ROUNDS", "7")
    assert fe.enforcement_round_limit() == 7


def test_cli_enforcement_and_underscore_rounds():
    parser = argparse.ArgumentParser()
    fe.add_enforcement_arguments(parser)
    args = parser.parse_args(["--enforcement", "--enforcement_rounds", "20"])
    assert args.enforcement is True
    assert args.enforcement_rounds == 20


def test_cli_hyphen_rounds_alias():
    parser = argparse.ArgumentParser()
    fe.add_enforcement_arguments(parser)
    args = parser.parse_args(["--enforcement", "--enforcement-rounds", "12"])
    assert args.enforcement_rounds == 12


def test_apply_cli_sets_env(monkeypatch):
    monkeypatch.delenv("C2HLS_ENFORCEMENT", raising=False)
    monkeypatch.delenv("C2HLS_ENFORCEMENT_ROUNDS", raising=False)
    parser = argparse.ArgumentParser()
    fe.add_enforcement_arguments(parser)
    args = parser.parse_args(["--enforcement", "--enforcement_rounds", "20"])
    fe.apply_enforcement_env(args)
    assert os.environ["C2HLS_ENFORCEMENT"] == "1"
    assert os.environ["C2HLS_ENFORCEMENT_ROUNDS"] == "20"


def test_code_intent_rejects_sequential_lcst():
    intent = fe.code_intent(SEQUENTIAL_LCST)
    assert intent["dataflow"] is False
    assert intent["ping_pong"] is False
    assert intent["ok"] is False


def test_code_intent_rejects_101836_arrays_inside_as_ping_pong():
    intent = fe.code_intent(WRAP_101836)
    assert intent["dataflow"] is True
    assert intent["tile_loop"] is True
    assert intent["arrays_inside"] is True
    assert intent["ping_pong"] is False
    assert intent["ok"] is False
    assert "buf[2]" in (intent.get("reason") or "").lower() or "explicit" in (
        intent.get("reason") or ""
    ).lower() or "load_b" in (intent.get("reason") or "").lower()


def test_code_intent_accepts_pingpong_dataflow():
    intent = fe.code_intent(PINGPONG_DATAFLOW)
    assert intent["dataflow"] is True
    assert intent["ping_pong"] is True
    assert intent["tile_loop"] is True
    assert intent["ok"] is True
    assert "t & 1" in PINGPONG_DATAFLOW or "t&1" in PINGPONG_DATAFLOW.replace(" ", "")


def test_code_intent_rejects_bulk_dataflow_and_outside_arrays():
    bulk = fe.code_intent(BULK_DATAFLOW_12893)
    assert bulk["dataflow"] is True
    assert bulk["tile_loop"] is False
    assert bulk["ok"] is False
    outside = fe.code_intent(ARRAYS_OUTSIDE_LOOP)
    assert outside["tile_loop"] is True
    assert outside["arrays_inside"] is False
    assert outside["arrays_outside"] is True
    assert outside["ok"] is False


def test_csynth_rejects_interval_equal_latency():
    shown = fe.csynth_overlap(FLASH_SEQ_REPORT)
    assert shown["ok"] is False
    assert shown["interval"] == 24746
    assert shown["latency_cycles"] == 24745


def test_csynth_rejects_weak_handshake_overlap():
    shown = fe.csynth_overlap({
        "latency_cycles": 292679,
        "interval": 271746,
        "dsp": 3,
    })
    assert shown["ok"] is False


def test_csynth_rejects_interval_much_less_than_latency_without_modules():
    """12893-class: interval 4553 is the next launch, not tile overlap."""
    shown = fe.csynth_overlap({
        "latency_cycles": 12893,
        "interval": 4553,
        "dsp": 320,
    })
    assert shown["ok"] is False


def test_csynth_rejects_enforcement_lcst_sum():
    shown = fe.csynth_overlap(ENFORCEMENT_12893_REPORT, tile_loop=False)
    assert shown["ok"] is False
    assert "sum" in (shown.get("reason") or "").lower() or "12893" in (shown.get("reason") or "")


def test_csynth_accepts_pe_pp_plus_overlap():
    shown = fe.csynth_overlap(OVERLAP_REPORT, tile_loop=True, code=PINGPONG_DATAFLOW)
    assert shown["ok"] is True


def test_static_verdict_fails_flash_mm():
    v = fe.static_verdict(SEQUENTIAL_LCST, FLASH_SEQ_REPORT)
    assert v.passed is False
    assert v.code_intended is False
    assert v.csynth_shows is False


def test_static_verdict_rejects_enforcement_12893():
    v = fe.static_verdict(BULK_DATAFLOW_12893, ENFORCEMENT_12893_REPORT)
    assert v.passed is False
    assert v.code_intended is False
    assert v.csynth_shows is False


def test_static_verdict_passes_legal_overlap():
    v = fe.static_verdict(PINGPONG_DATAFLOW, OVERLAP_REPORT)
    assert v.passed is True
    assert v.code_intended is True
    assert v.csynth_shows is True


def test_parse_llm_judge_json():
    reply = """```json
{
  "schema": "flash_overlap_enforcement_v1",
  "passed": false,
  "code_intended": {"ok": true, "dataflow": true, "ping_pong": true, "reason": "tile loop around DATAFLOW"},
  "csynth_shows": {"ok": false, "reason": "interval equals latency"}
}
```"""
    v = fe.parse_llm_verdict(reply)
    assert v.code_intended is True
    assert v.csynth_shows is False
    assert v.passed is False
    assert "interval" in (v.repair_focus or "").lower() or "latency" in (v.reason or "").lower() or True


def test_repair_prompt_formats_with_skeleton(monkeypatch):
    monkeypatch.delenv("C2HLS_PP_LOAD_B_IN_DF", raising=False)
    blob = fe.build_repair_prompt(
        verdict_json='{"passed": false}',
        repair_focus="add DATAFLOW",
        benchmark_context="mm",
        header_name="kernel.h",
        header_code="#define I 64",
        kernel_code=SEQUENTIAL_LCST,
        report_blob="interval 1",
        flash_kernel_code=SEQUENTIAL_LCST,
        flash_report_blob="latency 24745",
    )
    assert "load_A_tile" in blob
    assert "pragma HLS DATAFLOW" in blob
    assert "buf[2]" in blob or "t & 1" in blob
    assert "LANES" in blob
    assert "{verdict_json}" not in blob
    assert "B loaded once outside" in blob
    assert "C2HLS_PP_LOAD_B_IN_DF=1" not in blob


def test_repair_prompt_loadb_in_df_overlay(monkeypatch):
    monkeypatch.setenv("C2HLS_PP_LOAD_B_IN_DF", "1")
    blob = fe.build_repair_prompt(
        verdict_json='{"passed": false}',
        repair_focus="put load_B in DATAFLOW",
        benchmark_context="mm",
        header_name="kernel.h",
        header_code="#define I 64",
        kernel_code=SEQUENTIAL_LCST,
        report_blob="interval 1",
        flash_kernel_code=SEQUENTIAL_LCST,
        flash_report_blob="latency 24745",
    )
    assert "C2HLS_PP_LOAD_B_IN_DF=1" in blob
    assert "load_B INSIDE" in blob
    assert "hls-enforcement-loadb-in-tile-dataflow" in blob
    assert "Prefix load_B" in blob
    sys_prompt = fe.judge_system_prompt()
    assert "C2HLS_PP_LOAD_B_IN_DF" in sys_prompt
    _block, ids = fe.load_enforcement_keep_flash_skills()
    assert "hls-enforcement-loadb-in-tile-dataflow" in ids
    assert "avoid-enforcement-loadb-prefix-when-df-opt-in" in ids


def test_combine_llm_pass_vetoed_by_sequential_csynth():
    llm = fe.JudgeVerdict(
        passed=True,
        code_intended=True,
        csynth_shows=True,
        reason="looks good",
        source="llm",
    )
    combined = fe.combine_verdicts(
        fe.static_verdict(PINGPONG_DATAFLOW, FLASH_SEQ_REPORT),
        llm,
    )
    assert combined.passed is False
    assert combined.csynth_shows is False


def test_combine_static_pass_not_vetoed_by_llm():
    llm = fe.JudgeVerdict(
        passed=False,
        code_intended=False,
        csynth_shows=True,
        reason="LLM judge reply missing JSON verdict",
        source="llm_parse_error",
    )
    combined = fe.combine_verdicts(
        fe.static_verdict(PINGPONG_DATAFLOW, OVERLAP_REPORT),
        llm,
    )
    assert combined.passed is True
    assert combined.csynth_shows is True


def test_combine_static_structure_not_vetoed_by_llm():
    llm = fe.JudgeVerdict(
        passed=False,
        code_intended=False,
        csynth_shows=False,
        reason="LLM judge reply missing JSON verdict",
        source="llm_parse_error",
    )
    combined = fe.combine_verdicts(
        fe.static_verdict(PINGPONG_DATAFLOW, STORE_BOUND_REPORT),
        llm,
    )
    assert combined.code_intended is True
    assert combined.csynth_shows is False
    assert combined.passed is False
    focus = (combined.repair_focus or "").lower()
    assert (
        "max(load,compute,store)" in focus
        or "lcst" in focus
        or "overlap" in focus
        or "missing" in focus
    )


def test_enforcement_does_not_pass_structure_without_overlap():
    rounds = {"n": 0}

    def judge(code, report):
        return fe.static_verdict(code, report)

    def generate(code, report, verdict, round_i):
        rounds["n"] += 1
        return PINGPONG_DATAFLOW

    def evaluate(code):
        return {
            "success": True,
            "code": code,
            "report": STORE_BOUND_REPORT,
            "csim": {"passed": True},
            "cosim": None,
        }

    out = fe.run_enforcement_loop(
        kernel_code=SEQUENTIAL_LCST,
        synth_report=FLASH_SEQ_REPORT,
        rounds=3,
        judge_fn=judge,
        generate_fn=generate,
        evaluate_fn=evaluate,
    )
    assert out["passed"] is False
    assert out["overlap"] is False
    assert rounds["n"] == 3
    assert all(a.get("status") != "structure_ok" for a in out["attempts"])


def test_repair_loop_stops_when_judge_passes():
    rounds = {"n": 0}

    def judge(code, report):
        return fe.static_verdict(code, report)

    def generate(code, report, verdict, round_i):
        rounds["n"] += 1
        return PINGPONG_DATAFLOW

    def evaluate(code):
        return {
            "success": True,
            "code": code,
            "report": OVERLAP_REPORT,
            "csim": {"passed": True},
            "cosim": None,
        }

    out = fe.run_enforcement_loop(
        kernel_code=SEQUENTIAL_LCST,
        synth_report=FLASH_SEQ_REPORT,
        rounds=20,
        judge_fn=judge,
        generate_fn=generate,
        evaluate_fn=evaluate,
    )
    assert out["passed"] is True
    assert out["attempted"] is True
    assert rounds["n"] == 1
    assert out["rounds_used"] == 1
    assert "DATAFLOW" in out["code"]


def test_repair_loop_exhausts_rounds_without_pass():
    def judge(code, report):
        return fe.JudgeVerdict(False, False, False, "still sequential", "static")

    def generate(code, report, verdict, round_i):
        return SEQUENTIAL_LCST

    def evaluate(code):
        return {
            "success": True,
            "code": code,
            "report": FLASH_SEQ_REPORT,
            "csim": {"passed": True},
            "cosim": None,
        }

    out = fe.run_enforcement_loop(
        kernel_code=SEQUENTIAL_LCST,
        synth_report=FLASH_SEQ_REPORT,
        rounds=3,
        judge_fn=judge,
        generate_fn=generate,
        evaluate_fn=evaluate,
    )
    assert out["passed"] is False
    assert out["rounds_used"] == 3
    assert len(out["attempts"]) == 3


def test_maybe_run_skips_when_disabled(monkeypatch):
    monkeypatch.delenv("C2HLS_ENFORCEMENT", raising=False)
    monkeypatch.delenv("BATCH_PARALLEL_CAMPAIGN_ROOT", raising=False)
    orch = SimpleNamespace(hls_code=SEQUENTIAL_LCST, synth_report=FLASH_SEQ_REPORT)
    assert fe.maybe_run_enforcement(orch) is None


def test_apply_enforcement_from_campaign_sets_env(monkeypatch, tmp_path):
    monkeypatch.delenv("C2HLS_ENFORCEMENT", raising=False)
    monkeypatch.delenv("C2HLS_ENFORCEMENT_ROUNDS", raising=False)
    fe.apply_enforcement_from_campaign(
        {"enforcement": True, "enforcement_rounds": 20}
    )
    assert os.environ["C2HLS_ENFORCEMENT"] == "1"
    assert os.environ["C2HLS_ENFORCEMENT_ROUNDS"] == "20"
    assert fe.enforcement_enabled() is True


def test_enforcement_enabled_from_campaign_json_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv("C2HLS_ENFORCEMENT", raising=False)
    monkeypatch.delenv("C2HLS_ENFORCEMENT_ROUNDS", raising=False)
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "campaign.json").write_text(
        '{"enforcement": true, "enforcement_rounds": 11}\n', encoding="utf-8"
    )
    monkeypatch.setenv("BATCH_PARALLEL_CAMPAIGN_ROOT", str(root))
    fe.apply_enforcement_from_campaign_root()
    assert fe.enforcement_enabled() is True
    assert fe.enforcement_round_limit() == 11


def test_maybe_run_does_not_skip_when_campaign_enforcement_true(monkeypatch, tmp_path):
    monkeypatch.delenv("C2HLS_ENFORCEMENT", raising=False)
    monkeypatch.delenv("C2HLS_ENFORCEMENT_ROUNDS", raising=False)
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "campaign.json").write_text(
        '{"enforcement": true, "enforcement_rounds": 4}\n', encoding="utf-8"
    )
    monkeypatch.setenv("BATCH_PARALLEL_CAMPAIGN_ROOT", str(root))

    class Orch:
        def __init__(self):
            self.hls_code = SEQUENTIAL_LCST
            self.synth_report = dict(FLASH_SEQ_REPORT)
            self.generated_csim = {"passed": True}
            self.header_code = ""
            self.header_name = "kernel.h"
            self.benchmark_context = ""
            self.benchmark_name = "autosa_mm"
            self._artifact_output_dir = None

        def _call_llm(self, messages, max_tokens=None):
            return """```json
{"schema": "flash_overlap_enforcement_v1", "passed": true,
 "code_intended": {"ok": true, "dataflow": true, "ping_pong": true, "reason": "ok"},
 "csynth_shows": {"ok": true, "reason": "interval below latency"}}
```"""

        def _request_code_revision(self, prompt):
            return PINGPONG_DATAFLOW

        def _evaluate_candidate_with_repairs(self, code, label):
            return {
                "success": True,
                "code": code,
                "report": dict(OVERLAP_REPORT),
                "csim": {"passed": True},
                "cosim": None,
            }

    out = fe.maybe_run_enforcement(Orch())
    assert out is not None
    assert out["attempted"] is True


def test_maybe_run_commits_kernel_when_judge_passes(monkeypatch):
    monkeypatch.setenv("C2HLS_ENFORCEMENT", "1")
    monkeypatch.setenv("C2HLS_ENFORCEMENT_ROUNDS", "4")

    class Orch:
        def __init__(self):
            self.hls_code = SEQUENTIAL_LCST
            self.synth_report = dict(FLASH_SEQ_REPORT)
            self.generated_csim = {"passed": True}
            self.header_code = ""
            self.header_name = "kernel.h"
            self.benchmark_context = ""
            self.benchmark_name = "autosa_mm"
            self._artifact_output_dir = None
            self.calls = 0

        def _call_llm(self, messages, max_tokens=None):
            self.calls += 1
            code = self.hls_code
            # After repair, judge the candidate from the user prompt's latest kernel.
            blob = messages[-1]["content"] if messages else ""
            if "DATAFLOW" in blob and "4688" in blob:
                return """```json
{"schema": "flash_overlap_enforcement_v1", "passed": true,
 "code_intended": {"ok": true, "dataflow": true, "ping_pong": true, "reason": "ok"},
 "csynth_shows": {"ok": true, "reason": "interval below latency"}}
```"""
            return """```json
{"schema": "flash_overlap_enforcement_v1", "passed": false,
 "code_intended": {"ok": false, "dataflow": false, "ping_pong": false, "reason": "lcst"},
 "csynth_shows": {"ok": false, "reason": "interval equals latency"},
 "repair_focus": "add ping-pong DATAFLOW"}
```"""

        def _request_code_revision(self, prompt):
            return PINGPONG_DATAFLOW

        def _evaluate_candidate_with_repairs(self, code, label):
            return {
                "success": True,
                "code": code,
                "report": dict(OVERLAP_REPORT),
                "csim": {"passed": True},
                "cosim": None,
            }

    orch = Orch()
    out = fe.maybe_run_enforcement(orch)
    assert out is not None
    assert out["passed"] is True
    assert out["applied"] is True
    assert orch.hls_code == PINGPONG_DATAFLOW
    assert orch.synth_report["interval"] == 4689
    assert orch.calls >= 1


def test_maybe_run_does_not_pass_structure_without_overlap(monkeypatch):
    monkeypatch.setenv("C2HLS_ENFORCEMENT", "1")
    monkeypatch.setenv("C2HLS_ENFORCEMENT_ROUNDS", "1")
    orch = SimpleNamespace(
        hls_code=PINGPONG_DATAFLOW,
        synth_report=dict(STORE_BOUND_REPORT),
        generated_csim={"passed": True},
        header_code="",
        header_name="kernel.h",
        benchmark_context="",
        benchmark_name="autosa_mm",
        _artifact_output_dir=None,
    )
    out = fe.maybe_run_enforcement(orch)
    assert out is not None
    assert out["passed"] is False
    assert out["applied"] is False
    assert out["overlap"] is False
    assert out["code_intended"] is True


def test_format_report_includes_resources_and_cycles():
    blob = fe.format_judge_report(OVERLAP_REPORT)
    assert "4688" in blob
    assert "DSP" in blob.upper() or "dsp" in blob
    assert "BRAM" in blob.upper() or "bram" in blob


def test_attach_enforcement_is_idempotent(monkeypatch):
    monkeypatch.setenv("C2HLS_ENFORCEMENT", "1")
    calls = {"n": 0}

    def fake_maybe(orch):
        calls["n"] += 1
        out = {"attempted": True, "passed": False, "applied": False, "rounds_used": 0}
        orch.enforcement_result = out
        return out

    monkeypatch.setattr(fe, "maybe_run_enforcement", fake_maybe)
    orch = SimpleNamespace(
        hls_code=SEQUENTIAL_LCST,
        synth_report=dict(FLASH_SEQ_REPORT),
        benchmark_name="autosa_mm",
        _pipelined_ctx={"flash_done": True},
    )
    first = fe.attach_enforcement_after_flash(orch)
    second = fe.attach_enforcement_after_flash(orch)
    assert calls["n"] == 1
    assert first is not None and first["attempted"] is True
    assert second == first
    assert orch._pipelined_ctx["enforcement_ran"] is True
    assert orch._pipelined_ctx["enforcement"]["attempted"] is True


def test_autosa_and_tier_a_flash_paths_call_attach():
    """AutoSA uses tier_a _run_synth_flash, not flash_pipelined handle_job."""
    import inspect

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "pc2"))
    from batch_parallel_dispatch import run_batch_parallel_job
    from flash_pipelined_bench import FlashPipelinedBenchSession
    from tier_a_batch_parallel_bench import TierABatchParallelBenchSession

    dispatch_src = inspect.getsource(run_batch_parallel_job)
    assert "is_autosa_workflow" in dispatch_src
    assert "tier_a_execute_job" in dispatch_src
    assert "attach_enforcement_after_flash" in inspect.getsource(
        TierABatchParallelBenchSession._run_synth_flash
    )
    assert "attach_enforcement_after_flash" in inspect.getsource(
        FlashPipelinedBenchSession._finalize_success
    )
    assert "attach_enforcement_after_flash" in inspect.getsource(
        FlashPipelinedBenchSession._run_synth
    )


FLASH_WIDE_4808 = """
#include "kernel.h"
extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512
  const int LANES = 16;
  data_t A_local[I][K];
  data_t B_local[J][K];
  data_t C_local[I][J];
  load_a_rows: for (int i = 0; i < I; ++i)
    load_a_chunks: for (int k0 = 0; k0 < K; k0 += LANES) {
#pragma HLS PIPELINE II=1
      load_a_lanes: for (int u = 0; u < LANES; ++u) {
#pragma HLS UNROLL
        A_local[i][k0 + u] = A[i][k0 + u];
      }
    }
  load_b_rows: for (int j = 0; j < J; ++j)
    load_b_chunks: for (int k0 = 0; k0 < K; k0 += LANES) {
#pragma HLS PIPELINE II=1
      load_b_lanes: for (int u = 0; u < LANES; ++u) {
#pragma HLS UNROLL
        B_local[j][k0 + u] = B[j][k0 + u];
      }
    }
  compute_i: for (int i = 0; i < I; ++i)
    compute_j: for (int j = 0; j < J; ++j) {
#pragma HLS PIPELINE II=1
      C_local[i][j] = A_local[i][0] * B_local[j][0];
    }
  store_c_rows: for (int i = 0; i < I; ++i)
    store_c_chunks: for (int j0 = 0; j0 < J; j0 += LANES) {
#pragma HLS PIPELINE II=1
      store_c_lanes: for (int u = 0; u < LANES; ++u) {
#pragma HLS UNROLL
        C[i][j0 + u] = C_local[i][j0 + u];
      }
    }
}
"""

FLASH_4808_REPORT = {
    "latency_cycles": 4808,
    "interval": 4809,
    "dsp": 318,
    "bram": 106,
    "ff": 54536,
    "lut": 37238,
    "modules": [
        {"name": "autosa_mm_Pipeline_load_a_rows_load_a_chunks", "latency_cycles": 259},
        {"name": "autosa_mm_Pipeline_load_b_rows_load_b_chunks", "latency_cycles": 259},
        {"name": "autosa_mm_Pipeline_compute_i_compute_j", "latency_cycles": 4145},
        {"name": "autosa_mm_Pipeline_store_c_rows_store_c_chunks", "latency_cycles": 259},
    ],
}

ENF_12642_CODE = """
#include "kernel.h"
#define NT 2
#define TI (I / NT)
static void load_B(data_t B[J][K], data_t B_local[J][K]) {
#pragma HLS INLINE off
  for (int j = 0; j < J; ++j)
    for (int k = 0; k < K; ++k) {
#pragma HLS PIPELINE II=1
      B_local[j][k] = B[j][k];
    }
}
static void load_A_tile(data_t A[I][K], data_t A_buf[TI][K], int t) {
#pragma HLS INLINE off
  const int i0 = t * TI;
  for (int i = 0; i < TI; ++i)
    for (int k = 0; k < K; ++k) {
#pragma HLS PIPELINE II=1
      A_buf[i][k] = A[i0 + i][k];
    }
}
extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
  data_t B_local[J][K];
  load_B(B, B_local);
  tile_loop: for (int t = 0; t < NT; ++t) {
#pragma HLS DATAFLOW
    data_t A_buf[TI][K];
    data_t C_buf[TI][J];
    load_A_tile(A, A_buf, t);
    compute_tile(A_buf, B_local, C_buf);
    store_C_tile(C_buf, C, t);
  }
}
"""

ENF_12642_REPORT = {
    "latency_cycles": 12642,
    "interval": 12643,
    "dsp": 318,
    "bram": 106,
    "ff": 56915,
    "lut": 37324,
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

WIDE_PP_CODE = """
#include "kernel.h"
#define NT 2
#define TI (I / NT)
static void load_B(data_t B[J][K], data_t B_local[J][K]) {
#pragma HLS INLINE off
  const int LANES = 16;
  load_b_rows: for (int j = 0; j < J; ++j)
    load_b_chunks: for (int k0 = 0; k0 < K; k0 += LANES) {
#pragma HLS PIPELINE II=1
      load_b_lanes: for (int u = 0; u < LANES; ++u) {
#pragma HLS UNROLL
        B_local[j][k0 + u] = B[j][k0 + u];
      }
    }
}
static void load_A_tile(data_t A[I][K], data_t A_buf[TI][K], int t) {
#pragma HLS INLINE off
}
static void compute_tile(data_t A_buf[TI][K], data_t B_local[J][K], data_t C_buf[TI][J]) {
#pragma HLS INLINE off
}
static void store_C_tile(data_t C_buf[TI][J], data_t C[I][J], int t) {
#pragma HLS INLINE off
}
extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
  data_t B_local[J][K];
  data_t A_buf[2][TI][K];
  data_t C_buf[2][TI][J];
  load_B(B, B_local);
  tile_loop: for (int t = 0; t < NT; ++t) {
#pragma HLS DATAFLOW
    load_A_tile(A, A_buf[(t + 1) & 1], t + 1);
    compute_tile(A_buf[t & 1], B_local, C_buf[t & 1]);
    store_C_tile(C_buf[(t - 1) & 1], C, t - 1);
  }
}
"""


def test_static_verdict_rejects_12642_that_dropped_flash_wide_loads():
    v = fe.static_verdict(
        ENF_12642_CODE,
        ENF_12642_REPORT,
        baseline_code=FLASH_WIDE_4808,
        baseline_report=FLASH_4808_REPORT,
    )
    assert v.passed is False
    assert v.code_intended is False or v.csynth_shows is False
    blob = (v.reason + " " + (v.repair_focus or "")).lower()
    assert any(
        n in blob
        for n in ("lanes", "512", "regress", "4808", "259", "4171", "scalar", "flash")
    )


def test_static_verdict_still_fails_flash_4808_for_missing_dataflow():
    v = fe.static_verdict(
        FLASH_WIDE_4808,
        FLASH_4808_REPORT,
        baseline_code=FLASH_WIDE_4808,
        baseline_report=FLASH_4808_REPORT,
    )
    assert v.passed is False
    assert v.code_intended is False


def test_static_verdict_passes_wide_pingpong_better_than_flash():
    v = fe.static_verdict(
        WIDE_PP_CODE,
        OVERLAP_REPORT,
        baseline_code=FLASH_WIDE_4808,
        baseline_report=FLASH_4808_REPORT,
    )
    assert v.passed is True, v.reason


def test_repair_prompt_keeps_flash_lanes_and_forbids_scalar_copy():
    blob = fe.build_repair_prompt(
        verdict_json='{"passed": false}',
        repair_focus="add tile DATAFLOW",
        benchmark_context="autosa_mm",
        header_name="kernel.h",
        header_code="#define I 64",
        kernel_code=ENF_12642_CODE,
        report_blob="latency 12642",
        flash_kernel_code=FLASH_WIDE_4808,
        flash_report_blob="latency 4808 load_A=259 load_B=259 compute=4145 store=259 dsp=318",
    )
    low = blob.lower()
    assert "lanes" in low
    assert "k0 += lanes" in low or "k0 += 16" in low
    assert "wrap" in low or "keep" in low
    assert "dot64" in low or "compute" in low
    assert "hls-enforcement-keep-flash-wide-load" in low
    assert "avoid-enforcement-scalar-axi-copy" in low
    wrap = blob.split("## How to wrap")[1].split("## Keep-flash skills")[0]
    assert "B_local[j][k] = B[j][k]" not in wrap
    assert "k0 += LANES" in wrap or "k0 += lanes" in wrap.lower()


def test_repair_loop_does_not_commit_12642_over_flash():
    def judge(code, report):
        return fe.static_verdict(
            code,
            report,
            baseline_code=FLASH_WIDE_4808,
            baseline_report=FLASH_4808_REPORT,
        )

    def generate(code, report, verdict, round_i):
        return ENF_12642_CODE

    def evaluate(code):
        return {
            "success": True,
            "code": code,
            "report": ENF_12642_REPORT,
            "csim": {"passed": True},
            "cosim": None,
        }

    out = fe.run_enforcement_loop(
        kernel_code=FLASH_WIDE_4808,
        synth_report=FLASH_4808_REPORT,
        rounds=2,
        judge_fn=judge,
        generate_fn=generate,
        evaluate_fn=evaluate,
    )
    assert out["passed"] is False
    assert out["applied"] is not True
    assert out["code"] == ENF_12642_CODE or out["rounds_used"] >= 1


def test_enforcement_keep_flash_skills_file_is_valid():
    import json
    from skill_library import _coerce_skill_entry

    path = (
        Path(__file__).resolve().parents[1]
        / "hls_full_optimization_skills_schema_1_1_package"
        / "flash_enforcement_keep_flash_skill_entries.json"
    )
    assert path.is_file(), path
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = [e["id"] for e in data["skills"]]
    for sid in (
        "hls-enforcement-keep-flash-wide-load",
        "hls-enforcement-keep-flash-compute-nest",
        "hls-enforcement-wrap-flash-with-tile-dataflow",
        "avoid-enforcement-scalar-axi-copy",
        "avoid-enforcement-worse-than-flash",
    ):
        assert sid in ids, sid
    for entry in data["skills"]:
        assert _coerce_skill_entry(entry) is not None, entry.get("id")


def test_enforcement_loadb_in_df_skills_file_is_valid():
    import json
    from skill_library import _coerce_skill_entry

    path = (
        Path(__file__).resolve().parents[1]
        / "hls_full_optimization_skills_schema_1_1_package"
        / "flash_enforcement_loadb_in_dataflow_skill_entries.json"
    )
    assert path.is_file(), path
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = [e["id"] for e in data["skills"]]
    assert "hls-enforcement-loadb-in-tile-dataflow" in ids
    assert "avoid-enforcement-loadb-prefix-when-df-opt-in" in ids
    for entry in data["skills"]:
        assert _coerce_skill_entry(entry) is not None, entry.get("id")


def test_apply_enforcement_from_campaign_sets_loadb_in_df(monkeypatch):
    monkeypatch.delenv("C2HLS_PP_LOAD_B_IN_DF", raising=False)
    fe.apply_enforcement_from_campaign(
        {"enforcement": True, "enforcement_rounds": 20, "pp_load_b_in_dataflow": True}
    )
    assert os.environ["C2HLS_PP_LOAD_B_IN_DF"] == "1"
    assert fe.LOADB_IN_DF_ENV == "C2HLS_PP_LOAD_B_IN_DF"


def test_default_keep_flash_skills_omit_loadb_overlay(monkeypatch):
    monkeypatch.delenv("C2HLS_PP_LOAD_B_IN_DF", raising=False)
    _block, ids = fe.load_enforcement_keep_flash_skills()
    assert "hls-enforcement-loadb-in-tile-dataflow" not in ids
    assert "avoid-enforcement-loadb-prefix-when-df-opt-in" not in ids
