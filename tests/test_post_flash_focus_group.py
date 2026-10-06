"""Focus group: analysis pack, coverage gate, prompt roles, two-round loop."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import post_flash_focus_group as fg


HEADER_1024 = """
typedef float data_t;
#define I 1024
#define J 1024
#define K 1024
"""

WINNER_INCOMPLETE_KERNEL = """
#include "kernel.h"
extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
  const int PE   = 16;
  const int SIMD = 32;
  const int TI = 64;
  const int TJ = 64;
  const int TK = 64;
  data_t local_A[TI][TK];
  data_t local_B[TJ][TK];
  data_t Crow[PE][TJ];
  loop_i0: for (int i0 = 0; i0 < I; i0 += TI) {
    loop_j0: for (int j0 = 0; j0 < J; j0 += TJ) {
      load_A: for (int i = 0; i < TI; i++) {
        load_A_k: for (int k = 0; k < TK; k++) {
#pragma HLS PIPELINE II=1
          local_A[i][k] = A[i0 + i][k];
        }
      }
      load_B: for (int j = 0; j < TJ; j++) {
        load_B_k: for (int k = 0; k < TK; k++) {
#pragma HLS PIPELINE II=1
          local_B[j][k] = B[j0 + j][k];
        }
      }
      compute_k0: for (int k0 = 0; k0 < TK; k0 += SIMD) {
        compute_j: for (int j = 0; j < TJ; j++) {
#pragma HLS PIPELINE II=1
          pe_mac: for (int p = 0; p < PE; ++p) {
#pragma HLS UNROLL
            data_t partial = 0;
            simd_k: for (int s = 0; s < SIMD; ++s) {
#pragma HLS UNROLL
              partial += local_A[p][k0 + s] * local_B[j][k0 + s];
            }
            Crow[p][j] += partial;
          }
        }
      }
      store_C: for (int i = 0; i < PE; i++) {
        store_C_j: for (int j = 0; j < TJ; j++) {
#pragma HLS PIPELINE II=1
          C[i0 + i][j0 + j] = Crow[i][j];
        }
      }
    }
  }
}
"""

FULL_K_PACKED_STORE_KERNEL = """
#include "kernel.h"
#define PE    64
#define SIMD  16
#define TJ    64
#define TK    64
extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
  data_t local_A[PE][TK];
  data_t local_B[TJ][TK];
  data_t Crow[PE][TJ];
  tile_i: for (int i0 = 0; i0 < I; i0 += PE) {
    tile_j: for (int j0 = 0; j0 < J; j0 += TJ) {
      tile_k: for (int k0 = 0; k0 < K; k0 += TK) {
        load_A: for (int idx = 0; idx < PE * TK; idx += SIMD) {
#pragma HLS PIPELINE II=1
          local_A[idx >> 6][idx & 63] = A[i0][k0];
        }
        load_B: for (int idx = 0; idx < TJ * TK; idx += SIMD) {
#pragma HLS PIPELINE II=1
          local_B[idx >> 6][idx & 63] = B[j0][k0];
        }
        compute_k0: for (int kk = 0; kk < TK; kk += SIMD) {
          compute_j: for (int j = 0; j < TJ; ++j) {
#pragma HLS PIPELINE II=1
            Crow[0][j] += local_A[0][kk] * local_B[j][kk];
          }
        }
      }
      store_C: for (int idx = 0; idx < PE * TJ; idx += SIMD) {
#pragma HLS PIPELINE II=1
        C[i0][j0] = Crow[0][0];
      }
    }
  }
}
"""


def _scope(scope_id, kind="loop", parent=None, trip=1, latency=100, pipeline_ii=1, pipelined=True):
    return {
        "scope_id": scope_id,
        "name": scope_id.split("/")[-1],
        "kind": kind,
        "parent": parent,
        "tripcount": trip,
        "trip": trip,
        "latency_cycles": latency,
        "pipeline_ii": pipeline_ii,
        "pipelined": "yes" if pipelined else "no",
    }


def _incomplete_report():
    """PE 16 x SIMD 32, one TK=64 K-slice, store PE rows only. Top 1,443,073."""
    top = "autosa_mm"
    ij = f"{top}/loop_i0_loop_j0"
    return {
        "latency_cycles": 1443073,
        "lut": 1000,
        "dsp": 2592,
        "ff": 2000,
        "bram": 90,
        "uram": 0,
        "feedback": {
            "scopes": [
                _scope(top, kind="module", trip=1, latency=1443073, pipelined=False),
                _scope(ij, parent=top, trip=256, latency=1443072, pipeline_ii=None, pipelined=False),
                _scope(f"{ij}/load_A", parent=ij, trip=4096, latency=4168),
                _scope(f"{ij}/load_B", parent=ij, trip=4096, latency=4168),
                _scope(f"{ij}/compute_j", parent=ij, trip=128, latency=364),
                _scope(f"{ij}/store_C", parent=ij, trip=1024, latency=1093),
            ]
        },
    }


def _full_k_report(*, latency=3101953):
    """PE 64 x SIMD 16, K tiled 16 times, packed store. Floor 1,048,576."""
    top = "autosa_mm"
    ij = f"{top}/tile_i_tile_j"
    tk = f"{ij}/tile_k"
    return {
        "latency_cycles": latency,
        "lut": 1000,
        "dsp": 5248,
        "ff": 2000,
        "bram": 106,
        "uram": 0,
        "feedback": {
            "scopes": [
                _scope(top, kind="module", trip=1, latency=latency, pipelined=False),
                _scope(ij, parent=top, trip=256, latency=latency - 1, pipeline_ii=None, pipelined=False),
                _scope(tk, parent=ij, trip=16, latency=10000, pipeline_ii=None, pipelined=False),
                _scope(f"{tk}/load_A", parent=tk, trip=256, latency=328),
                _scope(f"{tk}/load_B", parent=tk, trip=256, latency=328),
                _scope(f"{tk}/compute_j", parent=tk, trip=256, latency=380),
                _scope(f"{ij}/store_C", parent=ij, trip=256, latency=325),
            ]
        },
    }


class FakeOrch:
    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = 0
        self.messages = []
        self.part = "xcu280-fsvh2892-2L-e"
        self.clock_ns = 3.33
        self.gpt_model = "fake-model"

    def _call_llm(self, messages):
        self.messages.append(messages)
        idx = self.calls
        self.calls += 1
        if idx < len(self.replies):
            return self.replies[idx]
        system = " ".join(m.get("content", "") for m in messages if m.get("role") == "system")
        if "```kernel" in system or "code editor" in system.lower() or "chair" in system.lower():
            return (
                "```kernel\n"
                + FULL_K_PACKED_STORE_KERNEL
                + "\n```\n"
            )
        return "**targets:** compute_j\n**actions:** keep full I,J,K\n**avoid:** shrinking K\n"


def test_enabled_default_off(monkeypatch):
    monkeypatch.delenv("C2HLS_FOCUS_GROUP", raising=False)
    assert fg.focus_group_enabled() is False


def test_enabled_on(monkeypatch):
    monkeypatch.setenv("C2HLS_FOCUS_GROUP", "1")
    assert fg.focus_group_enabled() is True


def test_rounds_default_two_repair_default_three(monkeypatch):
    monkeypatch.delenv("C2HLS_FOCUS_GROUP_ROUNDS", raising=False)
    monkeypatch.delenv("C2HLS_FOCUS_GROUP_REPAIR_ROUNDS", raising=False)
    assert fg.focus_round_limit() == 2
    assert fg.repair_round_limit() == 3


def test_parse_pe_simd_from_const_and_define():
    assert fg.parse_pe_simd_from_kernel(WINNER_INCOMPLETE_KERNEL) == (16, 32)
    assert fg.parse_pe_simd_from_kernel(FULL_K_PACKED_STORE_KERNEL) == (64, 16)


def test_classify_scope_region_by_name():
    assert fg.classify_scope_region(_scope("autosa_mm/load_A")) == "load"
    assert fg.classify_scope_region(_scope("autosa_mm/load_B_k")) == "load"
    assert fg.classify_scope_region(_scope("autosa_mm/compute_j")) == "compute"
    assert fg.classify_scope_region(_scope("autosa_mm/pe_mac")) == "compute"
    assert fg.classify_scope_region(_scope("autosa_mm/store_C")) == "store"
    assert fg.classify_scope_region(_scope("autosa_mm/wb_j")) == "store"
    assert fg.classify_scope_region(_scope("autosa_mm/tile_k")) == "other"


def test_arithmetic_floor_pe_times_simd():
    assert fg.arithmetic_floor(1024, 1024, 1024, 64, 16) == 1048576
    assert fg.arithmetic_floor(1024, 1024, 1024, 16, 32) == 2097152
    assert fg.arithmetic_floor(1024, 1024, 1024, None, 16) is None


def test_analysis_pack_region_cycles_and_floor():
    pack = fg.build_analysis_pack(
        header_code=HEADER_1024,
        kernel_code=FULL_K_PACKED_STORE_KERNEL,
        report=_full_k_report(),
    )
    assert pack["i"] == 1024
    assert pack["j"] == 1024
    assert pack["k"] == 1024
    assert pack["pe"] == 64
    assert pack["simd"] == 16
    assert pack["arithmetic_floor"] == 1048576
    assert pack["regions"]["compute"]["ii1"] is True
    assert pack["regions"]["load"]["latency_cycles"] > 0
    assert pack["sequential_tax"]["sum_region_latency"] >= pack["regions"]["compute"]["latency_cycles"]
    text = fg.render_analysis_pack(pack)
    assert "1048576" in text
    assert "load" in text
    assert "compute" in text
    assert "store" in text


def test_coverage_gate_rejects_k_slice_and_partial_store():
    pack = fg.build_analysis_pack(
        header_code=HEADER_1024,
        kernel_code=WINNER_INCOMPLETE_KERNEL,
        report=_incomplete_report(),
    )
    gate = fg.coverage_gate(pack)
    assert gate.ok is False
    assert pack["compute_trip_total"] < pack["arithmetic_floor"] * 0.5
    assert "K" in gate.reason or "compute" in gate.reason or "store" in gate.reason


def test_coverage_gate_passes_full_k_packed_store():
    pack = fg.build_analysis_pack(
        header_code=HEADER_1024,
        kernel_code=FULL_K_PACKED_STORE_KERNEL,
        report=_full_k_report(),
    )
    gate = fg.coverage_gate(pack)
    assert gate.ok is True
    assert pack["compute_trip_total"] == pack["arithmetic_floor"]


def test_prompt_roles_specialists_plan_only_chair_emits_kernel():
    prompts = fg.prompt_text_for_docs()
    assert fg.focus_round_limit.__name__
    assert "plan only" in prompts["analyst_system"].lower() or "do not output" in prompts["analyst_system"].lower()
    assert "```kernel" not in prompts["analyst_system"]
    for name in fg.SPECIALIST_IDS:
        system = prompts["specialist_systems"][name]
        assert "plan only" in system.lower() or "do not output" in system.lower()
        assert "```kernel" not in system
        assert name.replace("_", " ") in system.lower() or name in system.lower() or name.split("_")[0] in system.lower()
    assert "```kernel" in prompts["chair_system"]
    assert "plan only" not in prompts["chair_system"].lower()
    assert "coverage" in prompts["repair_user"].lower() or "numerical" in prompts["repair_user"].lower()
    assert "C2HLS_FOCUS_GROUP_REPAIR" not in prompts["chair_system"]
    # Repair is legality, not a focus-round counter.
    assert "focus round" not in prompts["repair_user"].lower()


def test_specialist_order_and_count():
    assert fg.SPECIALIST_IDS == (
        "load",
        "compute",
        "store",
        "ii_pipeline",
        "unroll",
        "memory_partition",
    )


def test_run_focus_group_two_rounds_chair_is_only_kernel_writer(tmp_path, monkeypatch):
    import c2hls

    out_dir = tmp_path / "focus_group"
    monkeypatch.setenv("C2HLS_FOCUS_GROUP_ROUNDS", "2")
    monkeypatch.setenv("C2HLS_FOCUS_GROUP_REPAIR_ROUNDS", "3")
    monkeypatch.setattr(c2hls, "compile_check_cpp", lambda *a, **k: (True, ""))

    reports = [
        _full_k_report(latency=2500000),
        _full_k_report(latency=2000000),
    ]
    state = {"i": 0}

    def fake_synth(*args, **kwargs):
        idx = min(state["i"], len(reports) - 1)
        state["i"] += 1
        return {
            "synth": {"success": True, "report": reports[idx]},
            "csim": {"passed": True},
            "cosim": None,
        }

    monkeypatch.setattr(c2hls, "_run_synth_csim_cosim", fake_synth)
    orch = FakeOrch()
    outcome = fg.run_focus_group(
        bench="autosa_mm",
        kernel_code=FULL_K_PACKED_STORE_KERNEL,
        header_code=HEADER_1024,
        header_name="kernel.h",
        report=_full_k_report(latency=3101953),
        orchestrator=orch,
        out_dir=out_dir,
        testbench_code="// tb",
        top_function="autosa_mm",
        part="xcu280-fsvh2892-2L-e",
        clock_ns=3.33,
    )
    assert outcome.success is True
    transcript = json.loads((out_dir / "focus_group_transcript.json").read_text())
    assert transcript["schema"] == "post_flash_focus_group_transcript_v1"
    assert len(transcript["rounds"]) == 2
    assert transcript["rounds"][0]["accepted"] is True
    assert outcome.result["latency_cycles"] == 2000000

    kernel_writer_calls = 0
    specialist_calls = 0
    analyst_calls = 0
    for msgs in orch.messages:
        system = " ".join(m.get("content", "") for m in msgs if m.get("role") == "system")
        if "chair" in system.lower() or "code editor" in system.lower():
            kernel_writer_calls += 1
            assert "```kernel" in system
        elif "theoretical" in system.lower() or "analyst" in system.lower():
            analyst_calls += 1
            assert "```kernel" not in system
        elif "specialist" in system.lower() or "plan only" in system.lower():
            specialist_calls += 1
            assert "```kernel" not in system
    assert analyst_calls == 2
    assert specialist_calls == 12
    assert kernel_writer_calls == 2
    assert (out_dir / "selected.cpp").is_file()


def test_coverage_gate_rejects_faster_incomplete_candidate():
    seed = fg.build_analysis_pack(
        header_code=HEADER_1024,
        kernel_code=FULL_K_PACKED_STORE_KERNEL,
        report=_full_k_report(latency=3101953),
    )
    cand = fg.build_analysis_pack(
        header_code=HEADER_1024,
        kernel_code=WINNER_INCOMPLETE_KERNEL,
        report=_incomplete_report(),
    )
    assert fg.coverage_gate(seed).ok is True
    assert fg.coverage_gate(cand).ok is False
    assert fg.should_accept(cand, seed, part="xcu280-fsvh2892-2L-e") is False


def test_maybe_chain_skipped_when_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("C2HLS_FOCUS_GROUP", raising=False)
    called = {"n": 0}

    def boom(**kwargs):
        called["n"] += 1
        raise AssertionError("run_focus_group should not run")

    monkeypatch.setattr(fg, "run_focus_group_for_kernel_dir", boom)
    out = fg.maybe_chain_focus_group(
        bench="autosa_mm",
        bench_dir=tmp_path,
        kernel_dir=tmp_path,
        orchestrator=FakeOrch(),
    )
    assert out is None
    assert called["n"] == 0


def test_maybe_chain_runs_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("C2HLS_FOCUS_GROUP", "1")
    called = {}

    def fake_run(**kwargs):
        called.update(kwargs)
        return fg.FocusGroupOutcome(
            bench="autosa_mm",
            success=True,
            out_dir=str(tmp_path / "focus_group"),
            result={"latency_cycles": 1},
        )

    monkeypatch.setattr(fg, "run_focus_group_for_kernel_dir", fake_run)
    out = fg.maybe_chain_focus_group(
        bench="autosa_mm",
        bench_dir=tmp_path / "bench",
        kernel_dir=tmp_path / "pe64_simd16",
        orchestrator=FakeOrch(),
    )
    assert out is not None
    assert out.success is True
    assert called["kernel_dir"] == tmp_path / "pe64_simd16"


def test_run_focus_group_raises_completion_token_floor(tmp_path, monkeypatch):
    import c2hls

    monkeypatch.setenv("C2HLS_FOCUS_GROUP_ROUNDS", "1")
    monkeypatch.setenv("C2HLS_FOCUS_GROUP_REPAIR_ROUNDS", "1")
    monkeypatch.setattr(c2hls, "compile_check_cpp", lambda *a, **k: (True, ""))
    monkeypatch.setattr(
        c2hls,
        "_run_synth_csim_cosim",
        lambda *a, **k: {
            "synth": {"success": True, "report": _full_k_report(latency=2000000)},
            "csim": {"passed": True},
            "cosim": None,
        },
    )
    orch = FakeOrch()
    orch.max_completion_tokens = 8192
    fg.run_focus_group(
        bench="autosa_mm",
        kernel_code=FULL_K_PACKED_STORE_KERNEL,
        header_code=HEADER_1024,
        header_name="kernel.h",
        report=_full_k_report(latency=3101953),
        orchestrator=orch,
        out_dir=tmp_path / "focus_group",
        testbench_code="// tb",
        top_function="autosa_mm",
        part="xcu280-fsvh2892-2L-e",
        clock_ns=3.33,
    )
    assert orch.max_completion_tokens >= 65536


def test_dse_v2_winner_hook_calls_focus_group(monkeypatch, tmp_path):
    import post_flash_dse_v2 as v2

    monkeypatch.setenv("C2HLS_FOCUS_GROUP", "1")
    seen = {}

    def fake_fg(**kwargs):
        seen["kernel_dir"] = kwargs["kernel_dir"]
        return None

    monkeypatch.setattr("post_flash_focus_group.maybe_chain_focus_group", fake_fg)
    outcome = type("O", (), {"success": True, "result": {"winner_trial_id": "pe64_simd16"}})()
    v2._maybe_run_focus_group_after_winner(
        bench="autosa_mm",
        bench_dir=tmp_path / "bench",
        cell_dir=tmp_path / "cell",
        orchestrator=FakeOrch(),
        outcome=outcome,
    )
    assert seen["kernel_dir"] == tmp_path / "cell" / "dse_v2" / "pe64_simd16"
