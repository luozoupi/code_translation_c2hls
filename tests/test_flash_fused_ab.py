"""On-chip flash must reject fused A+B loads (run-1 / 871 failure mode)."""
from __future__ import annotations

import json
from pathlib import Path

import autosa_flow_gates as g
import prompt_c2hls as prompts
from c2hls import flash_fused_ab_reject_error

REPO = Path(__file__).resolve().parents[1]
ONCHIP_JSON = (
    REPO / "hls_full_optimization_skills_schema_1_1_package"
    / "flash_onchip_wide_gemm_skill_entries.json"
)

# Run-1 autosa_mm_hcl: one pipeline writes A_loc and B_loc together.
_FUSED_RUN1 = """
extern "C" void autosa_mm_hcl(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
    data_t A_loc[I][K];
    data_t B_loc[J][K];
    load_A_B:
    for (int word = 0; word < (I * K) / LANES; ++word) {
#pragma HLS PIPELINE II=1
        int row = word / (K / LANES);
        int col0 = (word % (K / LANES)) * LANES;
        load_lanes:
        for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
            A_loc[row][col0 + l] = A[row][col0 + l];
            B_loc[row][col0 + l] = B[row][col0 + l];
        }
    }
}
"""

# Champion 940-class: nested i / j += LANES, separate load_A then load_B.
_SEPARATE_CHAMPION = """
extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
    data_t A_local[I][K];
    data_t B_local[J][K];
    load_A_rows: for (int i = 0; i < I; ++i) {
      load_A_cols: for (int j = 0; j < K; j += LANES) {
#pragma HLS PIPELINE II=1
        load_A_lanes: for (int u = 0; u < LANES; ++u) {
#pragma HLS UNROLL
          A_local[i][j + u] = A[i][j + u];
        }
      }
    }
    load_B_rows: for (int j = 0; j < J; ++j) {
      load_B_cols: for (int k = 0; k < K; k += LANES) {
#pragma HLS PIPELINE II=1
        load_B_lanes: for (int u = 0; u < LANES; ++u) {
#pragma HLS UNROLL
          B_local[j][k + u] = B[j][k + u];
        }
      }
    }
}
"""


def test_detects_run1_fused_label_and_zipped_body():
    assert g.flash_fused_ab_in_code(_FUSED_RUN1) is True
    assert g.flash_fused_ab_in_code(_SEPARATE_CHAMPION) is False
    assert g.flash_fused_ab_in_code("") is False


def test_detects_zip_without_fused_label():
    zipped = """
    for (int t = 0; t < N; ++t) {
#pragma HLS PIPELINE II=1
        A_local[t] = A[t];
        B_local[t] = B[t];
    }
    """
    assert g.flash_fused_ab_in_code(zipped) is True


def test_detects_fused_module_in_csynth_report():
    report = {
        "dsp": 5344,
        "latency_cycles": 906,
        "feedback": {
            "scopes": [
                {"name": "autosa_mm_hcl_Pipeline_load_A_B", "latency_cycles": 282},
            ]
        },
    }
    assert g.flash_fused_ab_in_report(report) is True
    assert g.flash_fused_ab_in_report({"dsp": 5344}) is False


def test_fused_ab_error_text():
    err = g.flash_fused_ab_error(_FUSED_RUN1)
    assert err is not None
    low = err.lower()
    assert "rejected" in low
    assert "load_a_b" in low or "fused" in low
    assert "load_a" in low and "load_b" in low
    assert g.flash_fused_ab_error(_SEPARATE_CHAMPION) is None


def test_onchip_guidance_forbids_fused_ab():
    text = g.flash_onchip_initial_guidance().lower()
    assert "load_a" in text
    assert "load_b" in text
    assert "load_a_b" in text
    assert "never" in text or "do not fuse" in text or "not fuse" in text
    assert "zip" in text or "lockstep" in text or "same loop" in text


def test_onchip_pack_has_fused_ab_avoid():
    data = json.loads(ONCHIP_JSON.read_text(encoding="utf-8"))
    ids = [s["id"] for s in data["skills"]]
    assert "avoid-onchip-fused-ab-lockstep" in ids
    blob = json.dumps(data).lower()
    assert "load_a_b" in blob
    assert "never" in blob or "do not fuse" in blob


def test_c2hls_helper_only_when_onchip_flash(monkeypatch):
    monkeypatch.delenv("C2HLS_FLASH_ONCHIP", raising=False)
    assert flash_fused_ab_reject_error("flash", _FUSED_RUN1) is None
    monkeypatch.setenv("C2HLS_FLASH_ONCHIP", "1")
    assert flash_fused_ab_reject_error("flash", _FUSED_RUN1)
    assert flash_fused_ab_reject_error("tiling", _FUSED_RUN1) is None
    assert flash_fused_ab_reject_error("flash", _SEPARATE_CHAMPION) is None


def test_fused_ab_repair_prompt():
    err = g.flash_fused_ab_error(_FUSED_RUN1)
    prompt = prompts.hls_flash_fused_ab_fix.format(
        reject_reason=err,
        hls_code=_FUSED_RUN1,
        header_code="// kernel.h",
        attempt_history="",
    )
    head = prompt[:900].lower()
    assert "hard reject" in head or "rejected" in head
    assert "load_a" in head
    assert "load_b" in head
    assert "load_a_b" in prompt.lower()
    assert "```cpp" in prompt
