"""Overlap ablation: PE array + fuse + ping-pong DATAFLOW, not the stream FIFO pack."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import post_flash_overlap as pfo
from skill_library import _coerce_skill_entry


REQUIRED_SKILL_IDS = (
    "hls-overlap-fuse-crow-init-drain",
    "hls-overlap-pingpong-dataflow-on-pe-array",
    "avoid-overlap-stream-pe-fifo-pack",
)


def test_overlap_skills_file_is_valid():
    path = pfo.resolve_overlap_skills_path()
    assert path.is_file(), path
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = [entry["id"] for entry in data["skills"]]
    for sid in REQUIRED_SKILL_IDS:
        assert sid in ids, sid
    for entry in data["skills"]:
        skill = _coerce_skill_entry(entry)
        assert skill is not None, entry.get("id")
        assert skill.template.strip()
        assert skill.required_steps
        assert skill.guards


def test_overlap_skill_count_and_prompt():
    block, meta = pfo.build_overlap_skills_prompt_block()
    assert meta["skill_count"] == 3
    assert set(REQUIRED_SKILL_IDS) <= set(meta["skill_ids"])
    low = block.lower()
    assert "ping-pong" in low
    assert "mm_pe" in low
    assert "fuse" in low


def test_rejects_stream_pe_fifo_pack():
    streamish = (
        "#define PE_NUM 16\n"
        "#pragma HLS DATAFLOW\n"
        "hls::stream<ap_uint<128> > fifo_B[17];\n"
        + "\n".join(
            f"mm_pe(fifo_A[{i}], fifo_B[{i}], fifo_B[{i+1}], fifo_C[{i}]);"
            for i in range(16)
        )
    )
    assert pfo.is_stream_pe_fifo_pack(streamish) is True
    pingpong = (
        "#define PE 16\n#define SIMD 4\n"
        "for (int i0 = 0; i0 < I; i0 += PE) {\n"
        "#pragma HLS DATAFLOW\n"
        "  load_A_tile(A, Atile, i0);\n"
        "  compute_tile(Atile, local_B, Ctile);\n"
        "  store_C_tile(C, Ctile, i0);\n"
        "}\n"
    )
    assert pfo.is_stream_pe_fifo_pack(pingpong) is False
    ok_report = {
        "dsp": 352,
        "feedback": {"scopes": [{"name": "compute_j", "pipeline_ii": 1}]},
    }
    assert pfo.architecture_ok(streamish, ok_report, "autosa_mm") is False
    assert pfo.architecture_ok(pingpong, ok_report, "autosa_mm") is True


def test_artifact_tag_is_overlap():
    paths = pfo.artifact_paths(Path("/tmp/cell"), "autosa_mm")
    assert paths["kernel"].name == "autosa_mm_overlap.cpp"
    assert paths["result"].name == "autosa_mm_overlap_result.json"


def test_overlap_launcher_supports_mm_only():
    text = (
        Path(__file__).resolve().parents[1]
        / "scripts/pc2/start_autosa_pe_overlap.sh"
    ).read_text(encoding="utf-8")
    assert "--mm-only" in text
    assert "20260830_mmflow" in text or "C2HLS_MM_MATRIX_ROOT" in text
