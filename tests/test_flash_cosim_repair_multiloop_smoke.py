"""Smoke tests for multi-loop cosim repair (offline, no LLM/Vitis)."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.pc2.flash_cosim_repair_lib import (  # noqa: E402
    DEFAULT_MAX_REPAIR_LOOPS,
    RepairJob,
    build_followup_diagnose_prompt,
    build_followup_repair_prompt,
    job_dir,
    loop_dir,
    repair_loop_out_dir,
    run_repair_batch,
    run_repair_job,
    validation_error_text,
)


def _make_job(tmp_path: Path, **overrides) -> RepairJob:
    final_cpp = tmp_path / "source.cpp"
    final_cpp.write_text('extern "C" void workload() {}\n', encoding="utf-8")
    defaults = {
        "index": 0,
        "job_id": "test_cell",
        "repair_variant": "noskills",
        "source_cell_id": "test_cell",
        "source_artifact_basename": "flash_test",
        "source_artifact_key": "test",
        "bench": "hlsfactory_2mm",
        "setup_tag": "test_setup",
        "cell_dir": str(tmp_path),
        "final_cpp": str(final_cpp),
        "cosim_error": "ERROR: cosim mismatch",
        "source_cosim_work_dir": "/tmp/cosim_work",
        "matrix_family": "test",
        "variant": "test",
        "mode": "flash",
    }
    defaults.update(overrides)
    return RepairJob(**defaults)


def _fake_llm_factory():
    call_n = {"n": 0}

    def fake_llm(messages):
        call_n["n"] += 1
        if call_n["n"] % 2 == 1:
            return f"diagnosis turn {call_n['n']}"
        return '```cpp\nextern "C" void workload() { /* fixed */ }\n```'

    return fake_llm


def test_default_max_loops_is_one():
    sig = inspect.signature(run_repair_job)
    assert sig.parameters["max_loops"].default == DEFAULT_MAX_REPAIR_LOOPS
    assert DEFAULT_MAX_REPAIR_LOOPS == 1


def test_loop_dir_and_repair_loop_out_dir(tmp_path):
    root = tmp_path / "session"
    assert repair_loop_out_dir(root, "cell_a", 1, max_loops=1) == job_dir(root, "cell_a")
    assert repair_loop_out_dir(root, "cell_a", 2, max_loops=10) == loop_dir(root, "cell_a", 2)


def test_validation_error_text_prefers_cosim():
    validation = {
        "csim": {"passed": False, "error": "csim compile failed"},
        "synth": {"passed": False, "error": "synth failed"},
        "cosim": {"passed": False, "error": "SIGSEGV"},
    }
    assert validation_error_text(validation).startswith("cosim: SIGSEGV")


def test_followup_prompts_include_loop_metadata():
    prompt = build_followup_diagnose_prompt(
        bench="hlsfactory_correlation",
        top_function="workload",
        hls_code="void workload() {}",
        error_text="cosim: still bad",
        repair_variant="noskills",
        loop_index=3,
        max_loops=10,
        prior_diagnosis="prior root cause",
    )
    assert "Repair attempt: 3 of 10" in prompt
    assert "prior root cause" in prompt

    repair = build_followup_repair_prompt(
        bench="hlsfactory_correlation",
        top_function="workload",
        hls_code="void workload() {}",
        error_text="cosim: still bad",
        diagnosis="new plan",
        repair_variant="noskills",
        loop_index=4,
        max_loops=10,
    )
    assert "Repair attempt: 4 of 10" in repair
    assert "new plan" in repair


def _assert_multiloop_stops_early_on_cosim_pass(tmp_path: Path) -> None:
    with patch("scripts.pc2.flash_cosim_repair_lib.call_llm") as mock_llm, patch(
        "scripts.pc2.flash_cosim_repair_lib.validate_repaired_code"
    ) as mock_validate:
        mock_llm.side_effect = _fake_llm_factory()
        validate_n = {"n": 0}

        def fake_validate(bench, hls_code, repair_tag=""):
            validate_n["n"] += 1
            passed = validate_n["n"] >= 2
            return {
                "csim": {"success": passed, "passed": passed, "error": ""},
                "synth": {"success": True, "error": "", "latency_cycles": 42},
                "cosim": {
                    "success": passed,
                    "passed": passed,
                    "error": "" if passed else "cosim: mismatch",
                },
            }

        mock_validate.side_effect = fake_validate

        job = _make_job(tmp_path)
        session_root = tmp_path / "all_avoids_new"
        result = run_repair_job(job, session_root, max_loops=10, force=True)

        assert result["cosim_passed"] is True
        assert result["max_loops"] == 10
        assert result["loops_used"] == 2
        assert len(result["loops"]) == 2
        assert (session_root / "jobs" / job.job_id / "loops" / "loop_01" / "repaired.cpp").exists()
        assert (session_root / "jobs" / job.job_id / "loops" / "loop_02" / "repaired.cpp").exists()
        assert not (session_root / "jobs" / job.job_id / "loops" / "loop_03").exists()


def _assert_multiloop_runs_all_loops_when_still_failing(tmp_path: Path) -> None:
    with patch("scripts.pc2.flash_cosim_repair_lib.call_llm") as mock_llm, patch(
        "scripts.pc2.flash_cosim_repair_lib.validate_repaired_code"
    ) as mock_validate:
        mock_llm.side_effect = _fake_llm_factory()
        mock_validate.return_value = {
            "csim": {"success": False, "passed": False, "error": "csim fail"},
            "synth": {"success": True, "error": "", "latency_cycles": 1},
            "cosim": {"success": False, "passed": False, "error": "cosim: still failing"},
        }

        job = _make_job(tmp_path)
        session_root = tmp_path / "noskills_old"
        result = run_repair_job(job, session_root, max_loops=3, force=True)

        assert result["cosim_passed"] is False
        assert result["loops_used"] == 3
        assert len(result["loops"]) == 3
        assert mock_llm.call_count == 6


def _assert_single_loop_result_schema_unchanged(tmp_path: Path) -> None:
    with patch("scripts.pc2.flash_cosim_repair_lib.call_llm") as mock_llm, patch(
        "scripts.pc2.flash_cosim_repair_lib.validate_repaired_code"
    ) as mock_validate:
        mock_llm.side_effect = _fake_llm_factory()
        mock_validate.return_value = {
            "csim": {"success": True, "passed": True, "error": ""},
            "synth": {"success": True, "error": "", "latency_cycles": 9},
            "cosim": {"success": True, "passed": True, "error": ""},
        }

        job = _make_job(tmp_path)
        session_root = tmp_path / "single"
        result = run_repair_job(job, session_root, max_loops=1, force=True)

        assert result["cosim_passed"] is True
        assert "loops" not in result
        assert "max_loops" not in result
        assert (session_root / "jobs" / job.job_id / "diagnose_prompt.txt").exists()
        assert not (session_root / "jobs" / job.job_id / "loops").exists()


def _assert_run_repair_batch_dry_run_includes_max_loops(tmp_path: Path) -> None:
    with patch("scripts.pc2.flash_cosim_repair_lib.discover_failures_for_artifact") as mock_discover:
        mock_discover.return_value = [
            {
                "source_cell_id": "cell_x",
                "source_artifact_basename": "flash_all_new_skills_avoids_global_20260621_020847",
                "source_artifact_key": "all_avoids_new",
                "bench": "hlsfactory_2mm",
                "setup_tag": "tag",
                "cell_dir": "/tmp/cell",
                "final_cpp": "/tmp/final.cpp",
                "cosim_error": "fail",
                "source_cosim_work_dir": "/tmp/work",
                "matrix_family": "m",
                "variant": "v",
                "mode": "flash",
            }
        ]
        run_root = tmp_path / "repair_run"
        cosim_root = tmp_path / "cosim"
        cosim_root.mkdir()

        summary = run_repair_batch(
            "all_avoids_new",
            run_root,
            cosim_root,
            max_loops=10,
            dry_run=True,
        )
        assert summary["max_loops"] == 10
        assert summary["dry_run"] is True
        assert summary["total"] == 1


def test_multiloop_stops_early_on_cosim_pass(tmp_path):
    _assert_multiloop_stops_early_on_cosim_pass(tmp_path)


def test_multiloop_runs_all_loops_when_still_failing(tmp_path):
    _assert_multiloop_runs_all_loops_when_still_failing(tmp_path)


def test_single_loop_result_schema_unchanged(tmp_path):
    _assert_single_loop_result_schema_unchanged(tmp_path)


def test_run_repair_batch_dry_run_includes_max_loops(tmp_path):
    _assert_run_repair_batch_dry_run_includes_max_loops(tmp_path)


def test_multiloop_start_script_exists():
    script = REPO_ROOT / "scripts" / "pc2" / "start_cosim_repair_multiloop_session.sh"
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "max-loops" in text
    assert "cosim_repair_ml10_" in text
    assert "flash_cosim_repair_multiloop" in text


def main() -> int:
    test_default_max_loops_is_one()
    test_loop_dir_and_repair_loop_out_dir(Path("/tmp/c2hls_multiloop_test_paths"))
    test_validation_error_text_prefers_cosim()
    test_followup_prompts_include_loop_metadata()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        _assert_multiloop_stops_early_on_cosim_pass(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        _assert_multiloop_runs_all_loops_when_still_failing(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        _assert_single_loop_result_schema_unchanged(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        _assert_run_repair_batch_dry_run_includes_max_loops(Path(tmp))
    test_multiloop_start_script_exists()
    print("test_flash_cosim_repair_multiloop_smoke: all ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
