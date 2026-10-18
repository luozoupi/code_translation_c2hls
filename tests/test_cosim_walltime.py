"""Separate simulated hardware latency, Vitis phase time, and invocation time."""
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

import cosim_walltime as wall
import hls_eval


def run_python(tmp_path, source, phase_timeout=.15, invocation_timeout=3):
    return wall.run_guarded_command(
        [sys.executable, "-c", source], invocation_timeout=invocation_timeout,
        phase_timeout=phase_timeout, marker_path=tmp_path / "phase.tsv",
        log_path=tmp_path / "output.log", poll_seconds=.01,
        cleanup_grace_seconds=.03, cleanup_kill_seconds=.5,
    )


def phase_script(tmp_path, before=0, phase=.02, tail=""):
    marker = str(tmp_path / "phase.tsv")
    return (
        "import time; from pathlib import Path; "
        f"p=Path({marker!r}); time.sleep({before!r}); "
        "p.write_text('start\\t'+str(time.time()*1000)+'\\n'); "
        f"time.sleep({phase!r}); "
        "p.write_text(p.read_text()+'end\\t'+str(time.time()*1000)+'\\n'); "
        + tail
    )


@pytest.mark.parametrize("raw", ["0", "-1", "inf", "nan", "invalid"])
def test_invalid_opt_in_fails_before_launch(raw):
    with pytest.raises(ValueError, match="positive finite"):
        wall.configured_timeout({wall.ENV_NAME: raw})


def test_unset_optional_and_positive_float():
    assert wall.configured_timeout({}) is None
    assert wall.configured_timeout({wall.ENV_NAME: "0.125"}) == .125


def test_pre_phase_synthesis_time_does_not_consume_phase_cap(tmp_path):
    result = run_python(tmp_path, phase_script(tmp_path, before=.3, phase=.02), phase_timeout=.2)
    assert result["classification"] == "completed"
    assert result["invocation_wall_seconds"] >= .3
    assert .01 <= result["cosim_wall_seconds"] < .2
    assert result["phase_markers_complete"]
    assert not result["cosim_wall_seconds_censored"]


def test_missing_phase_uses_only_whole_invocation_watchdog(tmp_path):
    result = run_python(tmp_path, "import time; time.sleep(5)",
                        phase_timeout=.04, invocation_timeout=.25)
    assert result["timed_out"]
    assert result["timeout_phase"] == "vitis_invocation"
    assert result["classification"] == "invocation_wall_time_limit"
    assert result["cosim_wall_seconds"] is None
    assert result["invocation_wall_seconds"] >= .25


def test_phase_timeout_cleans_detached_descendant_and_preserves_peer(tmp_path):
    ready = tmp_path / "detached.pid"
    child = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(20)"
    )
    source = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}],start_new_session=True); "
        + phase_script(tmp_path, before=.15, phase=20)
    )
    peer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
    try:
        result = run_python(tmp_path, source)
        assert result["timeout_phase"] == "cosim_design"
        assert result["classification"] == "cosim_wall_time_limit"
        assert result["cosim_wall_seconds"] >= .15
        assert result["cosim_wall_seconds_censored"]
        assert not result["owned_process_cleanup"]["remaining_owned_pids"]
        pid = int(ready.read_text())
        assert wall.process_info(pid) is None or wall.process_info(pid)["state"] == "Z"
        assert any(x["pid"] == pid and x["signal"] == signal.SIGKILL
                   for x in result["owned_process_cleanup"]["signaled"])
        assert peer.poll() is None
    finally:
        peer.terminate()
        peer.wait(timeout=3)


def test_pid_reuse_identity_is_not_signaled(monkeypatch):
    sent = []
    monkeypatch.setattr(wall.os, "kill", lambda *args: sent.append(args))
    if hasattr(signal, "pidfd_send_signal"):
        monkeypatch.setattr(wall.signal, "pidfd_send_signal", lambda *args: sent.append(args))
    info = wall.process_info(os.getpid())
    assert not wall.signal_identity(os.getpid(), str(int(info["start_ticks"]) + 1), signal.SIGTERM)
    assert sent == []


def test_full_output_on_disk_bounded_returned_tail(tmp_path):
    result = run_python(tmp_path, phase_script(tmp_path, tail="print('x'*400000)"))
    assert result["classification"] == "completed"
    assert Path(result["log_path"]).stat().st_size > 400000
    assert len(result["log"].encode()) <= 131072


def test_stale_marker_refused(tmp_path):
    (tmp_path / "phase.tsv").write_text("start\t1\n")
    with pytest.raises(FileExistsError, match="stale"):
        run_python(tmp_path, "raise AssertionError('must not execute')")


def test_malformed_or_reversed_markers_not_completed(tmp_path):
    marker = tmp_path / "phase.tsv"
    marker.write_text("start\tnan\nend\t1000\n")
    assert wall.read_markers(marker) == {}
    marker.write_text("start\t2000\nend\t1000\n")
    assert wall.read_markers(marker) == {"start": 2}


def test_tcl_marker_wraps_only_cosim_and_flushes(tmp_path):
    script = wall.tcl_phase_command("cosim_design -trace_level none", tmp_path / "phase.tsv")
    assert script.index('"start') < script.index("cosim_design") < script.index('"end')
    assert script.count("flush $c2hls_phase_marker") == 2
    assert "csynth_design" not in script


def test_marker_script_executes_in_tcl(tmp_path):
    tclsh = shutil.which("tclsh")
    if not tclsh:
        pytest.skip("Tcl interpreter is not installed")
    marker = tmp_path / "directory with spaces" / "phase.tsv"
    marker.parent.mkdir()
    script = wall.tcl_phase_command("after 25", marker)
    result = subprocess.run([tclsh], input=script, text=True, capture_output=True, timeout=3)
    assert result.returncode == 0 and not result.stderr
    times = wall.read_markers(marker)
    assert times["end"] - times["start"] >= .02


def fake_materialize(*args, **kwargs):
    return {"src_file": "kernel.cpp", "tb_file": "tb.cpp", "hdr_file": "", "extra_files": []}


def test_default_cosim_keeps_original_runner_without_markers(tmp_path, monkeypatch):
    monkeypatch.delenv(wall.ENV_NAME, raising=False)
    with patch.object(hls_eval, "_materialize_inputs", side_effect=fake_materialize), \
         patch.object(hls_eval, "_run_vitis_cmd", return_value=("cosim done with 0 errors", False)) as runner, \
         patch.object(hls_eval, "_parse_lat_rpt_cycles", return_value=12):
        result = hls_eval.run_cosim("kernel", "tb", work_dir=str(tmp_path))
    runner.assert_called_once()
    assert result["passed"]
    assert "c2hls_phase_marker" not in (tmp_path / "run_cosim.tcl").read_text()


@pytest.mark.parametrize("classification,complete,timed_out,returncode", [
    ("cosim_wall_time_limit", False, True, -15),
    ("phase_marker_missing", False, False, 0),
    ("tool_error", True, False, 1),
    ("process_cleanup_incomplete", True, False, 0),
])
def test_guarded_cosim_cannot_accept_printed_pass_without_complete_run(
    tmp_path, monkeypatch, classification, complete, timed_out, returncode,
):
    monkeypatch.setenv(wall.ENV_NAME, ".25")
    guarded = {"log": "cosim done with 0 errors", "timed_out": timed_out,
               "classification": classification, "phase_markers_complete": complete,
               "timeout_phase": "cosim_design" if timed_out else None,
               "returncode": returncode}
    with patch.object(hls_eval, "_materialize_inputs", side_effect=fake_materialize), \
         patch.object(hls_eval, "_vitis_shell_exports", return_value="true"), \
         patch.object(hls_eval, "configure_temp_env", return_value=tmp_path), \
         patch.object(wall, "run_guarded_command", return_value=guarded), \
         patch.object(hls_eval, "_parse_lat_rpt_cycles", return_value=12):
        result = hls_eval.run_cosim("kernel", "tb", work_dir=str(tmp_path))
    assert not result["passed"] and not result["success"]
    script = (tmp_path / "run_cosim.tcl").read_text()
    assert script.index("csynth_design") < script.index('"start') < script.index("cosim_design")


def test_summary_keeps_timing_and_resource_timeout_distinct():
    from c2hls import _summarize_test_result
    result = {
        "success": False, "passed": False, "timed_out": True,
        "error": "Cosim phase timed out after 3s of wall time",
        "classification": "cosim_wall_time_limit", "timeout_phase": "cosim_design",
        "invocation_wall_seconds": 9, "cosim_wall_seconds": 3.1,
        "cosim_wall_seconds_censored": True, "wall_timeout_seconds": 3,
        "timing_scope": wall.TIMING_SCOPE, "log_path": "/private/run.log",
    }
    summary = _summarize_test_result(result, True)
    assert summary["status"] == "timeout"
    for key, value in result.items():
        assert summary[key] == value


def dump(values):
    return '==BEGIN DUMP_ARRAYS==\nbegin dump: out\n' + values + '\nend dump: out\n==END DUMP_ARRAYS==\n'


def run_guarded_fixture(tmp_path, monkeypatch, text, golden, **scan_kwargs):
    path = tmp_path / 'complete.log'
    path.write_text(text)
    scanned = wall.scan_validation_log(path, **scan_kwargs)
    guarded = {'log': wall._tail(path), 'timed_out': False, 'classification': 'completed',
               'phase_markers_complete': True, 'returncode': 0, **scanned}
    monkeypatch.setenv(wall.ENV_NAME, '3')
    with patch.object(hls_eval, '_materialize_inputs', side_effect=fake_materialize), \
         patch.object(hls_eval, '_vitis_shell_exports', return_value='true'), \
         patch.object(hls_eval, 'configure_temp_env', return_value=tmp_path), \
         patch.object(wall, 'run_guarded_command', return_value=guarded), \
         patch.object(hls_eval, '_parse_lat_rpt_cycles', return_value=123):
        return hls_eval.run_cosim('kernel', 'tb', work_dir=str(tmp_path), golden_output_text=golden)


def test_large_complete_c_and_rtl_dumps_are_validated_beyond_tail(tmp_path, monkeypatch):
    # More than 128 KiB per dump, similar to larger PolyBench output arrays.
    values = ('1.000000 ' * 20 + '\n') * 1600
    golden = dump(values)
    text = golden + 'Starting C post checking\n' + golden + 'cosim done with 0 errors\n'
    result = run_guarded_fixture(tmp_path, monkeypatch, text, golden)
    assert len(text) > 500000
    assert result['passed'] and result['success']
    assert result['correctness']['dump_count'] == 2
    assert result['validation_log_scan']['complete']
    assert len(result['log'].encode()) <= 131072
    assert 'validation_log' not in result


def test_earlier_bad_c_dump_cannot_be_hidden_by_warnings_and_good_rtl(tmp_path, monkeypatch):
    golden, bad = dump('1 2 3'), dump('1 999 3')
    text = bad + ('WARNING: simulator activity\n' * 20000)
    text += 'Starting C post checking\n' + golden + 'cosim done with 0 errors\n'
    result = run_guarded_fixture(tmp_path, monkeypatch, text, golden)
    assert not result['passed'] and not result['success']
    assert result['correctness']['reason'] == 'golden_mismatch'
    assert result['correctness']['dump_count'] == 2


def test_early_tool_error_cannot_be_hidden_by_later_pass(tmp_path, monkeypatch):
    golden = dump('1 2 3')
    text = 'ERROR: an earlier tool phase failed\n' + ('WARNING: progress\n' * 20000)
    text += golden + 'Starting C post checking\n' + golden + 'cosim done with 0 errors\n'
    result = run_guarded_fixture(tmp_path, monkeypatch, text, golden)
    assert result['tool_failure_marker_seen']
    assert result['correctness']['passed']  # Numerical dumps alone cannot hide a tool failure.
    assert not result['passed'] and not result['success']


def test_evidence_limit_fails_closed_without_serializing_partial_good_dump(tmp_path, monkeypatch):
    golden = dump('1 2 3')
    text = golden + 'Starting C post checking\n' + golden + 'cosim done with 0 errors\n'
    result = run_guarded_fixture(tmp_path, monkeypatch, text, golden, max_evidence_bytes=100)
    assert not result['passed'] and not result['success']
    assert not result['validation_log_scan']['complete']
    assert result['correctness']['reason'] == 'incomplete_full_log_validation_evidence'
    assert 'validation_log' not in result


@pytest.mark.parametrize('prefix', ['==BEGIN DUMP_ARRAYS==\n', '==BEGIN\n DUMP_ARRAYS==\n', '==END DUMP_ARRAYS==\n'])
def test_earlier_malformed_outer_markers_cannot_disappear(tmp_path, monkeypatch, prefix):
    golden = dump('1 2 3')
    text = prefix + ('WARNING: progress\n' * 20000)
    text += 'Starting C post checking\n' + golden + 'cosim done with 0 errors\n'
    result = run_guarded_fixture(tmp_path, monkeypatch, text, golden)
    assert not result['validation_log_scan']['complete']
    assert not result['passed'] and not result['success']
