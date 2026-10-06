"""Csim must fail when the AutoSA bench prints mismatches and still exits 0.

Vitis then logs ``CSim done with 0 errors`` / ``csim_design finished successfully``.
The bench text is ``Failed with %d errors!`` or ``Passed!`` (see the n1024
testbench and ``HEAP_TESTBENCH``).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

import hls_eval
from c2hls import _summarize_test_result

# Testbench stdout, then the Vitis lines that previously counted as a pass.
_MISMATCH_LOG = """\
Failed with 1048576 errors!
INFO: [SIM 211-1] CSim done with 0 errors.
INFO: [HLS 200-111] Finished Command csim_design CPU user time: 12 seconds. Elapsed time: 13 seconds.
csim_design finished successfully
"""

_PASS_LOG = """\
Passed!
INFO: [SIM 211-1] CSim done with 0 errors.
INFO: [HLS 200-111] Finished Command csim_design CPU user time: 12 seconds. Elapsed time: 13 seconds.
csim_design finished successfully
"""

# Vitis wording that must not be treated as the bench mismatch line.
_UNRELATED_FAILED_LOG = """\
WARNING: [SIM 211-100] Failed to open optional waveform database
INFO: [SIM 211-1] CSim done with 0 errors.
csim_design finished successfully
Passed!
"""

# The printf format in a dumped source line is not a reported mismatch.
_FORMAT_STRING_LOG = """\
printf("Failed with %d errors!\\n", err);
INFO: [SIM 211-1] CSim done with 0 errors.
csim_design finished successfully
Passed!
"""

# Exit 0 is the Vitis line. No bench ``Passed!`` and no ``Failed with``.
_VITIS_ZERO_ERRORS_NO_TB_LINE = """\
INFO: [SIM 211-1] CSim done with 0 errors.
INFO: [HLS 200-111] Finished Command csim_design CPU user time: 12 seconds. Elapsed time: 13 seconds.
csim_design finished successfully
"""


def _run_csim(monkeypatch, log: str, work_dir: Path) -> dict:
    monkeypatch.setattr(hls_eval, "_run_vitis_cmd", lambda cmd, timeout: (log, False))
    return hls_eval.run_csim(
        "void autosa_mm() {}",
        "int main(){return 0;}\n",
        work_dir=str(work_dir),
    )


def test_mismatch_line_is_csim_failure_when_vitis_reports_zero_errors(tmp_path, monkeypatch):
    result = _run_csim(monkeypatch, _MISMATCH_LOG, tmp_path)
    assert result["passed"] is False
    assert result["success"] is False
    summary = _summarize_test_result(result, True)
    assert summary["passed"] is False
    assert summary["status"] == "failed"
    assert summary["ran"] is True
    assert "1048576" in (summary.get("error") or result.get("error") or "")


def test_passed_bang_with_exit_zero_stays_csim_pass(tmp_path, monkeypatch):
    result = _run_csim(monkeypatch, _PASS_LOG, tmp_path)
    assert result["passed"] is True
    assert result["success"] is True
    summary = _summarize_test_result(result, True)
    assert summary["passed"] is True
    assert summary["status"] == "passed"


def test_vitis_zero_errors_without_bench_line_is_csim_pass(tmp_path, monkeypatch):
    log = _VITIS_ZERO_ERRORS_NO_TB_LINE
    assert "Passed!" not in log
    assert "Failed with" not in log
    assert hls_eval.csim_log_passed(log) is True
    result = _run_csim(monkeypatch, log, tmp_path)
    assert result["passed"] is True
    assert result["success"] is True
    summary = _summarize_test_result(result, True)
    assert summary["passed"] is True
    assert summary["status"] == "passed"


def test_unrelated_vitis_failed_line_does_not_override_a_real_pass(tmp_path, monkeypatch):
    result = _run_csim(monkeypatch, _UNRELATED_FAILED_LOG, tmp_path)
    assert result["passed"] is True
    assert result["success"] is True


def test_printf_format_string_is_not_a_mismatch(tmp_path, monkeypatch):
    result = _run_csim(monkeypatch, _FORMAT_STRING_LOG, tmp_path)
    assert result["passed"] is True
    assert result["success"] is True


def test_package_csim_checker_rejects_mismatch_despite_zero_errors():
    from autosa_dse_hls_validate_lib import _csim_passed

    assert _csim_passed(_MISMATCH_LOG) is False
    assert _csim_passed(_PASS_LOG) is True
    assert _csim_passed(_UNRELATED_FAILED_LOG) is True
    assert _csim_passed(_VITIS_ZERO_ERRORS_NO_TB_LINE) is True
