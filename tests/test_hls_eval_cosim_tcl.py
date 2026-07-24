"""Cosim TCL generation must keep linker paths for Fir Apptainer."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hls_eval import (  # noqa: E402
    _config_compile_jobs_tcl,
    _cosim_design_tcl,
    _cosim_tcl_library_path_setup,
    _vitis_jobs,
)


class HlsEvalCosimTclTests(unittest.TestCase):
    def test_default_sets_container_safe_library_path(self) -> None:
        os.environ.pop("C2HLS_COSIM_LIBRARY_PATH", None)
        line = _cosim_tcl_library_path_setup()
        self.assertIn("set ::env(LIBRARY_PATH)", line)
        self.assertIn("/lib/x86_64-linux-gnu", line)
        self.assertNotIn("unset ::env(LIBRARY_PATH)", line)

    def test_empty_override_disables_line(self) -> None:
        os.environ["C2HLS_COSIM_LIBRARY_PATH"] = ""
        self.assertEqual(_cosim_tcl_library_path_setup(), "")

    def test_vitis_jobs_from_slurm_cpus(self) -> None:
        prev_jobs = os.environ.pop("C2HLS_VITIS_JOBS", None)
        prev_slurm = os.environ.get("SLURM_CPUS_PER_TASK")
        try:
            os.environ["SLURM_CPUS_PER_TASK"] = "16"
            self.assertEqual(_vitis_jobs(), 16)
            self.assertEqual(_config_compile_jobs_tcl(), "config_compile -jobs 16\n")
            self.assertEqual(
                _cosim_design_tcl(trace_level=""),
                'cosim_design -argv "-XsimJobs 16"',
            )
        finally:
            if prev_jobs is None:
                os.environ.pop("C2HLS_VITIS_JOBS", None)
            else:
                os.environ["C2HLS_VITIS_JOBS"] = prev_jobs
            if prev_slurm is None:
                os.environ.pop("SLURM_CPUS_PER_TASK", None)
            else:
                os.environ["SLURM_CPUS_PER_TASK"] = prev_slurm

    def test_vitis_jobs_explicit_override(self) -> None:
        prev_jobs = os.environ.get("C2HLS_VITIS_JOBS")
        prev_slurm = os.environ.get("SLURM_CPUS_PER_TASK")
        try:
            os.environ["C2HLS_VITIS_JOBS"] = "8"
            os.environ["SLURM_CPUS_PER_TASK"] = "16"
            self.assertEqual(_vitis_jobs(), 8)
        finally:
            if prev_jobs is None:
                os.environ.pop("C2HLS_VITIS_JOBS", None)
            else:
                os.environ["C2HLS_VITIS_JOBS"] = prev_jobs
            if prev_slurm is None:
                os.environ.pop("SLURM_CPUS_PER_TASK", None)
            else:
                os.environ["SLURM_CPUS_PER_TASK"] = prev_slurm

    def test_vitis_jobs_single_omits_tcl(self) -> None:
        prev_jobs = os.environ.pop("C2HLS_VITIS_JOBS", None)
        prev_slurm = os.environ.pop("SLURM_CPUS_PER_TASK", None)
        try:
            self.assertEqual(_vitis_jobs(), 1)
            self.assertEqual(_config_compile_jobs_tcl(), "")
            self.assertEqual(_cosim_design_tcl(trace_level=""), "cosim_design")
        finally:
            if prev_jobs is not None:
                os.environ["C2HLS_VITIS_JOBS"] = prev_jobs
            if prev_slurm is not None:
                os.environ["SLURM_CPUS_PER_TASK"] = prev_slurm


if __name__ == "__main__":
    unittest.main()
