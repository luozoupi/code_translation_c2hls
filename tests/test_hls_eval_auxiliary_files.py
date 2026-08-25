from __future__ import annotations

import subprocess
from unittest.mock import patch

import hls_eval


def test_vitis_timeout_cleanup_never_uses_unbounded_communicate(tmp_path) -> None:
    class _Pipe:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class _Process:
        pid = 12345

        def __init__(self):
            self.stdout = _Pipe()
            self.communicate_timeouts = []

        def communicate(self, timeout=None):
            self.communicate_timeouts.append(timeout)
            raise subprocess.TimeoutExpired(
                cmd="vitis-run", timeout=timeout, output="partial\n"
            )

        def kill(self):
            return None

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="vitis-run", timeout=timeout)

    process = _Process()
    with (
        patch.object(hls_eval, "configure_temp_env", return_value=tmp_path),
        patch.object(hls_eval, "_vitis_shell_exports", return_value="export A=1"),
        patch.object(hls_eval.subprocess, "Popen", return_value=process),
        patch.object(hls_eval, "_descendant_pids", return_value=set()),
        patch.object(hls_eval.os, "killpg"),
    ):
        output, timed_out = hls_eval._run_vitis_cmd("vitis-run", timeout=1)

    assert timed_out is True
    assert process.communicate_timeouts == [1, 5, 5]
    assert all(timeout is not None for timeout in process.communicate_timeouts)
    assert process.stdout.closed is True
    assert "could not drain" in output


def test_csim_adds_auxiliary_runtime_files_as_testbench_inputs(
    tmp_path,
) -> None:
    work_dir = tmp_path / "csim"
    with patch.object(
        hls_eval,
        "_run_vitis_cmd",
        return_value=("CSim done with 0 errors", False),
    ):
        result = hls_eval.run_csim(
            "void workload() {}",
            "int main() { return 0; }",
            work_dir=str(work_dir),
            extra_files=[
                {"path": "input.data", "content": "1\n"},
                {"path": "nested/check.data", "content": "1\n"},
                {"path": "kernel_description.md", "content": "metadata\n"},
                {"path": "hls_eval_config.toml", "content": "metadata = true\n"},
                {"path": "top.txt", "content": "workload\n"},
            ],
        )

    assert result["passed"] is True
    tcl = (work_dir / "run_csim.tcl").read_text(encoding="utf-8")
    expected = (
        "add_files -tb [list "
        "{testbench.cpp} "
        "{input.data} "
        "{nested/check.data}"
        "]"
    )
    assert expected in tcl
    assert "kernel_description.md" not in tcl
    assert "hls_eval_config.toml" not in tcl
    assert "top.txt" not in tcl


def test_testbench_file_list_is_deduplicated() -> None:
    assert hls_eval._tcl_testbench_files(["a", "a", "b"]).splitlines() == [
        "add_files -tb [list {a} {b}]",
    ]


def test_cosim_uses_relative_auxiliary_runtime_paths(tmp_path) -> None:
    work_dir = tmp_path / "cosim"
    with patch.object(
        hls_eval,
        "_run_vitis_cmd",
        return_value=("C/RTL co-simulation finished: PASS", False),
    ):
        result = hls_eval.run_cosim(
            "void workload() {}",
            "int main() { return 0; }",
            work_dir=str(work_dir),
            extra_files=[
                {"path": "input.data", "content": "1\n"},
                {"path": "kernel_description.md", "content": "metadata\n"},
            ],
        )

    assert result["passed"] is True
    tcl = (work_dir / "run_cosim.tcl").read_text(encoding="utf-8")
    assert (
        "add_files -tb [list {testbench.cpp} {input.data}]"
        in tcl
    )
    assert "kernel_description.md" not in tcl
