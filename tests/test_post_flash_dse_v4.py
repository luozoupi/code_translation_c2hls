"""Offline gate for DSE v4. No model call and no Vitis."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import post_flash_dse_v2 as v2
import post_flash_dse_v3 as v3
import post_flash_dse_v4 as v4

FLASH_144M = (
    Path(__file__).resolve().parents[1]
    / "artifacts/pc2/autosa_mm_ijk_sweep_20260923/n1024/cells/"
    "dse2_90_f1_r09_camp/variants/autosa_aav_n_90/autosa_mm/"
    "deepseek-v4-flash__flash__autosa__aav_n_90/dse_v2/pe16_simd32/autosa_mm.cpp"
)
S0 = "S_0(p0 + 128 * c0, 128 * c1 + 32 * c4 + c6, 8 * c2 + c7)"
KERNEL0_DATAFLOW_ERROR = (
    "top function is autosa_mm; kernel0 is absent; DATAFLOW pragma is absent"
)


def _clear_v4(monkeypatch) -> None:
    monkeypatch.delenv("C2HLS_DSE_V4", raising=False)
    monkeypatch.delenv("C2HLS_DSE_V4_CONFIG", raising=False)
    monkeypatch.delenv("C2HLS_DSE_V3", raising=False)
    monkeypatch.delenv("C2HLS_DSE_V3_HARNESS", raising=False)


def _st0() -> dict:
    return v4.load_v4_config("st0_c1")


def _gold_cpp(cfg: dict) -> str:
    return v4.resolve_gold_kernel_path(cfg).read_text(encoding="utf-8")


def _drop_pe_wrappers(code: str, keep: int) -> str:
    start, end = v4._function_span(code, "kernel0")
    assert start >= 0
    body = code[start : end + 1]
    pattern = re.compile(
        r"(?:/\*\s*Module Call\s*\*/\s*)?PE_wrapper\s*\((?:[^;]*?)\)\s*;",
        re.S,
    )
    seen = {"n": 0}

    def repl(match: re.Match[str]) -> str:
        seen["n"] += 1
        return match.group(0) if seen["n"] <= keep else ""

    new_body = pattern.sub(repl, body)
    return code[:start] + new_body + code[end + 1 :]


def test_only_st0_c1_config():
    rows = v4.load_v4_configs()
    assert len(rows) == 30
    assert len({row["id"] for row in rows}) == 30
    assert rows[0]["id"] == "st0_c1"
    for row in rows:
        assert v4.resolve_instruction_path(row).is_file()
        assert v4.resolve_gold_kernel_path(row).is_file()
        assert v4.expected_k_max(row) == 1023
    cfg = _st0()
    assert cfg["space_time"] == 0
    assert cfg["array_part"] == [128, 128, 8]
    assert cfg["latency"] == [1, 32]
    assert cfg["simd"] == [8]
    assert cfg["pe"] == [128]
    assert cfg["candidate_id"] == 1
    assert v4.pe_count(cfg) == 128
    assert v4.expected_k_max(cfg) == 1023
    assert v4.s0_k_expression(cfg).replace(" ", "") == "8*c2+c7"
    assert v4.repair_round_limit() == 12
    assert v4.v4_csynth_kwargs()["allow_compile_jobs"] is False
    assert v4.v4_csynth_kwargs()["cosim"] is False
    assert v4.v4_csynth_kwargs()["part"] == "xcu280-fsvh2892-2L-e"


def test_st0_c1_prompt_has_s0_and_order_not_flash(monkeypatch):
    _clear_v4(monkeypatch)
    prompt = v4.format_v4_initial_user(_st0())
    assert S0 in prompt
    assert "Order to follow" in prompt
    assert "A_IO_L2_in_serialize" in prompt
    assert "autosa_mm_final.cpp" not in prompt
    assert "Do not emit AutoSA kernel0" not in prompt


def test_pe_signature_and_store_path_follow_space_time():
    st0 = v4.expected_pe_signature(v4.load_v4_config("st0_c1"))
    st3 = v4.expected_pe_signature(v4.load_v4_config("st3_c1"))
    st4 = v4.expected_pe_signature(v4.load_v4_config("st4_c9"))
    st4_float = v4.expected_pe_signature(v4.load_v4_config("st4_c10"))
    assert st0.startswith("void PE(int idx,")
    assert "int idy" not in st0
    assert "fifo_C_drain_out" in st0
    assert "fifo_A_out" not in st0
    assert st3.startswith("void PE(int idx, int idy,")
    assert "fifo_A_out" in st3 and "fifo_C_drain_out" in st3
    assert "fifo_C_in" not in st3
    assert "fifo_C_in" in st4 and "fifo_C_out" in st4
    assert "fifo_C_drain_out" not in st4
    assert "hls::stream<float> &fifo_A_in" in st4_float
    for cid in ("st0_c1", "st3_c1", "st4_c9", "st4_c10"):
        cfg = v4.load_v4_config(cid)
        fail = v4.first_failure(v4.read_gold_kernel(cfg), cfg)
        assert fail is None, (cid, None if fail is None else fail.error)
        repair = v4.repair_user(
            "simd_port", config=cfg, kernel_code="void PE();", error="bad"
        )
        assert v4.expected_pe_signature(cfg) in repair
    st4_cfg = v4.load_v4_config("st4_c9")
    start, end = v4._function_span(v4.read_gold_kernel(st4_cfg), "kernel0")
    gold = v4.read_gold_kernel(st4_cfg)
    injected = gold[:end] + "\n  C_drain_IO_L1_out(C);\n" + gold[end:]
    err = v4.check_modules(injected, st4_cfg)
    assert "C_drain_" in err and "expected 0" in err
    k_repair = v4.repair_user(
        "full_k", config=st4_cfg, kernel_code="void PE();", error="bad"
    )
    assert "p1" in k_repair and "idy" in k_repair


def test_csim_bench_calls_kernel0_and_timeouts_fit_one_pass():
    import hls_eval

    prev_csim = hls_eval.CSIM_TIMEOUT
    prev_env = {
        key: os.environ.get(key)
        for key in ("C2HLS_CSIM_TIMEOUT", "C2HLS_SYNTH_TIMEOUT")
    }
    try:
        v4.apply_v4_timeouts()
        assert hls_eval.CSIM_TIMEOUT == 86400
        assert os.environ["C2HLS_CSIM_TIMEOUT"] == "86400"
        assert os.environ["C2HLS_SYNTH_TIMEOUT"] == "86400"
        assert v4.JOB_WALL == "12:00:00"
    finally:
        hls_eval.CSIM_TIMEOUT = prev_csim
        for key, value in prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    shared = (
        Path(__file__).resolve().parents[1]
        / "artifacts/pc2/autosa_mm_ijk_benches/n1024/autosa_mm/testbench.cpp"
    ).read_text(encoding="utf-8")
    assert "autosa_mm(A, B, C)" in shared
    for cid, a_n, marker in (
        ("st0_c1", 8388608, "A_from[(128 * c0 + c3) * 1024 + (8 * c2 + c5)]"),
        ("st3_c1", 16777216, "host_serialize_A"),
        ("st4_c9", 8388608, "A_from[(512 * c0 + 32 * c4 + c5) * 1024 + (32 * c2 + 4 * c3 + c6)]"),
        ("st4_c10", 2097152, "host_serialize_B"),
    ):
        cfg = v4.load_v4_config(cid)
        bench = v4.n1024_kernel0_testbench(cfg)
        assert "void kernel0(A_t16 *A, B_t16 *B, C_t16 *C)" in bench
        assert "kernel0((A_t16 *)packed_A, (B_t16 *)packed_B, (C_t16 *)packed_C)" in bench
        assert "autosa_mm(" not in bench
        assert 'printf("Passed!\\n");' in bench
        assert 'printf("Failed with %d errors!\\n", err);' in bench
        assert f"malloc((size_t){a_n} * sizeof(float))" in bench
        assert marker in bench
        repair = v4.repair_user("csim", config=cfg, kernel_code="void PE();", error="Failed with 1048576 errors!")
        assert "kernel0(A_t16 *A, B_t16 *B, C_t16 *C)" in repair
        assert "host_serialize_A" in repair
        assert "autosa_mm" in repair
    for cfg in v4.load_v4_configs():
        bench = v4.n1024_kernel0_testbench(cfg)
        assert "kernel0((A_t16 *)packed_A" in bench
        assert "autosa_mm(" not in bench


def test_gold_passes_checks_2_through_9(monkeypatch):
    cfg = _st0()
    gold = _gold_cpp(cfg)

    def boom(*_args, **_kwargs):
        raise AssertionError("classify_k_coverage must not be used for v4 full_k")

    monkeypatch.setattr(v3, "classify_k_coverage", boom)
    fail = v4.first_failure(gold, cfg)
    assert fail is None
    checked = v4.read_gold_kernel(cfg)
    assert "typedef ap_uint<512> A_t16" in checked
    for check_id in v4.STRUCTURAL_CHECK_IDS:
        err = v4.STRUCTURAL_CHECKERS[check_id](checked, cfg)
        assert err == "", (check_id, err)


def test_flash_144m_first_failure_is_kernel0_dataflow_no_csynth():
    cfg = _st0()
    flash = FLASH_144M.read_text(encoding="utf-8")
    called = {"csynth": 0}

    def csynth(_code: str, _cfg: dict) -> str:
        called["csynth"] += 1
        return "csynth should not run"

    fail = v4.first_failure(flash, cfg, csynth_fn=csynth)
    assert fail is not None
    assert fail.check_id == "kernel0_dataflow"
    assert fail.error == KERNEL0_DATAFLOW_ERROR
    assert called["csynth"] == 0
    repair = v4.repair_user(
        fail.check_id, config=cfg, kernel_code=flash, error=fail.error
    )
    assert "kernel0(A_t16 *A, B_t16 *B, C_t16 *C)" in repair
    assert "#pragma HLS DATAFLOW" in repair
    assert KERNEL0_DATAFLOW_ERROR in repair
    assert called["csynth"] == 0


def test_pe_count_failure_does_not_call_csynth():
    cfg = _st0()
    gold = _gold_cpp(cfg)
    short = _drop_pe_wrappers(gold, keep=64)
    called = {"csynth": 0, "compile": 0}

    def compile_fn(_code: str, _cfg: dict) -> str:
        called["compile"] += 1
        return "compile should not run"

    def csynth_fn(_code: str, _cfg: dict) -> str:
        called["csynth"] += 1
        return "csynth should not run"

    fail = v4.first_failure(short, cfg, compile_fn=compile_fn, csynth_fn=csynth_fn)
    assert fail is not None
    assert fail.check_id == "pe_count"
    assert "64" in fail.error
    assert "128" in fail.error
    assert called["compile"] == 0
    assert called["csynth"] == 0


def test_v2_unchanged_when_v4_unset(monkeypatch):
    _clear_v4(monkeypatch)
    assert v4.dse_v4_enabled() is False
    assert v3.dse_v3_enabled() is False
    assert v2.dse_v2_system_prompt() == v2._SYSTEM_V2
    assert "Do not emit AutoSA kernel0" not in v2.dse_v2_system_prompt()


def test_v4_wins_over_v3_rules(monkeypatch):
    _clear_v4(monkeypatch)
    monkeypatch.setenv("C2HLS_DSE_V3", "1")
    monkeypatch.setenv("C2HLS_DSE_V4", "1")
    assert v4.dse_v4_enabled() is True
    assert v3.dse_v3_enabled() is False
    assert "Do not emit AutoSA kernel0" not in v2.dse_v2_system_prompt()
    assert "512-bit beats" not in v2.dse_v2_system_prompt()
    prompt = v4.format_v4_initial_user(_st0())
    assert "Do not emit AutoSA kernel0" not in prompt


def test_one_config_dry_run_does_not_call_the_model(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/pc2/run_dse_v4_one.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            "st4_c10",
            "--out-dir",
            str(tmp_path / "run"),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert "config=st4_c10" in proc.stdout
    assert "submit=no" in proc.stdout
    assert not (tmp_path / "run" / "dse_v4_summary.json").exists()


def test_sbatch_writer_emits_thirty_and_does_not_submit(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/pc2/write_dse_v4_sbatch.sh"
    proc = subprocess.run(
        ["bash", str(script), "--out-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert "submitted=no" in proc.stdout
    jobs = sorted(tmp_path.glob("*.sbatch.sh"))
    assert len(jobs) == 30
    joined = "\n".join(path.read_text(encoding="utf-8") for path in jobs)
    assert "C2HLS_DSE_V4=1" in joined
    assert "C2HLS_DSE_V4_REPAIR_ROUNDS=12" in joined
    assert "unset C2HLS_DSE_SOURCE_KERNEL" in joined
    assert "unset C2HLS_DSE_V3" in joined
    assert "--cpus-per-task=8" in joined
    assert "--mem=64G" in joined
    assert "--time=12:00:00" in joined
    assert "C2HLS_CSIM_TIMEOUT=86400" in joined
    assert "C2HLS_SYNTH_TIMEOUT=86400" in joined
    assert "st0_c1" in joined and "st4_c10" in joined
    manifest = (tmp_path / "manifest.txt").read_text(encoding="utf-8")
    assert "submitted=no" in manifest
    for line in script.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert not stripped.startswith("sbatch")
        assert " sbatch " not in f" {stripped} "
    one = subprocess.run(
        [
            "bash",
            str(script),
            "--out-dir",
            str(tmp_path / "one"),
            "--config",
            "st0_c1",
            "--endpoint-url",
            "http://login5:18140/v1",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert "submitted=no" in one.stdout
    lone = (tmp_path / "one" / "st0_c1.sbatch.sh").read_text(encoding="utf-8")
    assert "export OPENAI_BASE_URL=http://login5:18140/v1" in lone
    assert "export CHATHLS_API_BASE=http://login5:18140/v1" in lone
    assert "export C2HLS_MODEL=deepseek-v4-flash" in lone
    assert "api.deepseek.com" not in lone
    refused = subprocess.run(
        [
            "bash",
            str(script),
            "--out-dir",
            str(tmp_path / "bad"),
            "--config",
            "st0_c1",
            "--endpoint-url",
            "https://api.deepseek.com/v1",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert refused.returncode != 0


def test_start_one_dry_run_does_not_launch_proxy_or_job(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/pc2/start_dse_v4_one.sh"
    text = script.read_text(encoding="utf-8")
    assert "start_dedicated_deepseek_proxy.sh" in text
    assert "sbatch --export=ALL" in text
    assert "OPENAI_BASE_URL" in text
    assert "CHATHLS_API_BASE" in text
    proc = subprocess.run(
        [
            "bash",
            str(script),
            "--config",
            "st0_c1",
            "--out-dir",
            str(tmp_path / "st0"),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert "config=st0_c1" in proc.stdout
    assert "model=deepseek-v4-flash" in proc.stdout
    assert "proxy=no" in proc.stdout
    assert "submit=no" in proc.stdout
    assert not (tmp_path / "st0" / "launch.json").exists()
    job = (tmp_path / "st0" / "st0_c1.sbatch.sh").read_text(encoding="utf-8")
    assert "C2HLS_DSE_V4_CONFIG=st0_c1" in job
    assert "OPENAI_BASE_URL=" not in job
