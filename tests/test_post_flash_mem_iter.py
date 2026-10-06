"""Generic mem-iter: bounded prompt, worst-latency pick, gold ban."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _row(
    it: int,
    *,
    status: str = "ok",
    lat: int | None = 1000,
    worst: int | None = None,
    dsp: int = 300,
    csim: str = "pass",
    kernel: str = "",
    error: str = "",
    note: str = "",
    ii: int | None = 1,
) -> dict:
    worst_v = worst if worst is not None else lat
    body = kernel or f"void autosa_mm() {{ int x{it} = {it}; }}\n"
    return {
        "iter": it,
        "status": status,
        "latency_cycles": lat,
        "latency_cycles_worst": worst_v,
        "dsp": dsp,
        "bram": 8,
        "ff": 100,
        "lut": 200,
        "uram": 0,
        "csim": csim,
        "kernel": body,
        "error": error,
        "note": note,
        "scopes": [
            {
                "scope_id": f"loop_{it}",
                "kind": "loop",
                "pipeline_ii": ii,
                "latency_cycles": worst_v,
                "trip_count": 64,
                "dsp": dsp,
                "bram": 8,
            }
        ],
        "ii_violations": [f"ii: miss on loop_{it}"] if status == "ok" and ii and ii > 1 else [],
    }


def test_default_rounds_and_token_budgets():
    from post_flash_mem_iter import (
        mem_iter_context_tokens,
        mem_iter_max_tokens,
        mem_iter_prompt_tokens,
        mem_iter_rounds,
    )

    assert mem_iter_rounds() == 50
    assert mem_iter_prompt_tokens() == 32768
    assert mem_iter_max_tokens() == 65536
    assert mem_iter_context_tokens() == 131072


def test_select_winner_is_min_worst_latency_not_last():
    from post_flash_mem_iter import select_winner

    rows = [
        _row(0, lat=5000, worst=5000, note="seed"),
        _row(1, lat=2000, worst=8000),
        _row(2, status="compile_fail", lat=None, worst=None, csim="", kernel="bad", error="nope"),
        _row(3, lat=3000, worst=3100),
        _row(4, lat=1000, worst=9000, csim="fail", status="ok"),
        _row(5, lat=4000, worst=4000),
    ]
    winner = select_winner(rows)
    assert winner is not None
    assert winner["iter"] == 3
    assert winner["latency_cycles_worst"] == 3100


def test_prompt_has_full_table_and_at_most_three_kernels():
    from post_flash_mem_iter import build_mem_prompt, count_fenced_kernels, pick_kernel_bodies

    rows = [
        _row(0, lat=9000, worst=9000, note="seed"),
        _row(1, lat=4000, worst=4200, note="ok"),
        _row(2, status="synth_fail", lat=None, worst=None, csim="", kernel="broken()", error="csynth"),
        _row(3, lat=3500, worst=3600),
        _row(4, lat=8000, worst=8100),
        _row(5, lat=3300, worst=3400),
        _row(6, status="compile_fail", lat=None, worst=None, csim="", kernel="fail6", error="undeclared"),
        _row(7, lat=5000, worst=5100),
    ]
    bodies = pick_kernel_bodies(rows)
    labels = [label for label, _ in bodies]
    assert "best-so-far" in labels
    assert "last-accepted" in labels
    assert "last-failed" in labels
    assert len(bodies) <= 3
    prompt = build_mem_prompt(rows)
    assert count_fenced_kernels(prompt) <= 3
    assert "QoR table (all prior iters)" in prompt
    for it in range(8):
        assert f"{it} " in prompt
    low = prompt.lower()
    assert "gt_code" not in low
    assert "gold hls" not in low
    assert "goal code" not in low
    assert "ground-truth" not in low


def test_prompt_gold_ban():
    import pytest
    from post_flash_mem_iter import assert_no_gold, build_mem_prompt

    rows = [_row(0, kernel="void autosa_mm() { /* ok */ }\n")]
    text = build_mem_prompt(rows)
    assert_no_gold(text)
    with pytest.raises(ValueError, match="gold"):
        assert_no_gold("please copy the gold HLS kernel")


def test_over_budget_drops_deltas_keeps_three_bodies():
    from post_flash_mem_iter import build_mem_prompt, count_fenced_kernels

    huge = "int buf[] = {" + ",".join(["1"] * 200) + "};\nvoid autosa_mm() {}\n"
    rows = [_row(i, lat=8000 + i * 10, worst=8000 + i * 10, kernel=huge + f"// iter {i}\n") for i in range(8)]
    rows[1]["latency_cycles"] = 100
    rows[1]["latency_cycles_worst"] = 100
    rows[2]["status"] = "compile_fail"
    rows[2]["csim"] = ""
    rows[2]["latency_cycles"] = None
    rows[2]["latency_cycles_worst"] = None
    prompt = build_mem_prompt(rows, prompt_token_budget=900)
    assert count_fenced_kernels(prompt) == 3
    assert "```kernel" in prompt
    assert "QoR table (all prior iters)" in prompt
    assert "best-so-far" in prompt


def test_dse_v2_runner_imports_c2hls_orchestrator():
    text = (REPO / "scripts/pc2/run_post_flash_dse_v2.py").read_text(encoding="utf-8")
    assert "from c2hls import C2HLSOrchestrator" in text
    assert "HLSOrchestrator()" not in text
    assert "C2HLSOrchestrator(" in text


def test_call_mem_llm_records_timeout_instead_of_raising(monkeypatch):
    from post_flash_mem_iter import call_mem_llm

    monkeypatch.setenv("C2HLS_LLM_TIMEOUT_RETRIES", "2")
    monkeypatch.setenv("C2HLS_LLM_RETRY_BACKOFF_S", "0")
    monkeypatch.setenv("C2HLS_LLM_LOCK", "0")

    class Boom:
        n = 0

        def _call_llm(self, messages, max_tokens=None):
            Boom.n += 1
            raise TimeoutError("Request timed out.")

    reply, err = call_mem_llm(Boom(), [{"role": "user", "content": "x"}], 128)
    assert reply == ""
    assert "timed out" in err.lower()
    assert Boom.n == 2


def test_call_mem_llm_retries_timeout_then_returns_reply(monkeypatch):
    from post_flash_mem_iter import call_mem_llm

    monkeypatch.setenv("C2HLS_LLM_TIMEOUT_RETRIES", "4")
    monkeypatch.setenv("C2HLS_LLM_RETRY_BACKOFF_S", "0")
    monkeypatch.setenv("C2HLS_LLM_LOCK", "0")

    class Flaky:
        n = 0

        def _call_llm(self, messages, max_tokens=None):
            Flaky.n += 1
            if Flaky.n < 3:
                raise TimeoutError("Request timed out.")
            return "```kernel\nvoid autosa_mm() {}\n```"

    reply, err = call_mem_llm(Flaky(), [{"role": "user", "content": "x"}], 128)
    assert err == ""
    assert "autosa_mm" in reply
    assert Flaky.n == 3


def test_resolve_mem_seed_uses_parent_family(tmp_path):
    from post_flash_mem_iter import parent_family_from_cell, resolve_mem_seed

    (tmp_path / "autosa_mm_selected.cpp").write_text("selected", encoding="utf-8")
    (tmp_path / "autosa_mm_selected_report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "autosa_mm_stream.cpp").write_text("stream", encoding="utf-8")
    (tmp_path / "autosa_mm_stream_report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "autosa_mm_flash_opt.cpp").write_text("flash", encoding="utf-8")
    (tmp_path / "autosa_mm_flash_opt_report.json").write_text("{}", encoding="utf-8")
    cpp, _rpt = resolve_mem_seed(tmp_path, parent_family="stream")
    assert cpp.name == "autosa_mm_stream.cpp"
    cpp, _rpt = resolve_mem_seed(tmp_path, parent_family="flash")
    assert cpp.name == "autosa_mm_flash_opt.cpp"
    assert parent_family_from_cell({"stream": 1, "dse": "v2"}) == "stream"
    assert parent_family_from_cell({"enf": 1, "dse": "v1"}) == "enf"
    assert parent_family_from_cell({"skill": "one_shot"}) == "oneshot"
