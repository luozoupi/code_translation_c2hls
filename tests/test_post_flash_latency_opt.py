from __future__ import annotations
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import post_flash_latency_opt as plo
import c2hls  # requires project .venv (openai); run via .venv/bin/python -m pytest


class FakeOrch:
    """Minimal orchestrator stub: returns canned LLM replies in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.part = "xcu280-fsvh2892-2L-e"
        self.clock_ns = 4.0
        self.gpt_model = "fake-model"

    def _call_llm(self, messages):
        idx = self.calls
        self.calls += 1
        return self.replies[idx]


def _fake_load_benchmark_inputs_factory():
    fake_inputs = {
        "meta": {
            "part": "xcu280-fsvh2892-2L-e",
            "clock_ns": 4.0,
            "translated_hls_top": "kernel_atax",
        },
        "header_code": "// header\n",
        "header_name": "atax.h",
        "testbench_code": "// tb\n",
        "extra_files": [],
        "benchmark_context": "- atax kernel",
    }
    return lambda bench_dir: fake_inputs


def _fake_run_synth_csim_cosim_factory(reports):
    state = {"i": 0}

    def _fake(*args, **kwargs):
        idx = state["i"]
        state["i"] += 1
        report = reports[idx]
        return {
            "synth": {"success": True, "report": report},
            "csim": {"passed": True},
            "cosim": None,
        }

    return _fake


def test_enabled_default_off():
    os.environ.pop("C2HLS_POST_FLASH_LATENCY_OPT", None)
    assert plo.latency_opt_enabled() is False


def test_enabled_on():
    os.environ["C2HLS_POST_FLASH_LATENCY_OPT"] = "1"
    assert plo.latency_opt_enabled() is True
    del os.environ["C2HLS_POST_FLASH_LATENCY_OPT"]


def test_rounds_defaults():
    os.environ.pop("C2HLS_LATENCY_OPT_ROUNDS", None)
    os.environ.pop("C2HLS_LATENCY_OPT_REPAIR_ROUNDS", None)
    assert plo.latency_round_limit() == 3
    assert plo.repair_round_limit() == 3


def test_under_budget_u280():
    part = "xcu280-fsvh2892-2L-e"
    report = {"lut": 1000, "dsp": 10, "ff": 2000, "bram": 0, "uram": 0}
    assert plo.under_device_budget(report, part, budget_pct=100.0) is True
    report_over = {"lut": 1000, "dsp": 20000, "ff": 2000, "bram": 0, "uram": 0}
    assert plo.under_device_budget(report_over, part, budget_pct=100.0) is False


def test_should_accept_candidate():
    best = {"latency_cycles": 1000, "report": {"lut": 1, "dsp": 1, "ff": 1, "bram": 0, "uram": 0}}
    cand_ok = {"latency_cycles": 800, "report": {"lut": 1, "dsp": 1, "ff": 1, "bram": 0, "uram": 0}}
    assert plo.should_accept(cand_ok, best, part="xcu280-fsvh2892-2L-e") is True
    cand_worse = {"latency_cycles": 1200, "report": cand_ok["report"]}
    assert plo.should_accept(cand_worse, best, part="xcu280-fsvh2892-2L-e") is False
    cand_over = {"latency_cycles": 100, "report": {"lut": 1, "dsp": 99999, "ff": 1, "bram": 0, "uram": 0}}
    assert plo.should_accept(cand_over, best, part="xcu280-fsvh2892-2L-e") is False


def test_should_accept_legalization_when_no_best():
    cand = {"latency_cycles": 5000, "report": {"lut": 1, "dsp": 1, "ff": 1, "bram": 0, "uram": 0}}
    assert plo.should_accept(cand, best=None, part="xcu280-fsvh2892-2L-e") is True


def test_render_analysis_pack_includes_scopes_and_templates():
    report = {
        "latency_cycles": 10000,
        "lut": 100, "dsp": 8, "ff": 200, "bram": 0, "uram": 0,
        "feedback": {
            "summary": {
                "loop_count": 2, "pipelined_loops": 1, "bottleneck_count": 1,
                "scopes_with_negative_slack": 0, "high_severity_bottlenecks": 1,
            },
            "scopes": [
                {
                    "scope_id": "k/outer", "kind": "loop", "latency_cycles": 9000,
                    "trip_count": 64, "interval": 64, "pipelined": "no", "pipeline_ii": None,
                    "dsp": 0, "lut": 50,
                },
                {
                    "scope_id": "k/inner", "kind": "loop", "latency_cycles": 128,
                    "trip_count": 64, "interval": 2, "pipelined": "yes", "pipeline_ii": 2,
                    "dsp": 8, "lut": 40,
                },
            ],
            "bottlenecks": [
                {
                    "kind": "ii_target_miss", "severity": "high",
                    "scope_id": "k/inner", "evidence": "II=2 target=1",
                },
                {
                    "kind": "non_pipelined_hot_loop", "severity": "high",
                    "scope_id": "k/outer", "evidence": "not pipelined",
                },
            ],
        },
    }
    text = plo.render_latency_analysis_pack(report, part="xcu280-fsvh2892-2L-e")
    assert "10000" in text
    assert "k/outer" in text and "k/inner" in text
    assert "ii_target_miss" in text
    assert "pipeline" in text.lower()
    assert "guided" in text.lower()


def test_template_actions_resource_pressure():
    report = {
        "latency_cycles": 10,
        "lut": 1200000, "dsp": 1, "ff": 1, "bram": 0, "uram": 0,
        "feedback": {"scopes": [], "bottlenecks": [], "summary": {}},
    }
    text = plo.render_latency_analysis_pack(report, part="xcu280-fsvh2892-2L-e")
    assert "80" in text or "pressure" in text.lower() or "budget" in text.lower()


def test_plan_and_modify_prompts_structure():
    docs = plo.prompt_text_for_docs()
    assert "analyst" in docs["plan_system"].lower() or "plan" in docs["plan_system"].lower()
    assert "target" in docs["plan_user"].lower() or "action" in docs["plan_user"].lower()
    assert "kernel" in docs["modify_system"].lower()
    assert "kernel" in docs["modify_user"].lower()
    assert "error" in docs["repair_user"].lower() or "fix" in docs["repair_user"].lower()


def test_plan_mentions_scope():
    plan = "**targets:** k/inner\n**actions:** pipeline k/inner"
    assert plo.plan_mentions_scope(plan, ["k/inner", "k/outer"]) is True
    assert plo.plan_mentions_scope("generic plan", ["k/inner"]) is False
    assert plo.plan_mentions_scope("", ["k/inner"]) is False


def test_artifact_paths():
    cell = Path("/tmp/cell")
    flash = plo.artifact_paths(cell, "atax", "flash_final")
    df = plo.artifact_paths(cell, "atax", "dataflow")
    assert flash["kernel"].name == "atax_latency_opt.cpp"
    assert flash["trajectory"].name == "atax_latency_opt_trajectory.json"
    assert flash["manifest"].name == "atax_latency_opt_manifest.json"
    assert df["kernel"].name == "atax_dataflow_latency_opt.cpp"
    assert df["trajectory"].name == "atax_dataflow_latency_opt_trajectory.json"


def test_trajectory_helpers():
    seed = {
        "latency_cycles": 10000,
        "resources": {"lut": 100, "dsp": 8},
        "under_budget": True,
        "validated": True,
    }
    traj = plo.new_trajectory(
        bench="atax",
        source_role="flash_final",
        part="xcu280-fsvh2892-2L-e",
        budget_pct=100.0,
        N=3,
        R=3,
        seed=seed,
    )
    assert traj["schema"] == plo.TRAJECTORY_SCHEMA
    assert traj["benchmark"] == "atax"
    assert traj["source_role"] == "flash_final"
    assert traj["seed"] == seed
    assert traj["best_so_far"] is None
    assert traj["rounds"] == []
    assert traj["final"] is None

    plo.append_round_event(
        traj,
        {
            "round": 1,
            "phase": "plan",
            "repair_index": None,
            "plan_summary": "pipeline k/inner",
            "validated": False,
            "latency_cycles": None,
            "resources": {},
            "under_budget": None,
            "decision": None,
            "reason": None,
        },
    )
    assert len(traj["rounds"]) == 1
    assert traj["rounds"][0]["phase"] == "plan"
    assert traj["rounds"][0]["plan_summary"] == "pipeline k/inner"

    plo.append_round_event(
        traj,
        {
            "round": 1,
            "phase": "optimize",
            "validated": True,
            "latency_cycles": 8500,
            "resources": {"lut": 110, "dsp": 8},
            "under_budget": True,
            "decision": "accept",
            "reason": "lower latency under budget",
        },
    )
    assert len(traj["rounds"]) == 2
    assert traj["rounds"][1]["phase"] == "optimize"
    assert traj["rounds"][1]["decision"] == "accept"

    plo.set_best_so_far(
        traj,
        round_idx=1,
        latency_cycles=8500,
        resources={"lut": 110, "dsp": 8},
        kernel_sha256="deadbeef",
    )
    assert traj["best_so_far"]["round"] == 1
    assert traj["best_so_far"]["latency_cycles"] == 8500
    assert traj["best_so_far"]["kernel_sha256"] == "deadbeef"

    plo.finalize_trajectory(
        traj,
        success=True,
        final_latency=8500,
        seed_latency=10000,
    )
    assert traj["final"]["latency_cycles"] == 8500
    assert abs(traj["final"]["speedup_vs_seed"] - (10000 / 8500)) < 1e-9
    assert traj["final"]["under_budget"] is True
    assert traj["final"]["success"] is True


def test_resolve_prefers_latency_opt(tmp_path):
    from post_flash_mem_parallel import resolve_selected_kernel
    bench = "atax"
    (tmp_path / f"{bench}_selected.cpp").write_text("selected", encoding="utf-8")
    (tmp_path / f"{bench}_latency_opt.cpp").write_text("opt", encoding="utf-8")
    (tmp_path / f"{bench}_latency_opt_result.json").write_text(
        '{"success": true}', encoding="utf-8"
    )
    path, role = resolve_selected_kernel(tmp_path, bench)
    assert path.name == f"{bench}_latency_opt.cpp"
    assert role == "latency_opt"


def test_resolve_include_post_passes_false_returns_selected(tmp_path):
    from post_flash_mem_parallel import resolve_selected_kernel
    bench = "atax"
    (tmp_path / f"{bench}_selected.cpp").write_text("selected", encoding="utf-8")
    (tmp_path / f"{bench}_latency_opt.cpp").write_text("opt", encoding="utf-8")
    (tmp_path / f"{bench}_latency_opt_result.json").write_text(
        '{"success": true}', encoding="utf-8"
    )
    path, role = resolve_selected_kernel(tmp_path, bench, include_post_passes=False)
    assert path.name == f"{bench}_selected.cpp"
    assert role == "selected"


def test_resolve_latency_source_kernel_flash_final_ignores_own_output(tmp_path):
    bench = "atax"
    (tmp_path / f"{bench}_selected.cpp").write_text("selected", encoding="utf-8")
    (tmp_path / f"{bench}_latency_opt.cpp").write_text("opt", encoding="utf-8")
    (tmp_path / f"{bench}_latency_opt_result.json").write_text(
        '{"success": true}', encoding="utf-8"
    )
    path, role, _report = plo.resolve_latency_source_kernel(tmp_path, bench, "flash_final")
    assert path.name == f"{bench}_selected.cpp"
    assert role == "selected"


def test_resolve_latency_source_kernel_flash_final_prefers_pragma_opt(tmp_path):
    bench = "atax"
    (tmp_path / f"{bench}_selected.cpp").write_text("selected", encoding="utf-8")
    (tmp_path / f"{bench}_pragma_opt.cpp").write_text("pragma", encoding="utf-8")
    (tmp_path / f"{bench}_pragma_opt_result.json").write_text(
        '{"success": true}', encoding="utf-8"
    )
    path, role, _report = plo.resolve_latency_source_kernel(tmp_path, bench, "flash_final")
    assert path.name == f"{bench}_pragma_opt.cpp"
    assert role == "pragma_opt"


def test_resolve_latency_source_kernel_dataflow_prefers_dataflow_pragma_opt(tmp_path):
    bench = "atax"
    (tmp_path / f"{bench}_dataflow.cpp").write_text("dataflow", encoding="utf-8")
    (tmp_path / f"{bench}_dataflow_result.json").write_text(
        '{"success": true}', encoding="utf-8"
    )
    (tmp_path / f"{bench}_dataflow_pragma_opt.cpp").write_text("dataflow-pragma", encoding="utf-8")
    (tmp_path / f"{bench}_dataflow_pragma_opt_result.json").write_text(
        '{"success": true}', encoding="utf-8"
    )
    path, role, _report = plo.resolve_latency_source_kernel(tmp_path, bench, "dataflow")
    assert path.name == f"{bench}_dataflow_pragma_opt.cpp"
    assert role == "dataflow_pragma_opt"


def test_resolve_latency_source_kernel_dataflow_falls_back_to_dataflow_cpp(tmp_path):
    bench = "atax"
    (tmp_path / f"{bench}_dataflow.cpp").write_text("dataflow", encoding="utf-8")
    (tmp_path / f"{bench}_dataflow_result.json").write_text(
        '{"success": true}', encoding="utf-8"
    )
    path, role, _report = plo.resolve_latency_source_kernel(tmp_path, bench, "dataflow")
    assert path.name == f"{bench}_dataflow.cpp"
    assert role == "dataflow"


def test_run_latency_opt_for_cell_accepts_lower_latency(tmp_path, monkeypatch):
    bench = "atax"
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()

    seed_code = 'extern "C" void kernel_atax(int *x) {\n  x[0] = x[0] + 1;\n}\n'
    (cell_dir / f"{bench}_selected.cpp").write_text(seed_code)

    monkeypatch.setattr(c2hls, "_load_benchmark_inputs", _fake_load_benchmark_inputs_factory())
    monkeypatch.setattr(c2hls, "compile_check_cpp", lambda *a, **k: (True, ""))

    reports = [
        {"latency_cycles": 1000, "lut": 10, "dsp": 1, "ff": 10, "bram": 0, "uram": 0},
        {"latency_cycles": 500, "lut": 10, "dsp": 1, "ff": 10, "bram": 0, "uram": 0},
    ]
    monkeypatch.setattr(c2hls, "_run_synth_csim_cosim", _fake_run_synth_csim_cosim_factory(reports))

    monkeypatch.setenv("C2HLS_LATENCY_OPT_ROUNDS", "1")
    monkeypatch.setenv("C2HLS_LATENCY_OPT_REPAIR_ROUNDS", "1")
    monkeypatch.setenv("C2HLS_POST_FLASH_LATENCY_OPT", "1")

    plan_reply = (
        "**targets:** k/loop\n"
        "**actions:** PIPELINE II=1 on k/loop\n"
        "**avoid:** unroll\n"
        "**risk:** low"
    )
    modify_reply = (
        '```kernel\n'
        'extern "C" void kernel_atax(int *x) {\n'
        '#pragma HLS PIPELINE II=1\n'
        '  x[0] = x[0] + 1;\n'
        '}\n'
        '```'
    )
    orch = FakeOrch([plan_reply, modify_reply])

    outcome = plo.run_latency_opt_for_cell(
        bench=bench,
        bench_dir=bench_dir,
        cell_dir=cell_dir,
        orchestrator=orch,
        source_role="flash_final",
        skip_existing=True,
    )

    assert outcome.success is True
    assert outcome.result["latency_cycles"] == 500
    assert orch.calls == 2  # one plan call, one modify call — no repairs needed

    paths = plo.artifact_paths(cell_dir, bench, "flash_final")
    assert paths["kernel"].is_file()
    assert "PIPELINE" in paths["kernel"].read_text(encoding="utf-8")

    traj = json.loads(paths["trajectory"].read_text())
    phases = [r["phase"] for r in traj["rounds"]]
    assert "plan" in phases
    assert "optimize" in phases
    accept_events = [r for r in traj["rounds"] if r.get("decision") == "accept"]
    assert len(accept_events) == 1
    assert traj["final"]["latency_cycles"] == 500
    assert traj["final"]["success"] is True


def test_run_latency_opt_for_cell_rejects_over_budget_candidate(tmp_path, monkeypatch):
    bench = "atax"
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()

    seed_code = 'extern "C" void kernel_atax(int *x) {\n  x[0] = x[0] + 1;\n}\n'
    (cell_dir / f"{bench}_selected.cpp").write_text(seed_code)

    monkeypatch.setattr(c2hls, "_load_benchmark_inputs", _fake_load_benchmark_inputs_factory())
    monkeypatch.setattr(c2hls, "compile_check_cpp", lambda *a, **k: (True, ""))

    reports = [
        {"latency_cycles": 1000, "lut": 10, "dsp": 1, "ff": 10, "bram": 0, "uram": 0},
        {"latency_cycles": 100, "lut": 10, "dsp": 99999, "ff": 10, "bram": 0, "uram": 0},
    ]
    monkeypatch.setattr(c2hls, "_run_synth_csim_cosim", _fake_run_synth_csim_cosim_factory(reports))

    monkeypatch.setenv("C2HLS_LATENCY_OPT_ROUNDS", "1")
    monkeypatch.setenv("C2HLS_LATENCY_OPT_REPAIR_ROUNDS", "1")
    monkeypatch.setenv("C2HLS_POST_FLASH_LATENCY_OPT", "1")

    plan_reply = (
        "**targets:** k/loop\n"
        "**actions:** UNROLL factor=64 on k/loop\n"
        "**avoid:** none\n"
        "**risk:** large DSP growth"
    )
    modify_reply = (
        '```kernel\n'
        'extern "C" void kernel_atax(int *x) {\n'
        '#pragma HLS UNROLL factor=64\n'
        '  x[0] = x[0] + 1;\n'
        '}\n'
        '```'
    )
    orch = FakeOrch([plan_reply, modify_reply])

    outcome = plo.run_latency_opt_for_cell(
        bench=bench,
        bench_dir=bench_dir,
        cell_dir=cell_dir,
        orchestrator=orch,
        source_role="flash_final",
        skip_existing=True,
    )

    assert outcome.success is True
    assert outcome.result["latency_cycles"] == 1000  # best stays at seed; candidate rejected

    paths = plo.artifact_paths(cell_dir, bench, "flash_final")
    assert paths["kernel"].read_text(encoding="utf-8") == seed_code
    traj = json.loads(paths["trajectory"].read_text())
    reject_events = [r for r in traj["rounds"] if r.get("decision") == "reject_budget"]
    assert len(reject_events) == 1
    assert traj["best_so_far"]["latency_cycles"] == 1000
    assert traj["final"]["latency_cycles"] == 1000


def test_promote_latency_opt_flash_updates_selected(tmp_path):
    bench = "atax"
    cell = tmp_path
    code = 'extern "C" void kernel_atax() {}\n'
    report = {"latency_cycles": 1828, "lut": 1, "dsp": 1, "ff": 1, "bram": 0, "uram": 0}
    (cell / f"{bench}_selected.cpp").write_text("// old selected\n", encoding="utf-8")
    (cell / f"{bench}_selected_report.json").write_text(
        '{"latency_cycles": 5000}\n', encoding="utf-8"
    )
    (cell / f"{bench}_flow_manifest.json").write_text(
        json.dumps({
            "schema": "flash_flow_manifest_v1",
            "selected_from": "flash_opt",
            "latency_cycles": {"phase_b": 10000, "flash_opt": 5000, "selected": 5000},
            "files": {"selected": f"{bench}_selected.cpp"},
        })
        + "\n",
        encoding="utf-8",
    )
    (cell / f"{bench}_latency_opt.cpp").write_text(code, encoding="utf-8")
    result_payload = {"success": True, "latency_cycles": 1828}
    promotion = plo.promote_latency_opt_as_selected(
        cell_dir=cell,
        bench=bench,
        source_role="flash_final",
        code=code,
        report=report,
        result_payload=result_payload,
    )
    assert promotion["selected_stage"] == "latency_opt"
    assert (cell / f"{bench}_selected.cpp").read_text(encoding="utf-8") == code
    sel_rep = json.loads((cell / f"{bench}_selected_report.json").read_text())
    assert sel_rep["latency_cycles"] == 1828
    manifest = json.loads((cell / f"{bench}_flow_manifest.json").read_text())
    assert manifest["selected_from"] == "latency_opt"
    assert manifest["latency_cycles"]["selected"] == 1828


def test_promote_latency_opt_dataflow_updates_result(tmp_path):
    bench = "atax"
    cell = tmp_path
    code = 'extern "C" void kernel_atax() {}\n'
    report = {"latency_cycles": 1828, "lut": 1, "dsp": 1, "ff": 1, "bram": 0, "uram": 0}
    (cell / f"{bench}_dataflow_result.json").write_text(
        json.dumps({
            "success": True,
            "latency_cycles": 5633,
            "synth_report": {"latency_cycles": 5633},
        })
        + "\n",
        encoding="utf-8",
    )
    (cell / f"{bench}_dataflow_latency_opt.cpp").write_text(code, encoding="utf-8")
    result_payload = {"success": True, "latency_cycles": 1828}
    promotion = plo.promote_latency_opt_as_selected(
        cell_dir=cell,
        bench=bench,
        source_role="dataflow",
        code=code,
        report=report,
        result_payload=result_payload,
    )
    assert promotion["selected_stage"] == "dataflow_latency_opt"
    data = json.loads((cell / f"{bench}_dataflow_result.json").read_text())
    assert data["latency_cycles"] == 1828
    assert data["selected_stage"] == "dataflow_latency_opt"
    assert data["selected_kernel"] == f"{bench}_dataflow_latency_opt.cpp"
    assert data["pre_latency_opt"]["latency_cycles"] == 5633


if __name__ == "__main__":
    import tempfile

    test_enabled_default_off()
    test_enabled_on()
    test_rounds_defaults()
    test_under_budget_u280()
    test_should_accept_candidate()
    test_should_accept_legalization_when_no_best()
    test_render_analysis_pack_includes_scopes_and_templates()
    test_template_actions_resource_pressure()
    test_plan_and_modify_prompts_structure()
    test_plan_mentions_scope()
    test_artifact_paths()
    test_trajectory_helpers()
    with tempfile.TemporaryDirectory() as tmp:
        test_promote_latency_opt_flash_updates_selected(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_promote_latency_opt_dataflow_updates_result(Path(tmp))
    print("test_post_flash_latency_opt: ok (run via pytest for mocked round-loop tests)")
