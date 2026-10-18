import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import analyze_qor_review_ab as analysis
import freeze_qor_ab_seeds as seeds_module
import run_qor_review_ab as runner
from evaluation_repro import FINGERPRINT_SCHEMA, build_run_fingerprint, sha256_json
from phase_b_manifest import canonical_json_sha256, text_sha256, file_sha256


def test_seed_recovery_uses_measured_hash_not_first_reply():
    code = "void f() { int x = 1; }"
    report = {"latency_cycles": 20}
    event = {"label":"[Phase B]", "report_sha256":canonical_json_sha256(report),
             "code_sha256":text_sha256(code), "synthesis_status":"passed", "correctness_status":"passed",
             "timing_met":True, "resource_fit":True}
    result = {"baseline_report":report, "baseline_csim":{"passed":True,"ran":True},
              "synthesis_evaluations":{"events":[event]}}
    history = {"event_history":[{"role":"assistant","content":"```cpp\nvoid broken() {}\n```"},
                                {"role":"assistant","content":"```cpp\n"+code+"\n```"}]}
    recovered, _ = seeds_module.recover_baseline(result, history)
    assert recovered == code
    event["code_sha256"] = "incorrect"
    with pytest.raises(ValueError, match="no saved reply"):
        seeds_module.recover_baseline(result, history)
    result["baseline_csim"]["ran"] = False
    with pytest.raises(ValueError, match="executed passing"):
        seeds_module.recover_baseline(result, history)


def registration():
    return json.loads((REPO / "artifacts/experiment_matrix/deepseek_flash_reviewer_ab27_20260925.json").read_text())


def test_full_matrix_is_paired_and_interleaved():
    config = registration()
    cells = runner.cells_for(config)
    assert len(cells) == len({c["cell_id"] for c in cells}) == 324
    assert [c["cell_id"] for c in cells[:4]] == config["smoke_cell_ids"]
    pairs = {}
    for c in cells:
        pairs.setdefault(c["pair_id"], []).append(c)
    assert len(pairs) == 162
    assert all({c["arm"] for c in group} == {"review_off","review_on"} for group in pairs.values())
    assert {tuple(c["arm"] for c in group) for group in pairs.values()} == {("review_off","review_on"),("review_on","review_off")}


def test_arm_environment_preserves_budget_and_model(tmp_path, monkeypatch):
    config = registration()
    key = tmp_path / "key.txt"
    key.write_text("test-not-a-secret")
    config["key_file"] = str(key)
    monkeypatch.setenv("C2HLS_QOR_DESIGN_SWEEP", "1")
    monkeypatch.setenv("C2HLS_LLM_SEED", "999")
    manifest = {"skill_snapshot_sha256":"test"}
    cells = runner.cells_for(config)
    a,b = [runner.cell_environment(config, manifest, c) for c in cells[:2]]
    assert a["C2HLS_QOR_PROFILE_REVIEW"] == "0" and b["C2HLS_QOR_PROFILE_REVIEW"] == "1"
    for env in (a,b):
        assert env["C2HLS_QOR_DESIGN_SWEEP"] == "0"
        assert env["C2HLS_LLM_CANDIDATE_BUDGET"] == env["C2HLS_SYNTHESIS_EVAL_BUDGET"] == "5"
        assert env["C2HLS_SWEEP_ATTEMPTS_PER_CANDIDATE"] == "2"
        assert env["C2HLS_MODEL"] == "deepseek-flash"
        assert env["C2HLS_LLM_SEED"] == "42"
    assert a["C2HLS_PHASE_B_SEED_MANIFEST"] == b["C2HLS_PHASE_B_SEED_MANIFEST"]
    assert a["C2HLS_VITIS_USER_HOME"] != b["C2HLS_VITIS_USER_HOME"]


def fixture_result(tmp_path, on=False):
    config = {"model":"deepseek-flash", "review_version":"profile_skill_tradeoff_v1",
              "phase_b_manifest":str(tmp_path / "seeds.json")}
    code="void f() {}"
    report={"latency_cycles":100,"latency_cycles_worst":100,"dsp":10}
    csim={"ran":True,"passed":True}
    seed={"code_sha256":text_sha256(code),"csynth_report_sha256":canonical_json_sha256(report),
          "csim_sha256":canonical_json_sha256(csim),"input_c_sha256":"input","header_sha256":"header",
          "csynth_report":report}
    seeds={"entries":{"tiny":seed}}
    Path(config["phase_b_manifest"]).write_text(json.dumps(seeds))
    cell={"cell_id":"tiny_flash_r1_" + ("review_on" if on else "review_off"), "benchmark":"tiny",
          "strategy":"flash","arm":"review_on" if on else "review_off","repeat":1,"pair_id":"tiny_flash_r1", "status":"running",
          "result_path":str(tmp_path / "tiny_multistep_results.json")}
    payload={"search":{"qor_profile_review":on,"sample_id":cell["cell_id"]},
             "budgets":{"llm_candidate_budget":"5","candidate_budget":"5"}}
    result={"success":True,"frozen_phase_b":{**seed,"manifest_sha256":file_sha256(Path(config["phase_b_manifest"]))},
            "baseline_report":report,"baseline_csim":csim,"final_report":report,"csim":csim,"hls_code":code,
            "candidate_feasibility":{"feasible":True},"reference_isolation_audit":{"passed":True},
            "skill_snapshot_integrity":{"unchanged":True},"llm_usage":{"calls":1,"events":[{"model":"deepseek-flash"}]},
            "synthesis_evaluations":{"count":1,"events":[]},
            "qor_design_sweep":{"enabled":False,"attempted":False},
            "run_fingerprint":{"schema_version":FINGERPRINT_SCHEMA,"payload":payload,"sha256":sha256_json(payload)}}
    Path(cell["result_path"]).write_text(json.dumps(result))
    (tmp_path / "tiny_history.json").write_text(json.dumps({"event_history":[{"role":"user","content":"HARDWARE/PERFORMANCE DESIGN REVIEW:" if on else "generic feedback"}]}))
    return config,seeds,cell,result


def test_validator_labels_measured_baseline_fallback(tmp_path):
    config,seeds,cell,_=fixture_result(tmp_path)
    row=analysis.analyze_cell(cell,config,seeds)
    assert row["audit_ok"] and row["feasible"] and row["cycles"]==100
    assert row["baseline_retained"] and row["feasible_generated_candidates"]==0


@pytest.mark.parametrize("design", [{"enabled":True}, {"attempted":True}, {"candidates":[{"id":"unexpected"}]}])
def test_validator_rejects_real_parameter_search_artifacts(tmp_path, design):
    config,seeds,cell,result=fixture_result(tmp_path)
    result["qor_design_sweep"]=design
    Path(cell["result_path"]).write_text(json.dumps(result))
    row=analysis.analyze_cell(cell,config,seeds)
    assert not row["audit_ok"]
    assert "parameter-search artifact unexpectedly present" in row["issues"]


@pytest.mark.parametrize("change,expected",[
    ("seed","frozen baseline identity mismatch"),
    ("search","parameter search unexpectedly executed"),
    ("budget","five-call"),
    ("model","observed models"),
    ("cosim","COSIM"),
    ("review","reviewer evidence leaked"),
])
def test_validator_rejects_confounded_results(tmp_path,change,expected):
    config,seeds,cell,result=fixture_result(tmp_path)
    if change=="seed": result["frozen_phase_b"]["code_sha256"]="wrong"
    elif change=="search": result["qor_synthesis_evaluation_count"]=1
    elif change=="budget": result["llm_usage"]["calls"]=6
    elif change=="model": result["llm_usage"]["events"]=[{"model":"other"}]
    elif change=="cosim": result["selected_winner_cosim_count"]=1
    else: result["qor_profile_reviews"]=[{"review_version":"profile_skill_tradeoff_v1"}]
    Path(cell["result_path"]).write_text(json.dumps(result))
    row=analysis.analyze_cell(cell,config,seeds)
    assert not row["audit_ok"] and row["cycles"] is None
    assert any(expected in v for v in row["issues"])


def test_unpaired_invalid_results_never_get_imputed_cycles():
    off={"pair_id":"p","benchmark":"tiny","strategy":"flash","repeat":1,"arm":"review_off","feasible":True,"cycles":100,"seed_code_sha256":"x"}
    on={**off,"arm":"review_on","feasible":False,"cycles":None}
    pair=analysis.pair_rows([off,on])[0]
    assert not pair["valid_pair"] and pair["speedup_on_over_off"] is None
    on.update(feasible=True,cycles=50)
    assert analysis.pair_rows([off,on])[0]["speedup_on_over_off"]==2
    on["seed_code_sha256"]="different"
    assert not analysis.pair_rows([off,on])[0]["valid_pair"]


def test_seed_and_sample_identity_change_resume_fingerprint(tmp_path):
    manifest=tmp_path/'seed.json';manifest.write_text('{}')
    args=dict(repo=tmp_path,benchmark_dir=tmp_path,benchmark='tiny',model_id='deepseek-flash',
              model_label='deepseek_flash',skill_mode='matched',steps=[],profile='test')
    env={'C2HLS_PHASE_B_SEED_MANIFEST':str(manifest),'C2HLS_SAMPLE_ID':'r1'}
    first=build_run_fingerprint(**args,environ=env)
    env['C2HLS_SAMPLE_ID']='r2'
    second=build_run_fingerprint(**args,environ=env)
    assert first['sha256']!=second['sha256']
    manifest.write_text('{"changed":true}')
    assert build_run_fingerprint(**args,environ=env)['sha256']!=second['sha256']
