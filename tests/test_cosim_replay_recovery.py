import copy
import hashlib
import json
from pathlib import Path
import sys
import types
import pytest
from scripts import recover_sigmetrics_timeout_replay as recovery


def dump(p, value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value))


@pytest.fixture
def cohort(tmp_path):
    raw=tmp_path/'raw'; path=raw/'measurements'/'done.json'
    dump(path, {'validation_id':'done','kernel_sha256':'kernel','cosim':{'status':'timeout'}})
    state={'tasks':[{'id':'done','kind':'historical','status':'timeout','measurement_sha256':recovery.sha(path)},
                    {'id':'interrupted','kind':'historical','status':'running'},
                    {'id':'new','kind':'historical','status':'pending'}]}
    manifest={'raw_root':str(raw),'rows':[{'validation_id':k,'kernel_sha256':'kernel'} for k in ['done','interrupted','new']]}
    return tmp_path,state,manifest,path


def test_completed_timeout_is_preserved_and_only_missing_jobs_recover(cohort):
    root,state,manifest,path=cohort
    before=path.read_bytes()
    finished,pending,hashes=recovery.inventory(root,state,manifest)
    assert [t['id'] for t in finished]==['done']
    assert [t['id'] for t in pending]==['interrupted','new']
    assert path.read_bytes()==before
    assert hashes[str(path)]==recovery.sha(path)
    assert state['tasks'][1]['status']=='running'


@pytest.mark.parametrize('damage',['missing','changed_hash','wrong_kernel','wrong_identity','extra','wrong_status'])
def test_corrupt_evidence_blocks_recovery(cohort,damage):
    root,state,manifest,path=cohort
    if damage=='missing': path.unlink()
    elif damage=='changed_hash': path.write_text(path.read_text()+'\n')
    elif damage=='extra': dump(path.parent/'unregistered.json',{})
    elif damage=='wrong_status': state['tasks'][0]['status']='pass'
    else:
        value=json.loads(path.read_text())
        value['kernel_sha256' if damage=='wrong_kernel' else 'validation_id']='incorrect'
        dump(path,value);state['tasks'][0]['measurement_sha256']=recovery.sha(path)
    with pytest.raises(ValueError): recovery.inventory(root,state,manifest)


def test_finished_measurement_is_adopted_without_reexecution(cohort):
    root,state,manifest,path=cohort
    state['tasks'][0].update(status='running')
    finished,pending,_=recovery.inventory(root,state,manifest)
    assert finished[0]['status']=='timeout'
    assert len(pending)==2


def test_diagnostic_selection_is_separate_and_deterministic():
    tasks=[{'id':'a','status':'pass','estimate_cycles':1},
           {'id':'c','status':'timeout','estimate_cycles':100},
           {'id':'b','status':'timeout','estimate_cycles':100},
           {'id':'d','status':'fail','estimate_cycles':2}]
    assert recovery.select_diagnostics(tasks)==[tasks[2]]


def test_kernel_pass_requires_measured_cycles_and_correct_identity(tmp_path):
    path=tmp_path/'result.json';task={'id':'a','kernel_sha256':'k'}
    exe={'returncode':0,'process_timed_out':False}
    for value in [None,0,-1]:
        dump(path,{'validation_id':'a','kernel_sha256':'k','cosim':{'status':'pass','passed':True,'kernel_runtime_cycles':value}})
        with pytest.raises(ValueError): recovery.classify_measurement(task,path,exe)
    dump(path,{'validation_id':'a','kernel_sha256':'k','cosim':{'status':'pass','passed':True,'kernel_runtime_cycles':10}})
    assert recovery.classify_measurement(task,path,exe)['status']=='pass'
    assert recovery.classify_measurement(task,path,{'returncode':124,'process_timed_out':True})['status']=='execution_error'


def test_dead_controller_guard_rejects_live_identity(monkeypatch):
    monkeypatch.setattr(recovery,'process_info',lambda _: {'start_ticks':'42'})
    with pytest.raises(RuntimeError): recovery.require_dead({'pid':1,'pid_start_ticks':'42'})
    recovery.require_dead({'pid':1,'pid_start_ticks':'43'})


def test_descendant_rejects_cycles_and_unrelated_processes():
    assert recovery.descendant(3,1,{3:(2,'xsim'),2:(1,'worker')})
    assert not recovery.descendant(3,1,{3:(2,'xsim'),2:(3,'worker')})


def test_summary_keeps_six_hour_diagnostics_out_of_original_denominator():
    manifest={'workers':{'recovery':2,'diagnostic':1},'legacy_historical_counts':{'pass':33,'timeout':26},'legacy_parent_counts':{'pass':1,'timeout':3}}
    state={'status':'running','updated_at':'now','tasks':[{'cohort':'recovery','status':'pass'}, {'cohort':'diagnostic','status':'pass'}]}
    s=recovery.summarize(state,manifest)
    assert s['combined_three_hour_replay']=={'pass':34,'timeout':26}
    assert s['cohorts']['diagnostic']=={'pass':1}


def test_real_subprocess_writes_measurement_and_observation_sidecar(tmp_path,monkeypatch):
    class Monitor:
        def __init__(self,scratch_dirs,jsonl_path,root_pids=()): self.path=jsonl_path
        def sample(self): self.path.write_text('{"observational":true}\n')
        def finish(self,outcome=None): return {'outcome':outcome}
    monkeypatch.setitem(sys.modules,'cosim_runtime_monitor',types.SimpleNamespace(CosimRuntimeMonitor=Monitor))
    source=tmp_path/'source';worker=source/'scripts/run_historical_cosim.py';worker.parent.mkdir(parents=True)
    worker.write_text('''import json,sys\nfrom pathlib import Path\na=sys.argv\nroot=Path(a[a.index('--output')+1]);ident=a[a.index('--validation-id')+1]\np=root/'raw'/'measurements'/(ident+'.json');p.parent.mkdir(parents=True,exist_ok=True)\np.write_text(json.dumps({'validation_id':ident,'kernel_sha256':'k','cosim':{'status':'pass','passed':True,'kernel_runtime_cycles':123}}))\n''')
    output=tmp_path/'output';output.mkdir()
    hist=output/(tmp_path.name+'_hist')
    task={'id':hashlib.sha256(str(tmp_path).encode()).hexdigest(),'kernel_sha256':'k','historical_output':str(hist),'timeout_seconds':10}
    manifest={'historical_source':str(source),'outer_worker_overhead_seconds':5,'observation_interval_seconds':0.01}
    result=recovery.run_task(output,task,manifest)
    assert result['status']=='pass'
    assert (Path(result['telemetry_path'])/'summary.json').exists()
    with pytest.raises(FileExistsError): recovery.run_task(output,task,manifest)
