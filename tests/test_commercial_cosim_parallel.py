import copy
import importlib.util
import json
from pathlib import Path
import signal
from types import SimpleNamespace
import pytest

SCRIPT = Path(__file__).parents[1]/'scripts/queue_commercial_cosim_parallel.py'
spec = importlib.util.spec_from_file_location('commercial_parallel', SCRIPT)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def task(identity='a', status='pending'):
    return {'id':identity, 'status':status, 'historical_output':'/unused/a',
            'kernel_sha256':'hash', 'timeout_seconds':10800, 'cohort':'recovery',
            'benchmark':'hlsfactory_gemm'}


def state(tasks, status='stopped'):
    return {'status':status, 'tasks':tasks, 'failure':None}


def test_clean_drain_only_continues_pending_and_preserves_completed():
    before=state([task('a','pass'),task('b','running'),task('c')],'running')
    after=state([task('a','pass'),task('b','timeout'),task('c')])
    assert queue.require_clean_drain(before,after)==[task('c')]


@pytest.mark.parametrize('status',['running','stopped_on_error','interrupted','prepared'])
def test_uncertain_drain_cannot_resume(status):
    with pytest.raises(ValueError,match='clean drain'):
        queue.require_clean_drain(state([task()]),state([task()],status))


@pytest.mark.parametrize('key,value',[('timeout_seconds',3600),('kernel_sha256','other'),
                                     ('historical_output','elsewhere'),('cohort','diagnostic')])
def test_protocol_changes_are_not_hidden_by_handoff(key,value):
    before=state([task()]);after=copy.deepcopy(before)
    after['tasks'][0][key]=value
    with pytest.raises(ValueError,match='protocol changed'):
        queue.require_clean_drain(before,after)


def test_terminal_result_cannot_change():
    before=state([task(status='pass')]);after=state([task(status='timeout')])
    with pytest.raises(ValueError,match='Completed recovery outcome changed'):
        queue.require_clean_drain(before,after)


@pytest.mark.parametrize('next_status',['pending','execution_error'])
def test_active_job_cannot_be_rerolled_after_drain(next_status):
    with pytest.raises(ValueError,match='did not finish cleanly'):
        queue.require_clean_drain(state([task(status='running')]),state([task(status=next_status)]))


def test_remaining_active_job_rejects_handoff():
    with pytest.raises(ValueError,match='active tasks'):
        queue.require_clean_drain(state([task(status='running')]),state([task(status='running')]))


def test_pending_record_with_measurement_is_not_retried(tmp_path):
    t=task();t['historical_output']=str(tmp_path)
    (tmp_path/'raw/measurements').mkdir(parents=True)
    (tmp_path/'raw/measurements/a.json').write_text('{}')
    with pytest.raises(ValueError,match='execution evidence'):
        queue.require_pending_pristine([t])


def previous_stop(signal_value=signal.SIGTERM):
    return {'status':'running','failure':None,'pid':12,'pid_start_ticks':'34','boot_id':'b',
            'stop_requested':{'signal':signal_value}}


IDENTITY={'pid':12,'start_ticks':'34','boot_id':'b'}


def test_live_controller_requires_durable_drain_before_commercial_dispatch():
    fig=SimpleNamespace(alive=lambda _:True)
    s=previous_stop();s.pop('stop_requested')
    assert queue.drain_admission(IDENTITY,s,fig)['commercial_allowed'] is False
    result=queue.drain_admission(IDENTITY,previous_stop(),fig)
    assert result['commercial_allowed'] is True and result['peer_pid']==12


@pytest.mark.parametrize('key,value',[('pid',13),('pid_start_ticks','35'),('boot_id','wrong'),
                                     ('status','stopped'),('failure',{'error':'x'})])
def test_peer_admission_authenticates_identity_and_recorded_state(key,value):
    previous=previous_stop();previous[key]=value
    assert not queue.drain_admission(IDENTITY,previous,SimpleNamespace(alive=lambda _:True))['commercial_allowed']


def test_dead_peer_is_never_whitelisted():
    assert queue.drain_admission(IDENTITY,previous_stop(),SimpleNamespace(alive=lambda _:False))=={
        'alive':False,'commercial_allowed':False,'peer_pid':None}


def test_old_operational_summary_reaches_complete_only_after_actual_recovery_completion():
    baseline=state([task('a','pass'),task('b')])
    identity={'pid':10,'start_ticks':'11','boot_id':'b','campaign':'new'}
    active=dict(task('b','running'),lane='recovery')
    ongoing=queue.recovery_operational_state(baseline,[active],identity)
    assert ongoing['status']=='running'
    assert ongoing['tasks'][0]==baseline['tasks'][0]
    active['status']='timeout'
    finished=queue.recovery_operational_state(baseline,[active],identity)
    assert finished['status']=='complete' and finished['pid']==10
    assert 'lane' not in finished['tasks'][1]
    assert baseline['tasks'][1]['status']=='pending'


def test_old_operational_state_never_overwrites_terminal_identity():
    with pytest.raises(ValueError,match='completed recovery outcome'):
        queue.recovery_operational_state(state([task('a','pass')]),[dict(task('a','timeout'),lane='recovery')],{})


def test_commercial_lane_error_does_not_mark_recovery_failed():
    identity={'pid':10,'start_ticks':'11','boot_id':'b','campaign':'new'}
    merged=queue.recovery_operational_state(state([task('a','pass')]),
                [dict(task('b','execution_error'),lane='commercial')],identity)
    assert merged['status']=='complete' and merged['failure'] is None


def test_summary_counts_legacy_plus_terminal_drain_plus_remaining_once():
    s={'status':'running','updated_at':'now','tasks':[
        dict(task('new1','timeout'),lane='recovery'),dict(task('new2','running'),lane='recovery'),
        dict(task('c','pass'),lane='commercial')],
       'recovery_baseline_counts':{'pass':5,'timeout':24},
       'recovery_legacy_counts':{'pass':33,'timeout':26},
       'commercial_distinct_validations':144,'lane_failures':{}}
    report=queue.summarize(s)
    assert report['combined_three_hour_replay']=={'pass':38,'timeout':51,'running':1}
    assert report['counts']['commercial']=={'pass':1} and report['model_calls']==0


def test_resource_ownership_follows_only_exact_registered_ancestry():
    table={10:{'ppid':1},11:{'ppid':10},12:{'ppid':11},20:{'ppid':1}}
    assert queue.descendant(12,10,table)
    assert not queue.descendant(20,10,table)
    assert not queue.descendant(12,99,table)


def test_completed_measurement_hash_and_archive_are_preserved(tmp_path):
    archive=tmp_path/'result.tar.zst';archive.write_bytes(b'archive')
    measurement=tmp_path/'measurement.json'
    measurement.write_text(json.dumps({'archive':str(archive),'cosim':{'status':'pass'}}))
    t=task(status='pass');t.update(measurement_path=str(measurement),measurement_sha256=queue.sha(measurement))
    files=queue.completed_files(state([t]))
    assert files=={str(measurement):queue.sha(measurement),str(archive):queue.sha(archive)}
    archive.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Frozen evidence changed'):
        queue.check_files(files)


@pytest.mark.parametrize('observed_start,expected_unknown',[('22',[]),('reused',[20,21])])
def test_resource_gate_rechecks_peer_start_ticks_after_admission(tmp_path,monkeypatch,observed_start,expected_unknown):
    table={10:{'ppid':1,'state':'S','start_ticks':'11'},
           20:{'ppid':1,'state':'S','start_ticks':observed_start},
           21:{'ppid':20,'state':'S','start_ticks':'23'}}
    fig=SimpleNamespace(process_table=lambda:table)
    monkeypatch.setattr(queue.os,'getpid',lambda:10)
    monkeypatch.setattr(queue.os,'getloadavg',lambda:(0,0,0))
    original=Path.read_text
    def read_text(path,*args,**kwargs):
        if str(path) in ('/proc/20/comm','/proc/21/comm'):
            return 'xsim'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read_text)
    config={'minimum_free_gib':0,'minimum_available_memory_gib':0,'maximum_load_per_cpu':.7}
    result=queue.resource_gate(tmp_path,config,fig,{'pid':20,'start_ticks':'22'})
    assert result['unrelated_vitis_pids']==expected_unknown
    assert result['allowed'] is (not expected_unknown)


def test_duplicate_after_drain_task_id_is_rejected():
    before=state([task('a')]);after=state([task('a'),task('a')])
    with pytest.raises(ValueError,match='identity changed'):
        queue.require_clean_drain(before,after)
