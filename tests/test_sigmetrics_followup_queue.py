import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('followup', Path(__file__).parents[1]/'scripts/queue_sigmetrics_followups.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def group(name='kernel', mode='flash'):
    return {'id': name+'__'+mode, 'mode': mode, 'benchmark': name,
            'status': 'pending', 'next_attempt': 1, 'attempts': []}


def test_timeout_consumes_attempt_and_exhausts_exactly_k():
    g = group()
    for k in range(1, 6):
        assert g['next_attempt'] == k
        g['status'] = 'running'
        queue.apply_attempt(g, {'status': 'timeout', 'attempt': k}, 5)
        assert g['status'] == ('pending' if k < 5 else 'exhausted')
    assert len(g['attempts']) == 5
    assert queue.next_group([g]) is None


def test_first_pass_stops_group_without_additional_calls():
    g = group(); g['status'] = 'running'
    queue.apply_attempt(g, {'status': 'pass', 'attempt': 1}, 5)
    assert g['status'] == 'pass' and g['passed_on_attempt'] == 1
    assert queue.next_group([g]) is None


@pytest.mark.parametrize('status', ['execution_error', 'evidence_error'])
def test_uncertain_attempt_is_never_freely_retried(status):
    g = group(); g['status'] = 'running'
    queue.apply_attempt(g, {'status': status, 'attempt': 1}, 5)
    assert g['status'] == status
    assert g['next_attempt'] == 1
    assert queue.next_group([g]) is None


def test_retry_rotates_after_all_fresh_groups():
    a,b = group('a'),group('b')
    a['next_attempt'] = 2
    assert queue.next_group([a,b]) is b


def test_mismatched_or_duplicate_attempt_fails_closed():
    g = group(); g['status'] = 'running'
    with pytest.raises(ValueError): queue.apply_attempt(g, {'status':'pass','attempt':2}, 5)
    queue.apply_attempt(g, {'status':'pass','attempt':1}, 5)
    with pytest.raises(ValueError): queue.apply_attempt(g, {'status':'pass','attempt':1}, 5)


def test_curve_preserves_pending_and_blocked_denominator():
    a,b,c = group('a'),group('b'),group('c')
    a.update(status='pass',passed_on_attempt=2,attempts=[{'status':'timeout'},{'status':'pass'}])
    c['status']='blocked_input'
    state={'status':'running','updated_at':'now','historical_tasks':[],
           'passk_groups':[a,b,c],'paused_lanes':{}}
    result=queue.summary(state, {'passk_modes':['flash'],'k':5})
    curve=result['success_within_k']['flash']
    assert curve['denominator_all_packaged_kernels']==3
    assert curve['observed_successes_within_k']=={'1':0,'2':1,'3':1,'4':1,'5':1}
    assert result['passk_attempt_counts']=={'timeout':1,'pass':1}


@pytest.mark.parametrize('status,live,allowed', [
    ('running',True,False), ('running',False,False),
    ('complete',True,False), ('complete',False,True),
    ('stopped_on_error',False,False),
    ('complete_with_quarantined_failure',False,True)])
def test_waits_for_terminal_state_and_process_exit(tmp_path,monkeypatch,status,live,allowed):
    folder=tmp_path/'existing';folder.mkdir()
    (folder/'summary.json').write_text(json.dumps({'status':status}))
    (folder/'controller_identity.json').write_text('{}')
    monkeypatch.setattr(queue,'alive',lambda _:live)
    gate=queue.dependencies({'wait_for_campaigns':[str(folder)]})
    assert gate['allowed'] is allowed


def test_missing_or_reused_pid_does_not_authorize_signalling():
    assert queue.alive({'pid':999999999,'boot_id':'wrong','start_ticks':'0','source_runner':'x'}) is False


@pytest.mark.parametrize('reason,status', [('candidate_csim_failed','failed_csim'),
                                        ('invalid_native_golden','blocked_input'), ('worker_error','execution_error')])
def test_pre_rtl_nonpasses_keep_their_validation_stage(reason,status):
    m={'cosim':{'status':'not_run','skip_reason':reason}}
    assert queue.historical_status(m,{'returncode':0})==status
    assert queue.historical_status(m,{'returncode':1})=='execution_error'


def test_historical_cleanup_uncertainty_pauses_lane_even_after_pass():
    measurement={'cosim':{'status':'pass', 'passed':True, 'success':True,
                         'kernel_runtime_cycles':100, 'correctness':{'passed':True}}}
    assert queue.historical_status(measurement,{'returncode':0})=='pass'
    assert queue.historical_status(measurement,{'returncode':0,'cleanup_survivors':[123]})=='execution_error'
