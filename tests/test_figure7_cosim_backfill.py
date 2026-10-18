import importlib.util
import json
from pathlib import Path
import pytest


SCRIPT = Path(__file__).parents[1]/'scripts/queue_figure7_cosim_backfill.py'
spec = importlib.util.spec_from_file_location('figure7_queue', SCRIPT)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


@pytest.mark.parametrize('status,alive,allowed', [
    ('running', True, False), ('running', False, False),
    ('stopped_on_failure', False, False), ('stopped', False, False),
    ('complete', True, False), ('complete', False, True),
    ('complete_with_failures', False, True)])
def test_recovery_must_complete_and_exit_before_dispatch(tmp_path, monkeypatch, status, alive, allowed):
    (tmp_path/'summary.json').write_text(json.dumps({'status': status}))
    (tmp_path/'controller_identity.json').write_text('{}')
    monkeypatch.setattr(queue, 'alive', lambda identity: alive)
    gate = queue.dependency_gate({'wait_for_recovery': str(tmp_path)})
    assert gate['allowed'] is allowed


def test_proc_enumeration_ignores_disappearing_process_directory(tmp_path, monkeypatch):
    # A process can exit between /proc enumeration and its stat read. Avoid a
    # recursive pathlib glob, whose directory scan can raise outside the try.
    (tmp_path/'10').mkdir()
    (tmp_path/'11').mkdir()
    fields = ['S', '1'] + ['0']*17 + ['12345']
    (tmp_path/'10/stat').write_text('10 (process with spaces) ' + ' '.join(fields))
    (tmp_path/'version').write_text('not a pid')
    monkeypatch.setattr(Path, 'glob', lambda *args: pytest.fail('unsafe /proc glob used'))
    result = queue.process_table(tmp_path)
    assert result == {10: {'pid': 10, 'ppid': 1, 'start_ticks': '12345', 'state': 'S'}}


@pytest.mark.parametrize('skip,status', [('candidate_csim_failed','failed_csim'),
                                       ('invalid_native_golden','blocked_input'),
                                       ('worker_error','execution_error')])
def test_native_or_csim_block_is_not_reported_as_rtl_failure(skip, status):
    result = queue.classify_measurement({'cosim': {'status':'not_run','skip_reason':skip}}, {'returncode':0})
    assert result == status


@pytest.mark.parametrize('field,value', [('passed',False),('success',False),
                                        ('correctness',{'passed':False}),('kernel_runtime_cycles',None)])
def test_pass_requires_independent_output_check_and_measured_cycles(field,value):
    cosim = {'status':'pass','passed':True,'success':True,'correctness':{'passed':True},'kernel_runtime_cycles':100}
    cosim[field]=value
    assert queue.classify_measurement({'cosim':cosim}, {'returncode':0}) == 'evidence_error'


def test_cleanup_uncertainty_takes_precedence_over_recorded_pass():
    cosim = {'status':'pass','passed':True,'success':True,'correctness':{'passed':True},'kernel_runtime_cycles':100}
    assert queue.classify_measurement({'cosim':cosim}, {'returncode':0}) == 'pass'
    assert queue.classify_measurement({'cosim':cosim}, {'returncode':0,'cleanup_survivors':[123]}) == 'execution_error'
    assert queue.classify_measurement({'cosim':cosim,'archive_error':'not verified'}, {'returncode':0}) == 'execution_error'


def test_existing_measurement_cannot_be_silently_rerun(tmp_path):
    raw=tmp_path/'raw'
    (raw/'measurements').mkdir(parents=True)
    (raw/'measurements/id.json').write_text('{}')
    with pytest.raises(ValueError, match='previously started'):
        queue.execute_task(tmp_path/'campaign', {}, {'raw_root':str(raw)}, {'validation_id':'id'}, None, None)


def test_stale_or_reused_pid_never_counts_as_live():
    assert not queue.alive({'pid':999999999,'start_ticks':'0','boot_id':'wrong','source_runner':'x'})


def test_uncertain_status_is_visible_without_fabricating_an_outcome():
    state={'status':'stopped_on_error','updated_at':'now','tasks':[
        {'validation_id':'a','status':'pass'}, {'validation_id':'b','status':'execution_error'},
        {'validation_id':'c','status':'pending'}], 'failure':{'validation_id':'b'}}
    report=queue.summarize(state)
    assert report['counts']=={'pass':1,'execution_error':1,'pending':1}
    assert report['distinct_validations']==3
    assert report['model_calls']==0
