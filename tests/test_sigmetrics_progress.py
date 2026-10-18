"""Progress reports must preserve denominators and separate execution from validation."""
import json
from pathlib import Path

from scripts.snapshot_sigmetrics_progress import (
    main_study_progress, rtl_outcome, select_recovery_scheduler, summarize_api,
)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_controller_selection_changes_only_when_queue_owner_changes():
    parallel = {'pid': 10, 'workers': 4, 'status': 'running'}
    coexist = {'pid': 20, 'workers': 4, 'status': 'waiting_for_previous_controller'}
    assert select_recovery_scheduler({'pid': 10}, parallel, coexist)['selected_from'] == 'parallel'
    assert select_recovery_scheduler({'pid': 20}, parallel, coexist)['selected_from'] == 'coexistence'
    assert select_recovery_scheduler({'pid': 30}, parallel, coexist)['status'] == 'no_matching_scheduler_receipt'


def test_settled_and_unresolved_ledger_amounts_are_separate():
    result = summarize_api([
        {'status': 'settled', 'reserved_usd': 9, 'charged_or_reserved_usd': 0.25},
        {'status': 'reserved', 'reserved_usd': 2, 'charged_or_reserved_usd': 2},
    ], 300)
    assert result['settled_estimate_usd'] == 0.25
    assert result['unresolved_reservation_usd'] == 2
    assert result['estimated_charged_or_reserved_usd'] == 2.25


def test_partial_rtl_success_is_not_a_five_case_pass():
    result = {'rtl_executed_pass': True, 'cosim': {'ran': True, 'passed': True,
              'validation_suite': {'planned_cases': 5, 'executed_cases': 1, 'all_cases_passed': True}}}
    assert rtl_outcome(result)['outcome'] == 'incomplete_or_unverified'
    result['cosim']['validation_suite']['executed_cases'] = 5
    assert rtl_outcome(result)['outcome'] == 'pass_all_five'
    result['cosim']['passed'] = False
    assert rtl_outcome(result)['outcome'] == 'fail'
    result['cosim']['error'] = 'simulation timed out after 3600 seconds'
    assert rtl_outcome(result)['outcome'] == 'timeout'


def test_registered_cells_and_actual_run_ids_define_main_totals(tmp_path):
    campaign = tmp_path / 'sigmetrics_recipe_main_20260930'
    specs = [{'cell_id': f'cell{i}', 'benchmark': 'atax', 'model': 'model',
              'arm': 'none', 'repeat': 1, 'status': 'pending'} for i in range(216)]
    put(campaign / 'manifest.json', {'schema_version': 'c2hls.sigmetrics-recipe-main.v1', 'cells': specs})
    cells = [dict(c) for c in specs]
    for i, status in enumerate(['complete', 'blocked_parent', 'running', 'accounting_or_execution_failure']):
        cells[i]['status'] = status
    parent_path = campaign / 'parents' / 'atax'
    put(campaign / 'state.json', {'status': 'running', 'cells': cells,
        'parents': [{'benchmark': 'atax', 'path': str(parent_path), 'status': 'running'}]})
    put(campaign / 'cells/cell0/result.json', {'success': False, 'rtl_executed_pass': False,
         'cosim': {'ran': True, 'passed': False, 'status': 'failed',
                   'validation_suite': {'planned_cases': 5, 'executed_cases': 2, 'all_cases_passed': False}}})
    ledger = {'reservations': [
        {'run_id': campaign.name + '__cell0', 'status': 'settled', 'charged_or_reserved_usd': 1},
        {'run_id': campaign.name + '__cell2', 'status': 'reserved', 'charged_or_reserved_usd': 2},
        {'run_id': campaign.name + '__unregistered', 'status': 'settled', 'charged_or_reserved_usd': 100},
        {'run_id': 'pilot__cell0', 'status': 'settled', 'charged_or_reserved_usd': 100},
    ]}
    files = [p for p in tmp_path.rglob('*') if p.is_file()]
    before = {p: p.read_bytes() for p in files}
    result = main_study_progress(campaign, ledger)
    assert result['registered_cells'] == 216
    assert result['registration_count_matches']
    assert result['cell_counts'] == {'pending': 212, 'blocked': 1, 'running': 1, 'completed': 1, 'failed': 1, 'unknown': 0}
    assert result['rtl_outcomes'] == {'fail': 1}
    assert result['saved_results'] == 1
    assert result['api']['requests'] == 2
    assert result['api']['estimated_charged_or_reserved_usd'] == 3
    assert result['parents'][0]['stage'] == 'public_c_validation'
    assert before == {p: p.read_bytes() for p in files}


def test_transient_incomplete_result_does_not_become_a_pass(tmp_path):
    campaign = tmp_path / 'main'
    put(campaign / 'manifest.json', {'cells': [{'cell_id': 'one', 'status': 'pending'}]})
    put(campaign / 'state.json', {'cells': [{'cell_id': 'one', 'status': 'complete'}], 'parents': []})
    result_path = campaign / 'cells/one/result.json'
    result_path.parent.mkdir(parents=True)
    result_path.write_text('{')
    result = main_study_progress(campaign, {'reservations': []})
    assert result['cell_counts']['completed'] == 1
    assert result['saved_results'] == 0
    assert result['rtl_outcomes'] == {}
    assert result['read_issues'][0]['error'] == 'JSONDecodeError'


def test_absent_main_campaign_remains_unregistered(tmp_path):
    result = main_study_progress(tmp_path / 'absent', {})
    assert result['status'] == 'not_prepared'
    assert result['registered_cells'] == 0
