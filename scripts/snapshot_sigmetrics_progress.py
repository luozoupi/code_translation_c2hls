#!/usr/bin/env python3
"""Write a dated read-only snapshot of the revision pilots and API ledger."""
from __future__ import annotations
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/sigmetrics2027'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def select_recovery_scheduler(state, parallel, coexist):
    """Use the scheduler owning the current queue state, never a stale PID."""
    pid = (state or {}).get('pid')
    for kind, candidate in (('coexistence', coexist), ('parallel', parallel)):
        if candidate and candidate.get('pid') == pid:
            return {**candidate, 'selected_from': kind}
    return {'status': 'no_matching_scheduler_receipt', 'pid': pid, 'workers': None}


def summarize_api(requests, cap=None):
    settled = sum(float(r.get('charged_or_reserved_usd', 0)) for r in requests if r.get('status') == 'settled')
    reserved = sum(float(r.get('charged_or_reserved_usd', 0)) for r in requests if r.get('status') != 'settled')
    return {'cap_usd': cap, 'settled_estimate_usd': settled, 'unresolved_reservation_usd': reserved,
            'estimated_charged_or_reserved_usd': settled + reserved,
            'requests': len(requests), 'statuses': dict(Counter(r.get('status', 'unknown') for r in requests)),
            'overruns': sum(bool(r.get('overrun')) for r in requests),
            'billing_scope': 'actual ledger entries only; settled token-cost estimates plus unresolved reservations, not invoice totals'}


def rtl_outcome(result, selected=True):
    if not result:
        return {'outcome': 'not_recorded', 'executed_cases': 0, 'planned_cases': 5}
    cosim = result.get('cosim') or {}
    suite = cosim.get('validation_suite') or {}
    executed = suite.get('executed_cases', 0)
    planned = suite.get('planned_cases')
    evidence = {'executed_cases': executed, 'planned_cases': planned, 'reported_status': cosim.get('status'),
                'kernel_runtime_cycles': None}
    all_five = (cosim.get('passed') is True and planned == 5 and executed == 5
                and suite.get('all_cases_passed') is True and (not selected or result.get('rtl_executed_pass') is True))
    if all_five:
        evidence.update(outcome='pass_all_five', kernel_runtime_cycles=cosim.get('kernel_runtime_cycles'))
    elif cosim.get('status') == 'timeout' or 'timed out' in str(cosim.get('error', '')).lower():
        evidence['outcome'] = 'timeout'
    elif cosim.get('passed') is False and (cosim.get('ran') or executed):
        evidence['outcome'] = 'fail'
    elif not cosim.get('ran') and not executed:
        evidence['outcome'] = 'not_run'
    else:
        evidence['outcome'] = 'incomplete_or_unverified'
    return evidence


def main_study_progress(campaign, ledger):
    """Derive progress from registered cells and durable results, without writes."""
    read_issues = []
    def observed(path):
        try:
            return read(path)
        except (OSError, json.JSONDecodeError) as exc:
            read_issues.append({'path': str(path), 'error': type(exc).__name__})
            return None
    manifest = observed(campaign / 'manifest.json')
    if not manifest:
        return {'status': 'not_prepared', 'registered_cells': 0, 'read_issues': read_issues}
    state = observed(campaign / 'state.json') or {}
    summary = observed(campaign / 'summary.json') or {}
    specs = manifest.get('cells', [])
    current = {c['cell_id']: c for c in state.get('cells', [])}
    rows, parent_rows = [], []
    for spec in specs:
        identity = spec['cell_id']
        status = current.get(identity, spec).get('status', 'pending')
        if status == 'complete':
            category = 'completed'
        elif status.startswith('blocked'):
            category = 'blocked'
        elif status == 'running':
            category = 'running'
        elif status == 'pending':
            category = 'pending'
        elif 'failure' in status or status.startswith('failed'):
            category = 'failed'
        else:
            category = 'unknown'
        path = campaign / 'cells' / identity / 'result.json'
        result = observed(path)
        row = {k: spec.get(k) for k in ('cell_id', 'benchmark', 'model', 'arm', 'repeat')}
        row.update(status=status, category=category, result_saved=result is not None,
                   success=result.get('success') if result else None, rtl=rtl_outcome(result))
        if result:
            row.update(result_path=str(path), accounting_complete=(result.get('accounting_reconciliation') or {}).get('complete'),
                       selected_code_sha256=result.get('selected_code_sha256'))
        if current.get(identity, {}).get('reason'):
            row['reason'] = current[identity]['reason']
        rows.append(row)
    for parent in state.get('parents', []):
        path = Path(parent['path']) / 'measurement.json'
        measurement = observed(path)
        stage = parent['status']
        if stage == 'running':
            stage = 'recipe_and_prompt_qualification' if measurement else 'public_c_validation'
        parent_rows.append({'benchmark': parent['benchmark'], 'status': parent['status'], 'stage': stage,
                            'measurement_saved': measurement is not None, 'rtl': rtl_outcome(measurement, selected=False),
                            'tool_counts': (measurement.get('tool_flows') or {}).get('counts') if measurement else None})
    prefix = campaign.name + '__'
    registered_run_ids = {prefix + spec['cell_id'] for spec in specs}
    requests = [r for r in ledger.get('reservations', []) if r.get('run_id') in registered_run_ids]
    counts = Counter(row['category'] for row in rows)
    normalized = {k: counts[k] for k in ('pending', 'blocked', 'running', 'completed', 'failed', 'unknown')}
    return {'status': state.get('status', summary.get('status', 'registered')), 'campaign': str(campaign),
            'schema_version': manifest.get('schema_version'), 'registered_cells': len(specs),
            'expected_registered_cells': 216, 'registration_count_matches': len(specs) == 216,
            'state_updated_at': state.get('updated_at'), 'controller_pid': state.get('pid'),
            'cell_counts': normalized, 'raw_cell_statuses': dict(Counter(r['status'] for r in rows)),
            'parent_counts': dict(Counter(p['status'] for p in parent_rows)), 'parents': parent_rows,
            'rtl_outcomes': dict(Counter(row['rtl']['outcome'] for row in rows if row['result_saved'])),
            'saved_results': sum(row['result_saved'] for row in rows), 'cells': rows,
            'resource_gate': state.get('resource_gate'), 'stop_requested': state.get('stop_requested'),
            'api': summarize_api(requests), 'read_issues': read_issues,
            'scope': 'registered cells retained including blocked parents; controller completion and five-input RTL validation reported separately'}


def snapshot():
    started=datetime.now(timezone.utc);out=ART/'status'/started.strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    historic=Path('/home/luo00466/code_translation-c2hls-hpca2027/results_sweeps/july_cosim_backfill_20260928')
    parent=read(ART/'parent_pilot_20260928/atax/measurement.json')
    pilot=ROOT/'results_sweeps/sigmetrics_recipe_pilot_20260928'
    ledger=read(ART/'api_budget.json') or {};requests=ledger.get('reservations',[])
    recovery=ROOT/'results_sweeps/historical_registry_recovery_20260928'
    recovery_state=read(recovery/'state.json')
    parallel=read(recovery/'parallel_scheduler_status.json')
    coexist=read(recovery/'coexist_scheduler_status.json')
    scheduler=select_recovery_scheduler(recovery_state,parallel,coexist)
    try:
        proc=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True,timeout=5)
        sessions=[s for s in proc.stdout.splitlines() if s.startswith('c2hls_')]
    except (OSError,subprocess.TimeoutExpired):sessions=[]
    data={'read_started_at':started.isoformat(),'read_finished_at':datetime.now(timezone.utc).isoformat(),
          'historical_state':read(historic/'state.json'),'historical_summary':read(historic/'summary.json'),
          'validator_pilot':read(ROOT/'results_sweeps/sigmetrics_validator_pilot_20260928/summary.json'),
          'registry_recovery':read(ROOT/'results_sweeps/historical_registry_recovery_20260928/migration_summary.json'),
          'registry_recovery_state':recovery_state,
          'registry_recovery_summary':read(ROOT/'results_sweeps/historical_registry_recovery_20260928/summary.json'),
          'registry_recovery_launcher':read(ROOT/'results_sweeps/historical_registry_recovery_20260928/launcher_status.json'),
          'registry_recovery_parallel_scheduler':scheduler,
          'registry_recovery_original_parallel_scheduler':parallel,
          'registry_recovery_coexist_scheduler':coexist,
          'recipe_main':main_study_progress(ROOT/'results_sweeps/sigmetrics_recipe_main_20260930',ledger),
          'native_inputs':{k:v for k,v in (read(ART/'core_inputs_v1/native_validation.json') or {}).items() if k!='cases'},
          'parent':{'success':parent['success'],'tool_counts':parent['tool_flows']['counts'],
                    'rtl_cycles':parent.get('cosim',{}).get('kernel_runtime_cycles')} if parent else {'status':'pending_or_running'},
          'recipe_pilot':read(pilot/'state.json') or {'status':'not_prepared'},
          'api':summarize_api(requests,ledger.get('limit_usd')),'tmux_sessions':sessions,
          'scope':'dated snapshot; no continuing monitoring after command returns'}
    rows=[]
    for path in sorted(pilot.glob('cells/*/result.json')):
        result=read(path);rows.append({'cell':path.parent.name,'success':result.get('success'),
           'rtl_executed_pass':result.get('rtl_executed_pass'),'selected_code_sha256':result.get('selected_code_sha256'),
           **result.get('accounting_reconciliation',{}),'result_path':str(path),'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    data['completed_cells']=rows
    data['read_finished_at']=datetime.now(timezone.utc).isoformat()
    (out/'snapshot.json').write_text(json.dumps(data,indent=2)+'\n')
    if rows:
        with (out/'cells.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    historical=data['historical_summary'] or {}
    lines=['# SIGMETRICS revision progress',f"\nRead window: {data['read_started_at']}–{data['read_finished_at']}.",
           f"\nHistorical queue: {(data['historical_state'] or {}).get('status')}; {historical.get('measurement_status_counts')}.",
           f"\nValidator pilot: {(data['validator_pilot'] or {}).get('measurement_status_counts')}.",
           f"\nNative inputs: {data['native_inputs'].get('passed')}/{data['native_inputs'].get('total')} passed.",
           f"\nFresh public-C parent: {data['parent']}.",f"\nRecipe pilot: {data['recipe_pilot']['status']}; {len(rows)} saved cells.",
           f"\nShared API ledger: settled estimate ${data['api']['settled_estimate_usd']:.6f}; unresolved reservations ${data['api']['unresolved_reservation_usd']:.6f}; combined ${data['api']['estimated_charged_or_reserved_usd']:.6f} / ${data['api']['cap_usd']}. Actual ledger entries, not invoices or the unspent campaign ceiling; {data['api']['statuses']}.",
           f"\nHistorical registry recovery: {(data['registry_recovery_state'] or {}).get('status', 'prepared_not_launched')}; "
           f"{(data['registry_recovery_summary'] or {}).get('finished_new_validations', 0)}/"
           f"{(data['registry_recovery'] or {}).get('distinct_validations', 0)} finished, "
           f"{len((data['registry_recovery_state'] or {}).get('active', []))} active; "
           f"{(data['registry_recovery_summary'] or {}).get('measurement_status_counts', {})}.",
           f"\nRecovery scheduling transition: {(data['registry_recovery_parallel_scheduler'] or {}).get('status', 'original_controller')}; "
           f"requested worker limit: {(data['registry_recovery_parallel_scheduler'] or {}).get('workers', 2)}.",
           '\nRefresh with `python3 scripts/snapshot_sigmetrics_progress.py`. This command does not run a persistent monitor.']
    main=data['recipe_main']
    lines.insert(-1, f"\nMain study: {main['status']}; {main['registered_cells']}/216 cells registered; {main.get('cell_counts', {})}. Parent states: {main.get('parent_counts', {})}.")
    if main['registered_cells']:
        lines.insert(-1, f"\nMain-study durable RTL outcomes: {main['rtl_outcomes']}; {main['saved_results']} saved results. A completed controller cell is not necessarily a five-input RTL pass.")
        lines.insert(-1, f"\nMain-study API ledger (registered run IDs only): settled estimate ${main['api']['settled_estimate_usd']:.6f}; unresolved reservations ${main['api']['unresolved_reservation_usd']:.6f}; {main['api']['requests']} recorded requests.")
        active_parents=[p['benchmark']+': '+p['stage'] for p in main['parents'] if p['status'] in ('running','qualifying')]
        if active_parents: lines.insert(-1, '\nActive parent stages: '+', '.join(active_parents)+'.')
        if main['read_issues']: lines.insert(-1, f"\nMain-study artifacts unavailable during this read: {main['read_issues']}. Refresh for a new snapshot.")
    if coexist and coexist.get('pid') != (recovery_state or {}).get('pid'):
        lines.insert(-1, f"\nSeparate coexistence handover status: {coexist.get('status')}; it does not yet own the recorded queue state.")
    (out/'report.md').write_text('\n'.join(lines)+'\n');print(str(out))

if __name__=='__main__':snapshot()
