#!/usr/bin/env python3
"""Snapshot delivery, costs and selected RTL evidence; never alter pilot records."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_cell(spec, result, requests, preview):
    expected = hashlib.sha256(json.dumps(preview['messages'], ensure_ascii=True,
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    issues = []
    if len(requests) > 1:
        issues.append('more_than_one_provider_request')
    if any(r['prompt_sha256'] != expected for r in requests):
        issues.append('sent_prompt_differs_from_approved_preview')
    if any(r['status'] != 'settled' for r in requests):
        issues.append('provider_usage_unsettled')
    row = {'cell': spec['cell_id'], 'model': spec['model'], 'arm': spec['arm'],
           'controller_status': spec['status'], 'result_saved': result is not None,
           'provider_requests': len(requests), 'approved_prompt_match': bool(requests) and not any(r['prompt_sha256'] != expected for r in requests),
           'input_tokens': sum(r.get('input_tokens', 0) for r in requests),
           'output_tokens': sum(r.get('output_tokens', 0) for r in requests),
           'cached_tokens': sum(r.get('cached_tokens', 0) for r in requests),
           'estimated_or_reserved_usd': sum(r['charged_or_reserved_usd'] for r in requests),
           'issues': issues}
    if result is None:
        return row
    reference = result.get('reference_validation') or {}
    row['expert_reference_status'] = ('not_evaluated' if reference.get('reference_source') == 'independent_public_oracle'
                                      else result.get('ground_truth_status'))
    row['metadata_warnings'] = (['legacy_top_level_ground_truth_label']
        if row['expert_reference_status'] == 'not_evaluated' and result.get('ground_truth_status') != 'not_evaluated' else [])
    selected = result.get('selected_code_sha256')
    parent = (result.get('frozen_phase_b') or {}).get('code_sha256')
    cosim = result.get('cosim') or {}
    suite = cosim.get('validation_suite') or {}
    cases = suite.get('cases') or []
    rtl_pass = (cosim.get('passed') is True and cosim.get('success') is True
        and result.get('rtl_executed_pass') is True and bool(selected)
        and selected == result.get('cosim_target_code_sha256')
        and suite.get('planned_cases') == 5 and suite.get('executed_cases') == 5
        and len(cases) == 5 and suite.get('all_cases_passed') is True
        and all(c.get('passed') is True and c.get('success') is True
                and isinstance(c.get('kernel_runtime_cycles'), int)
                and not isinstance(c.get('kernel_runtime_cycles'), bool)
                and c['kernel_runtime_cycles'] > 0 for c in cases))
    if cosim.get('passed') and not rtl_pass:
        issues.append('passing_rtl_claim_lacks_complete_source_and_five_case_evidence')
    if rtl_pass and cosim['kernel_runtime_cycles'] != max(c['kernel_runtime_cycles'] for c in cases):
        issues.append('rtl_latency_is_not_worst_of_five')
    counts = (result.get('tool_flows') or {}).get('counts') or {}
    if counts.get('rtl_cosim', 0) != suite.get('executed_cases', 0):
        issues.append('rtl_flow_count_mismatch')
    accounting = result.get('accounting_reconciliation') or {}
    if not accounting.get('complete'):
        issues.extend(accounting.get('issues') or ['missing_accounting_reconciliation'])
    evaluations = (result.get('synthesis_evaluations') or {}).get('events') or []
    row.update(run_success=result.get('success'), independently_checked_rtl_pass=rtl_pass,
               parent_retained=bool(selected and parent and selected == parent),
               selected_code_sha256=selected, selected_rtl_cycles=cosim.get('kernel_runtime_cycles') if rtl_pass else None,
               selected_csynth_estimate_cycles=(result.get('final_report') or result.get('synth_report') or {}).get('latency_cycles'),
               generated_candidate_count=len(evaluations),
               generated_candidate_failures=[e.get('failure_class') for e in evaluations if e.get('failure_class')],
               selection_syntheses=counts.get('csynth', 0), rtl_resyntheses=counts.get('rtl_cosim', 0),
               csim_flows=counts.get('csim', 0), native_oracle_flows=counts.get('native_oracle', 0),
               native_compile_flows=counts.get('native_compile', 0),
               accounting_complete=accounting.get('complete') is True, issues=sorted(set(issues)))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, default=ROOT/'results_sweeps/sigmetrics_recipe_pilot_20260928')
    args = parser.parse_args(); pilot = args.pilot.resolve()
    manifest = json.loads((pilot/'manifest.json').read_text())
    state = json.loads((pilot/'state.json').read_text())
    ledger = json.loads(Path(manifest['api_budget']).read_text())
    stamp = datetime.now(timezone.utc)
    output = ROOT/'artifacts/sigmetrics2027/pilot_reports'/stamp.strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    rows = []
    for spec in state['cells']:
        path = pilot/'cells'/spec['cell_id']/'result.json'
        result = json.loads(path.read_text()) if path.exists() else None
        requests = [r for r in ledger['reservations'] if r['run_id'] == 'recipe_pilot_v1_'+spec['cell_id']]
        preview = json.loads((ROOT/'artifacts/sigmetrics2027/recipe_prompt_preview_final_20260928'/(spec['cell_id']+'_messages.json')).read_text())
        row = summarize_cell(spec, result, requests, preview)
        if path.exists():
            row.update(result_path=str(path), result_sha256=sha(path))
        rows.append(row)
    data = {'captured_at':stamp.isoformat(),'scope':'one-kernel engineering pilot; no comparative efficacy inference',
        'controller_status':state['status'],'manifest_sha256':sha(pilot/'manifest.json'),
        'authorized_manifest_unchanged':json.loads((pilot/'export_authorization.json').read_text())['manifest_sha256']==sha(pilot/'manifest.json'),
        'all_six_prompts_delivered':all(r['provider_requests']==1 and r['approved_prompt_match'] for r in rows),
        'engineering_gate_passed':len(rows)==6 and all(r['result_saved'] and r.get('accounting_complete') and not r['issues'] for r in rows),
        'pilot_estimated_or_reserved_usd':sum(r['estimated_or_reserved_usd'] for r in rows),
        'campaign_estimated_or_reserved_usd':sum(r['charged_or_reserved_usd'] for r in ledger['reservations']),
        'api_cap_usd':ledger['limit_usd'],'cells':rows}
    (output/'report.json').write_text(json.dumps(data,indent=2)+'\n')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (output/'cells.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=['# Prepared-prompt pilot report',f"\nCaptured {stamp.isoformat()}. One-kernel engineering pilot; no comparative efficacy inference.",
           f"\nController: **{state['status']}**. All six approved prompts delivered: **{data['all_six_prompts_delivered']}**. Accounting/evidence gate: **{data['engineering_gate_passed']}**.",
           f"\nPilot estimated cost plus unsettled reservations: **${data['pilot_estimated_or_reserved_usd']:.6f}**. Entire campaign: ${data['campaign_estimated_or_reserved_usd']:.6f} / ${data['api_cap_usd']:.2f}.",
           '\n| Cell | Provider requests | Saved | Five-case RTL pass | Parent retained | RTL cycles (max) | Estimated/reserved USD |',
           '|---|---:|---|---|---|---:|---:|']
    for r in rows:
        lines.append(f"| {r['cell']} | {r['provider_requests']} | {r['result_saved']} | {r.get('independently_checked_rtl_pass','pending')} | {r.get('parent_retained','pending')} | {r.get('selected_rtl_cycles','')} | {r['estimated_or_reserved_usd']:.6f} |")
    lines+=['\nFrozen pilot records retain a legacy top-level ground_truth_status label. For public-oracle runs, expert_reference_status is explicitly not_evaluated in this report; the raw records and active source remain unchanged.', '\nA retained parent is not a successful new generation. Selection syntheses, CSim and RTL resyntheses are separately tabulated in `cells.csv`. The parent construction cost is in the frozen parent measurement.',
            '\nCorrectness/feasibility failures remain observations. Missing accounting or inconsistent provenance prevents the engineering gate from passing. The full efficacy matrix is not launched by this report.']
    (output/'report.md').write_text('\n'.join(lines)+'\n');print(str(output))

if __name__=='__main__':main()
