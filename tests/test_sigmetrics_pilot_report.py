import hashlib
import importlib.util
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location('pilot_report',Path(__file__).resolve().parents[1]/'scripts/report_sigmetrics_recipe_pilot.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def fixture():
    preview={'messages':[{'role':'user','content':'public code'}]}
    digest=hashlib.sha256(json.dumps(preview['messages'],ensure_ascii=True,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    cell={'cell_id':'model_none','model':'model','arm':'none','status':'complete'}
    requests=[{'prompt_sha256':digest,'status':'settled','input_tokens':10,'output_tokens':5,'charged_or_reserved_usd':0.001}]
    result={'selected_code_sha256':'parent','cosim_target_code_sha256':'parent','frozen_phase_b':{'code_sha256':'parent'},
            'rtl_executed_pass':True,'success':True,'accounting_reconciliation':{'complete':True,'issues':[]},
            'cosim':{'passed':True,'success':True,'kernel_runtime_cycles':9,
                     'validation_suite':{'planned_cases':5,'executed_cases':5,'all_cases_passed':True,
                                         'cases':[{'passed':True,'success':True,'kernel_runtime_cycles':n} for n in [1,2,9,4,5]]}},
            'tool_flows':{'counts':{'rtl_cosim':5,'csynth':1}},'synthesis_evaluations':{'events':[{'failure_class':'timing_failure'}]}}
    return cell,result,requests,preview


def test_parent_retention_is_separate_from_generated_failure():
    row=report.summarize_cell(*fixture())
    assert row['independently_checked_rtl_pass'] and row['parent_retained']
    assert row['generated_candidate_failures']==['timing_failure'] and row['selected_rtl_cycles']==9
    assert not row['issues']


def test_wrong_payload_and_incomplete_rtl_evidence_fail():
    cell,result,requests,preview=fixture();requests[0]['prompt_sha256']='wrong'
    result['cosim']['validation_suite']['cases'].pop()
    row=report.summarize_cell(cell,result,requests,preview)
    assert not row['approved_prompt_match'] and not row['independently_checked_rtl_pass']
    assert 'sent_prompt_differs_from_approved_preview' in row['issues']


def test_false_latency_aggregation_is_rejected():
    cell,result,requests,preview=fixture();result['cosim']['kernel_runtime_cycles']=1
    row=report.summarize_cell(cell,result,requests,preview)
    assert 'rtl_latency_is_not_worst_of_five' in row['issues']


def test_pending_cell_is_not_a_failed_or_successful_sample():
    cell,result,requests,preview=fixture()
    row=report.summarize_cell(cell,None,[],preview)
    assert not row['result_saved'] and row['provider_requests']==0
    assert 'independently_checked_rtl_pass' not in row
