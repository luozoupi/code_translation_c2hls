import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('recipe_pilot',Path(__file__).resolve().parents[1]/'scripts/run_sigmetrics_recipe_pilot.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)


def fixture():
    result={'llm_usage':{'events':[{'input_tokens':10,'output_tokens':5,'cached_tokens':0,'prompt_sha256':'prompt'}]},
            'synthesis_evaluations':{'count':1},'candidate_ledger':{'complete':True},
            'reference_isolation_audit':{'passed':True},'selected_code_sha256':'code','cosim_target_code_sha256':'code',
            'rtl_executed_pass':True,'cosim':{'passed':True,'validation_suite':{'executed_cases':5,'all_cases_passed':True}},
            'tool_flows':{'counts':{'csynth':1,'rtl_cosim':5},'total_flow_invocations':11,
                          'events':[{'kind':'rtl_cosim','code_sha256':'code'} for _ in range(5)]}}
    rows=[{'status':'settled','input_tokens':10,'output_tokens':5,'cached_tokens':0,'prompt_sha256':'prompt','charged_or_reserved_usd':0.001}]
    return result,rows


def test_reconciles_calls_tokens_and_selected_rtl():
    result,rows=fixture();assert pilot.reconcile(result,rows)['complete']


def test_hidden_resynthesis_or_wrong_source_rejected():
    result,rows=fixture();result['tool_flows']['counts']['rtl_cosim']=6
    assert 'rtl_case_flow_count_mismatch' in pilot.reconcile(result,rows)['issues']
    result,rows=fixture();result['tool_flows']['events'][2]['code_sha256']='different'
    assert 'rtl_flow_source_mismatch' in pilot.reconcile(result,rows)['issues']


def test_unknown_provider_usage_prevents_gate():
    result,rows=fixture();rows[0]['status']='reserved'
    assert not pilot.reconcile(result,rows)['complete']


def test_correctness_failure_is_preserved_but_accounted():
    result,rows=fixture();result['rtl_executed_pass']=False;result['cosim']['passed']=False
    result['cosim']['validation_suite'].update(executed_cases=2,all_cases_passed=False)
    result['tool_flows']['counts']['rtl_cosim']=2;result['tool_flows']['events']=result['tool_flows']['events'][:2]
    assert pilot.reconcile(result,rows)['complete']
