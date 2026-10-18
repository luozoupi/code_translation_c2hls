import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

SCRIPT=Path(__file__).parents[1]/'scripts/prepare_commercial_cosim_backfill.py'
spec=importlib.util.spec_from_file_location('commercial_prepare',SCRIPT)
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)


def evidence():
    selection={'record_id':'id','paper_source_join_status':'exact_selected_source_linked',
      'paper_join_checks':{'cycles':True,'source':True},'toolchain':{},
      'kernel_source':'k','kernel_sha256':'h','source_result_json':'r','source_result_sha256':'h',
      'history_json':'history','history_sha256':'h'}
    benchmark={'validation_protocol':prep.PROTOCOL,'size_overrides':{},
      'historical_testbench_identity':'unavailable_in_archive','prompt_inputs_match':True,
      'input_hashes':{'testbench.cpp':'hash'},'native_golden':{'success':True,'output_sha256':'gold'}}
    history={'history_sha256':'h','plain_matches_public_ignoring_outer_whitespace':True,
      'header_matches_public_ignoring_outer_whitespace':True}
    native={'record_id':'id','kernel_sha256':'h','input_hashes':benchmark['input_hashes'],
      'gold_output_sha256':'gold','status':'pass'}
    return selection,benchmark,history,native


@pytest.mark.parametrize('status',['pass','output_mismatch'])
def test_native_mismatch_is_retained_for_independent_vitis_csim_confirmation(status):
    s,b,h,n=evidence();n['status']=status
    prep.require_evidence(s,b,h,n,SimpleNamespace(file_sha=lambda _: 'h'),{})


@pytest.mark.parametrize('status',['compile_failed','timeout','missing_output'])
def test_preflight_infrastructure_uncertainty_cannot_be_ignored(status):
    s,b,h,n=evidence();n['status']=status
    with pytest.raises(ValueError,match='infrastructure error'):
        prep.require_evidence(s,b,h,n,SimpleNamespace(file_sha=lambda _: 'h'),{})


def test_do_not_substitute_small_input_for_paper_kernel():
    s,b,h,n=evidence();b['size_overrides']={'N':10}
    with pytest.raises(ValueError,match='Full-size'):
        prep.require_evidence(s,b,h,n,SimpleNamespace(file_sha=lambda _: 'h'),{})


def test_do_not_claim_recovered_historical_harness():
    s,b,h,n=evidence();b['historical_testbench_identity']='assumed_same'
    with pytest.raises(ValueError,match='Full-size'):
        prep.require_evidence(s,b,h,n,SimpleNamespace(file_sha=lambda _: 'h'),{})


def test_native_evidence_must_match_current_registered_source_and_inputs():
    s,b,h,n=evidence();n['input_hashes']={}
    with pytest.raises(ValueError,match='exact saved design'):
        prep.require_evidence(s,b,h,n,SimpleNamespace(file_sha=lambda _: 'h'),{})


def test_archived_report_is_not_enough_without_archived_source_match():
    s,b,h,n=evidence();s['paper_join_checks']['source']=False
    with pytest.raises(ValueError,match='not source/report linked'):
        prep.require_evidence(s,b,h,n,SimpleNamespace(file_sha=lambda _: 'h'),{})
