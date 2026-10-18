#!/usr/bin/env python3
"""Measure an unchanged public-C parent; no model or expert HLS input is used."""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import c2hls
import hls_eval
from experiment_telemetry import capture_tool_calls, tool_summary, tool_scope
from phase_b_manifest import SCHEMA_VERSION, canonical_json_sha256, text_sha256, toolchain_fingerprint
from validation_suite import use_validation_suite


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--benchmark',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.inputs.read_text())
    entry=next(e for e in manifest['entries'] if e['benchmark']==args.benchmark)
    inputs=c2hls._load_benchmark_inputs(entry['benchmark_dir']);meta=inputs['meta'];code=inputs['c_code']
    (out/'parent.cpp').write_text(code)
    started=datetime.now(timezone.utc).isoformat();result={}
    with capture_tool_calls() as events, use_validation_suite(args.inputs,args.benchmark) as suite:
        arguments=dict(hls_code=code,header_code=inputs['header_code'],header_name=inputs['header_name'],
                       top_function=meta['translated_hls_top'],part=meta['part'],clock_ns=meta['clock_ns'])
        with tool_scope('parent_construction'):
            result['csim']=hls_eval.run_csim(**arguments,testbench_code=inputs['testbench_code'])
            if result['csim']['passed']:
                result['synthesis']=hls_eval.run_hls_synthesis(**arguments)
            if result.get('synthesis',{}).get('success'):
                result['cosim']=hls_eval.run_cosim(**arguments,testbench_code=inputs['testbench_code'],interface_depths=meta.get('cosim_depths',{}))
        result['validation_manifest_sha256']=suite['manifest_sha256']
    result.update(benchmark=args.benchmark,started_at=started,finished_at=datetime.now(timezone.utc).isoformat(),
                  origin='unchanged_public_c',model_calls=0,tool_flows=tool_summary(events))
    result['success']=result.get('cosim',{}).get('passed') is True
    (out/'measurement.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['success']:
        csim=c2hls._summarize_test_result(result['csim'],True);report=result['synthesis']['report']
        parent={'benchmark':args.benchmark,'code_path':'parent.cpp','code_sha256':text_sha256(code),
                'input_c_sha256':entry['input_c_sha256'],'header_sha256':entry['header_sha256'],
                'csim':csim,'csim_sha256':canonical_json_sha256(csim),
                'csynth_report':report,'csynth_report_sha256':canonical_json_sha256(report),
                'source_artifacts':{'measurement_path':str(out/'measurement.json'),
                                    'measurement_sha256':text_sha256((out/'measurement.json').read_text()),
                                    'construction_policy':'unchanged_public_c; all five CSim and RTL cases required; no expert files'},
                'cosim':c2hls._summarize_test_result(result['cosim'],True)}
        payload={'schema_version':SCHEMA_VERSION,'toolchain':toolchain_fingerprint(vitis_version='2023.2',part=meta['part'],clock_ns=meta['clock_ns']),
                 'entries':{args.benchmark:parent}}
        (out/'phase_b_manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'benchmark':args.benchmark,'success':result['success'],'tool_counts':result['tool_flows']['counts'],
                      'rtl_cycles':result.get('cosim',{}).get('kernel_runtime_cycles')}),flush=True)
    return 0 if result['success'] else 1

if __name__=='__main__':raise SystemExit(main())
