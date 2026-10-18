#!/usr/bin/env python3
"""Execute all prespecified native CPU oracles and save hash-bound goldens."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import c2hls
from experiment_telemetry import capture_tool_calls,tool_summary


def validate_case(item):
    entry,case=item
    bench=Path(entry['benchmark_dir']);meta=json.loads((bench/'metadata.json').read_text())
    inputs={'meta':meta,'c_code':(bench/'plain.cpp').read_text(),
            'header_code':(bench/meta['header_file']).read_text(),'header_name':meta['header_file'],
            'testbench_code':Path(case['testbench_path']).read_text(),'extra_files':[]}
    with capture_tool_calls() as events:
        result=c2hls._prepare_independent_golden(inputs,shape_registry_path=case['shape_registry_path'])
    output=bench/f"golden_case{case['case_id']}.txt"
    if output.exists():raise ValueError('native output already exists; use a new input cohort')
    if result['success']:output.write_text(result['output'])
    return {'benchmark':entry['benchmark'],'case_id':case['case_id'],'success':result['success'],
            'error':result.get('error'),'provenance':result.get('provenance'),
            'golden_output_path':str(output) if result['success'] else None,
            'testbench_sha256':case['testbench_sha256'],
            'tool_flows':tool_summary(events)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('manifest',type=Path);args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text());out=args.manifest.parent/'native_validation.json'
    if out.exists():raise ValueError('native validation already recorded')
    cases=[(entry,case) for entry in manifest['entries'] for case in entry['cases']]
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(validate_case,cases))
    result={'finished_at':datetime.now(timezone.utc).isoformat(),'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            'cases':records,'passed':sum(r['success'] for r in records),'total':len(records),'model_calls':0,'hls_calls':0}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'total':len(records),'failures':[{'benchmark':r['benchmark'],'case_id':r['case_id'],'error':r['error']} for r in records if not r['success']]}))
    return 0 if result['passed']==len(records) else 1


if __name__=='__main__':raise SystemExit(main())
