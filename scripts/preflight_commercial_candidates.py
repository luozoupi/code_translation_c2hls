#!/usr/bin/env python3
"""Native-only checks of exact archived Figure 7 selections against CPU goldens."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_commercial_archive_inputs import native_run, sha, write
from golden_output import compare_hlsfactory_dumps


def verify_selected(row, bench):
    if row['paper_source_join_status'] != 'exact_selected_source_linked':
        raise ValueError('selected source is not linked to paper slot')
    if sha(row['kernel_source']) != row['kernel_sha256']:
        raise ValueError('archived selected source changed')
    for name, digest in bench['input_hashes'].items():
        if sha(Path(bench['staged_benchmark']) / name) != digest:
            raise ValueError('staged public input changed')
    golden = bench['native_golden']
    if not golden['success'] or sha(golden['output']) != golden['output_sha256']:
        raise ValueError('native golden missing or changed')


def prepare(inventory, audit, output, workers):
    rows = json.loads(inventory.read_text())
    benches = {r['benchmark']: r for r in json.loads(audit.read_text())['benchmarks']}
    output.mkdir(parents=True, exist_ok=False)
    def check(row):
        bench = benches[row['benchmark']]
        verify_selected(row, bench)
        case = hashlib.sha256(row['record_id'].encode()).hexdigest()[:24]
        result = native_run(Path(row['kernel_source']), Path(bench['staged_benchmark']), output / case)
        result.update(record_id=row['record_id'], benchmark=row['benchmark'],
                      paper_slots=row['priority_paper_slots'], kernel_sha256=row['kernel_sha256'],
                      input_hashes=bench['input_hashes'], gold_output_sha256=bench['native_golden']['output_sha256'])
        if result['success']:
            golden = Path(bench['native_golden']['output']).read_text()
            candidate = Path(result['output']).read_text()
            specs = {key: {'shape': value['shape'], 'kind': value['kind']} for key, value in bench['outputs'].items()}
            comparison = compare_hlsfactory_dumps(golden, candidate, specs=specs, default_atol=1e-6, default_rtol=1e-5).to_dict()
            result.update(comparison=comparison, success=comparison['passed'],
                          status='pass' if comparison['passed'] else 'output_mismatch')
        write(output / case / 'result.json', result)
        print(row['record_id'], result['status'], flush=True)
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(check, rows))
    report = {'schema_version': 'c2hls.commercial-candidate-native-preflight.v1',
        'created_at': datetime.now(timezone.utc).isoformat(), 'inventory': str(inventory),
        'inventory_sha256': sha(inventory), 'input_audit': str(audit), 'input_audit_sha256': sha(audit),
        'script_sha256': sha(__file__), 'native_runner_sha256': sha(Path(__file__).with_name('audit_commercial_archive_inputs.py')),
        'counts': dict(Counter(r['status'] for r in results)), 'rows': results,
        'comparison_atol': 1e-6, 'comparison_rtol': 1e-5,
        'interpretation': 'native preflight only; Vitis CSim and RTL verification are still required',
        'source_modified': False, 'historical_testbench_identity_proven': False, 'vitis_or_api_invoked': False}
    write(output / 'report.json', report)
    print(json.dumps(report['counts']), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    prepare(a.inventory.resolve(), a.audit.resolve(), a.output.resolve(), a.workers)


if __name__ == '__main__':
    main()
