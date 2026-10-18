#!/usr/bin/env python3
"""Version saved validation identities without replacing historical evidence."""
from __future__ import annotations
import argparse
import copy
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from hlsfactory_registry import registry_identity
from scripts.scan_historical_cosim import atomic, digest, file_sha, frozen_evaluator_hashes, validation_identity
from scripts.run_historical_cosim import verify


def prepare(source, output, raw_root, mode):
    if output.exists() or raw_root.exists():
        raise ValueError('use new output and raw directories; historical artifacts must not be overwritten')
    original = json.loads(source.read_text())
    candidates = []
    for row in original['rows']:
        if row['inventory_status'] != 'queued':
            continue
        bench = Path(row['staged_benchmark'])
        meta = json.loads((bench / 'metadata.json').read_text())
        contract = registry_identity(meta, (bench / (meta.get('testbench_file') or 'testbench.cpp')).read_text(), environ={})
        imported = contract['path'].endswith('_fc27133.json')
        if mode == 'registry' and not imported:
            continue
        if mode == 'pilot' and row['benchmark'] != 'hlsfactory_atax':
            continue
        candidates.append((imported, row, contract))
    if mode == 'pilot':
        # Engineering smoke selection, not an efficacy sample: one smallest
        # saved ATAX design per harness, with stable identity as tie-breaker.
        candidates = [min((item for item in candidates if item[0] == cohort),
                          key=lambda item: (item[1]['csynth_cycles'], item[1]['validation_id']))
                      for cohort in (False, True)]
    rows, preflight = [], []
    for _, source_row, contract in candidates:
        row = copy.deepcopy(source_row)
        row['previous_validation_id'] = row['validation_id']
        row['previous_manifest_sha256'] = file_sha(source)
        row['golden_registry'] = contract
        row['validation_id'] = digest(validation_identity(row['kernel_sha256'], row['input_hashes'], contract))
        assert row['validation_id'] != row['previous_validation_id']
        row['existing_cosim'] = None
        verify(row)
        rows.append(row)
        if mode == 'pilot':
            from paper_baselines import load_public_benchmark_inputs
            inputs = load_public_benchmark_inputs(row['staged_benchmark'], environ={
                'C2HLS_PART': original['toolchain']['part'], 'C2HLS_CLOCK_NS': str(original['toolchain']['clock_ns'])})
            provenance = inputs.independent_golden_provenance
            if provenance['shape_registry']['sha256'] != contract['sha256']:
                raise ValueError('native preflight used a different registry')
            preflight.append({'validation_id': row['validation_id'], 'registry': contract,
                              'native_golden': provenance, 'model_calls': 0, 'hls_calls': 0})
    if not rows:
        raise ValueError('no eligible rows')
    manifest = {**{k: v for k, v in original.items() if k not in ('rows', 'code_hashes', 'raw_root', 'subset')},
                'schema_version': 'c2hls.historical-cosim-backfill.v2',
                'created_at': datetime.now(timezone.utc).isoformat(),
                'source_manifest': str(source), 'source_manifest_sha256': file_sha(source),
                'mode': mode, 'preparation_script_sha256': file_sha(Path(__file__)),
                'subset': {'selection': 'one smallest-CSYNTH ATAX per harness' if mode == 'pilot' else 'all imported-registry queued rows',
                           'efficacy_sample': False},
                'raw_root': str(raw_root), 'rows': rows, 'code_hashes': frozen_evaluator_hashes()}
    raw_root.mkdir(parents=True)
    for name in ('measurements', 'logs', 'vitis_archives'):
        (raw_root / name).mkdir()
    atomic(output / 'manifest.json', manifest)
    atomic(output / 'native_preflight.json', preflight)
    atomic(output / 'migration_summary.json', {'rows': len(rows), 'distinct_validations': len({r['validation_id'] for r in rows}),
           'cohorts': dict(Counter(r['golden_registry']['path'] for r in rows)), 'source_artifacts_modified': False})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--mode', choices=['pilot', 'registry'], required=True)
    args = parser.parse_args()
    manifest = prepare(args.source.resolve(), args.output.resolve(), args.raw_root.resolve(), args.mode)
    print(json.dumps({'output': str(args.output), 'rows': len(manifest['rows']),
                      'distinct_validations': len({r['validation_id'] for r in manifest['rows']})}))


if __name__ == '__main__':
    main()
