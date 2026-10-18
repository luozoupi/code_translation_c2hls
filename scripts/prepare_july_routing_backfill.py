#!/usr/bin/env python3
"""Prepare a provenance-checked July routing COSIM queue; never launch tools.

Old outcomes, including unsuccessful or not-run validations, are evidence to
retain, not invitations to retry. A separate scheduler owns execution.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HPCA = Path('/home/luo00466/code_translation-c2hls-hpca2027')
CAMPAIGN = 'july_routing_cosim_20261006'
REGISTRY_FILES = ('configs/hlsfactory_output_shapes.json',
                  'configs/hlsfactory_output_shapes_fc27133.json')


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(
        value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def safe_relative(name):
    p = Path(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p) == '.':
        raise ValueError(f'unsafe input path: {name}')
    return p


def verify_files(root, hashes):
    for name, sha in hashes.items():
        path = Path(root) / safe_relative(name)
        if file_sha(path) != sha:
            raise ValueError(f'input hash mismatch: {path}')


def registry_contract(row, frozen_source):
    bench = Path(row['staged_benchmark'])
    meta = read(bench / 'metadata.json')
    tb_name = meta.get('testbench_file') or 'testbench.cpp'
    safe_relative(tb_name)
    actual = file_sha(bench / tb_name)
    if row['input_hashes'].get(tb_name) != actual:
        raise ValueError('testbench differs from historical input hash')
    matches = []
    for name in REGISTRY_FILES:
        p = frozen_source / name
        entry = read(p).get('benchmarks', {}).get(str(meta.get('benchmark')))
        if isinstance(entry, dict) and entry.get('testbench_sha256') == actual:
            matches.append(dict(path=name, sha256=file_sha(p), testbench_sha256=actual))
    if len(matches) != 1:
        raise ValueError(f'expected one audited registry contract; found {len(matches)}')
    recorded = row.get('golden_registry')
    if recorded is not None and recorded != matches[0]:
        raise ValueError('recorded golden registry disagrees with exact harness contract')
    return matches[0]


def semantic_identity(row, toolchain, registry):
    """Policy labels and paths do not change a saved design's validation task."""
    return dict(code_sha256=row['kernel_sha256'], inputs=row['input_hashes'],
                toolchain=toolchain, golden_registry=registry)


def audit_row(row, toolchain, frozen_source, *, require_result):
    verify_files(row['staged_benchmark'], row['input_hashes'])
    if file_sha(row['staged_kernel']) != row['kernel_sha256']:
        raise ValueError('staged kernel hash mismatch')
    if require_result:
        result = Path(row.get('resolved_result') or row['source_result'])
        expected = row.get('source_result_sha256')
        if not expected or file_sha(result) != expected:
            raise ValueError(f'source result missing or changed: {result}')
    registry = registry_contract(row, frozen_source)
    key = digest(semantic_identity(row, toolchain, registry))
    return registry, key


def routing_labels(row):
    """Retain exact labels; early skill_on labels need their campaign suffix."""
    mode, campaign = row.get('skill_mode'), row.get('campaign', '')
    policy = mode
    if mode in ('skill_on', 'skill_off'):
        for tag in ('selective_positive', 'all_positive', 'none'):
            if f'_{tag}_' in campaign:
                policy = tag
                break
    sample = re.search(r'_sample(\d+)$', str(mode))
    return dict(routing_policy=policy, historical_skill_mode=mode,
                historical_sample_index=int(sample.group(1)) if sample else None,
                repeat_interpretation='recorded dose sample index' if sample else
                    'historical campaign cell; no repeat inferred')


def coverage_index(manifest_paths, frozen_source):
    index = defaultdict(list)
    receipts, issues = [], []
    for path in dict.fromkeys(Path(p).resolve() for p in manifest_paths):
        data = read(path)
        receipts.append(dict(path=str(path), sha256=file_sha(path)))
        raw = Path(data['raw_root'])
        seen = set()
        for row in data.get('rows', []):
            vid = row.get('validation_id')
            measurement_path = raw / 'measurements' / (str(vid) + '.json')
            embedded = row.get('existing_cosim')
            # Registered replay jobs also reserve identities even before a result.
            pending_other = (path.parent.name != 'july_cosim_backfill_20260928'
                             and row.get('inventory_status') == 'queued')
            if not measurement_path.is_file() and not embedded and not pending_other:
                continue
            marker = (vid, row.get('record_id'))
            if marker in seen:
                continue
            seen.add(marker)
            try:
                registry, key = audit_row(row, data['toolchain'], frozen_source,
                                          require_result=False)
                measurement = read(measurement_path) if measurement_path.is_file() else {}
                measure = measurement.get('cosim', embedded or {})
                if measurement and (measurement.get('kernel_sha256') != row['kernel_sha256']
                                    or measurement.get('validation_id') != vid):
                    raise ValueError('prior measurement identity mismatch')
                ref = dict(manifest=str(path), manifest_sha256=receipts[-1]['sha256'],
                           validation_id=vid, status=measure.get('status', 'reserved_in_existing_queue'),
                           source_record_id=row.get('record_id'),
                           measurement=str(measurement_path) if measurement else None,
                           measurement_sha256=file_sha(measurement_path) if measurement else None,
                           validation_scope=measure.get('validation_scope', 'historical_embedded'),
                           registry=registry)
                if ref not in index[key]:
                    index[key].append(ref)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                issues.append(dict(manifest=str(path), validation_id=vid,
                                   record_id=row.get('record_id'), error=str(exc)))
    return index, receipts, issues


def classify(row, covered, toolchain, frozen_source):
    rec = copy.deepcopy(row)
    rec.update(routing_labels(row))
    rec['prior_inventory_status'] = row.get('inventory_status')
    rec['queue_status'] = 'excluded'
    if row.get('kind') != 'agentic':
        rec['queue_reason'] = 'outside_scope_non_agentic'
    elif row.get('inventory_status') == 'blocked':
        rec['queue_status'], rec['queue_reason'] = 'blocked', row.get('reason') or 'historical_provenance_block'
    elif row.get('historical_csim_passed') is not True:
        rec['queue_reason'] = 'historical_csim_pass_not_recorded'
    elif row.get('inventory_status') not in ('queued', 'existing_cosim', 'predicted_longer_than_gold'):
        rec['queue_reason'] = 'historical_status_not_eligible'
    else:
        try:
            registry, key = audit_row(row, toolchain, frozen_source, require_result=True)
            rec.update(golden_registry=registry, semantic_identity_sha256=key,
                       source_and_inputs_verified=True)
            prior = covered.get(key, [])
            if prior:
                rec.update(queue_status='existing_evidence', queue_reason='semantic_identity_already_recorded',
                           existing_measurements=prior)
            elif row.get('existing_cosim'):
                rec.update(queue_status='existing_evidence', queue_reason='attributable_historical_cosim',
                           existing_measurements=[dict(status=row['existing_cosim'].get('status'),
                                                      source_result=row['source_result'])])
            else:
                rec.update(queue_status='queued', queue_reason='previously_csim_passed_selected_kernel_without_cosim')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rec.update(queue_status='blocked', queue_reason='input_audit: ' + str(exc))
    return rec


def stage_rows(inventory, raw, toolchain, code_hashes, campaign):
    staged, identities = [], {}
    for source in inventory:
        if source['queue_status'] != 'queued':
            continue
        key = source['semantic_identity_sha256']
        frozen_identity = dict(schema_version='c2hls.historical-validation-identity.v2',
                               **semantic_identity(source, toolchain, source['golden_registry']),
                               evaluator_hashes=code_hashes)
        frozen_sha = digest(frozen_identity)
        vid = digest(dict(schema_version='c2hls.july-routing-validation.v1',
                          campaign=campaign, frozen_validation_fingerprint=frozen_sha))
        source['new_validation_id'] = vid
        source['frozen_validation_fingerprint'] = frozen_sha
        row = copy.deepcopy(source)
        row.update(previous_validation_id=source.get('validation_id'), validation_id=vid,
                   inventory_status='queued', existing_cosim=None,
                   staged_kernel=str(raw / 'inputs' / vid / 'kernel.cpp'),
                   staged_benchmark=str(raw / 'inputs' / vid / 'benchmark'))
        if key not in identities:
            dest = raw / 'inputs' / vid
            (dest / 'benchmark').mkdir(parents=True)
            shutil.copyfile(source['staged_kernel'], dest / 'kernel.cpp')
            for name in source['input_hashes']:
                target = dest / 'benchmark' / safe_relative(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(source['staged_benchmark']) / name, target)
            verify_files(dest / 'benchmark', source['input_hashes'])
            if file_sha(dest / 'kernel.cpp') != source['kernel_sha256']:
                raise ValueError('copy changed kernel')
            identities[key] = vid
        elif identities[key] != vid:
            raise ValueError('inconsistent validation identity')
        staged.append(row)
    return staged


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def default_existing(source, repo):
    paths = [source, repo / 'results_sweeps/historical_registry_recovery_20260928/manifest.json',
             repo / 'results_sweeps/reviewer_ab_rtl_r1_20261001/manifest.json']
    for name in ('sigmetrics_timeout_replay_20261001', 'sigmetrics_timeout_recovery_20261006'):
        for p in sorted((repo / 'results_sweeps' / name).rglob('manifest.json')):
            if read(p).get('rows'):
                paths.append(p)
    return paths


def prepare(source, output, raw, frozen_source, fingerprint_manifest, existing):
    if output.exists() or raw.exists():
        raise ValueError('new output and raw directories are required; never overwrite a prepared campaign')
    original, reference = read(source), read(fingerprint_manifest)
    code_hashes = reference['code_hashes']
    verify_files(frozen_source, code_hashes)
    worker = frozen_source / 'scripts/run_historical_cosim.py'
    if original['toolchain'] != reference['toolchain']:
        raise ValueError('historical and frozen evaluator toolchains differ')
    covered, coverage_receipts, coverage_issues = coverage_index(existing, frozen_source)
    # A missing old artifact must not silently create an apparent new candidate.
    if coverage_issues:
        raise ValueError('existing coverage audit failed: ' + json.dumps(coverage_issues[:5]))
    inventory = [classify(row, covered, original['toolchain'], frozen_source)
                 for row in original['rows']]
    output.mkdir(parents=True)
    raw.mkdir(parents=True)
    for name in ('logs', 'measurements', 'vitis_archives'):
        (raw / name).mkdir()
    rows = stage_rows(inventory, raw, original['toolchain'], code_hashes, output.name)
    unique = {}
    for row in rows:
        unique.setdefault(row['validation_id'], row)
    created = datetime.now(timezone.utc).isoformat()
    # Stable content-hash order avoids ranking the interim sample by synthesis cycles.
    tasks = [dict(validation_id=vid, benchmark=row['benchmark'], status='pending')
             for vid, row in sorted(unique.items())]
    manifest = dict(schema_version='c2hls.historical-cosim-backfill.v2', created_at=created,
        campaign=output.name, rows=rows, code_hashes=code_hashes, toolchain=original['toolchain'],
        raw_root=str(raw), source_manifest=str(source), source_manifest_sha256=file_sha(source),
        date_window=original.get('date_window'), frozen_worker=str(worker),
        frozen_worker_sha256=file_sha(worker), frozen_source=str(frozen_source),
        evaluator_fingerprint_manifest=str(fingerprint_manifest),
        evaluator_fingerprint_manifest_sha256=file_sha(fingerprint_manifest),
        preparation_script_sha256=file_sha(Path(__file__)), prior_manifest_receipts=coverage_receipts,
        timeout_seconds=3600, csim_timeout_seconds=180,
        policy=dict(validation='independent native golden, candidate CSim, candidate RTL COSIM',
            timeout_scope='csynth_design plus cosim_design wall clock per frozen worker invocation',
            candidate_generation=False, historical_outputs_modified=False,
            retry_existing_outcomes=False, predicted_cycle_skips=False,
            cycle_skip_note='Direct frozen worker has no CSYNTH cycle-ratio skip; all eligible saved kernels are retained.',
            deduplication='kernel, all public input hashes, toolchain, exact registry contract',
            order='lexicographic new validation identity; no performance selection',
            historical_reference_isolation_flags='retained; COSIM does not cure generation leakage'))
    counts = Counter(row['queue_status'] for row in inventory)
    summary = dict(schema_version='c2hls.july-routing-backfill-summary.v1', prepared_at=created,
        status='prepared', historical_rows=len(inventory), inventory_status_counts=dict(counts),
        queued_rows=len(rows), distinct_validations=len(tasks), completed=0,
        queued_by_routing_policy=dict(Counter(r['routing_policy'] for r in rows)),
        queued_by_campaign=dict(Counter(r['campaign'] for r in rows)),
        queued_by_strategy=dict(Counter(r['strategy'] for r in rows)),
        reference_isolation_audit_counts=dict(Counter(str(r.get('historical_reference_audit_passed')) for r in rows)),
        launches_performed=0, model_calls=0, timeout_seconds=3600, csim_timeout_seconds=180)
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'state.json', dict(status='prepared', prepared_at=created, tasks=tasks))
    write_json(output / 'summary.json', summary)
    write_json(output / 'inventory.json', inventory)
    write_csv(output / 'inventory.csv', inventory)
    write_json(output / 'input_audit.json', dict(created_at=created, source_manifest_sha256=file_sha(source),
        frozen_evaluator_files_verified=len(code_hashes), prior_manifest_receipts=coverage_receipts,
        prior_coverage_identities=len(covered), coverage_issues=coverage_issues,
        queued_rows_source_and_input_verified=len(rows), staged_unique_designs_verified=len(tasks),
        inventory_sha256=file_sha(output / 'inventory.json'), manifest_sha256=file_sha(output / 'manifest.json')))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=HPCA / 'results_sweeps/july_cosim_backfill_20260928/manifest.json')
    parser.add_argument('--output', type=Path, default=REPO / 'results_sweeps' / CAMPAIGN)
    parser.add_argument('--raw-root', type=Path, default=REPO / 'artifacts/sigmetrics2027' / CAMPAIGN / 'raw')
    parser.add_argument('--frozen-source', type=Path, default=REPO / 'artifacts/sigmetrics2027/registry_recovery_20260928/source')
    parser.add_argument('--fingerprint-manifest', type=Path, default=REPO / 'results_sweeps/historical_registry_recovery_20260928/manifest.json')
    parser.add_argument('--existing-manifest', type=Path, action='append')
    args = parser.parse_args()
    summary = prepare(args.source.resolve(), args.output.resolve(), args.raw_root.resolve(),
        args.frozen_source.resolve(), args.fingerprint_manifest.resolve(),
        args.existing_manifest or default_existing(args.source, REPO))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
