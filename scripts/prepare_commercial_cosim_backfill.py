#!/usr/bin/env python3
"""Register archived commercial selections for new public-harness validation.

The historical CSim harness was not archived. This adapter never invents its
identity: every row states that the saved source is being revalidated using a
new full-size public harness matched to archived prompt plain/header inputs.
Native preflight mismatches remain eligible for independent Vitis CSim checks;
only the unchanged worker's successful Vitis CSim gate permits RTL co-simulation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'new_independent_full_size_public_testbench_validation'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def require_evidence(selected, benchmark, history, native, july, toolchain):
    if selected['paper_source_join_status'] != 'exact_selected_source_linked' or not all(selected['paper_join_checks'].values()):
        raise ValueError('Paper selection not source/report linked: '+selected['record_id'])
    if selected['toolchain'] != toolchain:
        raise ValueError('Toolchain differs from frozen validator')
    for key, digest in [('kernel_source','kernel_sha256'),('source_result_json','source_result_sha256'),('history_json','history_sha256')]:
        if july.file_sha(selected[key]) != selected[digest]:
            raise ValueError('Archived source evidence changed: '+key)
    if (benchmark['validation_protocol'] != PROTOCOL or benchmark['size_overrides']
            or benchmark['historical_testbench_identity'] != 'unavailable_in_archive'
            or benchmark['prompt_inputs_match'] is not True
            or benchmark['native_golden']['success'] is not True):
        raise ValueError('Full-size independent public input audit is incomplete')
    if (history['history_sha256'] != selected['history_sha256']
            or history['plain_matches_public_ignoring_outer_whitespace'] is not True
            or history['header_matches_public_ignoring_outer_whitespace'] is not True):
        raise ValueError('Archived prompt inputs differ from registered public inputs')
    if (native['record_id'] != selected['record_id'] or native['kernel_sha256'] != selected['kernel_sha256']
            or native['input_hashes'] != benchmark['input_hashes']
            or native['gold_output_sha256'] != benchmark['native_golden']['output_sha256']):
        raise ValueError('Native preflight does not describe this exact saved design and harness')
    # A mismatch is evidence to retain; it must not silently alter the selection
    # or numerical tolerance. Vitis CSim independently confirms or rejects it.
    if native['status'] not in ('pass', 'output_mismatch'):
        raise ValueError('Native preflight has an unresolved infrastructure error')


def adapt(selected, benchmark, native, source):
    return dict(record_id=selected['record_id'], benchmark=selected['benchmark'], kind='agentic',
        campaign=selected['source_grid_identity'], model=selected['model'], strategy=selected['mode'],
        skill_mode=selected['setup'], source_setup=selected['setup'], selected_step='archived_selected',
        source_summary=str(source), source_result=selected['source_result_json'],
        source_result_sha256=selected['source_result_sha256'], history_json=selected['history_json'],
        history_sha256=selected['history_sha256'], kernel_sha256=selected['kernel_sha256'],
        staged_kernel=selected['kernel_source'], staged_benchmark=benchmark['staged_benchmark'],
        input_hashes=benchmark['input_hashes'], golden_registry=benchmark['golden_registry'],
        source_evidence='archived_selected_source_and_synthesis_report_linked_to_paper',
        validation_protocol=PROTOCOL, historical_testbench_identity='unavailable_in_archive',
        historical_independent_output_check=selected['independent_historical_output_check'],
        historical_reference_audit_passed=None, historical_validation_fingerprint=None,
        current_preflight_native_golden_output_sha256=benchmark['native_golden']['output_sha256'],
        native_candidate_preflight={k:native[k] for k in ('status','success','comparison','output_sha256')},
        published_synthesis_cycles=selected['published_synthesis_cycles'],
        archived_reported_synthesis_cycles=selected['reported_synthesis_cycles'],
        paper_slots=selected['priority_paper_slots'], existing_cosim=None)


def prepare(source, inputs, native_path, output, raw, python):
    if output.exists() or raw.exists():
        raise ValueError('New output and raw paths are required')
    july = module('commercial_adapter_july_utilities', ROOT/'scripts/prepare_july_routing_backfill.py')
    selected = july.read(source)
    audit = july.read(inputs)
    native_report = july.read(native_path)
    if audit['issues'] or audit['original_harness_not_archived'] is not True:
        raise ValueError('Public input provenance is unresolved')
    benchmarks = {r['benchmark']:r for r in audit['benchmarks']}
    histories = {r['history']:r for r in audit['history_input_checks']}
    native = {r['record_id']:r for r in native_report['rows']}
    if len(native) != len(native_report['rows']) or set(native) != {r['record_id'] for r in selected}:
        raise ValueError('Candidate preflight denominator differs from selected inventory')
    reference_path = ROOT/'results_sweeps/historical_registry_recovery_20260928/manifest.json'
    reference = july.read(reference_path)
    frozen = ROOT/'artifacts/sigmetrics2027/registry_recovery_20260928/source'
    july.verify_files(frozen, reference['code_hashes'])
    existing = july.default_existing(july.HPCA/'results_sweeps/july_cosim_backfill_20260928/manifest.json', ROOT)
    existing.append(ROOT/'results_sweeps/july_routing_cosim_20261006/manifest.json')
    covered, receipts, issues = july.coverage_index(existing, frozen)
    if issues:
        raise ValueError('Existing semantic coverage audit failed: '+str(issues[:3]))
    inventory = []
    for selection in selected:
        bench = benchmarks[selection['benchmark']]
        check = native[selection['record_id']]
        require_evidence(selection, bench, histories[selection['history_json']], check, july, reference['toolchain'])
        row = adapt(selection, bench, check, source)
        registry, key = july.audit_row(row, reference['toolchain'], frozen, require_result=True)
        row.update(golden_registry=registry, semantic_identity_sha256=key)
        if key in covered:
            row.update(queue_status='existing_evidence', existing_measurements=covered[key])
        else:
            row.update(queue_status='queued')
        inventory.append(row)
    output.mkdir(parents=True)
    raw.mkdir(parents=True)
    for folder in ('logs','measurements','vitis_archives'):
        (raw/folder).mkdir()
    rows, unique = [], {}
    for source_row in inventory:
        if source_row['queue_status'] != 'queued':
            continue
        row = copy.deepcopy(source_row)
        fingerprint = july.digest(dict(schema_version='c2hls.commercial-public-harness-validation.v1',
            semantic_identity=july.semantic_identity(row, reference['toolchain'], row['golden_registry']),
            evaluator_hashes=reference['code_hashes'], validation_protocol=PROTOCOL))
        vid = july.digest(dict(campaign=output.name, validation_fingerprint=fingerprint))
        row.update(validation_id=vid, current_validation_fingerprint=fingerprint,
            inventory_status='queued', staged_kernel=str(raw/'inputs'/vid/'kernel.cpp'),
            staged_benchmark=str(raw/'inputs'/vid/'benchmark'))
        source_row['validation_id'] = vid
        if vid not in unique:
            destination = raw/'inputs'/vid
            (destination/'benchmark').mkdir(parents=True)
            shutil.copyfile(source_row['staged_kernel'], destination/'kernel.cpp')
            for name in row['input_hashes']:
                relative = july.safe_relative(name)
                target = destination/'benchmark'/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(source_row['staged_benchmark'])/relative, target)
            july.audit_row(row, reference['toolchain'], frozen, require_result=True)
            unique[vid] = row
        rows.append(row)
    created = datetime.now(timezone.utc).isoformat()
    ordered = sorted(unique.items(), key=lambda item: (item[1]['benchmark'], item[1]['model'], item[1]['record_id']))
    tasks = [dict(validation_id=vid, benchmark=row['benchmark'],status='pending') for vid,row in ordered]
    manifest = dict(schema_version='c2hls.commercial-public-harness-backfill.v1', created_at=created,
        campaign=output.name, rows=rows, code_hashes=reference['code_hashes'], toolchain=reference['toolchain'],
        raw_root=str(raw), frozen_worker=str(frozen/'scripts/run_historical_cosim.py'), frozen_source=str(frozen),
        timeout_seconds=3600, csim_timeout_seconds=180, validation_protocol=PROTOCOL,
        source_inventory=str(source), source_inventory_sha256=july.file_sha(source),
        public_input_audit=str(inputs), public_input_audit_sha256=july.file_sha(inputs),
        native_candidate_preflight=str(native_path), native_candidate_preflight_sha256=july.file_sha(native_path),
        prior_manifest_receipts=receipts, policy=dict(priority='Only Figure 7/Table 7 selected sources and recipe-disabled comparators; fixed source selections.',
          timeout_scope='3600 seconds for the frozen csynth_design plus cosim_design invocation; preceding Vitis CSim independently capped at 180 seconds',
          historical_testbench='Unavailable; do not claim exact historical-harness replay',
          validation='New independent full-size public harness, archived prompt plain/header matched; unchanged native-golden -> Vitis CSim -> RTL gate',
          native_mismatches='Retain all selections, no source or tolerance changes; RTL only after independent Vitis CSim passes.',
          generation=False, model_calls=0, performance_selection=False, retry_existing_outcomes=False,
          order='Lexicographic benchmark, model, record identity; no synthesis-performance sorting'))
    july.write_json(output/'manifest.json', manifest)
    july.write_json(output/'inventory.json', inventory)
    july.write_csv(output/'inventory.csv', inventory)
    july.write_json(output/'input_audit.json', dict(created_at=created, validation_protocol=PROTOCOL,
        selected_rows=len(selected), queued_rows=len(rows), distinct_validations=len(tasks),
        status_counts=dict(Counter(r['queue_status'] for r in inventory)), native_candidate_counts=native_report['counts'],
        public_input_audit=str(inputs), public_input_audit_sha256=july.file_sha(inputs),
        native_candidate_audit=str(native_path), native_candidate_audit_sha256=july.file_sha(native_path),
        coverage_issues=issues, existing_semantic_identities=len(covered), launches_performed=0))
    owned_source = output/'source/scripts'
    owned_source.mkdir(parents=True)
    for path in (Path(__file__), ROOT/'scripts/queue_figure7_cosim_backfill.py',
                 ROOT/'scripts/prepare_july_routing_backfill.py', ROOT/'scripts/resume_sigmetrics_recipe_main.py'):
        shutil.copyfile(path, owned_source/path.name)
    pinned = [source,inputs,native_path,reference_path,output/'manifest.json',output/'input_audit.json']
    pinned.extend(owned_source.iterdir())
    pinned.extend(Path(r['history_json']) for r in rows)
    config = dict(schema_version='c2hls.commercial-selected-queue.v1', created_at=created,
        campaign=str(output), python=str(python), python_sha256=july.file_sha(python),
        source=str(owned_source.parent), source_runner=str(owned_source/'queue_figure7_cosim_backfill.py'),
        owned_process_helper=str(owned_source/'resume_sigmetrics_recipe_main.py'),
        wait_for_recovery=str(ROOT/'results_sweeps/sigmetrics_timeout_recovery_20261006'), workers=1,
        outer_worker_timeout_seconds=5400, validation_protocol=PROTOCOL,
        policy='Prepared child for registered combined scheduling; no separate launcher.',
        pinned_files={str(p):july.file_sha(p) for p in set(pinned)})
    july.write_json(output/'queue_manifest.json',config)
    state = dict(status='queued',queued_at=created,updated_at=created,tasks=tasks,failure=None,
                 queue_manifest_sha256=july.file_sha(output/'queue_manifest.json'))
    july.write_json(output/'state.json',state)
    summary = dict(status='queued',updated_at=created,selected_rows=len(selected),distinct_validations=len(tasks),
        counts={'pending':len(tasks)}, model_calls=0,native_candidate_counts=native_report['counts'],validation_protocol=PROTOCOL)
    july.write_json(output/'summary.json',summary)
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','inputs','native','output','raw-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    args=parser.parse_args()
    prepare(args.source.resolve(),args.inputs.resolve(),args.native.resolve(),args.output.resolve(),args.raw_root.resolve(),args.python.resolve())


if __name__=='__main__':
    main()
