#!/usr/bin/env python3
"""Freeze Figure 7/Table 7 retrieval identities; never invent runnable kernels.

This is a read-only source audit plus an export of exact published selections.
It deliberately emits an empty eligible_rows.json until saved implementation
bytes and their original benchmark inputs can be joined to these records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path('/home/luo00466')
PAPER = HOME / '6a3058ea0542c2fa6f168fe8'
HPCA = HOME / 'code_translation-c2hls-hpca2027'
BUNDLE = HPCA / 'results_sweeps/results-20260730'
ARMS = {'flash_noskills': ('flash', 'no_skills'),
        'flash_curated': ('flash', 'curated'),
        'flash_allpos': ('flash', 'all_positive'),
        'ms_noskills': ('multistep', 'no_skills'),
        'ms_curated': ('multistep', 'curated'),
        'ms_allpos': ('multistep', 'all_positive')}
SOURCES = {
    'sonnet': (HOME / 'schema_records_newskills_sonnet.jsonl',
               '2c72d00e5a7120bbdd234cecf06271c2fa3e58e0805e67b117072ff1e28140a9'),
    'opus': (BUNDLE / 'schema_records_newskills_opus.jsonl',
             'c83745a19bf1b94d8b24408f69d00eb3dbf273528b212e4340cddb030cd590ff'),
    'gpt55': (BUNDLE / 'schema_records_newskills_gpt55.jsonl',
              'b8e94ffb7b27de4eb95ea28759077e35214077bc04fbf90ba9013862cecfa608')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                   separators=(',', ':')).encode()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT /
                        'artifacts/sigmetrics2027/figure7_backfill_20261007')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    aligned_path = PAPER / 'generated/model_expertise_aligned.csv'
    aligned = {(r['model_label'], r['benchmark']): r for r in read_csv(aligned_path)}
    by_model = {
        'sonnet': ('Claude Sonnet 4.6', 'newskills_sonnet_snapshot.csv', 'benchmark',
                   'best_arm', 'best_cycles', 'best_noskill_arm', 'best_noskill_cycles'),
        'opus': ('Claude Opus 4.8', 'july30_opus_selection.csv', 'bench',
                 'selected_setup', 'selected_cycles', 'best_expertise_disabled_setup',
                 'best_expertise_disabled_cycles'),
        'gpt55': ('GPT-5.5', 'newskills_gpt55_snapshot.csv', 'benchmark',
                  'selected_setup', 'selected_cycles', 'best_no_skills_setup',
                  'best_no_skills_cycles')}
    rows, selected_schema, receipts, counts = [], [], [], {}
    july = json.loads((ROOT / 'results_sweeps/july_routing_cosim_20261006/inventory.json').read_text())
    # The original Sonnet schema digest is already pinned by its paper manifest.
    sonnet_manifest = json.loads((PAPER / 'generated/newskills_sonnet_manifest.json').read_text())
    source_meta = sonnet_manifest['sources']['schema_records']
    expected_sonnet = source_meta['sha256']
    for model, (label, snapshot, bk, sk, ck, ok, ock) in by_model.items():
        path, expected = SOURCES[model]
        if model == 'sonnet':
            expected = expected_sonnet
        if sha(path) != expected:
            raise ValueError(f'Published raw source checksum mismatch: {path}')
        schema = [json.loads(line) for line in path.read_text().splitlines()]
        index = {}
        for lineno, record in enumerate(schema, 1):
            key = (record['problem']['group_path'][0],
                   record['implementation']['origin_version'], record['report_type'])
            if key in index:
                raise ValueError(f'Duplicate schema identity: {key}')
            index[key] = (lineno, record)
        counts[model] = dict(Counter(x['report_type'] for x in schema))
        candidate_rtl = [r for r in schema if r['report_type'] == 'rtl_sim'
                         and r['implementation']['origin_version'] is not None]
        if candidate_rtl:
            raise ValueError('Candidate RTL evidence now exists: audit policy must be reconsidered')
        receipts.append({'path': str(path), 'sha256': sha(path)})
        snapshot_path = PAPER / 'generated' / snapshot
        receipts.append({'path': str(snapshot_path), 'sha256': sha(snapshot_path)})
        for selected in read_csv(snapshot_path):
            bench = selected[bk]
            plot = aligned[(label, bench)]
            for role, arm_key, cycle_key in [('selected', sk, ck), ('recipe_disabled', ok, ock)]:
                arm = selected[arm_key]
                mode, policy = ARMS[arm]
                origin = f'enh_newskills__{model}__{mode}__{policy}'
                syn_line, syn = index[(bench, origin, 'hls_synth')]
                sw_line, sw = index[(bench, origin, 'sw_run')]
                cycles = int(float(selected[cycle_key]))
                report = syn['hls_synth']
                if int(report['PerformanceEstimates']['SummaryOfOverallLatency']['Worst-caseLatency']) != cycles:
                    raise ValueError(f'Published selected cycles disagree with schema: {model}/{bench}/{arm}')
                if report['status'] != 'pass' or sw['sw_run']['status'] != 'pass':
                    raise ValueError('Selected source lacks reported CSim/CSynth pass')
                expected_plot_cycles = plot['selected_cycles'] if role == 'selected' else plot['expertise_off_cycles']
                if int(expected_plot_cycles) != cycles:
                    raise ValueError('Aligned figure does not use the expected snapshot selection')
                record_id = f'{model}/{bench}/{arm}'
                # Cycle equality is only a retrieval hint, not a source identity.
                numeric_leads = [x for x in july if model in x.get('campaign', '')
                                 and x.get('benchmark') == 'hlsfactory_' + bench.replace('-', '_')
                                 and x.get('csynth_cycles') == cycles]
                row = dict(slot_id=f'{model}/{bench}/{role}', record_id=record_id,
                    model=model, model_label=label, benchmark=bench, role=role,
                    setup=arm, mode=mode, routing_policy=policy, published_synthesis_cycles=cycles,
                    common_table_aggregate=plot['common_table_aggregate'] == 'True',
                    origin_version=origin, schema_source=str(path), schema_sha256=sha(path),
                    hls_synth_line=syn_line, sw_run_line=sw_line,
                    hls_synth_record_sha256=digest(syn), sw_run_record_sha256=digest(sw),
                    historical_csim_passed=True, independent_c_output_check='not_recorded',
                    candidate_rtl_recorded=False, recorded_run=syn['run'],
                    toolchain=dict(part=syn['run']['device'], vitis_version=syn['run']['vitis_version'],
                                   clock_ns=float(report['UserAssignments']['TargetClockPeriod']),
                                   flow_target=report['UserAssignments']['FlowTarget']),
                    synthesis_user_assignments=report['UserAssignments'],
                    kernel_source=None, kernel_sha256=None, benchmark_input_hashes=None,
                    queue_status='blocked', queue_reason='missing_selected_source_and_original_input_provenance',
                    dedup_status='not_possible_without_kernel_and_input_hashes',
                    numerical_only_july_leads=[{'record_id': x['record_id'],
                         'campaign': x['campaign'], 'csynth_cycles': x['csynth_cycles']}
                         for x in numeric_leads],
                    historical_launcher_project_lead='/mnt/e/courses/UMN/c2hls/code_translation_c2hls',
                    project_lead_is_verified_source_location=False,
                    requested_artifacts=[f'hlsfactory_{bench}_selected.cpp or exact saved final/generated C++',
                        'full result JSON/history associating selected source with this synthesis report',
                        f'benchmarks/hlsfactory_{bench}/metadata.json',
                        f'benchmarks/hlsfactory_{bench}/plain.cpp',
                        f'benchmarks/hlsfactory_{bench}/testbench.cpp',
                        'all original headers/support files and any cosim-specific harness',
                        'HLS Tcl/cflags/top/part/clock settings',
                        'CSim log and any independent output-comparison record',
                        'any existing RTL reports/logs with exact source and input hashes'])
                rows.append(row)
                selected_schema.append({'slot_id': row['slot_id'], 'synthesis': syn, 'csim': sw})
    if len(rows) != 156:
        raise ValueError(f'Expected 156 plotted teacher arm slots, found {len(rows)}')
    reported_rtl = []
    for model, filename in [('deepseek', 'deepseek_v4_flash_flash_csynth_skills_vs_noskills.csv'),
                             ('grok', 'grok45_flash_csynth_skills_vs_noskills.csv')]:
        path = BUNDLE / filename
        receipts.append({'path': str(path), 'sha256': sha(path)})
        for item in read_csv(path):
            for arm in ['skills', 'noskills']:
                reported_rtl.append(dict(model=model, benchmark=item['bench'], arm=arm,
                    cohort=item['set'], source=str(path), source_sha256=sha(path),
                    recorded_report_relative_path=item[f'{arm}_report'],
                    reported_rtl_pass=item[f'{arm}_flash_pass'],
                    reported_rtl_cycles=item[f'{arm}_flash_cosim_cycles'],
                    evidence='reported_RTL_summary_not_independently_replayed',
                    raw_artifacts_recovered=False))
    summary = dict(schema='c2hls.figure7-source-audit.v1', paper_items=['Figure 7', 'Table 7'],
        teacher_slots=len(rows), teacher_unique_selections=len({x['record_id'] for x in rows}),
        common_table_teacher_slots=sum(x['common_table_aggregate'] for x in rows),
        eligible_validations=0, blocked_teacher_slots=len(rows),
        teacher_source_report_counts=counts, compact_reported_rtl_slots=len(reported_rtl),
        source_files=receipts, plotted_data={'path': str(aligned_path), 'sha256': sha(aligned_path)},
        gate='Do not queue until saved selected source and its exact original benchmark inputs are recovered.',
        interpretation='Teacher CSim passes mean tool completion; 26 RTL records per teacher schema belong only to benchmark baselines.')
    dump(args.output / 'teacher_selected_slots.json', rows)
    dump(args.output / 'selected_schema_evidence.json', selected_schema)
    dump(args.output / 'reported_rtl_source_paths.json', reported_rtl)
    dump(args.output / 'eligible_rows.json', [])
    dump(args.output / 'summary.json', summary)
    fields = ['slot_id', 'record_id', 'model', 'benchmark', 'role', 'setup',
              'published_synthesis_cycles', 'common_table_aggregate', 'origin_version',
              'schema_source', 'schema_sha256', 'hls_synth_line', 'sw_run_line',
              'hls_synth_record_sha256', 'queue_status', 'queue_reason']
    with (args.output / 'teacher_selected_slots.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({key: summary[key] for key in ['teacher_slots', 'teacher_unique_selections',
        'common_table_teacher_slots', 'eligible_validations', 'compact_reported_rtl_slots']}, indent=2))


if __name__ == '__main__':
    main()
