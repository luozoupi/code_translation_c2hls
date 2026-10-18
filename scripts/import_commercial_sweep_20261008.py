#!/usr/bin/env python3
"""Safely import a hash-manifested commercial-results archive and audit selections."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
from collections import Counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path('/home/luo00466/code_translation-c2hls-hpca2027/c2hls_newskills_commercial_sweep.tar.gz')
OUTPUT = ROOT / 'artifacts/sigmetrics2027/commercial_sweep_20261008/import'
ARCHIVE_ROOT = 'c2hls_newskills_commercial_sweep'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def content_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def candidate_binding(result, code, history):
    """Require code plus its selected report; numerical equality alone is insufficient."""
    report = result.get('final_report') or result.get('synth_report') or {}
    if not code:
        return {'status': 'missing_source', 'matches': []}
    if not isinstance(report.get('latency_cycles'), (int, float)) or report['latency_cycles'] <= 0:
        return {'status': 'source_report_unresolved', 'matches': []}
    matches = []
    if result.get('hls_code') == code:
        matches.append({'location': 'hls_code', 'equality': 'byte_exact',
                        'csim': result.get('csim'), 'report_equal': True})
    for key in ['generated_step_history', 'optimization_history']:
        for i, step in enumerate(result.get(key, [])):
            if step.get('code') == code and step.get('report') == report:
                matches.append({'location': f'{key}[{i}]', 'equality': 'byte_exact',
                                'step_name': step.get('step_name'), 'success': step.get('success'),
                                'csim': step.get('csim'), 'report_equal': True})
    # Promoted Phase-B baselines are not copied into generated_step_history.code.
    # Require the explicit baseline promotion, identical selected/baseline report,
    # and exact assistant code apart from surrounding Markdown whitespace.
    promotion = result.get('best_so_far_promotion') or {}
    if (not matches and promotion.get('promoted') is True and
            promotion.get('from_step_index') == -1 and promotion.get('from_step_name') == 'baseline'
            and result.get('baseline_report') == report):
        for i, message in enumerate(history.get('messages', [])):
            if message.get('role') != 'assistant':
                continue
            blocks = re.findall(r'```(?:cpp|c\+\+|c)?\s*\n(.*?)```', message.get('content', ''), re.S)
            for j, block in enumerate(blocks):
                if block.strip() == code.strip():
                    matches.append({'location': f'history.messages[{i}].code_block[{j}]',
                                    'equality': 'outer_whitespace_only', 'step_name': 'baseline',
                                    'explicit_baseline_promotion': True,
                                    'csim': result.get('baseline_csim'), 'report_equal': True})
    verified = any((m.get('csim') or {}).get('passed') is True for m in matches)
    return {'status': 'source_report_and_csim_linked' if verified else
            'source_report_linked_csim_unconfirmed' if matches else 'source_report_unresolved',
            'matches': matches}


def csv_dump(path, rows, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def audit_import(output):
    receipt = json.loads((output / 'import_receipt.json').read_text())
    payload = Path(receipt['payload_root'])
    slots_path = ROOT / 'artifacts/sigmetrics2027/figure7_backfill_20261007/teacher_selected_slots.json'
    slots = json.loads(slots_path.read_text())
    all_rows, by_id = [], {}
    for campaign in sorted((payload / 'results').iterdir()):
        match = re.fullmatch(r'results_matrix_u280_ENH_newskills_(.+)_(SONNET|OPUS|GPT55)', campaign.name)
        if not match:
            raise ValueError(f'Unknown commercial campaign: {campaign.name}')
        setup, model = match.groups()
        model = model.lower()
        for path in sorted(campaign.glob('*/*/*_results.json')):
            result = json.loads(path.read_text())
            benchmark = result['benchmark']
            if benchmark != path.parent.parent.name:
                raise ValueError(f'Result benchmark identity disagrees with its directory: {path}')
            source = path.parent / (benchmark + ('_final.cpp' if setup.startswith('ms_') else '_generated.cpp'))
            history_path = path.parent / (benchmark + '_history.json')
            history = json.loads(history_path.read_text()) if history_path.exists() else {}
            code = source.read_text() if source.exists() else None
            report = result.get('final_report') or result.get('synth_report') or {}
            record_id = f'{model}/{benchmark.removeprefix("hlsfactory_")}/{setup}'
            row = {'record_id': record_id, 'model': model,
                   'benchmark': benchmark, 'benchmark_label': benchmark.removeprefix('hlsfactory_'),
                   'setup': setup, 'mode': 'multistep' if setup.startswith('ms_') else 'flash',
                   'source_result_json': str(path), 'source_result_sha256': sha(path),
                   'kernel_source': str(source) if source.exists() else None,
                   'kernel_sha256': sha(source) if source.exists() else None,
                   'history_json': str(history_path) if history_path.exists() else None,
                   'history_sha256': sha(history_path) if history_path.exists() else None,
                   'result_success': result.get('success'),
                   'reported_synthesis_cycles': report.get('latency_cycles'),
                   'selected_report_sha256': digest(report), 'selected_report': report,
                   'run': result.get('run'), 'best_so_far_promotion': result.get('best_so_far_promotion'),
                   'binding': candidate_binding(result, code, history),
                   'cosim_status': result.get('cosim_status', {}).get('generated'),
                   'cosim_record': result.get('cosim'),
                   'independent_historical_output_check': 'not_recorded',
                   'original_testbench_bytes_available': False,
                   'priority_paper_slots': [], 'source_grid_identity': campaign.name}
            if record_id in by_id:
                raise ValueError(f'Duplicate result identity: {record_id}')
            all_rows.append(row)
            by_id[record_id] = row
    schemas, schema_receipts = {}, []
    for model in ['sonnet', 'opus', 'gpt55']:
        source = payload / 'schema' / f'schema_records_newskills_{model}.jsonl'
        parsed = [json.loads(line) for line in source.read_text().splitlines()]
        schemas[model] = {(r['problem']['group_path'][0], r['implementation']['origin_version'], r['report_type']): r for r in parsed}
        original = Path(next(s['schema_source'] for s in slots if s['model'] == model))
        original_records = [json.loads(line) for line in original.read_text().splitlines()]
        if parsed != original_records:
            raise ValueError(f'Archive schema records differ from the published source: {model}')
        schema_receipts.append({'model': model, 'imported_schema': str(source), 'imported_sha256': sha(source),
                                'paper_schema': str(original), 'paper_sha256': sha(original),
                                'parsed_records_identical': True, 'record_count': len(parsed)})
    joined = []
    for slot in slots:
        row = by_id.get(slot['record_id'])
        if row is None:
            joined.append(dict(slot, source_join_status='missing_archive_result'))
            continue
        schema = schemas[slot['model']]
        synth = schema[(slot['benchmark'], slot['origin_version'], 'hls_synth')]
        csim = schema[(slot['benchmark'], slot['origin_version'], 'sw_run')]
        if digest(synth) != slot['hls_synth_record_sha256'] or digest(csim) != slot['sw_run_record_sha256']:
            raise ValueError(f'Schema record mismatch for {slot["slot_id"]}')
        expected_cycles = int(slot['published_synthesis_cycles'])
        report = row['selected_report']
        run = row['run']
        toolchain = slot['toolchain']
        checks = {'published_cycles_equal': row['reported_synthesis_cycles'] == expected_cycles,
                  'model_equal': run.get('model', '').lower() == {
                      'sonnet': 'claude-sonnet-4-6', 'opus': 'claude-opus-4-8', 'gpt55': 'gpt55'}[slot['model']],
                  'part_equal': run.get('part') == toolchain['part'],
                  'clock_equal': float(run.get('clock_ns', -1)) == toolchain['clock_ns'],
                  'vitis_version_equal': run.get('vitis_version') == toolchain['vitis_version'],
                  'flow_target_equal': run.get('flow_target') == toolchain['flow_target'],
                  'source_result_csim_linked': row['binding']['status'] == 'source_report_and_csim_linked'}
        resources = synth['hls_synth']['AreaEstimates']['Resources']
        checks['resources_equal'] = all(int(report.get(k.lower(), -1)) == int(resources[k])
                                       for k in ['BRAM_18K', 'DSP', 'FF', 'LUT', 'URAM']
                                       if k != 'BRAM_18K') and int(report.get('bram', -1)) == int(resources['BRAM_18K'])
        status = 'exact_selected_source_linked' if all(checks.values()) else 'source_join_incomplete'
        row['priority_paper_slots'].append(slot['slot_id'])
        row['paper_join_checks'] = checks
        row['paper_source_join_status'] = status
        row['toolchain'] = toolchain
        row['published_synthesis_cycles'] = expected_cycles
        row['schema_hls_synth_record_sha256'] = digest(synth)
        joined.append(dict(slot, recovered_result=row['source_result_json'], recovered_kernel=row['kernel_source'],
                           recovered_kernel_sha256=row['kernel_sha256'], source_join_status=status,
                           source_join_checks=checks, source_binding=row['binding']))
    selected = [r for r in all_rows if r['priority_paper_slots']]
    skill_files = []
    skill_maps = []
    for name in ['gemm_flatten_90.json', 'flash_no_rmw_maxi.json']:
        path = payload / 'skills' / name
        document = json.loads(path.read_text())
        entries = {entry['id']: entry for entry in document['skills']}
        if len(entries) != len(document['skills']):
            raise ValueError(f'Duplicate skill identifier in {name}')
        skill_maps.append(entries)
        skill_files.append({'path': str(path), 'sha256': sha(path), 'skill_count': len(entries),
                            'skill_ids': sorted(entries)})
    skill_ids = set().union(*skill_maps)
    overlaps = {key: sorted(k for k in set(skill_maps[0][key]) | set(skill_maps[1][key])
                          if skill_maps[0][key].get(k) != skill_maps[1][key].get(k))
                for key in skill_maps[0].keys() & skill_maps[1].keys()}
    delivered = set()
    routing = set()
    policy_observations = {}
    for row in all_rows:
        row_delivered = set()
        row_routing = set()
        if row['history_json']:
            history = json.loads(Path(row['history_json']).read_text())
            for message in history.get('messages', []):
                row_delivered.update(re.findall(r'\[skill ([^\]\s]+)\]', message.get('content', '')))
        result = json.loads(Path(row['source_result_json']).read_text())
        for step in result.get('generated_step_history', []):
            identity = (step.get('routing_decision') or {}).get('skill_id')
            if identity:
                row_routing.add(identity)
        delivered.update(row_delivered)
        routing.update(row_routing)
        arm = policy_observations.setdefault(row['setup'], Counter())
        arm['rows'] += 1
        arm['with_rendered_skill_markers'] += bool(row_delivered)
        arm['with_recorded_routing_skill_ids'] += bool(row_routing)
    launch_settings = []
    for model in ['sonnet', 'opus', 'gpt55']:
        path = payload / 'scripts' / f'_enh_env_{model}_newskills.sh'
        source = path.read_text()
        # Explicit allowlist prevents accidental export of provider secrets.
        settings = {}
        for key in ['C2HLS_SKILLS_EXTRA', 'C2HLS_DYNAMIC_ROUTING', 'C2HLS_SKILL_LIBRARY_PERSIST', 'SKILLS_BASE']:
            found = re.search(r'(?m)^(?:export\s+)?' + key + r'=([^\n#]*)', source)
            settings[key] = found.group(1).strip() if found else None
        launch_settings.append({'model': model, 'path': str(path), 'sha256': sha(path), 'settings': settings})
    skill_audit = {'schema': 'c2hls.commercial-skills-audit.v1', 'archived_skill_files': skill_files,
                   'unique_archived_skill_ids': len(skill_ids), 'overlap_difference_fields': overlaps,
                   'rendered_prompt_skill_ids': sorted(delivered),
                   'rendered_prompt_ids_missing_from_archived_files': sorted(delivered - skill_ids),
                   'recorded_routing_skill_ids': sorted(routing),
                   'recorded_routing_ids_missing_from_archived_files': sorted(routing - skill_ids),
                   'arm_policy_observations': policy_observations,
                   'control_caveat': '77/78 ms_noskills runs retain skill-ID-based dynamic step routing and corresponding system messages, while none carries rendered [skill ...] recipe blocks. Interpret this arm as skill-prompt-disabled, not as a completely skill-library-free controller. No rendered marker is not proof of absence of all skill-derived text.',
                   'launcher_settings': launch_settings,
                   'provenance': 'Launchers name both archived extra libraries; rendered histories record actual prompt text. Their 129-ID union agrees with the recorded loaded-library count. The external base55 library and historical loader code are absent, so exact merged-store bytes and duplicate-tag precedence are not proven.',
                   'backfill_uses_library': False}
    dump(output / 'skill_catalog_audit.json', skill_audit)
    summary = {'schema': 'c2hls.commercial-source-audit.v1', 'archive_sha256': receipt['archive_sha256'],
               'grid_result_count': len(all_rows), 'grid_models': dict(Counter(r['model'] for r in all_rows)),
               'grid_setups': dict(Counter(r['setup'] for r in all_rows)),
               'grid_result_success': dict(Counter(str(r['result_success']) for r in all_rows)),
               'grid_source_binding': dict(Counter(r['binding']['status'] for r in all_rows)),
               'grid_generated_cosim_status': dict(Counter(str(r['cosim_status']) for r in all_rows)),
               'teacher_slots': len(joined), 'selected_distinct_records': len(selected),
               'selected_slot_source_join': dict(Counter(r['source_join_status'] for r in joined)),
               'selected_record_source_binding': dict(Counter(r['binding']['status'] for r in selected)),
               'source_grid_benchmarks': sorted(set(r['benchmark'] for r in all_rows)),
               'schema_receipts': schema_receipts, 'paper_slots_source': str(slots_path),
               'paper_slots_sha256': sha(slots_path),
               'original_testbench_recovered': False,
               'backfill_admission_requires': 'independently staged and hashed public testbench, native-output preflight, frozen evaluator and input provenance; historical CSim completion is not correctness proof',
               'skill_catalog_note': 'Archived99+33 skill files cover129 distinct IDs, explicitly named by the historical launchers, and original rendered prompts are retained. External base55 file and loader precedence are not proven; see skill_catalog_audit.json.'}
    dump(output / 'full_grid_inventory.json', all_rows)
    dump(output / 'selected_inventory.json', selected)
    dump(output / 'selected_slot_join.json', joined)
    dump(output / 'source_audit_summary.json', summary)
    fields = ['record_id', 'model', 'benchmark', 'setup', 'mode', 'reported_synthesis_cycles',
              'kernel_source', 'kernel_sha256', 'source_result_json', 'source_result_sha256',
              'history_json', 'history_sha256', 'result_success', 'cosim_status', 'paper_source_join_status']
    csv_dump(output / 'selected_inventory.csv', selected, fields)
    csv_dump(output / 'full_grid_inventory.csv', all_rows, fields)
    return summary


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name or '\x00' in name:
        raise ValueError(f'Unsafe archive name: {name!r}')
    if str(path) != name.rstrip('/') or path.parts[0] != ARCHIVE_ROOT:
        raise ValueError(f'Noncanonical or unexpected archive path: {name!r}')
    return path


def parse_manifest(raw):
    records = {}
    for line in raw.decode('utf-8').splitlines():
        match = re.fullmatch(r'([0-9a-fA-F]{64}) [ *](.+)', line)
        if not match:
            raise ValueError('Malformed checksum manifest')
        digest, rel = match.groups()
        safe_name(ARCHIVE_ROOT + '/' + rel)
        if rel in records:
            raise ValueError(f'Duplicate checksum name: {rel}')
        records[rel] = digest.lower()
    return records


def import_archive(archive, output):
    if output.exists():
        raise FileExistsError(f'Refusing to replace import: {output}')
    # Scan every entry and verify manifest coverage before creating any payload.
    with tarfile.open(archive, 'r:gz') as source:
        members = source.getmembers()
        seen = set()
        for member in members:
            safe_name(member.name)
            if member.name in seen:
                raise ValueError(f'Duplicate archive entry: {member.name}')
            seen.add(member.name)
            if not (member.isfile() or member.isdir()) or member.issym() or member.islnk():
                raise ValueError(f'Link or special archive entry: {member.name}')
        manifest_name = ARCHIVE_ROOT + '/MANIFEST.sha256'
        manifest = parse_manifest(source.extractfile(manifest_name).read())
        files = {m.name[len(ARCHIVE_ROOT) + 1:] for m in members if m.isfile() and m.name != manifest_name}
        if set(manifest) != files:
            raise ValueError(f'Manifest coverage mismatch: missing={files - set(manifest)}, extra={set(manifest) - files}')
        # Reject file/directory collisions, including implicit parents.
        regular = {m.name for m in members if m.isfile()}
        for member in members:
            if any(str(parent) in regular for parent in PurePosixPath(member.name).parents):
                raise ValueError(f'Archive ancestor is a regular file: {member.name}')
        output.mkdir(parents=True)
        payload = output / 'payload'
        payload.mkdir()
        hashes = []
        for member in members:
            target = payload / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with source.extractfile(member) as stream, target.open('xb') as out:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
                    out.write(chunk)
            rel = member.name[len(ARCHIVE_ROOT) + 1:]
            value = digest.hexdigest()
            if member.name != manifest_name and value != manifest[rel]:
                raise ValueError(f'Checksum mismatch: {rel}')
            target.chmod(0o444)
            hashes.append({'relative_path': rel, 'sha256': value, 'size': member.size})
        for parent, dirs, _ in os.walk(payload, topdown=False):
            Path(parent).chmod(0o555)
        receipt = {'schema': 'c2hls.commercial-archive-import.v1',
                   'created_at': datetime.now(timezone.utc).isoformat(),
                   'archive': str(archive.resolve()), 'archive_sha256': sha(archive),
                   'archive_bytes': archive.stat().st_size, 'archive_members': len(members),
                   'payload_bytes': sum(m.size for m in members),
                   'payload_root': str(payload / ARCHIVE_ROOT),
                   'manifest_verified_files': len(manifest), 'all_payload_checksums_verified': True,
                   'links_and_special_files': 0, 'payload_read_only': True, 'files': hashes}
        dump(output / 'import_receipt.json', receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if not args.audit_only:
        receipt = import_archive(args.archive, args.output)
        print(json.dumps({k: v for k, v in receipt.items() if k != 'files'}, indent=2))
    print(json.dumps(audit_import(args.output), indent=2))


if __name__ == '__main__':
    main()
