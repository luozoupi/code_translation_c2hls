#!/usr/bin/env python3
"""Offline, manually scoped goal audit. Never invokes a model/compiler/controller.

Historical goals are retrospective probes, not preregistered claims. Scope
correspondence is supplied by a human audit and checked literally; this does
not implement automatic loop matching or general source equivalence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def pointer(document, path):
    value = document
    for key in path.strip('/').split('/') if path else []:
        key = key.replace('~1', '/').replace('~0', '~')
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def evaluate_goal(goal, code, report, binding_ok=True):
    """Three-valued observation, independent of correctness or endpoint latency."""
    out = {'goal_id': goal['id'], 'target': goal.get('target'),
           'predicate': goal.get('predicate'), 'verdict': 'unknown',
           'observation': None, 'reason': ''}
    if not binding_ok:
        out['reason'] = 'source_or_report_binding_failed'
        return out
    if goal.get('evidence_kind') != 'compiler_loop_metric':
        out['reason'] = 'required_compiler_evidence_not_available'
        return out
    target = goal['target']
    if target.get('mapping') != 'manually_audited_exact_source_and_scope':
        out['reason'] = 'scope_correspondence_not_established'
        return out
    matches = [s for s in report.get('feedback', {}).get('scopes', [])
               if s.get('scope_id') == target.get('scope_id') and s.get('kind') == 'loop']
    if len(matches) != 1:
        out['reason'] = 'scope_missing_renamed_or_ambiguous'
        return out
    scope = matches[0]
    line = target.get('source_line', 0)
    lines = code.splitlines()
    if (not isinstance(line, int) or not 1 <= line <= len(lines)
            or lines[line - 1].strip() != target.get('source_line_text')
            or not str(scope.get('source_location', '')).endswith(':' + str(line))):
        out['reason'] = 'source_to_report_line_binding_failed'
        return out
    pred = goal['predicate']
    if pred.get('metric') not in {'pipeline_ii', 'trip_count', 'pipeline_depth'}:
        out['reason'] = 'unsupported_metric_no_substitution'
        return out
    observed, threshold = scope.get(pred['metric']), pred.get('threshold')
    if (isinstance(observed, bool) or not isinstance(observed, (int, float))
            or not math.isfinite(observed) or observed <= 0
            or isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)):
        out['reason'] = 'required_metric_missing_or_invalid'
        return out
    tests = {'le': lambda: observed <= threshold, 'eq': lambda: observed == threshold,
             'ge': lambda: observed >= threshold}
    if pred.get('operator') not in tests:
        out['reason'] = 'unsupported_predicate'
        return out
    out.update(observation=observed, verdict='achieved' if tests[pred['operator']]() else 'unmet',
               reason='bound_compiler_metric', scope_id=scope['scope_id'])
    return out


def review_case(case):
    result_path = Path(case['result_path'])
    if not result_path.is_file():
        document, file_ok = {}, False
    else:
        raw = result_path.read_bytes()
        file_ok = digest(raw) == case['result_sha256']
        document = json.loads(raw)
    try:
        code = pointer(document, case['source_pointer'])
        report = pointer(document, case['report_pointer'])
        binding_ok = (file_ok and isinstance(code, str)
                      and digest(code.encode()) == case['source_sha256']
                      and canonical_hash(report) == case['report_sha256'])
    except (KeyError, IndexError, ValueError, TypeError):
        code, report, binding_ok = '', {}, False
    reviews = [evaluate_goal(g, code, report, binding_ok) for g in case['goals']]
    by_id = {r['goal_id']: r for r in reviews}
    if len(by_id) != len(reviews):
        raise ValueError('Duplicate goal IDs in case')
    specs = {g['id']: g for g in case['goals']}
    def credit(key, active=()):
        if key in active or key not in by_id or by_id[key]['verdict'] != 'achieved':
            return False
        return all(credit(parent, (*active, key)) for parent in specs[key].get('prerequisites', []))
    for goal, review in zip(case['goals'], reviews):
        prerequisites = {key: by_id.get(key, {}).get('verdict', 'unknown')
                         for key in goal.get('prerequisites', [])}
        review['prerequisite_verdicts'] = prerequisites
        review['prerequisite_credit'] = {key: credit(key) for key in prerequisites}
        review['prerequisites_satisfied'] = all(review['prerequisite_credit'].values())
        review['eligible_for_goal_credit_with_prerequisites'] = credit(goal['id'])
    return {'case_id': case['case_id'], 'source_sha256': case['source_sha256'],
            'binding_ok': binding_ok,
            'recorded_correctness': case.get('recorded_correctness', 'unknown'),
            'recorded_feasibility': case.get('recorded_feasibility', 'unknown'),
            'endpoint_cycles': case.get('endpoint_cycles'),
            'parent_cycles': case.get('parent_cycles'),
            'goal_reviews': reviews,
            'controller_action': 'none_offline_only'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert manifest['schema_version'] == 1 and manifest['retrospective'] is True
    results = [review_case(c) for c in manifest['cases']]
    counts = {k: sum(g['verdict'] == k for c in results for g in c['goal_reviews'])
              for k in ['achieved', 'unmet', 'unknown']}
    out = {'schema_version': 1, 'retrospective': True, 'manifest_sha256': digest(args.manifest.read_bytes()),
           'script_sha256': digest(Path(__file__).read_bytes()), 'cases': results, 'goal_counts': counts,
           'limits': ['Post-hoc probes; no historical goal contract or causal enforcement effect.',
                      'Manual source/scope bindings; no automatic correspondence across renamed loops.',
                      'Correctness and feasibility are historical records, not new simulation.',
                      'No bank-port, traffic, or concurrency proof from source markers.',
                      'No controller state changed; unmet goals never rewrite historical selection.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'cases': len(results), 'goal_counts': counts, 'all_bindings_match': all(c['binding_ok'] for c in results)}))


if __name__ == '__main__':
    main()
