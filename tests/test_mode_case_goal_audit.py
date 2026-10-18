"""Conservative evidence semantics for offline goal review; no HLS calls."""
import copy
import importlib.util
import json
import tempfile
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('case_goals', Path(__file__).resolve().parents[1] / 'scripts/audit_sigmetrics_mode_case_goals.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class GoalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.code = 'for (int i=0; i<32; ++i) {\n}\n'
        self.goal = {'id': 'pipeline', 'evidence_kind': 'compiler_loop_metric',
                     'target': {'mapping': 'manually_audited_exact_source_and_scope',
                                'scope_id': 'kernel/compute', 'source_line': 1,
                                'source_line_text': self.code.splitlines()[0]},
                     'predicate': {'metric': 'pipeline_ii', 'operator': 'le', 'threshold': 1}}
        self.report = {'latency_cycles_worst': 80, 'feedback': {'scopes': [
            {'scope_id': 'kernel/compute', 'kind': 'loop', 'source_location': 'kernel.cpp:1',
             'pipeline_ii': 1, 'interval': 1, 'trip_count': 32}]}}

    def verdict(self):
        return MOD.evaluate_goal(self.goal, self.code, self.report)['verdict']

    def test_achieved_with_bound_loop(self):
        self.assertEqual(self.verdict(), 'achieved')

    def test_faster_endpoint_does_not_override_unmet_loop_goal(self):
        self.report['feedback']['scopes'][0]['pipeline_ii'] = 4
        self.assertEqual(self.verdict(), 'unmet')

    def test_enabling_latency_regression_does_not_negate_achieved_goal(self):
        self.report['latency_cycles_worst'] = 1000
        self.assertEqual(self.verdict(), 'achieved')

    def test_renamed_loop_is_unknown(self):
        self.report['feedback']['scopes'][0]['scope_id'] = 'kernel/renamed'
        self.assertEqual(self.verdict(), 'unknown')

    def test_missing_ii_does_not_fall_back_to_interval(self):
        self.report['feedback']['scopes'][0]['pipeline_ii'] = None
        self.assertEqual(self.verdict(), 'unknown')

    def test_module_interval_cannot_prove_loop_goal(self):
        self.report['feedback']['scopes'][0]['kind'] = 'module'
        self.assertEqual(self.verdict(), 'unknown')

    def test_source_or_report_mismatch_is_unknown(self):
        self.assertEqual(MOD.evaluate_goal(self.goal, self.code, self.report, False)['verdict'], 'unknown')
        self.report['feedback']['scopes'][0]['source_location'] = 'kernel.cpp:2'
        self.assertEqual(self.verdict(), 'unknown')

    def test_duplicate_scope_is_unknown(self):
        self.report['feedback']['scopes'].append(copy.deepcopy(self.report['feedback']['scopes'][0]))
        self.assertEqual(self.verdict(), 'unknown')

    def test_unsupported_hardware_evidence_remains_unknown(self):
        self.goal['evidence_kind'] = 'bank_port_binding'
        self.assertEqual(self.verdict(), 'unknown')

    def test_unproven_prerequisite_blocks_transitive_credit_not_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'record.json'
            raw = json.dumps({'code': self.code, 'report': self.report}).encode()
            path.write_bytes(raw)
            bank = {'id': 'banks', 'evidence_kind': 'bank_port_binding', 'prerequisites': []}
            pipe = copy.deepcopy(self.goal)
            pipe['prerequisites'] = ['banks']
            dependent = copy.deepcopy(self.goal)
            dependent.update(id='dependent', prerequisites=['pipeline'])
            case = {'case_id': 'test', 'result_path': str(path), 'result_sha256': MOD.digest(raw),
                    'source_pointer': '/code', 'report_pointer': '/report',
                    'source_sha256': MOD.digest(self.code.encode()),
                    'report_sha256': MOD.canonical_hash(self.report),
                    'goals': [bank, pipe, dependent]}
            reviews = MOD.review_case(case)['goal_reviews']
            self.assertEqual([r['verdict'] for r in reviews], ['unknown', 'achieved', 'achieved'])
            self.assertFalse(any(r['eligible_for_goal_credit_with_prerequisites'] for r in reviews))
            case['result_sha256'] = 'wrong'
            self.assertTrue(all(r['verdict'] == 'unknown' for r in MOD.review_case(case)['goal_reviews']))


if __name__ == '__main__':
    unittest.main()
