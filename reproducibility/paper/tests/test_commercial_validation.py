"""Integrity checks for the fixed-selection HLS/RTL evidence join."""
import copy
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import generate_sigmetrics_commercial_validation as chart


class CommercialValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panels, cls.evidence, _ = chart.load_data()
        cls.original = {model: chart.rows(ROOT / 'generated' / name) for model, name in (
            ('SONNET', 'newskills_sonnet_snapshot.csv'), ('OPUS', 'july30_opus_selection.csv'),
            ('GPT55', 'newskills_gpt55_snapshot.csv'))}
        cls.historical_synthesis = chart.rows(chart.HISTORICAL_SYNTHESIS)
        cls.historical_rtl_rows = chart.rows(ROOT / 'generated/july30_cosim_expertise.csv')
        cls.historical_rtl = chart.historical.read_cosim(ROOT / 'generated/july30_cosim_expertise.csv')['models']

    def test_selection_cannot_be_changed_to_a_different_rtl_winner(self):
        altered = copy.deepcopy(self.evidence)
        row = next(r for r in altered if r['model'] == 'SONNET' and r['kernel'] == '2mm' and r['role'] == 'selected')
        row['arm'] = 'flash_curated'
        with self.assertRaisesRegex(ValueError, 'original selected/off arm changed'):
            chart.commercial_panels(altered, self.original)

    def test_each_ratio_uses_two_cycle_counts_of_the_same_evidence_tier(self):
        for model in chart.MODELS.values():
            for pair in self.panels[model].values():
                selected, off = pair['selected'], pair['off']
                self.assertEqual(pair['synthesis_ratio'], float(off['paper_cycles']) / float(selected['paper_cycles']))
                if pair['rtl_ratio'] is not None:
                    self.assertEqual(pair['rtl_ratio'], float(off['cosim_cycles']) / float(selected['cosim_cycles']))
                else:
                    self.assertFalse(selected['status'] == off['status'] == 'passed')

    def test_missing_outcomes_are_not_zero_or_successes(self):
        sonnet = self.panels['Claude Sonnet 4.6']['cholesky']
        opus = self.panels['Claude Opus 4.8']['doitgen']
        self.assertEqual(sonnet['selected_status'], 'env_error')
        self.assertEqual(opus['selected_status'], opus['off_status'])
        self.assertEqual(opus['selected_status'], 'timeout')
        for pair in [sonnet, opus]:
            self.assertIsNone(pair['rtl_ratio'])
            self.assertGreater(pair['synthesis_ratio'], 0)
            self.assertFalse(pair['common'])
        altered = copy.deepcopy(self.evidence)
        next(row for row in altered if row['status'] == 'timeout')['cosim_cycles'] = '1'
        with self.assertRaisesRegex(ValueError, 'Incomplete RTL outcome'):
            chart.validate_evidence(altered)

    def test_same_passing_pairs_define_both_commercial_geometric_means(self):
        summaries = chart.summary(self.panels)
        for model, expected in [('Claude Sonnet 4.6', 24), ('Claude Opus 4.8', 25), ('GPT-5.5', 23)]:
            paired = [p for p in self.panels[model].values() if p['common']]
            self.assertEqual(len(paired), expected)
            for field, metric in [('synthesis_ratio', 'synthesis_off_over_selected_gm'),
                                  ('rtl_ratio', 'rtl_off_over_selected_gm')]:
                self.assertTrue(all(p[field] is not None for p in paired))
                self.assertAlmostEqual(math.exp(sum(math.log(p[field]) for p in paired)/expected), summaries[model][metric])
        self.assertAlmostEqual(summaries['Claude Opus 4.8']['synthesis_off_over_selected_gm'], 2.737497810204733, places=9)

    def test_original_grid_and_asymmetric_rerun_exclusions_remain(self):
        sonnet = self.panels['Claude Sonnet 4.6']
        self.assertEqual([b for b,p in sonnet.items() if p['partial']], ['doitgen'])
        gpt = self.panels['GPT-5.5']
        self.assertEqual({b for b,p in gpt.items() if p['partial']}, {'3mm', 'floyd-warshall', 'gemver'})
        grok = self.panels['Grok 4.5']
        asymmetric = [p for p in grok.values() if p['asymmetric']]
        self.assertEqual(len(asymmetric), 5)
        self.assertTrue(all(p['rtl_ratio'] is not None and not p['common'] for p in asymmetric))
        self.assertEqual(sum(p['common'] for p in grok.values()), 21)
        self.assertEqual(sum(p['common'] for p in self.panels['DeepSeek V4 Flash'].values()), 22)

    def test_pass_records_keep_evidence_tiers_and_ties(self):
        self.assertEqual(sum(r['status'] == 'passed' for r in self.evidence), 153)
        parsed = {(r['model'],r['kernel'],r['role']) for r in self.evidence
                  if r['rtl_evidence_tier'] == 'parsed_cycles_and_PASS_log'}
        self.assertEqual(parsed, {('OPUS','bicg','off'), ('OPUS','floyd-warshall','off')})
        summaries = chart.summary(self.panels)
        self.assertEqual([summaries[m]['rtl_ties'] for m in chart.MODELS.values()], [3,2,6])
        self.assertTrue(all(r['independent_reference_output_check'] == 'not_recorded_in_this_bundle' for r in self.evidence))

    def test_historical_synthesis_requires_same_implementation_and_rtl_outcome(self):
        for field, value, expected in [
            ('selected_source', 'another_campaign', 'source identity mismatch'),
            ('selected_rtl_cycles', '1', 'RTL cycles mismatch'),
            ('selected_pass', 'False', 'RTL pass mismatch'),
            ('cohort', 'another_cohort', 'cohort mismatch'),
            ('selected_synthesis_field', 'skills_latency_avg', 'synthesis metric changed'),
            ('selected_report', 'different_kernel_selected_report.json', 'selected-report link mismatch'),
        ]:
            with self.subTest(field=field):
                altered = copy.deepcopy(self.historical_synthesis)
                altered[0][field] = value
                with self.assertRaisesRegex(ValueError, expected):
                    chart.historical_synthesis_panels(altered, self.historical_rtl, self.historical_rtl_rows)

    def test_historical_bars_use_synthesis_pairs_and_preserve_existing_rtl_ratios(self):
        for model in self.historical_rtl:
            for bench, pair in self.panels[model].items():
                old = self.historical_rtl[model][bench]
                self.assertEqual(pair['rtl_ratio'], old['ratio'])
                source = pair['synthesis_evidence']
                if pair['synthesis_ratio'] is not None:
                    self.assertEqual(pair['synthesis_ratio'],
                                     float(source['off_synthesis_cycles']) / float(source['selected_synthesis_cycles']))
                else:
                    self.assertTrue(not source['off_synthesis_cycles'] or not source['selected_synthesis_cycles'])
        deep_atax = self.panels['DeepSeek V4 Flash']['atax']
        self.assertEqual(deep_atax['synthesis_ratio'], 133218 / 16356)
        self.assertAlmostEqual(deep_atax['rtl_ratio'], 161985 / 17175, places=9)

    def test_incomplete_historical_synthesis_cannot_change_the_summary_cohort(self):
        summaries = chart.summary(self.panels)
        deep, grok = summaries['DeepSeek V4 Flash'], summaries['Grok 4.5']
        self.assertEqual((deep['paired'], deep['synthesis_paired']), (22, 21))
        self.assertIsNone(deep['synthesis_off_over_selected_gm'])
        self.assertAlmostEqual(deep['rtl_off_over_selected_gm'], 2.553561897318187)
        self.assertEqual(set(deep['missing_synthesis_kernels']), {'ludcmp', 'trmm'})
        self.assertIsNone(self.panels['DeepSeek V4 Flash']['trmm']['synthesis_ratio'])
        self.assertGreater(self.panels['DeepSeek V4 Flash']['trmm']['rtl_ratio'], 0)
        self.assertEqual((grok['paired'], grok['synthesis_paired']), (21, 21))
        self.assertEqual(grok['missing_synthesis_kernels'], ['seidel-2d'])
        pairs = [p for p in self.panels['Grok 4.5'].values() if p['common']]
        self.assertAlmostEqual(grok['synthesis_off_over_selected_gm'],
                               math.exp(sum(math.log(p['synthesis_ratio']) for p in pairs) / 21))
        self.assertAlmostEqual(grok['rtl_off_over_selected_gm'], 1.8726536979590611)


if __name__ == '__main__':
    unittest.main()
