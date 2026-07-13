from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from golden_output import (  # noqa: E402
    ComparisonReason,
    CorrectnessStatus,
    NumericKind,
    OutputParseError,
    OutputSpec,
    compare_hlsfactory_dumps,
    compare_structured_outputs,
    parse_hlsfactory_dumps,
)


def dump(name: str, values: str, *, same_line: bool = False) -> str:
    separator = "" if same_line else "\n"
    return (
        "tool preamble\n"
        "==BEGIN DUMP_ARRAYS==\n"
        f"begin dump: {name}{separator}{values}\n"
        f"end   dump: {name}\n"
        "==END   DUMP_ARRAYS==\n"
        "tool epilogue\n"
    )


class HLSFactoryDumpParserTests(unittest.TestCase):
    def test_parses_same_line_first_value_and_multiple_named_blocks(self):
        text = (
            "==BEGIN DUMP_ARRAYS==\n"
            "begin dump: x1.250000 2.500000\n"
            "end   dump: x\n"
            "begin dump: table\n1 2 -3\n"
            "end dump: table\n"
            "==END DUMP_ARRAYS==\n"
        )
        outputs = parse_hlsfactory_dumps(text)

        self.assertEqual(set(outputs), {"x", "table"})
        self.assertEqual(outputs["x"].values, (1.25, 2.5))
        self.assertFalse(outputs["x"].integer_tokens)
        self.assertEqual(outputs["table"].values, (1, 2, -3))
        self.assertTrue(outputs["table"].integer_tokens)

    def test_rejects_non_numeric_and_unbalanced_dump(self):
        with self.assertRaises(OutputParseError) as non_numeric:
            parse_hlsfactory_dumps(dump("y", "1.0 broken 3.0"))
        self.assertEqual(non_numeric.exception.reason, ComparisonReason.MALFORMED_OUTPUT)

        with self.assertRaises(OutputParseError) as unbalanced:
            parse_hlsfactory_dumps("begin dump: y\n1.0\n")
        self.assertEqual(unbalanced.exception.reason, ComparisonReason.MALFORMED_OUTPUT)

    def test_rejects_missing_output(self):
        with self.assertRaises(OutputParseError) as context:
            parse_hlsfactory_dumps("Vitis HLS completed successfully")
        self.assertEqual(context.exception.reason, ComparisonReason.NO_OUTPUT)


class StructuredGoldenComparisonTests(unittest.TestCase):
    def test_tolerant_float_pass_and_mismatch(self):
        passing = compare_structured_outputs(
            {"y": [1.0, 2.0]},
            {"y": [1.0 + 2e-7, 2.0 - 2e-7]},
            default_atol=1e-6,
            default_rtol=0,
        )
        self.assertTrue(passing.passed)
        self.assertEqual(passing.reason, ComparisonReason.MATCH)
        self.assertEqual(passing.details["values_compared"], 2)

        failing = compare_structured_outputs(
            {"y": [1.0, 2.0]},
            {"y": [1.0, 2.01]},
            default_atol=1e-6,
            default_rtol=0,
        )
        self.assertEqual(failing.correctness_status, CorrectnessStatus.FAILED)
        self.assertEqual(failing.reason, ComparisonReason.FLOAT_MISMATCH)
        self.assertEqual(failing.details["mismatch_count"], 1)

    def test_count_mismatch_precedes_shape_check(self):
        result = compare_structured_outputs(
            {"matrix": [[1, 2], [3, 4]]},
            {"matrix": [[1, 2, 3]]},
        )
        self.assertEqual(result.reason, ComparisonReason.COUNT_MISMATCH)
        self.assertEqual(result.details["expected_count"], 4)
        self.assertEqual(result.details["actual_count"], 3)

    def test_equal_count_but_different_shape_fails(self):
        result = compare_structured_outputs(
            {"matrix": [[1, 2], [3, 4]]},
            {"matrix": [1, 2, 3, 4]},
        )
        self.assertEqual(result.reason, ComparisonReason.SHAPE_MISMATCH)
        self.assertEqual(result.details["expected_shape"], [2, 2])
        self.assertEqual(result.details["actual_shape"], [4])

    def test_integer_comparison_is_exact_even_with_large_float_tolerance(self):
        result = compare_structured_outputs(
            {"path": [1, 2, 3]},
            {"path": [1.0, 2.000001, 3.0]},
            default_atol=1.0,
            default_rtol=1.0,
        )
        self.assertEqual(result.reason, ComparisonReason.TYPE_MISMATCH)
        self.assertEqual(result.details["comparison"], "exact_integer")

        exact = compare_structured_outputs(
            {"path": [1, 2, 3]},
            {"path": [1.0, 2.0, 3.0]},
        )
        self.assertTrue(exact.passed)

    def test_integer_value_mismatch_has_integer_reason(self):
        result = compare_structured_outputs(
            {"path": [1, 2, 3]},
            {"path": [1, 9, 3]},
        )
        self.assertEqual(result.reason, ComparisonReason.INTEGER_MISMATCH)
        self.assertEqual(result.details["reported_mismatches"][0]["index"], [1])

    def test_nan_is_rejected_by_default_and_can_be_explicitly_allowed(self):
        rejected = compare_structured_outputs(
            {"y": [math.nan]},
            {"y": [math.nan]},
        )
        self.assertEqual(rejected.reason, ComparisonReason.NAN_MISMATCH)

        allowed = compare_structured_outputs(
            {"y": [math.nan]},
            {"y": [math.nan]},
            {"y": OutputSpec(kind=NumericKind.FLOAT, allow_nan=True)},
        )
        self.assertTrue(allowed.passed)

        # Non-finite values are strings in the result, so strict JSON never
        # emits the non-standard JavaScript NaN literal.
        json.dumps(rejected.to_dict(), allow_nan=False)
        self.assertEqual(
            rejected.to_dict()["details"]["reported_mismatches"][0]["expected"],
            "NaN",
        )

    def test_infinity_requires_opt_in_and_matching_sign(self):
        rejected = compare_structured_outputs(
            {"y": [math.inf]},
            {"y": [math.inf]},
        )
        self.assertEqual(rejected.reason, ComparisonReason.INFINITY_MISMATCH)

        allowed = compare_structured_outputs(
            {"y": [math.inf]},
            {"y": [math.inf]},
            {"y": {"kind": "float", "allow_infinity": True}},
        )
        self.assertTrue(allowed.passed)

        wrong_sign = compare_structured_outputs(
            {"y": [math.inf]},
            {"y": [-math.inf]},
            {"y": OutputSpec(kind=NumericKind.FLOAT, allow_infinity=True)},
        )
        self.assertEqual(wrong_sign.reason, ComparisonReason.INFINITY_MISMATCH)

    def test_missing_named_output_is_visible(self):
        result = compare_structured_outputs(
            {"x": [1], "y": [2]},
            {"x": [1], "z": [2]},
        )
        self.assertEqual(result.reason, ComparisonReason.OUTPUT_SET_MISMATCH)
        self.assertEqual(result.details["missing_outputs"], ["y"])
        self.assertEqual(result.details["unexpected_outputs"], ["z"])

    def test_ragged_sequence_is_invalid_output(self):
        result = compare_structured_outputs(
            {"matrix": [[1, 2], [3, 4]]},
            {"matrix": [[1], [2, 3]]},
        )
        self.assertEqual(result.correctness_status, CorrectnessStatus.INVALID_OUTPUT)
        self.assertEqual(result.reason, ComparisonReason.MALFORMED_OUTPUT)
        self.assertEqual(result.details["source"], "candidate")


class DumpGoldenComparisonTests(unittest.TestCase):
    def test_dump_float_tolerance_and_declared_shape(self):
        golden = dump("C", "1.000000 2.000000 3.000000 4.000000")
        candidate = dump("C", "1.000001 2.000000 3.000000 3.999999")
        result = compare_hlsfactory_dumps(
            golden,
            candidate,
            {"C": OutputSpec(shape=(2, 2), atol=2e-6, rtol=0)},
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.details["outputs"]["C"]["shape"], [2, 2])

    def test_declared_shape_count_mismatch_fails(self):
        result = compare_hlsfactory_dumps(
            dump("C", "1.0 2.0 3.0"),
            dump("C", "1.0 2.0 3.0"),
            {"C": OutputSpec(shape=(2, 2))},
        )
        self.assertEqual(result.reason, ComparisonReason.COUNT_MISMATCH)
        self.assertEqual(result.details["declared_count"], 4)

    def test_malformed_candidate_is_typed_and_does_not_raise(self):
        result = compare_hlsfactory_dumps(
            dump("y", "1.0 2.0"),
            dump("y", "1.0 not-a-number"),
        )
        self.assertEqual(result.correctness_status, CorrectnessStatus.INVALID_OUTPUT)
        self.assertEqual(result.reason, ComparisonReason.MALFORMED_OUTPUT)
        self.assertEqual(result.details["source"], "candidate")

    def test_no_candidate_output_is_typed_and_does_not_raise(self):
        result = compare_hlsfactory_dumps(
            dump("y", "1.0 2.0"),
            "INFO: C simulation exited with code 0",
        )
        self.assertEqual(result.correctness_status, CorrectnessStatus.INVALID_OUTPUT)
        self.assertEqual(result.reason, ComparisonReason.NO_OUTPUT)
        self.assertEqual(result.details["source"], "candidate")


if __name__ == "__main__":
    unittest.main()
