from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from paper_baselines import (
    METHOD_ONE_SHOT,
    METHOD_PRAGMA_ONLY,
    PaperBaselineEngine,
    PublicBenchmarkInputs,
    build_baseline_fingerprint,
    cpp_token_stream,
    enforce_baseline_contract,
    finalize_baseline_result,
    load_matching_resume,
    pragma_only_guard,
    strip_pragma_directives,
)
from reference_isolation import audit_messages
from scripts.normalize_hpca_freeze_index import _baseline_candidate_events


REPO = Path(__file__).resolve().parents[1]


def _inputs(root: Path) -> PublicBenchmarkInputs:
    return PublicBenchmarkInputs(
        benchmark_dir=root,
        benchmark="toy",
        c_code="void workload(int *a) { for (int i=0;i<4;i++) a[i]++; }",
        header_code="void workload(int *a);",
        header_name="kernel.h",
        testbench_code="int main(){return 0;}",
        extra_files=(),
        translated_hls_top="workload",
        part="xcu280-fsvh2892-2L-e",
        clock_ns=3.33,
        cosim_depths={"a": 4},
        independent_golden_provenance={"status": "not_required"},
    )


def _fenced(code: str) -> str:
    return f"```cpp\n{code}\n```"


class FakeRuntime:
    def __init__(self, replies: list[str], latencies: list[int] | None = None):
        self.replies = list(replies)
        self.latencies = list(latencies or [500, 400, 300, 200, 100])
        self.requests: list[dict] = []
        self.csim_codes: list[str] = []
        self.synth_codes: list[str] = []
        self.cosim_codes: list[str] = []

    def llm(self, messages, index, seed, seed_supported):
        self.requests.append(
            {
                "messages": copy.deepcopy(list(messages)),
                "index": index,
                "seed": seed,
                "seed_supported": seed_supported,
            }
        )
        return {
            "text": self.replies[index],
            "event": {
                "provider": "fake",
                "usage_available": True,
                "input_tokens": 8,
                "output_tokens": 2,
                "total_tokens": 10,
            },
        }

    def csim(self, code):
        self.csim_codes.append(code)
        return {"status": "passed", "ran": True, "passed": True, "success": True}

    def synth(self, code):
        self.synth_codes.append(code)
        marker_match = re.search(r"MARKER_(\d+)", code)
        assert marker_match
        marker = int(marker_match.group(1))
        latency = self.latencies[marker]
        return {
            "success": True,
            "report": {
                "latency_cycles_worst": latency,
                "latency_cycles": latency,
                "bram": 1,
                "dsp": 1,
                "ff": 10,
                "lut": 10,
                "slack_ns": 0.1,
            },
        }

    def cosim(self, code):
        self.cosim_codes.append(code)
        return {
            "status": "passed",
            "ran": True,
            "passed": True,
            "success": True,
            "cycles": 123,
        }

    @staticmethod
    def feasibility(report, csim):
        feasible = bool(report) and bool(csim.get("passed"))
        return {
            "schema_version": "c2hls.candidate-feasibility.v1",
            "feasible": feasible,
            "resource_fit": feasible,
            "timing_met": feasible,
            "reasons": [] if feasible else ["not_feasible"],
        }


class PragmaTokenGuardTests(unittest.TestCase):
    def test_allows_only_pragma_comment_and_formatting_changes(self):
        base = """
// public loop
void workload(int *a) {
#pragma HLS PIPELINE II=2
  for (int i = 0; i < 8; ++i) { a[i] += 1; }
}
"""
        candidate = """
void workload ( int* a ) { /* same behavior */
# pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=a cyclic factor=2
for(int i=0;i<8;++i){a[i]+=1;}
}
"""
        guard = pragma_only_guard(base, candidate)
        self.assertTrue(guard["passed"])
        self.assertTrue(guard["pragma_changed"])
        self.assertEqual(
            cpp_token_stream(strip_pragma_directives(base)[0]),
            cpp_token_stream(strip_pragma_directives(candidate)[0]),
        )

    def test_rejects_any_nonpragma_semantic_edit(self):
        base = "void f(int *a){for(int i=0;i<8;i++)a[i]+=1;}"
        for candidate in (
            "void f(int *a){for(int i=0;i<9;i++)a[i]+=1;}",
            "void f(float *a){for(int i=0;i<8;i++)a[i]+=1;}",
            "void f(int *a){for(int i=0;i<8;i++)a[i]-=1;}",
            "#define N 8\nvoid f(int *a){for(int i=0;i<N;i++)a[i]+=1;}",
        ):
            with self.subTest(candidate=candidate):
                guard = pragma_only_guard(base, candidate)
                self.assertFalse(guard["passed"])
                self.assertIsNotNone(guard["mismatch"])

    def test_removes_complete_backslash_continued_pragma_only(self):
        source = (
            "void f(int *a){\n"
            "#pragma HLS ARRAY_PARTITION variable=a \\\n"
            "  cyclic factor=4\n"
            "a[0]++;\n}\n"
        )
        stripped, pragmas = strip_pragma_directives(source)
        self.assertEqual(1, len(pragmas))
        self.assertNotIn("factor", cpp_token_stream(stripped))
        self.assertIn("a", cpp_token_stream(stripped))

    def test_raw_string_contents_cannot_hide_nonpragma_edits_or_fake_directives(self):
        base = 'const char *s=R"tag(\n#pragma NOT_A_DIRECTIVE\n/* alpha */\n)tag";\n'
        candidate = base.replace("alpha", "beta")
        stripped, pragmas = strip_pragma_directives(base)
        self.assertEqual((), pragmas)
        self.assertIn("#pragma NOT_A_DIRECTIVE", stripped)
        self.assertFalse(pragma_only_guard(base, candidate)["passed"])


class PaperBaselineEngineTests(unittest.TestCase):
    def test_best_of_five_is_independent_seeded_matched_and_selected_cosim_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            replies = [
                _fenced(f"void workload(int*a){{int MARKER_{i}; a[0]+={i};}}")
                for i in range(5)
            ]
            runtime = FakeRuntime(replies)
            engine = PaperBaselineEngine(
                inputs=_inputs(Path(tmp)),
                method=METHOD_ONE_SHOT,
                model_id="qwen3.6-27b",
                base_seed=11,
                seed_supported=True,
                llm_request=runtime.llm,
                csim=runtime.csim,
                synthesize=runtime.synth,
                cosim=runtime.cosim,
                feasibility=runtime.feasibility,
            )
            result = engine.run()

        self.assertTrue(result["success"])
        self.assertEqual(4, result["selected_candidate_index"])
        self.assertEqual(5, result["llm_usage"]["calls"])
        self.assertEqual(5, result["synthesis_evaluations"]["count"])
        self.assertEqual(5, result["synthesis_evaluation_count"])
        self.assertEqual(1, result["selected_winner_cosim_count"])
        self.assertEqual(0, result["post_route_implementation_count"])
        self.assertEqual(6, result["total_synthesis_calls"])
        self.assertEqual(6, result["total_tool_calls"])
        self.assertEqual(
            result["selected_code_sha256"], result["cosim_target_code_sha256"]
        )
        self.assertEqual(
            result["selected_code_sha256"], result["cosim"]["target_code_sha256"]
        )
        completion_times = [
            candidate["cumulative_elapsed_seconds"]
            for candidate in result["candidates"]
        ]
        self.assertEqual(sorted(completion_times), completion_times)
        self.assertTrue(all(value >= 0 for value in completion_times))
        self.assertEqual(5, len(runtime.csim_codes))
        self.assertEqual(5, len(runtime.synth_codes))
        self.assertEqual(1, len(runtime.cosim_codes))
        self.assertIn("MARKER_4", runtime.cosim_codes[0])
        self.assertEqual(123, result["executed_cosim_cycles"])
        self.assertEqual("passed", result["correctness_status"])
        self.assertEqual([11, 12, 13, 14, 15], [r["seed"] for r in runtime.requests])
        prompts = [r["messages"] for r in runtime.requests]
        self.assertTrue(all(prompt == prompts[0] for prompt in prompts))

    def test_claude_explicitly_records_seed_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            replies = [
                _fenced(f"void workload(int*a){{int MARKER_{i};}}")
                for i in range(5)
            ]
            runtime = FakeRuntime(replies)
            result = PaperBaselineEngine(
                inputs=_inputs(Path(tmp)),
                method=METHOD_ONE_SHOT,
                model_id="claude-sonnet-4-6",
                base_seed=3,
                seed_supported=False,
                llm_request=runtime.llm,
                csim=runtime.csim,
                synthesize=runtime.synth,
                cosim=runtime.cosim,
                feasibility=runtime.feasibility,
            ).run()
        self.assertEqual([None] * 5, [r["seed"] for r in runtime.requests])
        self.assertEqual(
            "unsupported_by_provider", result["llm_usage"]["seed_control"]
        )
        self.assertTrue(
            all(not event["seed_supported"] for event in result["llm_usage"]["events"])
        )
        self.assertEqual(
            [None] * 5,
            [
                item["effective_seed"]
                for item in result["llm_usage"]["candidate_seed_schedule"]
            ],
        )

    def test_real_engine_result_normalizes_to_candidate_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            replies = [
                _fenced(f"void workload(int*a){{int MARKER_{i};}}")
                for i in range(5)
            ]
            runtime = FakeRuntime(replies)
            result = PaperBaselineEngine(
                inputs=_inputs(Path(tmp)),
                method=METHOD_ONE_SHOT,
                model_id="qwen3.6-27b",
                base_seed=17,
                seed_supported=True,
                llm_request=runtime.llm,
                csim=runtime.csim,
                synthesize=runtime.synth,
                cosim=runtime.cosim,
                feasibility=runtime.feasibility,
            ).run()
        events = _baseline_candidate_events(result, "producer-run", "producer")
        self.assertEqual(5, len(events))
        self.assertEqual(50, events[-1]["cumulative_tokens"])
        self.assertEqual(5, events[-1]["cumulative_synthesis_evaluations"])
        self.assertTrue(events[-1]["selected_for_executed_cosim"])

    def test_pragma_only_revisions_are_independent_and_semantic_edit_is_rejected(self):
        initial = """void workload(int*a){int MARKER_0;
#pragma HLS PIPELINE II=4
for(int i=0;i<4;i++)a[i]+=1;}"""
        valid1 = initial.replace("II=4", "II=2")
        invalid = initial.replace("a[i]+=1", "a[i]+=2")
        valid3 = initial.replace("II=4", "II=1")
        valid4 = initial.replace("#pragma HLS PIPELINE II=4", "#pragma HLS UNROLL factor=2")
        replies = [_fenced(code) for code in (initial, valid1, invalid, valid3, valid4)]
        with tempfile.TemporaryDirectory() as tmp:
            runtime = FakeRuntime(replies, [40, 30, 20, 10, 5])
            result = PaperBaselineEngine(
                inputs=_inputs(Path(tmp)),
                method=METHOD_PRAGMA_ONLY,
                model_id="qwen3.6-27b",
                base_seed=5,
                seed_supported=True,
                llm_request=runtime.llm,
                csim=runtime.csim,
                synthesize=runtime.synth,
                cosim=runtime.cosim,
                feasibility=runtime.feasibility,
            ).run()

        self.assertTrue(result["success"])
        self.assertEqual(5, result["llm_usage"]["calls"])
        self.assertEqual(4, result["synthesis_evaluations"]["count"])
        self.assertEqual(5, result["total_synthesis_calls"])
        rejected = result["candidates"][2]
        self.assertEqual("non_pragma_token_edit", rejected["rejection_reason"])
        self.assertIn("cumulative_elapsed_seconds", rejected)
        self.assertFalse(rejected["guard"]["passed"])
        revision_prompts = [request["messages"] for request in runtime.requests[1:]]
        self.assertTrue(all(prompt == revision_prompts[0] for prompt in revision_prompts))
        self.assertEqual([5, 6, 7, 8, 9], [r["seed"] for r in runtime.requests])
        self.assertEqual(4, len(runtime.csim_codes))
        self.assertEqual(4, len(runtime.synth_codes))
        self.assertEqual(1, len(runtime.cosim_codes))

    def test_failed_csim_never_consumes_synthesis_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            replies = [
                _fenced(f"void workload(int*a){{int MARKER_{i};}}")
                for i in range(5)
            ]
            runtime = FakeRuntime(replies)
            runtime.csim = lambda code: {
                "status": "failed",
                "ran": True,
                "passed": False,
                "success": False,
            }
            result = PaperBaselineEngine(
                inputs=_inputs(Path(tmp)),
                method=METHOD_ONE_SHOT,
                model_id="qwen3.6-27b",
                base_seed=0,
                seed_supported=True,
                llm_request=runtime.llm,
                csim=runtime.csim,
                synthesize=runtime.synth,
                cosim=runtime.cosim,
                feasibility=runtime.feasibility,
            ).run()
        self.assertFalse(result["success"])
        self.assertEqual(5, result["llm_usage"]["calls"])
        self.assertEqual(0, result["synthesis_evaluations"]["count"])
        self.assertEqual(0, result["selected_winner_cosim_count"])
        self.assertEqual(0, result["total_synthesis_calls"])
        self.assertEqual([], runtime.synth_codes)
        self.assertEqual([], runtime.cosim_codes)

    def test_missing_initial_translation_emits_timed_zero_cost_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = FakeRuntime(["provider returned no fenced source"])
            result = PaperBaselineEngine(
                inputs=_inputs(Path(tmp)),
                method=METHOD_PRAGMA_ONLY,
                model_id="qwen3.6-27b",
                base_seed=0,
                seed_supported=True,
                llm_request=runtime.llm,
                csim=runtime.csim,
                synthesize=runtime.synth,
                cosim=runtime.cosim,
                feasibility=runtime.feasibility,
            ).run()
        self.assertEqual(5, result["candidate_count"])
        self.assertEqual(1, result["llm_usage"]["calls"])
        self.assertEqual(0, result["total_synthesis_calls"])
        self.assertTrue(
            all(
                "cumulative_elapsed_seconds" in candidate
                for candidate in result["candidates"]
            )
        )
        self.assertTrue(
            all(
                candidate.get("rejection_reason") == "initial_translation_missing"
                for candidate in result["candidates"][1:]
            )
        )

    def test_passing_cosim_without_executed_cycles_is_not_a_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            replies = [
                _fenced(f"void workload(int*a){{int MARKER_{i};}}")
                for i in range(5)
            ]
            runtime = FakeRuntime(replies)
            runtime.cosim = lambda code: {
                "status": "passed",
                "ran": True,
                "passed": True,
                "success": True,
                "kernel_runtime_cycles": None,
            }
            result = PaperBaselineEngine(
                inputs=_inputs(Path(tmp)),
                method=METHOD_ONE_SHOT,
                model_id="qwen3.6-27b",
                base_seed=0,
                seed_supported=True,
                llm_request=runtime.llm,
                csim=runtime.csim,
                synthesize=runtime.synth,
                cosim=runtime.cosim,
                feasibility=runtime.feasibility,
            ).run()
        self.assertFalse(result["success"])
        self.assertFalse(result["selected_cosim_measurement_valid"])
        self.assertEqual(6, result["total_synthesis_calls"])
        self.assertEqual("passed", result["correctness_status"])
        self.assertIn("without an executed cycle count", result["error"])

    def test_finalizer_persists_audit_bound_to_exact_transcript_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metadata.json").write_text("{}", encoding="utf-8")
            result = {
                "benchmark": "toy",
                "success": False,
                "csim": {"status": "not_run", "ran": False, "passed": False},
                "cosim": {"status": "not_run", "ran": False, "passed": False},
                "synthesis_evaluations": {"count": 0, "events": []},
                "llm_usage": {"calls": 0, "events": []},
                "_history": [
                    {"role": "user", "content": "Optimize this public kernel."}
                ],
                "_selected_code": "",
            }
            finalized = finalize_baseline_result(
                result,
                fingerprint={"schema_version": "test", "sha256": "0" * 64, "payload": {}},
                profile={"name": "test", "reference_blind": True},
                benchmark_dir=root,
                output_dir=root,
                elapsed_seconds=1.25,
            )
            transcript_path = root / finalized["run"]["transcript_file"]
            audit_path = root / finalized["run"][
                "reference_isolation_audit_path"
            ]
            transcript_digest = hashlib.sha256(transcript_path.read_bytes()).hexdigest()
            persisted_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertEqual(
            transcript_digest,
            finalized["reference_isolation_audit"]["transcript_sha256"],
        )
        self.assertEqual(persisted_audit, finalized["reference_isolation_audit"])
        self.assertEqual(1.25, finalized["run"]["search_elapsed_seconds"])

    def test_search_transcript_contains_no_expert_source_path_or_identifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "benchmark": "toy",
                        "gold_hls_baseline_file": "gold_secret.cpp",
                    }
                )
            )
            (root / "plain.cpp").write_text(
                "void workload(int *a){a[0]++;}"
            )
            (root / "gold_secret.cpp").write_text(
                "void expert_tile_coalescing_8192(int *a){a[0]+=99;}\n"
                "void helper_secret_variant_4096(int *a){a[0]+=2;}\n"
                "void third_secret_variant_2048(int *a){a[0]+=3;}\n"
            )
            replies = [
                _fenced(f"void workload(int*a){{int MARKER_{i};}}")
                for i in range(5)
            ]
            runtime = FakeRuntime(replies)
            engine = PaperBaselineEngine(
                inputs=_inputs(root),
                method=METHOD_ONE_SHOT,
                model_id="qwen3.6-27b",
                base_seed=0,
                seed_supported=True,
                llm_request=runtime.llm,
                csim=runtime.csim,
                synthesize=runtime.synth,
                cosim=runtime.cosim,
                feasibility=runtime.feasibility,
            )
            engine.run()
            audit = audit_messages(engine.history, benchmark_dir=root)
        self.assertTrue(audit["passed"], audit)
        transcript = json.dumps(engine.history)
        self.assertNotIn("gold_secret.cpp", transcript)
        self.assertNotIn("expert_tile_coalescing_8192", transcript)


class BaselineReproducibilityTests(unittest.TestCase):
    def _env(self):
        return {
            "C2HLS_REFERENCE_BLIND": "1",
            "C2HLS_GT_COMPARISON_IN_CONTROL": "0",
            "C2HLS_REFERENCE_CODE_IN_PROMPTS": "0",
            "C2HLS_REFERENCE_METRICS_IN_PROMPTS": "0",
            "C2HLS_COSIM_SELECTED_ONLY": "1",
            "C2HLS_FORCE_SELECTED_COSIM": "1",
            "C2HLS_FEASIBILITY_SELECTION": "1",
            "C2HLS_CORRECTNESS_BEFORE_SYNTH": "1",
            "C2HLS_TRANSCRIPT_AUDIT": "1",
            "C2HLS_LLM_CANDIDATE_BUDGET": "5",
            "C2HLS_SYNTHESIS_EVAL_BUDGET": "5",
            "C2HLS_LLM_TEMPERATURE": "0.2",
            "C2HLS_LLM_TOP_P": "0.95",
            "C2HLS_LLM_SEED": "0",
            "C2HLS_MODEL_REVISION": "weights-sha256-test",
            "C2HLS_VITIS_VERSION": "2023.2",
            "C2HLS_PART": "xcu280-fsvh2892-2L-e",
            "C2HLS_CLOCK_NS": "3.33",
            "C2HLS_STRATEGY": METHOD_ONE_SHOT,
        }

    def test_contract_rejects_oracle_or_budget_drift(self):
        env = self._env()
        enforce_baseline_contract(env)
        env["C2HLS_REFERENCE_BLIND"] = "0"
        with self.assertRaisesRegex(ValueError, "unsafe"):
            enforce_baseline_contract(env)
        env = self._env()
        env["C2HLS_SYNTHESIS_EVAL_BUDGET"] = "6"
        with self.assertRaisesRegex(ValueError, "requires"):
            enforce_baseline_contract(env)

    def test_full_fingerprint_exact_resume_and_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metadata.json").write_text("{}")
            (root / "plain.cpp").write_text("void workload(){}")
            (root / "testbench.cpp").write_text("int main(){return 0;}")
            (root / "gold_secret.cpp").write_text("void secret_v1(){}")
            inputs = _inputs(root)
            env = self._env()
            with mock.patch.dict(os.environ, env, clear=True):
                fingerprint = build_baseline_fingerprint(
                    repo=REPO,
                    inputs=inputs,
                    method=METHOD_ONE_SHOT,
                    model_id="qwen3.6-27b",
                    model_label="qwen_27b",
                    base_seed=11,
                    profile={"name": "hpca2027_reference_blind"},
                )
                (root / "gold_secret.cpp").write_text("void secret_v2_changed(){}")
                fingerprint_after_expert_change = build_baseline_fingerprint(
                    repo=REPO,
                    inputs=inputs,
                    method=METHOD_ONE_SHOT,
                    model_id="qwen3.6-27b",
                    model_label="qwen_27b",
                    base_seed=11,
                    profile={"name": "hpca2027_reference_blind"},
                )
            self.assertEqual(fingerprint, fingerprint_after_expert_change)
            self.assertEqual("11", fingerprint["payload"]["decoding"]["seed"])
            self.assertEqual(
                [11, 12, 13, 14, 15],
                [
                    item["effective_seed"]
                    for item in fingerprint["payload"]["paper_baseline"][
                        "candidate_seed_schedule"
                    ]
                ],
            )
            result_path = root / "result.json"
            result_path.write_text(json.dumps({"run_fingerprint": fingerprint}))
            self.assertIsNotNone(load_matching_resume(result_path, fingerprint))
            with mock.patch.dict(os.environ, env, clear=True):
                different_cli_seed = build_baseline_fingerprint(
                    repo=REPO,
                    inputs=inputs,
                    method=METHOD_ONE_SHOT,
                    model_id="qwen3.6-27b",
                    model_label="qwen_27b",
                    base_seed=12,
                    profile={"name": "hpca2027_reference_blind"},
                )
            self.assertNotEqual(fingerprint["sha256"], different_cli_seed["sha256"])
            with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                load_matching_resume(result_path, different_cli_seed)
            changed = copy.deepcopy(fingerprint)
            changed["payload"]["paper_baseline"]["method"] = METHOD_PRAGMA_ONLY
            changed["sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                load_matching_resume(result_path, changed)


if __name__ == "__main__":
    unittest.main()
