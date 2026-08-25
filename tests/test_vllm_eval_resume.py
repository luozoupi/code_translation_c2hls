from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_vllm_corpus_eval
from scripts import run_vllm_vitis_smoke


def _generation_args(root: Path, input_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input=str(input_path),
        output_dir=str(root),
        run_name="test",
        output_jsonl=str(root / "generation.jsonl"),
        summary_path=str(root / "generation.summary.json"),
        heartbeat_path=str(root / "generation.heartbeat.json"),
        model="test-model",
        base_url="http://example.invalid/v1",
        api_key="EMPTY",
        limit=0,
        seed=0,
        sample="first",
        row_index=[],
        benchmark=[],
        exclude_benchmark=["hlsfactory_doitgen"],
        unique_benchmarks=True,
        max_prompt_chars=0,
        temperature=0.0,
        top_p=None,
        max_tokens=128,
        timeout=1,
        retries=0,
        retry_backoff_seconds=0.0,
        max_consecutive_errors=3,
        inter_request_delay_seconds=0.0,
        resume=True,
    )


def _vitis_args(root: Path, input_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input_jsonl=[str(input_path)],
        output_dir=str(root / "compact"),
        run_name="test",
        work_root=str(root / "work"),
        heartbeat_path="",
        hlsfactory_root=str(root / "benches"),
        benchmark=[],
        limit=0,
        allow_non_stop=False,
        no_csim=False,
        correctness_first=True,
        resume=True,
        retry_failed=False,
        max_consecutive_infrastructure_errors=3,
    )


class VllmEvalResumeTest(unittest.TestCase):
    def test_generation_checkpoints_and_resumes_successes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.jsonl"
            records = [
                {
                    "benchmark": benchmark,
                    "suite": "hlsfactory",
                    "messages": [
                        {"role": "user", "content": f"Generate {benchmark}"},
                        {"role": "assistant", "content": ""},
                    ],
                }
                for benchmark in (
                    "hlsfactory_2mm",
                    "hlsfactory_doitgen",
                    "hlsfactory_bicg",
                )
            ]
            input_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            args = _generation_args(root, input_path)
            response = {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "```cpp\n#pragma HLS pipeline\n```"
                        },
                    }
                ],
                "usage": {"completion_tokens": 4},
            }

            with mock.patch.object(
                run_vllm_corpus_eval,
                "_post_chat_completion",
                return_value=response,
            ) as post:
                first = run_vllm_corpus_eval.run_eval(args)
                second = run_vllm_corpus_eval.run_eval(args)

            self.assertEqual(post.call_count, 2)
            self.assertEqual(first["selection"]["records_selected"], 2)
            self.assertEqual(first["selection"]["records_succeeded"], 2)
            self.assertEqual(second["state"], "complete")
            self.assertEqual(
                len((root / "generation.jsonl").read_text().splitlines()),
                2,
            )

    def test_vitis_journal_resumes_terminal_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "generation.jsonl"
            generated = []
            for benchmark in ("hlsfactory_2mm", "hlsfactory_bicg"):
                generated.append(
                    {
                        "status": "ok",
                        "finish_reason": "stop",
                        "record": {"benchmark": benchmark},
                        "response": {
                            "content": "```cpp\n#pragma HLS pipeline\n```"
                        },
                    }
                )
            input_path.write_text(
                "".join(json.dumps(record) + "\n" for record in generated)
            )
            args = _vitis_args(root, input_path)

            def fake_case(record, **_kwargs):
                benchmark = record["record"]["benchmark"]
                return {
                    "schema_version": "vllm_vitis_smoke_v1",
                    "benchmark": benchmark,
                    "synth": {
                        "status": "pass",
                        "runtime_seconds": 1.0,
                        "error": "",
                        "latency_cycles": 10,
                        "fmax_mhz": 300.0,
                        "bram": 1,
                        "dsp": 1,
                        "ff": 1,
                        "lut": 1,
                    },
                    "csim": {
                        "status": "pass",
                        "runtime_seconds": 1.0,
                        "error": "",
                    },
                }

            with mock.patch.object(
                run_vllm_vitis_smoke,
                "_run_case",
                side_effect=fake_case,
            ) as run_case:
                first = run_vllm_vitis_smoke.run(args)
                second = run_vllm_vitis_smoke.run(args)

            self.assertEqual(run_case.call_count, 2)
            self.assertEqual(first["counts"]["completed"], 2)
            self.assertEqual(second["counts"]["pending"], 0)
            self.assertEqual(
                len(
                    (
                        root / "compact" / "test" / "results.jsonl"
                    ).read_text().splitlines()
                ),
                2,
            )


if __name__ == "__main__":
    unittest.main()
