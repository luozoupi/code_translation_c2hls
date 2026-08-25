from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import c2hls  # noqa: E402
import export_schema_jsonl as schema_export  # noqa: E402


class ReferenceValidationCacheTests(unittest.TestCase):
    @staticmethod
    def _stable_unavailable_probe() -> dict:
        return {
            "ran": False,
            "returncode": None,
            "version": None,
            "executable": None,
            "executable_sha256": None,
            "output_sha256": None,
            "error": "not probed in non-paper cache test",
        }

    def _inputs(self, root: Path) -> dict:
        bench_dir = root / "hlsfactory_cache_test"
        bench_dir.mkdir()
        (bench_dir / "hls_baseline.cpp").write_text("void kernel(double a[4]) { a[0] = 1; }\n")
        return {
            "bench_dir": str(bench_dir),
            "bench_name": "hlsfactory_cache_test",
            "header_name": "kernel.h",
            "header_code": "void kernel(double a[4]);\n",
            "testbench_code": "int main() { double a[4] = {}; kernel(a); return 0; }\n",
            "extra_files": [],
            "meta": {
                "benchmark": "hlsfactory_cache_test",
                "source_repo": "HLSFactory",
                "hls_top": "kernel",
                "part": "xcu280-fsvh2892-2L-e",
                "clock_ns": 3.33,
                "supports_csim": True,
                "supports_cosim": True,
                "preferred_gt_file": "hls_baseline.cpp",
                "variants": [{
                    "name": "hlsfactory_cache_test_0_baseline",
                    "file": "hls_baseline.cpp",
                }],
            },
        }

    def _validation(self, cosim_status: str = "not_run") -> dict:
        cosim_passed = cosim_status == "passed"
        return {
            "benchmark_ready": True,
            "invalid_reason": "",
            "reference_source": "local_vitis",
            "selected_variant_file": "hls_baseline.cpp",
            "selected_variant_name": "hlsfactory_cache_test_0_baseline",
            "selected_variant_step": "baseline",
            "synthesis": {
                "status": "passed",
                "success": True,
                "report": {"latency_cycles": 100},
            },
            "csim": {
                "status": "passed",
                "supported": True,
                "ran": True,
                "passed": True,
            },
            "cosim": {
                "status": cosim_status,
                "supported": True,
                "ran": cosim_passed,
                "passed": cosim_passed,
                "kernel_runtime_cycles": 110 if cosim_passed else None,
            },
            "report": {"latency_cycles": 100},
            "workflow": [],
        }

    def test_exact_input_cache_hit_and_source_change_invalidation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inputs = self._inputs(root)
            validation = self._validation(cosim_status="passed")
            env = {
                c2hls.REFERENCE_CACHE_DIR_ENV: str(root / "cache"),
                c2hls.REFERENCE_CACHE_REQUIRE_COSIM_ENV: "0",
                "C2HLS_REFERENCE_VALIDATE_MODE": "trusted_external",
                "C2HLS_REFERENCE_BLIND": "0",
                "C2HLS_VITIS_VERSION": "2023.2",
                "C2HLS_FLOW_TARGET": "vitis",
            }
            with (
                patch.dict(os.environ, env, clear=False),
                patch(
                    "evaluation_repro._probe_vitis_version",
                    return_value=self._stable_unavailable_probe(),
                ),
                patch.object(
                    c2hls,
                    "_validate_gold_reference_uncached",
                    side_effect=lambda _: copy.deepcopy(validation),
                ) as validate_mock,
            ):
                first = c2hls.validate_gold_reference(inputs)
                second = c2hls.validate_gold_reference(inputs)
                self.assertEqual(validate_mock.call_count, 1)
                self.assertFalse(first["reference_cache"]["hit"])
                self.assertTrue(first["reference_cache"]["written"])
                self.assertTrue(second["reference_cache"]["hit"])
                self.assertEqual(second["reference_source"], "cached_local_vitis")

                Path(inputs["bench_dir"], "hls_baseline.cpp").write_text(
                    "void kernel(double a[4]) { a[0] = 2; }\n"
                )
                third = c2hls.validate_gold_reference(inputs)
                self.assertEqual(validate_mock.call_count, 2)
                self.assertFalse(third["reference_cache"]["hit"])

    def test_cache_identity_binds_probed_executable_and_settings_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inputs = self._inputs(root)
            settings = root / "settings64.sh"
            settings.write_text("# settings v1\n")
            base_probe = {
                "ran": True,
                "returncode": 0,
                "version": "2023.2",
                "executable": "/opt/vitis-run",
                "executable_sha256": "a" * 64,
                "output_sha256": "b" * 64,
                "error": "",
            }
            env = {
                "C2HLS_VITIS_VERSION": "2023.2",
                "C2HLS_VITIS_SETTINGS": str(settings),
                "C2HLS_REFERENCE_BLIND": "1",
            }
            with (
                patch.dict(os.environ, env, clear=False),
                patch("evaluation_repro._probe_vitis_version", return_value=base_probe),
            ):
                first = c2hls._reference_cache_descriptor(inputs)
                self.assertTrue(c2hls._paper_reference_cache_identity_complete(first))

            changed_probe = dict(base_probe, executable_sha256="c" * 64)
            with (
                patch.dict(os.environ, env, clear=False),
                patch("evaluation_repro._probe_vitis_version", return_value=changed_probe),
            ):
                changed_executable = c2hls._reference_cache_descriptor(inputs)
            self.assertNotEqual(first["fingerprint"], changed_executable["fingerprint"])

            settings.write_text("# settings v2\n")
            with (
                patch.dict(os.environ, env, clear=False),
                patch("evaluation_repro._probe_vitis_version", return_value=base_probe),
            ):
                changed_settings = c2hls._reference_cache_descriptor(inputs)
            self.assertNotEqual(first["fingerprint"], changed_settings["fingerprint"])

    def test_paper_cache_rejects_unexecuted_toolchain_probe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inputs = self._inputs(root)
            validation = self._validation(cosim_status="passed")
            env = {
                c2hls.REFERENCE_CACHE_DIR_ENV: str(root / "cache"),
                "C2HLS_REFERENCE_BLIND": "1",
                "C2HLS_VITIS_VERSION": "2023.2",
            }
            with (
                patch.dict(os.environ, env, clear=False),
                patch(
                    "evaluation_repro._probe_vitis_version",
                    return_value={
                        "ran": False,
                        "returncode": None,
                        "version": None,
                        "executable": None,
                        "executable_sha256": None,
                        "error": "missing",
                    },
                ),
            ):
                self.assertIsNone(
                    c2hls._write_reference_validation_cache(inputs, validation)
                )
                cached, provenance = c2hls._load_reference_validation_cache(inputs)
            self.assertIsNone(cached)
            self.assertEqual(
                "paper_cache_toolchain_identity_incomplete",
                provenance["rejection_reason"],
            )

    def test_partial_cosim_cache_is_reused_only_when_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inputs = self._inputs(root)
            validation = self._validation(cosim_status="not_run")
            base_env = {
                c2hls.REFERENCE_CACHE_DIR_ENV: str(root / "cache"),
                "C2HLS_REFERENCE_VALIDATE_MODE": "trusted_external",
                "C2HLS_REFERENCE_BLIND": "0",
                "C2HLS_VITIS_VERSION": "2023.2",
                "C2HLS_FLOW_TARGET": "vitis",
            }
            with (
                patch.dict(os.environ, dict(base_env, **{
                    c2hls.REFERENCE_CACHE_REQUIRE_COSIM_ENV: "0",
                }), clear=False),
                patch(
                    "evaluation_repro._probe_vitis_version",
                    return_value=self._stable_unavailable_probe(),
                ),
            ):
                cache_path = c2hls._write_reference_validation_cache(inputs, validation)
                self.assertIsNotNone(cache_path)
                cached = c2hls.validate_gold_reference(inputs)
            self.assertTrue(cached["reference_cache"]["hit"])
            self.assertEqual(cached["cosim"]["status"], "not_run")

            with (
                patch.dict(os.environ, dict(base_env, **{
                    c2hls.REFERENCE_CACHE_REQUIRE_COSIM_ENV: "1",
                }), clear=False),
                patch(
                    "evaluation_repro._probe_vitis_version",
                    return_value=self._stable_unavailable_probe(),
                ),
                patch.object(
                    c2hls,
                    "_validate_gold_reference_uncached",
                    return_value=copy.deepcopy(validation),
                ) as validate_mock,
            ):
                uncached = c2hls.validate_gold_reference(inputs)
            validate_mock.assert_called_once()
            self.assertFalse(uncached["reference_cache"]["hit"])
            self.assertEqual(
                uncached["reference_cache"]["rejection_reason"],
                "cached_reference_cosim_not_passed",
            )

    def test_multistep_jsonl_retains_reference_cache_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bench_dir = root / "hlsfactory_cache_test"
            bench_dir.mkdir()
            (bench_dir / "metadata.json").write_text(json.dumps({
                "benchmark": "hlsfactory_cache_test",
                "source_repo": "HLSFactory",
                "variants": [{"name": "hlsfactory_cache_test_0_baseline"}],
            }))
            result_path = root / "hlsfactory_cache_test_multistep_results.json"
            result_path.write_text(json.dumps({
                "run": {
                    "model": "qwen3.6-27b",
                    "part": "xcu280-fsvh2892-2L-e",
                    "clock_ns": 3.33,
                },
                "reference_validation": {
                    "reference_cache": {
                        "schema_version": "1.0",
                        "hit": True,
                        "fingerprint": "abc123",
                        "cosim_status": "not_run",
                    },
                },
                "baseline_report": {"latency_cycles": 100},
                "final_report": {"latency_cycles": 100},
                "steps": [],
            }))

            records = schema_export._records_from_multistep(
                bench_dir,
                result_path,
                "xcu280-fsvh2892-2L-e",
                3.33,
            )

        self.assertEqual(len(records), 1)
        origin_meta = records[0]["implementation"]["origin_meta"]
        self.assertTrue(origin_meta["reference_cache"]["hit"])
        self.assertEqual(origin_meta["reference_cache"]["cosim_status"], "not_run")


if __name__ == "__main__":
    unittest.main()
