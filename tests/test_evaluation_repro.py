from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import evaluation_repro as repro  # noqa: E402


class EvaluationProfileTests(unittest.TestCase):
    def test_paper_profile_forces_reference_blind_invariants(self):
        env = {
            "C2HLS_GT_AWARE_REVERT": "1",
            "C2HLS_PHASE8_BASELINE_ALIGN": "1",
            "C2HLS_PHASE5_GT_PREPOP": "1",
            "C2HLS_COSIM_SKIP_SLOWER_THAN_GOLD": "1",
            "C2HLS_SKILL_LIBRARY_PERSIST": "1",
            "C2HLS_SKILL_UPDATE_STATS": "1",
        }
        profile = repro.apply_evaluation_profile("paper", environ=env)
        self.assertTrue(profile["reference_blind"])
        self.assertEqual(profile["name"], repro.PAPER_PROFILE)
        for key, value in repro.REFERENCE_BLIND_OVERRIDES.items():
            self.assertEqual(env[key], value)

    def test_legacy_profile_does_not_rewrite_oracle_settings(self):
        env = {"C2HLS_GT_AWARE_REVERT": "1"}
        profile = repro.apply_evaluation_profile("legacy", environ=env)
        self.assertFalse(profile["reference_blind"])
        self.assertEqual(env["C2HLS_GT_AWARE_REVERT"], "1")


class FingerprintTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, dict[str, str]]:
        repo = root / "repo"
        bench = repo / "benchmarks" / "tiny"
        bench.mkdir(parents=True)
        for name in (
            "c2hls.py",
            "hls_eval.py",
            "prompt_c2hls.py",
            "run_agentic_sweep.py",
            "evaluation_repro.py",
        ):
            (repo / name).write_text(f"# {name}\n", encoding="utf-8")
        (repo / "skills").mkdir()
        (repo / "skills" / "skills.json").write_text('{"skills": []}\n')
        frozen = repo / "paper_eval" / "frozen" / "skills.json"
        frozen.parent.mkdir(parents=True)
        frozen.write_text(
            '{"schema":"1.1","skills":[{"id":"validated",'
            '"pattern":"p","strategy":"s"}]}\n',
            encoding="utf-8",
        )
        settings = repo / "settings64.sh"
        settings.write_text("# test Vitis settings\n", encoding="utf-8")
        (bench / "metadata.json").write_text(
            json.dumps({"benchmark": "tiny", "plain_c_file": "plain.cpp"})
        )
        (bench / "plain.cpp").write_text("int workload(int x) { return x; }\n")
        env = {
            "C2HLS_VITIS_VERSION": "2023.2",
            "C2HLS_VITIS_SETTINGS": str(settings),
            "C2HLS_PART": "xcu280-fsvh2892-2L-e",
            "C2HLS_CLOCK_NS": "3.33",
            "C2HLS_STRATEGY": "dynamic",
            "C2HLS_DYNAMIC_ROUTING": "1",
            "C2HLS_SKILL_LIBRARY_FROZEN": "1",
            "C2HLS_SKILL_LIBRARY_PATH": str(frozen),
            "C2HLS_SKILL_LIBRARY_PERSIST": "0",
            "C2HLS_SKILL_UPDATE_STATS": "0",
            "C2HLS_MODEL_REVISION": "weights-deadbeef",
            "C2HLS_LLM_TEMPERATURE": "0.2",
            "C2HLS_LLM_TOP_P": "0.95",
            "C2HLS_LLM_SEED": "7",
            "C2HLS_MAX_COMPLETION_TOKENS": "8192",
            "C2HLS_SYNTHESIS_EVAL_BUDGET": "5",
        }
        env.update(repro.REFERENCE_BLIND_OVERRIDES)
        env["C2HLS_SKILL_SNAPSHOT_SHA256"] = repro.skill_snapshot_manifest(
            repo, environ=env
        )["sha256"]
        return repo, bench, env

    def _build(self, repo: Path, bench: Path, env: dict[str, str]):
        with patch.object(
            repro,
            "_probe_vitis_version",
            return_value={
                "command": ["vitis-run", "--version"],
                "executable": "/opt/Xilinx/Vitis/2023.2/bin/vitis-run",
                "executable_sha256": "b" * 64,
                "ran": True,
                "returncode": 0,
                "version": "2023.2",
                "output_sha256": "a" * 64,
                "error": "",
            },
        ):
            return repro.build_run_fingerprint(
                repo=repo,
                benchmark_dir=bench,
                benchmark="tiny",
                model_id="qwen/tiny",
                model_label="qwen",
                skill_mode="skill_on",
                steps=["pipeline"],
                profile={"name": repro.PAPER_PROFILE},
                environ=env,
            )

    def test_fingerprint_is_stable_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo, bench, env = self._fixture(Path(tmpdir))
            first = self._build(repo, bench, env)
            second = self._build(repo, bench, env)
            self.assertTrue(repro.fingerprint_matches(first, second))

            (bench / "plain.cpp").write_text("int workload(int x) { return x + 1; }\n")
            changed_input = self._build(repo, bench, env)
            self.assertNotEqual(first["sha256"], changed_input["sha256"])

            (bench / "plain.cpp").write_text("int workload(int x) { return x; }\n")
            env["C2HLS_LLM_SEED"] = "8"
            changed_seed = self._build(repo, bench, env)
            self.assertNotEqual(first["sha256"], changed_seed["sha256"])

            env["C2HLS_LLM_SEED"] = "7"
            (repo / "prompt_c2hls.py").write_text("# changed prompt\n")
            changed_prompt = self._build(repo, bench, env)
            self.assertNotEqual(first["sha256"], changed_prompt["sha256"])

    def test_tampered_payload_or_digest_never_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo, bench, env = self._fixture(Path(tmpdir))
            original = self._build(repo, bench, env)
            tampered = json.loads(json.dumps(original))
            tampered["payload"]["toolchain"]["clock_ns"] = "5.0"
            self.assertFalse(repro.fingerprint_matches(tampered, original))
            self.assertFalse(repro.fingerprint_matches({}, original))

    def test_completeness_exposes_unresolved_revision_and_decoding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo, bench, env = self._fixture(Path(tmpdir))
            complete = self._build(repo, bench, env)
            self.assertTrue(repro.fingerprint_completeness(complete)["complete"])

            del env["C2HLS_MODEL_REVISION"]
            del env["C2HLS_LLM_SEED"]
            incomplete = self._build(repo, bench, env)
            report = repro.fingerprint_completeness(incomplete)
            self.assertFalse(report["complete"])
            self.assertIn("model_revision_unresolved", report["issues"])
            self.assertIn("decoding_seed_not_explicit", report["issues"])

    def test_skill_snapshot_mismatch_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo, bench, env = self._fixture(Path(tmpdir))
            env["C2HLS_SKILL_SNAPSHOT_SHA256"] = "0" * 64
            fingerprint = self._build(repo, bench, env)
            report = repro.fingerprint_completeness(fingerprint)
            self.assertIn("skill_snapshot_hash_mismatch", report["issues"])

    def test_configured_vitis_must_match_invoked_binary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo, bench, env = self._fixture(Path(tmpdir))
            fingerprint = self._build(repo, bench, env)
            fingerprint["payload"]["toolchain"]["vitis_version_probe"][
                "version"
            ] = "2024.1"
            report = repro.fingerprint_completeness(fingerprint)
            self.assertIn("actual_vitis_version_mismatch", report["issues"])

    def test_vitis_probe_must_have_executed_successfully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo, bench, env = self._fixture(Path(tmpdir))
            fingerprint = self._build(repo, bench, env)
            probe = fingerprint["payload"]["toolchain"]["vitis_version_probe"]
            probe.update({"ran": False, "returncode": 1, "error": "failed"})
            issues = repro.fingerprint_completeness(fingerprint)["issues"]
            self.assertIn("actual_vitis_probe_not_run", issues)
            self.assertIn("actual_vitis_probe_failed", issues)
            self.assertIn("actual_vitis_probe_error", issues)

    def test_role_specific_model_override_disqualifies_paper_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo, bench, env = self._fixture(Path(tmpdir))
            env["C2HLS_TRANSLATOR_MODEL"] = "unrevisioned-other-model"
            issues = repro.fingerprint_completeness(
                self._build(repo, bench, env)
            )["issues"]
            self.assertIn("agent_model_override_forbidden:translator", issues)

    def test_skill_on_requires_and_hashes_only_explicit_frozen_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo, bench, env = self._fixture(Path(tmpdir))
            fingerprint = self._build(repo, bench, env)
            skills = fingerprint["payload"]["skills"]
            self.assertEqual("explicit_frozen_snapshot", skills["source_mode"])
            self.assertEqual(1, skills["file_count"])
            self.assertEqual("paper_eval/frozen/skills.json", skills["files"][0]["path"])

            del env["C2HLS_SKILL_LIBRARY_PATH"]
            env["C2HLS_SKILL_SNAPSHOT_SHA256"] = repro.skill_snapshot_manifest(
                repo, environ=env
            )["sha256"]
            report = repro.fingerprint_completeness(self._build(repo, bench, env))
            self.assertIn("frozen_skill_path_missing", report["issues"])


class ProvenanceStatusTests(unittest.TestCase):
    def test_actual_call_model_decoding_and_revision_are_enforced(self):
        payload = {
            "model": {
                "id": "qwen/tiny",
                "revision": {"value": "weights-1", "resolved": True},
            },
            "decoding": {
                "temperature": "0.2",
                "top_p": "0.95",
                "seed": "7",
                "max_completion_tokens": "8192",
            },
        }
        fingerprint = {"payload": payload}
        event = {
            "provider": "openai",
            "model": "qwen/tiny",
            "model_revision": "weights-1",
            "max_tokens": 8192,
            "prompt_sha256": "a" * 64,
            "decoding": {
                "temperature": 0.2,
                "top_p": 0.95,
                "seed": 7,
                "seed_supported": True,
            },
        }
        result = {"llm_usage": {"calls": 1, "events": [event]}}
        self.assertEqual([], repro.effective_llm_call_issues(result, fingerprint))
        event["model"] = "other"
        event["decoding"]["top_p"] = 0.5
        issues = repro.effective_llm_call_issues(result, fingerprint)
        self.assertIn("llm_call_0:model_mismatch", issues)
        self.assertIn("llm_call_0:top_p_mismatch", issues)

    def test_predicted_cosim_skip_is_not_an_executed_timeout(self):
        status = repro.derive_status_taxonomy({
            "final_report": {"latency_cycles": 10},
            "csim": {"ran": True, "passed": True, "status": "passed"},
            "cosim": {
                "ran": False,
                "passed": False,
                "status": "timeout",
                "skip_reason": "predicted_longer_than_gold",
                "cosim_policy": {"classification": "predicted_timeout"},
            },
        })
        self.assertEqual(status["correctness_status"], "passed")
        self.assertEqual(status["cosim_execution_status"], "not_run")
        self.assertTrue(status["cosim_predicted_skip"])
        self.assertFalse(status["timeout"])

    def test_executed_timeout_and_tool_error_remain_distinct(self):
        timeout = repro.derive_status_taxonomy({
            "final_report": {"latency_cycles": 10},
            "cosim": {"ran": True, "status": "failed", "error": "timed out"},
        })
        self.assertEqual(timeout["cosim_execution_status"], "timeout")
        self.assertTrue(timeout["timeout"])

        tool = repro.derive_status_taxonomy({
            "synthesis_evaluations": {
                "events": [{
                    "synthesis_ran": True,
                    "success": False,
                    "status": "tool_failure",
                    "tool_failure": True,
                    "error": "vitis executable missing",
                }]
            }
        })
        self.assertEqual(tool["synthesis_status"], "tool_failure")
        self.assertTrue(tool["tool_failure"])

    def test_status_taxonomy_uses_typed_candidate_events(self):
        csim_only = repro.derive_status_taxonomy({
            "error": "no correct feasible candidate",
            "correctness_status": "failed",
            "csim": {"ran": True, "passed": False, "status": "failed"},
            "synthesis_evaluations": {
                "events": [{"synthesis_ran": False, "success": False}]
            },
        })
        self.assertEqual("failed", csim_only["correctness_status"])
        self.assertEqual("not_run", csim_only["synthesis_status"])
        self.assertFalse(csim_only["tool_failure"])

        synthesis_failed = repro.derive_status_taxonomy({
            "correctness_status": "passed",
            "synthesis_evaluations": {
                "events": [{
                    "synthesis_ran": True,
                    "success": False,
                    "status": "failed",
                }]
            },
        })
        self.assertEqual("failed", synthesis_failed["synthesis_status"])
        self.assertFalse(synthesis_failed["tool_failure"])

        infeasible_but_synthesized = repro.derive_status_taxonomy({
            "correctness_status": "passed",
            "error": "no correct feasible candidate",
            "synthesis_evaluations": {
                "events": [{
                    "synthesis_ran": True,
                    "success": True,
                    "status": "passed",
                }]
            },
        })
        self.assertEqual("passed", infeasible_but_synthesized["synthesis_status"])
        self.assertFalse(infeasible_but_synthesized["tool_failure"])

        synthesis_timeout = repro.derive_status_taxonomy({
            "correctness_status": "passed",
            "synthesis_evaluations": {
                "events": [{
                    "synthesis_ran": True,
                    "success": False,
                    "status": "timeout",
                    "timed_out": True,
                }]
            },
        })
        self.assertEqual("timeout", synthesis_timeout["synthesis_status"])
        self.assertTrue(synthesis_timeout["timeout"])

        provider_failure = repro.derive_status_taxonomy({
            "correctness_status": "not_run",
            "llm_usage": {"events": [{"error": "provider unavailable"}]},
            "synthesis_evaluations": {"events": []},
        })
        self.assertEqual("not_run", provider_failure["synthesis_status"])
        self.assertTrue(provider_failure["provider_failure"])
        self.assertTrue(provider_failure["tool_failure"])

    def test_attach_does_not_invent_effective_decoding_or_synthesis_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = Path(tmpdir) / "history.json"
            history.write_text(json.dumps({"messages": []}))
            payload = {
                "schema_version": repro.FINGERPRINT_SCHEMA,
                "decoding": {"temperature": "0.2", "top_p": "0.95", "seed": "0"},
                "model": {"revision": {"value": "rev", "resolved": True}},
                "toolchain": {"vitis_version": "2023.2", "part": "u280", "clock_ns": "3.33"},
                "budgets": {"candidate_budget": "5"},
                "skills": {"mode": "skill_off"},
            }
            fingerprint = {
                "schema_version": repro.FINGERPRINT_SCHEMA,
                "sha256": repro.sha256_json(payload),
                "payload": payload,
            }
            result: dict = {"run": {}}
            repro.attach_run_provenance(
                result,
                fingerprint=fingerprint,
                profile={"name": repro.PAPER_PROFILE, "reference_blind": True},
                elapsed_seconds=1.5,
                history_path=history,
            )
        self.assertIsNone(result["run"]["decoding"]["effective"])
        self.assertIsNone(result["run"]["synthesis_evaluations"])
        self.assertEqual(result["correctness_status"], "not_run")
        self.assertEqual(result["executed_cosim_status"], "not_run")
        self.assertFalse(result["predicted_cosim_skip"])
        self.assertEqual(result["timeout_status"], "none")
        self.assertEqual(result["tool_failure_status"], "none")
        self.assertIn(
            "effective_decoding_unreported",
            result["run"]["reproducibility"]["issues"],
        )
        self.assertIn(
            "synthesis_evaluation_count_missing",
            result["run"]["reproducibility"]["issues"],
        )


if __name__ == "__main__":
    unittest.main()
