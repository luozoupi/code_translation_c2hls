"""Compatibility and formatting contracts for versioned model inputs."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
from string import Formatter
import unittest

from hlsmith.prompt_profiles import DEFAULT_PROFILE, PROFILE_NAMES, load_profile


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "reproducibility/paper/prompts/prompt_c2hls.py"


def fields(value):
    return Counter(name for _, name, _, _ in Formatter().parse(value)
                   if name is not None)


class PromptProfiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = {key: value for key, value in runpy.run_path(str(FROZEN)).items()
                        if not key.startswith("_") and isinstance(value, (str, list, dict))}

    def test_default_strings_maps_and_sequences_match_frozen_source(self):
        profile = load_profile()
        self.assertEqual(profile.name, "paper-sigmetrics-2026")
        self.assertEqual(profile.exports, self.original)
        self.assertEqual(profile.metadata["source_sha256"], hashlib.sha256(FROZEN.read_bytes()).hexdigest())

    def test_default_rendering_preserves_every_original_byte(self):
        templates = load_profile().templates
        for name, value in self.original.items():
            if not isinstance(value, str):
                continue
            with self.subTest(template=name):
                values = {field: f"SENTINEL_{field}_{{literal}}" for field in fields(value)}
                self.assertEqual(templates[name].format(**values).encode(), value.format(**values).encode())

    def test_readable_templates_keep_all_placeholder_names_and_multiplicities(self):
        original, readable = load_profile(), load_profile("researcher-v1")
        self.assertEqual(set(original.templates), set(readable.templates))
        self.assertEqual(len(original.templates), 23)
        for name, value in original.templates.items():
            with self.subTest(template=name):
                self.assertEqual(fields(value), fields(readable.templates[name]))
                rendered = readable.templates[name].format(**{key: f"<{key}>" for key in fields(value)})
                self.assertTrue(rendered)
                self.assertNotEqual(value, readable.templates[name])

    def test_step_map_uses_selected_templates_and_sequences_stay_identical(self):
        readable = load_profile("researcher-v1").exports
        original = load_profile().exports
        for step, template in readable["OPTIMIZATION_PROMPTS"].items():
            self.assertEqual(template, readable[f"q_optimize_{step}"])
        for name, value in original.items():
            if isinstance(value, list):
                self.assertEqual(value, readable[name])

    def test_exports_are_isolated_between_runs_and_from_original_module(self):
        first = load_profile()
        values = first.exports
        values["DEFAULT_OPT_STEPS"].clear()
        values["OPTIMIZATION_PROMPTS"]["flash"] = "changed"
        self.assertEqual(first.exports, self.original)
        self.assertEqual(load_profile().exports, self.original)

    def test_hash_identifies_complete_exported_content(self):
        for name in PROFILE_NAMES:
            with self.subTest(profile=name):
                profile = load_profile(name)
                payload = json.dumps(profile.exports, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()
                self.assertEqual(profile.metadata["sha256"], hashlib.sha256(payload).hexdigest())
                self.assertEqual(profile.metadata, load_profile(name).metadata)
        self.assertNotEqual(load_profile().metadata["sha256"], load_profile("researcher-v1").metadata["sha256"])

    def test_import_and_default_loading_leave_optional_clients_and_profile_unloaded(self):
        source = """
import sys
import hlsmith.prompt_profiles as profiles
assert 'prompt_c2hls' not in sys.modules
assert 'hlsmith.prompts.researcher_v1' not in sys.modules
profiles.load_profile()
assert 'hlsmith.prompts.researcher_v1' not in sys.modules
assert 'openai' not in sys.modules
assert 'anthropic' not in sys.modules
assert 'matplotlib' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", source], cwd=ROOT, check=True)

    def test_unknown_profile_fails_with_available_names(self):
        with self.assertRaisesRegex(ValueError, "paper-sigmetrics-2026, researcher-v1"):
            load_profile("missing-profile")

    def test_readable_translation_preserves_harness_constraints(self):
        templates = load_profile("researcher-v1").templates
        for name in ("q_translate_c_to_hls", "q_translate_c_to_hls_functional", "q_optimize_flash", "q_optimize_coalescing"):
            for phrase in ("workload()", 'extern "C"', "bundle=control", "argument order", "pointer element"):
                with self.subTest(template=name, constraint=phrase):
                    self.assertIn(phrase, templates[name])
        for phrase in ("PIPELINE", "UNROLL", "DATAFLOW", "ARRAY_PARTITION"):
            self.assertIn(phrase, templates["q_translate_c_to_hls_functional"])
        self.assertIn("explicitly", templates["q_optimize_coalescing"])


if __name__ == "__main__":
    unittest.main()
