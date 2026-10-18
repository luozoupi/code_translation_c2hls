"""Versioned model-facing prompts, loaded only when a run selects a profile.

The default exports the original prompt module without changing its strings.
The researcher profile is a new experimental input, not a claim of equivalent
model behavior. Importing this module never imports model clients or tools.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
from string import Formatter
from types import MappingProxyType
from typing import Any, Mapping


DEFAULT_PROFILE = "paper-sigmetrics-2026"
PROFILE_NAMES = (DEFAULT_PROFILE, "researcher-v1")
_TEMPLATE_MODULE = "prompt_c2hls"


@dataclass(frozen=True)
class PromptProfile:
    """A prompt snapshot with the legacy runtime's public variable names."""

    name: str
    _exports: Mapping[str, Any]
    _metadata: Mapping[str, Any]

    @property
    def exports(self) -> dict[str, Any]:
        # Runs may modify step lists; never share mutable values between runs.
        return copy.deepcopy(dict(self._exports))

    @property
    def templates(self) -> Mapping[str, str]:
        return MappingProxyType({key: value for key, value in self._exports.items()
                                 if isinstance(value, str)})

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)


def _fields(template: str) -> list[str]:
    return sorted(name for _, name, _, _ in Formatter().parse(template)
                  if name is not None)


def load_profile(name: str = DEFAULT_PROFILE) -> PromptProfile:
    """Return fresh exports and stable content hashes for a known profile.

    ``exports`` can update the runtime's prompt globals. ``metadata`` should
    accompany every run fingerprint. Unknown names fail before any API work.
    """
    if name not in PROFILE_NAMES:
        raise ValueError(f"Unknown prompt profile {name!r}; choose {', '.join(PROFILE_NAMES)}")
    original = importlib.import_module(_TEMPLATE_MODULE)
    exports = {key: copy.deepcopy(value) for key, value in vars(original).items()
               if not key.startswith("_") and isinstance(value, (str, dict, list))}
    original_templates = {key: value for key, value in exports.items()
                          if isinstance(value, str)}
    source_hash = hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()
    if name == "researcher-v1":
        readable = importlib.import_module("hlsmith.prompts.researcher_v1")
        replacements = readable.TEMPLATES
        if replacements.keys() != original_templates.keys():
            raise ValueError("Researcher prompt names differ from the paper profile")
        for key, value in replacements.items():
            if _fields(value) != _fields(original_templates[key]):
                raise ValueError(f"Prompt placeholders differ for {key}")
        exports.update(replacements)
        # The legacy map stores template values, so rebuild it after replacing
        # the named templates. Step names and step sequences stay identical.
        exports["OPTIMIZATION_PROMPTS"] = {
            step: exports[f"q_optimize_{step}"]
            for step in exports["OPTIMIZATION_PROMPTS"]
        }
    serialized = json.dumps(exports, sort_keys=True, ensure_ascii=False,
                            separators=(",", ":")).encode("utf-8")
    metadata = {
        "name": name,
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "template_count": len(original_templates),
        "source_sha256": source_hash,
        "schema": "hlsmith.prompt-profile.v1",
    }
    return PromptProfile(name, MappingProxyType(exports), MappingProxyType(metadata))
