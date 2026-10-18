#!/usr/bin/env python3
"""Record a reporting-only correction without resealing generation inputs."""

import argparse
import fcntl
import json
from pathlib import Path
import shutil

from run_qor_paired_model_campaign import atomic_json, now, sha256


def amend(config, original):
    root = Path(config["report_root"])
    with (root / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((root / "campaign_state.json").read_text())
        if state["status"] != "smoke_failed":
            raise ValueError("auditor amendment requires a stopped, failed smoke gate")
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        auditor = Path(__file__).with_name("analyze_qor_review_ab.py").resolve()
        entry = next(e for e in manifest["input_files"] if Path(e["path"]).resolve() == auditor)
        if sha256(original) != entry["sha256"]:
            raise ValueError("original auditor does not match frozen source")
        for item in manifest["input_files"]:
            if item is not entry and sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError("generation/config/input drift; cannot apply reporting-only amendment")
        snapshot = root / "auditor_amendments" / now().replace(":", "-")
        snapshot.mkdir(parents=True, exist_ok=False)
        for name in ("manifest.json", "campaign_state.json"):
            shutil.copy2(root / name, snapshot / name)
        shutil.copy2(original, snapshot / "auditor.before.py")
        shutil.copy2(auditor, snapshot / "auditor.after.py")
        manifest.setdefault("analysis_amendments", []).append({
            "at":now(), "path":str(auditor), "old_sha256":entry["sha256"], "new_sha256":sha256(auditor),
            "reason":"Accept disabled/never-attempted QoR placeholder; use current runtime status in tables.",
            "generation_inputs_unchanged":True, "saved_samples_reused_without_new_generation":True,
            "snapshot":str(snapshot),
        })
        entry["sha256"] = sha256(auditor)
        atomic_json(path, manifest)
        print(json.dumps(manifest["analysis_amendments"][-1]))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--original-auditor",type=Path,required=True)
    args=parser.parse_args()
    amend(json.loads(args.config.read_text()),args.original_auditor)
