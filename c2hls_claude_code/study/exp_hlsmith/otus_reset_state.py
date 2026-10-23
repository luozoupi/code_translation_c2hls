#!/usr/bin/env python3
"""otus_reset_state.py [--apply] : prepare driver_state.json for the move to Otus (2026-10-22).

Run it only after driver.py has stopped: the running driver keeps the state in memory and writes it back.
For every tag that has not finished, it drops the Fir job id, so the driver resubmits on Otus instead of waiting on
a job that doesn't exist there. Wall-time tier, memory and try count stay as they are, so a rerun starts at the tier
it reached on Fir. A tag the freeze marked final as CANCELLED (the driver saw the scancel) is reopened the same way.
Free-agent tags (afw_*) have no job id: if the session ended on Fir after the driver stopped (exit_code in its
workspace), the tag is marked final with that exit, so the Otus driver doesn't resume a finished session. A session
with no exit_code is only reported: the driver would resume it on Otus, mixing the two clusters in one run.
Without --apply it only prints what would change. With --apply it first writes driver_state.json.bak_fir.
"""
import json, shutil, sys
from pathlib import Path

STATE = Path(__file__).resolve().parent / "driver_state.json"
st = json.loads(STATE.read_text())
changed, warn = [], []
AFW = STATE.parents[1] / "agentfw" / "runs"
for tag, rec in st.items():
    if tag.startswith("afw_") and "final" not in rec:
        ws = AFW / f"mm1024_hlsm_r{tag[-1]}"
        if (ws / "exit_code").exists():
            rec["final"] = f"exit {(ws / 'exit_code').read_text().strip()}"
            changed.append(f"{tag}: session ended on Fir after the driver stopped: final = {rec['final']}")
        elif (ws / "transcript.jsonl").exists():
            warn.append(f"{tag}: UNFINISHED agent session in {ws} (no exit_code). The Otus driver would resume it there, "
                        "mixing Fir and Otus in one run. Let it finish on Fir before the copy, or move the workspace "
                        "aside so the driver starts it fresh.")
    elif str(rec.get("final", "")).startswith("CANCELLED"):
        changed.append(f"{tag}: reopen (final was {rec.pop('final')}), drop Fir job {rec.pop('job', None)}")
    elif "final" not in rec and "job" in rec:
        changed.append(f"{tag}: drop Fir job {rec.pop('job')} (tier {rec['tier']}, tries {rec['tries']})")
print("\n".join(changed) or "nothing to change")
if warn:
    print("\n".join(warn))
if changed and "--apply" in sys.argv:
    shutil.copy2(STATE, STATE.with_name(STATE.name + ".bak_fir"))
    STATE.write_text(json.dumps(st, indent=1) + "\n")
    print(f"wrote {STATE} (backup {STATE.name}.bak_fir)")
