#!/usr/bin/env python3
"""otus_reset_state.py [--apply] : prepare driver_state.json for the move to Otus (2026-10-22).

Run it only after driver.py has stopped: the running driver keeps the state in memory and writes it back.
For every tag that has not finished, it drops the Fir job id, so the driver resubmits on Otus instead of waiting on
a job that doesn't exist there. Wall-time tier, memory and try count stay as they are, so a rerun starts at the tier
it reached on Fir. A tag the freeze marked final as CANCELLED (the driver saw the scancel) is reopened the same way.
Without --apply it only prints what would change. With --apply it first writes driver_state.json.bak_fir.
"""
import json, shutil, sys
from pathlib import Path

STATE = Path(__file__).resolve().parent / "driver_state.json"
st = json.loads(STATE.read_text())
changed = []
for tag, rec in st.items():
    if str(rec.get("final", "")).startswith("CANCELLED"):
        changed.append(f"{tag}: reopen (final was {rec.pop('final')}), drop Fir job {rec.pop('job', None)}")
    elif "final" not in rec and "job" in rec:
        changed.append(f"{tag}: drop Fir job {rec.pop('job')} (tier {rec['tier']}, tries {rec['tries']})")
print("\n".join(changed) or "nothing to change")
if changed and "--apply" in sys.argv:
    shutil.copy2(STATE, STATE.with_name(STATE.name + ".bak_fir"))
    STATE.write_text(json.dumps(st, indent=1) + "\n")
    print(f"wrote {STATE} (backup {STATE.name}.bak_fir)")
