from __future__ import annotations

from typing import Any


def rank_candidates(rows: list[dict]) -> list[dict]:
    """Rank candidates that passed HLS validation and architecture checks.

    Eligibility requires ``hls_csim_pass``, ``hls_csynth_pass``, ``architecture_ok``,
    and a non-null ``csynth_latency``. Ties break on ``csynth_dsp`` (missing = 0,
    higher is better), then ``cand_id``. Returns new dicts with 1-based
    ``queue_rank``; input rows are not mutated.
    """
    elig = [
        r
        for r in rows
        if r.get("hls_csim_pass")
        and r.get("hls_csynth_pass")
        and r.get("architecture_ok")
        and r.get("csynth_latency") is not None
    ]
    elig.sort(
        key=lambda r: (
            float(r["csynth_latency"]),
            -float(r.get("csynth_dsp") or 0),
            str(r.get("cand_id") or ""),
        )
    )
    ranked: list[dict[str, Any]] = []
    for i, r in enumerate(elig, start=1):
        out = dict(r)
        out["queue_rank"] = i
        ranked.append(out)
    return ranked
