# Enhanced-branch architecture review

The paper-evaluation worktree reviewed the public GitHub branch
`c2hls_enhanced_l` at commit `da5c145` ("Enhanced-framework repairs + Argo
Sonnet/GPT-5.5 campaign tooling; lean repo") on 2026-07-13.

The branch is useful as an architectural map: it keeps routing
(`bottleneck_router.py`), typed HLS feedback (`hls_feedback.py`), robustness
checks, skill-library logic, corpus/export tooling, and campaign analysis in
named modules around the main controller. The manuscript therefore describes
agents as typed logical roles and emphasizes the compiler-feedback/control
boundary rather than claiming novelty from concurrent personas.

It was not merged wholesale. Relative to the active experiment base, that
branch removes `run_agentic_sweep.py`, changes campaign/result layout, and does
not contain the HPCA reference-isolation, fingerprint, public-contract,
golden-output, matched-budget, or evidence-freeze contracts. A wholesale merge
would also make the still-running legacy queue irreproducible. The isolated
`hpca2027-paper-eval` branch instead preserves the active base and adds the
paper controls with tests, while reusing the enhanced branch only to clarify
component ownership and terminology.
