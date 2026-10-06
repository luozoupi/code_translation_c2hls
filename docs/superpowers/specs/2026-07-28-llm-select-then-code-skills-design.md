# LLM select-then-code skills variant

**Status:** approved (conversation 2026-07-27/28)  
**Goal:** New flash skills variant that selects library skills via an LLM pre-call, then codes with only the selected skills (plus optional short `own_knowledge`).

## Non-goals

- Do not change `aav_n` / `noskills` / `bare` / existing `llm_curated` behavior.
- No dataflow, lat_opt, RAG, or RAG2 for the smoketest.

## Behavior

### Mode / variant

| Knob | Value |
|------|--------|
| `C2HLS_SKILL_PROMPT_MODE` | `llm_select_then_code` |
| Fixed-cosim variant key | `aav_sel` |
| Skills JSON | same 90-skill gemm_flatten package as skills campaigns |
| Include avoids | yes (parity with `aav_n`) |

### Per codegen / repair turn

1. **Selector LLM** (no code output) receives:
   - current code, synth summary, bottlenecks/feedback, diagnostics/errors
   - **full library**, full-fidelity render (all skills+avoids; no truncation of steps/guards/template; authored `...` in JSON stays)
2. Selector returns JSON only:
   ```json
   {
     "selected_skill_ids": ["..."],
     "avoid_skill_ids": ["..."],
     "own_knowledge": []
   }
   ```
   - IDs must exist in library; unknown IDs dropped
   - `own_knowledge` optional (empty OK)
   - no cap on number of selected skills/avoids
   - **Broad selection policy:** include every skill that could reduce latency,
     raise compute throughput, or increase memory parallelism; prefer more over
     fewer; when unsure, include.
3. **Orchestrator** builds coder skill block:
   - full-fidelity render of all selected + avoid skills
   - if `own_knowledge` non-empty, append section labeled exactly:  
     `further notes: model HLS knowledge here, not library skills`
   - constraints on notes: short text, no full kernels, must not invent skill ids, prefer implementing selected skills first
4. **Coder LLM** receives that block + usual code/reports/errors; produces code only.

### Frequency

Re-run selector **before every** flash codegen and code-repair turn that would inject skills.

### Fallback

If selector fails / empty after validation → bottleneck fallback (existing helper), still full-fidelity render of fallback skills; record `used_fallback=true`.

### Artifacts

Persist each selection turn (ids, own_knowledge, raw reply, injected char counts) under cell flow / `skill_selection.json` (append list) without breaking existing `*_flash_skills.json` schema consumers.

## Smoketest

- Model: `deepseek-v4-flash`
- Benches: `hlsfactory_2mm`, `hlsfactory_heat-3d`
- Flash + cosim only; no dataflow / lat_opt / rag / rag2
