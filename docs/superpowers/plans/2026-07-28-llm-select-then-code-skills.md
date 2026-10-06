# LLM select-then-code skills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add flash variant `aav_sel` (`llm_select_then_code`) that selects uncapped full-fidelity library skills via LLM, then codes with only those skills (+ optional short `own_knowledge`), without changing `aav_n`.

**Architecture:** Extend `skill_library` render/parse helpers; add selector path in `SkillCurationAgent` / `c2hls.py` skill-injection branch; register `aav_sel` in `flash_fixed_cosim_lib`; smoke via flash-only batch config + selected cosim for 2mm/heat-3d on deepseek-v4-flash.

**Tech Stack:** Python 3 (c2hls, skill_library, prompt_c2hls), pytest, Slurm batch_parallel, DeepSeek proxy, Vitis HLS cosim.

**Spec:** `docs/superpowers/specs/2026-07-28-llm-select-then-code-skills-design.md`

---

## File map

| File | Responsibility |
|------|----------------|
| `skill_library.py` | Full-fidelity render; parse/resolve `own_knowledge`; build coder block |
| `prompt_c2hls.py` | Selector user prompt (full library, no skill-count caps) |
| `c2hls.py` | Mode `llm_select_then_code`; call selector every skill-injection turn |
| `scripts/pc2/flash_fixed_cosim_lib.py` | Variant `aav_sel` |
| `tests/test_llm_select_then_code.py` | Offline unit tests |
| `scripts/pc2/batch_parallel_hlsfactory_llm_sel_smoke.json` | 2-bench smoke config |
| `scripts/pc2/start_hlsfactory_llm_sel_smoke.sh` | DeepSeek flash-only + cosim smoke launcher |

---

### Task 1: Full-fidelity render + own_knowledge helpers (TDD)

**Files:**
- Create: `tests/test_llm_select_then_code.py`
- Modify: `skill_library.py`

- [x] **Step 1: Failing tests**
- [x] **Step 2: Run — expect FAIL (symbols missing)**
- [x] **Step 3: Implement helpers in `skill_library.py`**
- [x] **Step 4: Re-run tests — PASS**

---

### Task 2: Selector prompt + agent path

**Files:**
- Modify: `prompt_c2hls.py`
- Modify: `c2hls.py` (`GLOBAL_SKILL_PROMPT_MODES`, skill-injection branch, `SkillCurationAgent`)

- [x] **Step 1: Add `build_skill_selection_user_prompt(...)`** in `prompt_c2hls.py`
- [x] **Step 2: Add `SkillCurationAgent.select_then_code_for_flash(step_name)`**
- [x] **Step 3: Wire mode in injection**
- [x] **Step 4: Unit-test prompt contains full library marker and schema keys**

---

### Task 3: Register `aav_sel` variant

- [x] **Step 1: Add variant**
- [x] **Step 2: Confirm `configure_fixed_cosim_flash_env(aav_sel)` sets mode**

---

### Task 4: Smoke launcher (2mm + heat-3d, deepseek, flash+cosim only)

- [x] **Step 1: Config**
- [x] **Step 2: Launcher**
- [x] **Step 3: Launch smoke** — stamp `20260728_042219_llm_sel_smoke`

---

### Task 5: Regression guard

- [x] Run existing `tests/test_skill_curation_smoke.py` — PASS
- [x] Run `tests/test_llm_select_then_code.py` — PASS
- [x] Confirm `aav_n` env still sets `all_skills_avoids_global`
---

## Self-review

- Spec: no freeform-as-skills; own_knowledge optional with fixed label; full-fidelity; uncapped; re-select every turn; smoke constraints — covered in tasks 1–4.
- No placeholders left.
- Names: `llm_select_then_code`, `aav_sel`, `own_knowledge`, `render_skill_for_prompt_full` consistent.
