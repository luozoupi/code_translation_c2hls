# autosa_mm zero-shot / no-skills Implementation Plan

> **For agentic workers:** Execute inline in this session. Skip git commits unless the user asks. Do not submit HLS/LLM jobs without `--endpoint-url`.

**Goal:** Add `--flavor zero_shot|noskills` to the existing mm-flow launcher so the professor can compare zero-shot and no-skills against frozen with-skills, without re-injecting skills JSON on workers.

**Architecture:** Keep one launcher and one batch JSON. Flavor sets variant + env. New AutoSA flash variants (`autosa_noskills`, `autosa_zero_shot`) so `configure_autosa_campaign_env` does not call `aav_n_gf` (which always loads gemm_flatten + no_RMW). DSE/stream honor `C2HLS_POST_FLASH_NO_SKILLS=1` and return an empty skills block while still sending the PE-recipe text.

**Tech Stack:** bash launcher, `autosa_flash_lib.py`, `batch_parallel_autosa_lib.py`, `post_flash_dse.py`, `post_flash_stream.py`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-autosa-mm-zero-shot-noskills-design.md`

**Locked:** Do not overwrite `20260830_mmflow`. Default flavor stays with-skills (`mmflow`, gemm_flatten_v1). Cosim off. U280 3.33 ns. DeepSeek-v4-flash.

---

## File map

| Path | Role |
|------|------|
| `scripts/pc2/autosa_flash_lib.py` | `configure_autosa_flash_noskills_env`, `configure_autosa_flash_zero_shot_env`, `apply_mm_flow_flavor`, setup tags |
| `scripts/pc2/batch_parallel_autosa_lib.py` | Register variants; dispatch configure |
| `post_flash_dse.py` | `post_flash_no_skills()`; empty `build_dse_skills_prompt_block`; omit skills heading when empty |
| `post_flash_stream.py` | Same empty-block path |
| `scripts/pc2/start_autosa_mm_flow.sh` | `--flavor`; prefixes; campaign.json flavor fields |
| `scripts/pc2/start_batch_parallel_variant.sh` | Export no-skills / skip-phase-b / turns from campaign.json |
| `scripts/pc2/batch_parallel_config.py` | Restore flavor env from campaign.json |
| `tests/test_autosa_mm_flow_flavors.py` | Flavor env + launcher strings |
| `tests/test_post_flash_dse.py` | Empty skills when flag set |
| `tests/test_post_flash_stream.py` | Empty skills when flag set |
| `tests/test_autosa_flash_aav_n.py` | Dispatch accepts new variants; default aav_n_gf unchanged |

Reuse: existing zero-shot prompts (`C2HLS_FLASH_OPT_PROMPT_MODE=zero_shot`), `C2HLS_SKIP_PHASE_B`, `format_recipe_prompt`. Do not point mm at `benchmarks_cosim`.

Skip git commits unless the user asks.

---

### Task 1: Failing tests

**Files:**
- Create: `tests/test_autosa_mm_flow_flavors.py`
- Modify: `tests/test_post_flash_dse.py`, `tests/test_post_flash_stream.py`, `tests/test_autosa_flash_aav_n.py`

Cover:

- `C2HLS_POST_FLASH_NO_SKILLS=1` → DSE and stream skill blocks empty (`skill_count=0`, no `hls-dse-` / `hls-stream-` ids). Default (unset) still loads JSON.
- `configure_autosa_flash_noskills_env`: `skill_off`, no packaged JSON, no flash overlay, `C2HLS_POST_FLASH_NO_SKILLS=1`.
- `configure_autosa_flash_zero_shot_env`: plus `C2HLS_SKIP_PHASE_B=1`, `C2HLS_FLASH_OPT_PROMPT_MODE=zero_shot`.
- Dispatch `validate_variant` accepts `autosa_noskills` and `autosa_zero_shot` for `autosa_flash` workflow.
- `apply_mm_flow_flavor("noskills")` sets variant `autosa_noskills`, job prefix `mmns`, DSE+stream on.
- `apply_mm_flow_flavor("zero_shot")` sets variant `autosa_zero_shot`, job prefix `mmzs`, DSE+stream off, turns=1.
- `apply_mm_flow_flavor("skills")` does not clear gemm pack if already set.
- `start_autosa_mm_flow.sh` contains `--flavor`, `zero_shot`, `noskills`, `mmzs`, `mmns`; still contains `mmflow` and `gemm_flatten`.

Do not break `test_mm_flow_config_uses_mmflow` or `test_flow_launcher_accepts_32x8_pe_recipe`.

---

### Task 2: Empty DSE/stream skills path

When `C2HLS_POST_FLASH_NO_SKILLS` is truthy, `build_*_skills_prompt_block` returns `("", {skills_path: "", skill_count: 0, skill_ids: []})` without opening the JSON files.

Initial user prompt: if the block is empty, do not say “using the DSE/stream skills below” and omit the skills heading. Keep the PE-recipe via `format_recipe_prompt` (already in `benchmark_context`).

Default (flag unset) behavior and `prompt_text_for_docs()` stay as today.

---

### Task 3: AutoSA variants + flavor apply

Add:

- `VARIANT_NOSKILLS = "autosa_noskills"`
- `VARIANT_ZERO_SHOT = "autosa_zero_shot"`
- setup tags `flash__autosa__noskills` / `flash__autosa__zero_shot`

`_configure_autosa_runtime_env()` = current non-skill half of `_configure_autosa_flash_env` (strategy, part, clock, timeouts, cosim off). Skills half stays in `_configure_autosa_flash_env`.

`apply_mm_flow_flavor(flavor: str) -> dict` mutates `os.environ` for the **launcher** (prefixes, DSE/stream flags, turns, skill strip). Worker-side configure functions re-apply skill_off so `aav_n_gf` cannot leak in.

Register both variants in `AUTOSA_VARIANTS` and `configure_autosa_campaign_env`.

---

### Task 4: Launcher + campaign.json + worker export

`start_autosa_mm_flow.sh`:

- Parse `--flavor` (`skills` default). Unknown flavor → exit 2.
- `--pe-recipe` + `--flavor zero_shot` → exit 2.
- After common mm-flow exports, call `apply_mm_flow_flavor`.
- Echo flavor, skills on/off, dse, stream, turns.
- Write `flavor.txt`, `post_flash_no_skills`, `skip_phase_b` into campaign.json / campaign root.
- Proxy dir: `autosa_mm_flow_${flavor}_${STAMP}` (skills keeps `aav_n_gf` in the path as today: `autosa_mm_flow_aav_n_gf_${STAMP}`).

`apply_autosa_flow_from_campaign` restores `C2HLS_POST_FLASH_NO_SKILLS`, `C2HLS_SKIP_PHASE_B`, `C2HLS_FLASH_OPT_PROMPT_MODE`, `C2HLS_TURNS` when present.

`start_batch_parallel_variant.sh` adds those keys to the sbatch `--export` bits from campaign.json.

Do not change `batch_parallel_autosa_mm_flow.json` job_prefix (`mmflow`). Flavor overrides `PC2_BATCH_JOB_PREFIX` via env.

---

### Task 5: Verify

```
.venv/bin/python -m pytest \
  tests/test_autosa_mm_flow_flavors.py \
  tests/test_post_flash_dse.py \
  tests/test_post_flash_stream.py \
  tests/test_autosa_flash_aav_n.py \
  tests/test_post_flash_pe_recipe.py \
  tests/test_batch_parallel_job_prefix.py \
  tests/test_zero_shot_flash_prompt.py \
  -q
```

Dry-run (no submit):

```
./scripts/pc2/start_autosa_mm_flow.sh --flavor noskills --dry-run --stamp pytest_mmns
./scripts/pc2/start_autosa_mm_flow.sh --flavor zero_shot --dry-run --stamp pytest_mmzs
```

Do not start a real job. Do not commit.
