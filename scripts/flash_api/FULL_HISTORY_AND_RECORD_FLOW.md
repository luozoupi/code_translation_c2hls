# Full LLM history and skill records (commercial API)

How to get PC2-style full prompts in `*_history.json` and flow sidecars (`*_flash_skills.json`, manifests) when running flash via **commercial LLM APIs** — no PC2 cluster, vLLM, Slurm, or `local.env`.

Implemented on branch **`c2hls_enhanced_l_pc2_api_layout`** (commits `82c1933`, `578ca6e`).

## 1. Get the code

```bash
git fetch origin
git checkout c2hls_enhanced_l_pc2_api_layout
git pull
```

Alternatively, stay on `c2hls_enhanced_l` and cherry-pick:

```bash
git fetch origin
git cherry-pick 82c1933 578ca6e
```

## 2. Run flash via API (not `scripts/pc2/`)

```bash
cd /path/to/c2hls

# Preflight (Vitis + API keys; confirms OPENAI_BASE_URL is unset)
python3 scripts/flash_api/check_setup.py

# Example: same skill variant as PC2 aav_n
python3 scripts/flash_api/run_flash_batch.py --profile aav_n --model claude-sonnet-4-6

# Faster pilot (csynth + csim only, no cosim)
python3 scripts/flash_api/run_flash_batch.py --profile aav_n --skip-cosim
```

**Do not set `OPENAI_BASE_URL`** — that is for self-hosted models on PC2 only. Use `C2HLS_CLAUDE_KEY_FILE` / `C2HLS_OPENAI_KEY_FILE` (or `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`), same as `run_agentic_sweep.py`.

More examples: [README.md](README.md).

## 3. Artifacts per benchmark cell

Output root:

```
artifacts/flash_api/<artifact_prefix>_<stamp>/<bench>/<model_tag>__<setup_tag>/
```

| File | Contents |
|------|----------|
| `hlsfactory_<bench>_history.json` | Full LLM chat. **Skills are in the `[Step: flash]` user message** (Phase B has no skills). |
| `hlsfactory_<bench>_flash_skills.json` | Structured skill injection record (e.g. 90 skills for `aav_n`). |
| `hlsfactory_<bench>_flow_manifest.json` | Which kernel was selected (`phase_b` vs `flash_opt`). |
| `skills_source.json` | Copy of the skill catalog used for that cell. |

`C2HLS_RECORD_FLOW=1` is **on by default** for API runs (`scripts/flash_shared/team_env.py`), so sidecars are written without manual `export`.

To disable sidecars (history is still full):

```bash
export C2HLS_RECORD_FLOW=0
```

## 4. Verify skills are present

```bash
# In history.json — expect ~90 for profile aav_n
grep -c '\[skill ' artifacts/flash_api/.../hlsfactory_gemm_history.json

# Or read the sidecar
jq '.flash_opt.injected_skill_count' artifacts/flash_api/.../hlsfactory_gemm_flash_skills.json
```

In `*_history.json`, find the **large user message** starting with `[Step: flash]` and `GLOBAL SKILL LIBRARY`. Earlier Phase B translate messages intentionally contain **no** optimization skills.

## 5. Skill profiles (same variants as PC2)

| `--profile` | Skills |
|-------------|--------|
| `aav_n` | All new skills + avoids (90) |
| `nav_n` | All new skills, no avoids (73) |
| `aav_o` / `nav_o` | Old 55-skill catalog |
| `noskills_old` | None |

List all profiles:

```bash
python3 scripts/flash_api/run_flash_batch.py --list-profiles
```

Each run’s `manifest.json` includes `pc2_mirror` (e.g. `aav_n`) for comparison with PC2 artifacts.

## 6. What you can ignore

| Path | Reason |
|------|--------|
| `scripts/pc2/` | PC2 batch-parallel, vLLM, Slurm |
| `benchmarks_cosim/` | Not in git; API uses `benchmarks/` |
| `local.env` | PC2 cluster config only |

## 7. Minimum files involved (for reference)

| Component | Role |
|-----------|------|
| `c2hls.py` | Writes **full** optimization prompts to `*_history.json` |
| `flash_flow_artifacts.py` | Writes `*_flash_skills.json`, flow manifests, per-step `.cpp` copies |
| `scripts/flash_api/` | Commercial API entry points |
| `scripts/flash_shared/team_env.py` | Sets `C2HLS_RECORD_FLOW=1` by default for API |
