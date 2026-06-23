# Flash API — commercial LLM flash tests (team server)

Run the **same flash test matrix** as the PC2 Devstral experiments, on the **team
server** with **Claude / OpenAI APIs**. No PC2 cluster, no `local.env`, no vLLM,
no `--pc2`.

## Zero extra setup (if you already run agentic sweeps)

These scripts use the **same paths and API key files** as `run_agentic_sweep.py`:

- Vitis / XRT / U280 platform → `c2hls_paths.TEAM_DEFAULTS` (via `configure_site("team")`)
- Claude key → `C2HLS_CLAUDE_KEY_FILE` (default `/home/luo00466/claude-api-key.txt`)
- OpenAI key → `C2HLS_OPENAI_KEY_FILE` (default `/home/luo00466/gpt-key.txt`)
- Temp scratch → `C2HLS_TMP_ROOT` or `C2HLS_SWEEP_TMP_ROOT` (default `/mnt/data/luo00466/tmp`)

Optional: put overrides in repo-root `.env` (loaded automatically).

**Do not set `OPENAI_BASE_URL`** — that is for self-hosted models on PC2 only.

## Quick start

```bash
cd /path/to/c2hls

# Preflight + plan (no benchmarks executed)
python3 scripts/flash_api/run_flash_batch.py --profile nav_o --dry-run

# One variant, all 28 hlsfactory kernels, default Claude Sonnet (cosim on)
python3 scripts/flash_api/run_flash_batch.py --profile nav_o

# Faster pilot: csynth + csim only (matches PC2 flash runs)
python3 scripts/flash_api/run_flash_batch.py --profile nav_o --skip-cosim
# or: export C2HLS_FLASH_API_SKIP_COSIM=1

# OpenAI
python3 scripts/flash_api/run_flash_batch.py --profile aav_n --model gpt-4o

# Top-5 variants (same as flash_top5_comparison.tex)
bash scripts/flash_api/start_top5.sh --dry-run

# Full 10-mode deterministic matrix
bash scripts/flash_api/start_deterministic_matrix.sh
```

Preflight checks Vitis path, API key files, and that `OPENAI_BASE_URL` is unset.

## Layout

| Path | Role |
|------|------|
| `scripts/flash_api/` | Entry points for **commercial API** runs |
| `scripts/flash_shared/` | Variant + skill wiring shared with PC2 (no Slurm) |
| `scripts/pc2/` | PC2 + vLLM only — **team does not need this** |
| `artifacts/flash_api/` | API run outputs (gitignored) |
| `artifacts/pc2/` | PC2 outputs (gitignored) |

## Profiles ↔ PC2 tests

Each API run writes `pc2_mirror` in `manifest.json` (e.g. `nav_o`) so you can
compare against PC2 artifacts without re-running PC2.

| API `--profile` | Same test as PC2 | Skills library |
|-----------------|------------------|----------------|
| `nav_o` | No avoids (old) | `skills.json` |
| `aav_n` | All+avoids (new) | `(90skills).json` |
| `nav_n` | No avoids (new) | `(73skills).json` |
| `noskills_old` | Noskills (old) | none |
| `aav_o` | All+avoids (old) | `skills.json` |

Full list: `python3 scripts/flash_api/run_flash_batch.py --list-profiles`

## Artifact coordinates

```
artifacts/flash_api/<artifact_prefix>_<stamp>/<bench>/<model_tag>__<setup_tag>/
```

Example:

```
artifacts/flash_api/flash_all_skills_no_avoids_global_20260624_120000/
  hlsfactory_gemm/sonnet__flash__all_skills_no_avoids_global/
```

`matrix.json` at the artifact root matches the PC2 schema.

## What differs from PC2 runs

| | PC2 (`scripts/pc2/`) | Team API (`scripts/flash_api/`) |
|--|----------------------|----------------------------------|
| LLM | Devstral via vLLM | Claude / GPT via API |
| Paths | `local.env` + Otus modules | `TEAM_DEFAULTS` / `.env` |
| Artifacts | `artifacts/pc2/` | `artifacts/flash_api/` |
| Skill variants | identical | identical |
| Cosim | off (pilot; `local.env`) | **on by default**; `--skip-cosim` or `C2HLS_FLASH_API_SKIP_COSIM=1` |

PC2 cosim-repair / multiloop tooling lives only under `scripts/pc2/` — not used by flash API.

Latencies will differ by model; compare structure and geo-mean lat/GT, not bit-identical cycles.
