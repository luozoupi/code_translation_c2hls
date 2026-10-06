# Frozen flash 139484/10: why we cannot replay it, and the 16K V4.1 isolation

Lab notebook for Ahmad / Professor Fang. Quote **latency min/max + DSP** (never interval). Do not treat DSE as the slide story; DSE is only in this file because it is an internal confounder we already ruled out for *flash* QoR.

Canonical path: `docs/frozen-flash-139k-replication.md`  
Pin pack / order (not a frozen tree): `artifacts/pc2/flash16k_v41_replay/`  
Launcher: `scripts/pc2/start_autosa_mm_flash16k_v41.sh`

---

## 1. Question

Frozen AutoSA-mm **flash** on 2026-08-30 produced a legal but almost serial kernel: **139484–139484 cycles, DSP 10**, UF=8, sequential load / compute / store. Later flashes on the same skill *set* (pack sha `847494f44a9b4a6e`) finished (`finish_reason=stop`) and synthesized **4878–4878 / DSP 336**, **5390–5390 / DSP 336**, and nothink **12895–12895 / DSP 320**.

We cannot treat 139K as “the 90-skill flash result.” We need to know whether it was:

1. a **truncated 16K completion** on real V4-Flash,
2. a **thinking-channel** effect (CoT dumped into `content`),
3. a **V4 vs V4.1** backend change after the 2026-09-10 retirement,
4. ordinary **stochasticity**, or
5. an irreproducible run because we have **no snapshot model id**.

This experiment isolates **today’s API** (request `deepseek-v4-flash` → live `deepseek-flash` / V4.1-Flash) at **frozen 16384 tokens**, **frozen skill sequence**, thinking **on vs off**. Flash only. No DSE 2.0, no DSE v1, no stream.

---

## 2. Frozen fact sheet

| Field | Value |
| --- | --- |
| Campaign | `artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow` |
| Cell | `.../variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/` |
| Created / completed | 2026-08-30T11:18:19Z → 2026-08-30T11:31:31Z |
| Slurm | watch `2723764`, drain `2723765`, coord `2723766`, synth `2723767` |
| Endpoint | `http://login5:18092/v1`, requested model `deepseek-v4-flash` |
| Flavor | `aav_n_gf` / `gemm_flatten_v1` / `all_skills_avoids_global` |
| Skills | 129 injected; pack file 99 ids + no-RMW overlay 33 (3 id overlaps); pack sha **`847494f44a9b4a6e`** |
| ABI | `autosa_mm(A[I][K], B[J][K], C[I][J])`, I=J=K=64 float |
| Part / clock / Vitis | `xcu280-fsvh2892-2L-e`, **3.33 ns**, Vitis 2023.2 |
| **Flash QoR** | **latency 139484–139484 cycles, DSP 10** (`autosa_mm_flash_opt_report.json`; interval 139485 is *not* the quote) |
| Kernel | `#define UF 8`, sequential load_A/B/C → k-blocked MAC → store_C; no DATAFLOW; DSP 10 |
| Drain (flash call) | `finish_reason=length`, `content_len=59697`, `out=16384` |
| Assistant start | “We need respond with optimized kernel complete code…” (English CoT in `content`) |
| History `llm_usage` | **all zeros** (`calls=0`); do **not** use JSON usage — use drain/slurm `content_len` vs `out` |
| `max_tokens` | implicit **16384** (no `C2HLS_FLASH_MAX_TOKENS` in frozen `campaign.json`) |
| Thinking API field | **omitted** (hosted default ON). Not `thinking.type=disabled`. |
| Ping-pong | Fang lock: mmflow does not *enforce* ping-pong. Frozen **flash kernel itself** is sequential UF=8, not ping-pong. |
| What continued after flash | DSE v1 then stream (out of scope here). Stream/selected is **4292–4292 / DSP 320**, not 139K. |

Phase B on the same drain (not the 139K kernel):

| Call | `finish_reason` | `content_len` | `out` |
| --- | --- | --- | --- |
| Phase B translate | `length` | 31614 | 8191 |
| Phase B compile repair | `stop` | 882 | 939 |
| **Flash** | **`length`** | **59697** | **16384** |

Three closed ` ```cpp ` fences exist inside the truncated flash `content`; the last closed fence is the UF=8 kernel. Extra English continues after the fence until the 16K cap.

---

## 3. Ways investigated (evidence, not narrative)

### 3.1 Thinking unspecified vs off vs on

`c2hls.py` `_openai_compat_extra_body`: if `C2HLS_THINKING` is unset, DeepSeek `extra_body` has **no** `thinking` key (API default ON). `C2HLS_THINKING=disabled` sends `thinking: {type: disabled}`.

| Run | Date | Thinking sent | Flash `max_tokens` | Flash drain | Flash QoR |
| --- | --- | --- | --- | --- | --- |
| HLSFactory v4-flash (Jul 26–28 plans/campaigns) | 2026-07 | unspecified (no `thinking` in campaign.json) | typical 16K-era | English CoT often in `content` | not AutoSA-mm 139K |
| Frozen mmflow flash | 2026-08-30 | omitted | **16384** | `length`, `out=16384`, `content_len=59697` | **139484–139484 / 10** |
| Frozen mmflow DSE v1 16×4 (post-flash) | 2026-08-30 | n/a for flash | n/a | DSE is a later LLM on the 139K seed | DSE csynth **13160–13160 / 352**; not flash |
| DSE 2.0 A | 2026-09-14 `..._20260914_084009_dse_v2` | omitted (`thinking` absent) | **65536** | flash `stop`, `out=19792`, `content_len=4996` | **4878–4878 / 336** |
| DSE 2.0 B | 2026-09-14 `..._dse_v2b_..._084315_dse_v2_b` | omitted | **65536** | flash `stop`, `out=28272`, `content_len=3304` | **5390–5390 / 336** |
| DSE 2.0 nothink | 2026-09-14 `..._214846_dse_v2_nothink` | `disabled` | **65536** | flash `stop`, `out=746`, `content_len=2259` | **12895–12895 / 320** |

JSON `llm_usage` on frozen (and typically on batch cells) is zeros. Always quote drain:

```
LLM usage model=deepseek-v4-flash finish_reason=... content_len=... in=... out=... total=...
```

### 3.2 History stores `content` only

Frozen flash assistant `content` starts as first-person English (“We need respond with optimized kernel…”) then eventually emits fences. That **looks** like non-thinking English, but it is consistent with **thinking spilled into the content channel** when the dedicated thinking field is not persisted. History has no `reasoning_content`. Truncated CoT in `content` is not proof that thinking was off.

### 3.3 DeepSeek backend timeline

- **2026-07-24:** DeepSeek chat/reasoner cutoff (industry docs). Pre-cutoff and post-cutoff V4-Flash are not the same object even if the request id matches.
- **2026-09-10:** V4-Flash retired. Request id `deepseek-v4-flash` is routed to **V4.1-Flash**. Live response `model` is `deepseek-flash`.
- Frozen **2026-08-30** is **pre-retirement real V4-Flash**. We have **no snapshot id**. Today’s alias is not the frozen weights.

### 3.4 `max_tokens` 16384 vs 65536

Frozen flash: `finish_reason=length`, `out=16384`.  
Later DSE 2.0 launcher default: `C2HLS_FLASH_MAX_TOKENS=65536`. Those flashes `stop` with `out` 746–28272, all **below** 65536.

`c2hls.py` later grew `_FLASH_MAX_COMPLETION_FLOOR = 65536` when the env is **unset**. An explicit `16384` must not be raised to that floor (gated in this work).

### 3.5 Skills / prompt pack matched later — 139K is a *bad truncated flash*

DSE 2.0 A/B and nothink flash_skills:

- `skills_source.sha256` = **`847494f44a9b4a6e…`** (same file as frozen `skills_source.json`, byte-identical to the repo gemm_flatten pack).
- Injected **129** ids; **set-equal** to frozen.
- **Dump order is not frozen.** Current `_BASELINE_FIRST_SKILL_IDS` places `axi-burst-coalescing-narrow-safe` at index **1**. Frozen user prompt has `hls-avoid-zero-pipeline-submit` at index **1** and burst at index **18**.

Same pack + better token budget → **4878/336 and 5390/336** (thinking default) and **12895/320** (thinking off). 139K is therefore a **weak truncated salvage**, not “what the 90-skill pack does.”

### 3.6 DSE 2.0 seed chaining does not explain flash QoR

DSE 2.0 runs **after** flash on the flash kernel. Flash csynth in A/B/nothink is already 4878 / 5390 / 12895. Chaining cannot create those flash numbers retroactively. Frozen 139K is likewise a **flash** report, written before DSE/stream artifacts.

---

## 4. Ranked remaining causes

1. **Truncation at 16384** on a thinking-on completion (`finish_reason=length`). Frozen spent the whole budget on CoT-in-content and salvaged UF=8.
2. **Thinking channel vs `content`.** Default-on thinking + history that only keeps `content` produces the “non-thinking English” look. Nothink 65536 already yields a finished, better kernel (12895/320), so thinking is *causal* for length/quality, not only cosmetics.
3. **V4 vs V4.1.** Frozen was real V4-Flash. Today’s alias is V4.1. Even a perfect 16K replay may not return 139K.
4. **Stochasticity.** DSE 2.0 A vs B (4878 vs 5390) at the same 65536/default-thinking settings already moves QoR.
5. **No snapshot model id.** We cannot pin frozen weights. This experiment is “closest legal replay on today’s endpoint,” not a bit-identical rerun.

---

## 5. This experiment

### 5.1 Design

Two **flash-only** cells. Same everything except thinking:

| Arm | Thinking | API extra_body | `max_tokens` | Continuations | DSE / stream |
| --- | --- | --- | --- | --- | --- |
| think | `C2HLS_THINKING=api_default` (maps to omit) | **omit** `thinking` | **16384** | **0** (one shot, like frozen) | off |
| nothink | `C2HLS_THINKING=disabled` | `thinking: {type: disabled}` | **16384** | **0** | off |

ABI / part / clock / Vitis / flavor / `C2HLS_AUTOSA_FLOW=1` / enforcement off: same as frozen. Ping-pong still not enforced; we are not asking the model to emit ping-pong.

Phase B stays on orchestrator default **8192** (`C2HLS_LLM_MAX_TOKENS=8192`), matching frozen Phase B `out=8191`.

### 5.2 How skill order was pinned

Do **not** use a directory listing or today’s default injector order.

Extracted `[skill <id>]` markers from frozen **user** message `[Step: flash]` in `autosa_mm_history.json` (129 unique ids). That sequence **equals** `autosa_mm_flash_skills.json` `flash_opt.injected_skills` **and** `hls_full_optimization_skills_schema_1_1_package/flash_skill_order_20260830_mmflow.json`.

Launch copies that extraction to:

`artifacts/pc2/flash16k_v41_replay/flash_skill_order_from_frozen_user_prompt.json`

and sets `C2HLS_SKILL_PROMPT_ORDER_JSON` to that file.

Pack file is **byte-identical** to frozen `skills_source.json` (sha `847494f44a9b4a6e…`). Overlay `flash_no_RMW_m_axi_skill_entries.json` supplies the extra 33 ids (3 overlap the pack). Copies live under `flash16k_v41_replay/` so we do not mutate frozen trees. Workers still `configure_autosa_flash_aav_n_gf_env()`, which re-points the pack env at the repo gemm_flatten path; that path is the same bytes.

Phase B / flash **templates** still match frozen skeletons (`q_translate_c_to_hls_functional`, `q_optimize_flash`); frozen flash user text diverges only at `{synth_report}` substitution (expected).

### 5.3 Launch

```bash
# thinking ON (omit thinking field)
./scripts/pc2/start_autosa_mm_flash16k_v41.sh \
  --thinking-on --endpoint-url http://login5:18092/v1

# thinking OFF
./scripts/pc2/start_autosa_mm_flash16k_v41.sh \
  --thinking-off --endpoint-url http://login5:18092/v1
```

Gated code (default campaigns unchanged):

- `_flash_max_completion_tokens`: explicit values **below** 65536 (e.g. 16384) are honored, not raised to the floor.
- `_cpp_continuation_limit`: `C2HLS_CPP_CONTINUATIONS=0` is a real one-shot (previously `max(1, …)`).
- `init_campaign_json`: persists `skill_prompt_order_json` and `flash_only` (forces DSE/stream/DSE2 off).

### 5.4 Campaign paths and job ids

Launched 2026-09-14 23:01 UTC on `login5`. Proxy `http://login5:18092/v1` `/v1/models` listed `deepseek-v4-flash` (HTTP 200). Both drains were already in Phase B when helpers came up.

| Arm | Campaign root | thinking.txt | Helpers | Synth |
| --- | --- | --- | --- | --- |
| ON | `artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_flash16k_v41_think_20260914_230136_flash16k_v41_think` | `api_default` | watch `2987643`, drain `2987644`, coord `2987645` | `2987649` (`mmf16t-synth-n0-…`) |
| OFF | `artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_flash16k_v41_nothink_20260914_230136_flash16k_v41_nothink` | `disabled` | watch `2987646`, drain `2987647`, coord `2987648` | `2987650` (`mmf16n-synth-n0-…`) |

`campaign.json` on both: `flash_only=true`, `flash_max_tokens=16384`, `llm_max_tokens=8192`, `cpp_continuations=0`, `post_flash_dse/dse_v2/post_flash_stream=false`. Job prefixes `mmf16t` / `mmf16n`.

Dry-run leftover (not a result cell): `..._flash16k_v41_think_20260915_dryrun_think`.

In-flight drain (not QoR; csynth may still be running):

| Arm | Call | `finish_reason` | `content_len` | `out` |
| --- | --- | --- | --- | --- |
| ON | Phase B | `stop` | 678 | 348 |
| OFF | Phase B | `stop` | 721 | 252 |
| OFF | likely flash (`in≈51214` skill dump) | `stop` | 1987 | 691 |

Nothink 16K already `stop`s a short flash (`out=691`), unlike frozen flash `length`/`out=16384`. Confirm after history exists. Do not quote latency until `autosa_mm_flash_opt_report.json` is written.

### 5.5 How to read success (do not wait on HLS forever)

After drain logs a flash `LLM usage` line, the synth worker still has to csynth (~minutes). Harvest:

```text
CELL=$CAMPAIGN/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf
# 1) tokens
grep 'LLM usage' $CAMPAIGN/flow/gpu_drain.log
# 2) QoR — latency min/max + DSP
python3 -c "import json; r=json.load(open('$CELL/autosa_mm_flash_opt_report.json'));
print(r['latency_cycles'], r['latency_cycles_worst'], r['dsp'])"
```

Success criteria to log:

| Field | Frozen | What to record |
| --- | --- | --- |
| `finish_reason` | `length` | `length` vs `stop` |
| `content_len` | 59697 | integer |
| `out` | 16384 | must be ≤16384; 16384 + `length` = truncated like frozen |
| latency min/max | 139484–139484 | from report, not interval |
| DSP | 10 | |
| kernel | UF=8 sequential | `autosa_mm_flash_opt.cpp` |

Confirm injected order: `python3` compare `[skill …]` in new history user flash message to `flash16k_v41_replay/flash_skill_order_from_frozen_user_prompt.json`.

---

## 6. How we will interpret results

- **Recovering ~139K / DSP 10 with thinking on + 16K** supports “truncation on V4.1 still produces the weak UF=8 salvage.” Closest thing we can get to replaying frozen flash without V4 weights.
- **Finished kernel (`stop`) even at 16K**, thinking on, means V4.1’s thinking channel (or tighter `content`) **changed the failure mode** relative to frozen V4.
- **Nothink 16K vs prior nothink 65536 (12895–12895 / 320)** isolates **max_tokens vs thinking**. If 16K nothink truncates or gets much worse than 12895, the 64K budget was load-bearing once thinking is off. If 16K nothink still `stop`s near 12895, thinking (not the 16K cap) was the main lever vs 139K.
- **Neither arm near 139K, both better** (like 4.8K–13K class): V4.1 + current decode path no longer emit the UF=8 salvage; 139K is historical V4 truncation, not a property of the skill pack.
- **Flash fails to extract a kernel** (`_call_llm_for_complete_cpp` returns None on an unclosed fence when continuations=0): remaining mismatch vs frozen, which salvaged a closed inner fence. Document `finish_reason` / fences; do not silently turn continuations back on in this pair.

---

## 7. Open caveats

- **No frozen snapshot id.** Today’s `deepseek-v4-flash` request is V4.1 (`deepseek-flash` in the response).
- **Temperature / seed** not pinned (API default). A/B already showed 4878 vs 5390.
- **Skill dump order** in DSE 2.0 flashes was baseline-first, not frozen. This pair pins frozen order; that is *more* matched than DSE 2.0, so it is not a confounder vs 139K, but it is a difference vs 4878/5390.
- **Workers re-point** `C2HLS_PACKAGED_SKILLS_JSON` at the repo pack. Bytes match frozen; if someone edits the repo pack later, re-check `MANIFEST.json`.
- **Overlay** is still applied by `configure_autosa_flash_aav_n_gf_env` from the repo overlay file (same 33 ids as frozen injection).
- **Continuations:** frozen had one flash HTTP call. We set `C2HLS_CPP_CONTINUATIONS=0`. Current code, if the fence is unclosed, returns None instead of salvaging. Frozen’s last fence *was* closed.
- **Phase B** still runs (frozen did). Repair call may differ. Flash user prompt includes Phase B code + synth report, so Phase B drift can still move flash.
- **`cpp` continuation / complete-kernel guards** did not exist in the same form on 2026-08-30; we disabled extra LLM rounds but not the closed-fence extractor.
- Frozen mmflow **continued** into DSE then stream; those QoR numbers (13160/352 DSE, 4292/320 stream) are **not** this test.
- Do not overwrite frozen trees: `20260830_mmflow`, `20260909_*`, `20260910_234214`, `20260911_075217`, pe16 champion, AutoSA rank-1, `repro2`, `repro2_enf`, `repro2_bdf`, `repro3`, `repro4`.

---

## 8. File map

| Path | Role |
| --- | --- |
| `scripts/pc2/start_autosa_mm_flash16k_v41.sh` | Thin wrapper; does not rewrite frozen launchers in place |
| `scripts/pc2/start_autosa_mm_flow.sh` | Existing mmflow launcher; `C2HLS_FLASH_ONLY=1` skips DSE/stream |
| `artifacts/pc2/flash16k_v41_replay/` | Prompt-extracted order JSON, pack/overlay copies, `MANIFEST.json` |
| Campaign `NOTES.md` / `thinking.txt` / `flash_only.txt` | Pointers back here |
