# Compact 2-D systolic mesh HLS search

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact 2-D systolic PE mesh to the existing autosa_mm HLS search (enumerate → instantiate → csim+csynth → rank) without calling AutoSA or emitting `kernel0`.

**Architecture:** Keep the 1-D 12-point chain grid. Extend `PeRecipe` with `layout`/`pe_i`/`pe_j`. Enumerate a mesh grid (`PE_I,PE_J ∈ {4,8,16,32}`, SIMD ∈ {2,4,8}, DSP ≤ 85% of 9024, `PE_I×PE_J ≤ 128`). Instantiate a 2-D stream mesh (`mesh_pe`, A per-PE, B along I, C per-PE out) under the same `autosa_mm` ABI. One `ranking.jsonl` for chain ∪ mesh.

**Tech Stack:** Python 3, existing `PeRecipe` / `architecture_ok_for_recipe` / `validate_candidate` / `_run_synth_csim_cosim`, pytest, Vitis HLS 2023.2, U280 `xcu280-fsvh2892-2L-e` 3.33 ns.

**Locked:** Aug 18 slide 4285 vs 4228. Do not overwrite `autosa_mm` / `autosa_mm_32x8` recipes or `20260830_mmflow`. Cosim off. No AutoSA binary. Do not emit `kernel0`. Commit only if the user asks.

**Spec:** `docs/superpowers/specs/2026-08-30-compact-2d-systolic-mesh-search-design.md`

---

## File map

| Path | Role |
|------|------|
| `post_flash_pe_recipe.py` | Add `layout`, `pe_i`, `pe_j` defaults on `PeRecipe` |
| `compact_pe_search.py` | `candidate_id_mesh`, `search_candidate_id`, `enumerate_mm_mesh_recipes` |
| `compact_pe_mesh_instantiate.py` | 2-D mesh C++ emitter |
| `compact_pe_instantiate.py` | `instantiate_mm` dispatches on `layout` |
| `post_flash_stream.py` | Mesh branch in `architecture_ok_for_recipe` only |
| `compact_pe_validate.py` | Use `search_candidate_id`; extra result fields |
| `compact_pe_search_main.py` | Enumerate chain + mesh |
| `scripts/pc2/start_autosa_mm_pe_search.sh` | Dry-run prints mesh ids; walltime 12:00:00 |
| `scripts/pc2/compact_pe_search.sbatch.sh` | `#SBATCH --time=12:00:00` |
| `tests/test_compact_pe_mesh.py` | Mesh grid + instantiate + architecture_ok |
| `tests/test_compact_pe_search_main.py` | Dry-run includes `mesh8x4_simd8` |

Reuse: 1-D instantiate path, `rank_candidates`, `_run_synth_csim_cosim`. Do not rewrite `architecture_ok(bench=...)`.

---

### Task 1: PeRecipe mesh fields (defaults)

**Files:**
- Modify: `post_flash_pe_recipe.py` (`PeRecipe` dataclass)
- Test: `tests/test_post_flash_pe_recipe.py` (add one test; existing tests must still pass)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_post_flash_pe_recipe.py`:

```python
def test_pe_recipe_mesh_fields_default_to_chain(monkeypatch):
    monkeypatch.delenv("C2HLS_PE_RECIPE", raising=False)
    rec = recipe_for("autosa_mm")
    assert rec.layout == "chain"
    assert rec.pe_i == 0
    assert rec.pe_j == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_post_flash_pe_recipe.py::test_pe_recipe_mesh_fields_default_to_chain -v`

Expected: FAIL (`AttributeError: layout` or similar)

- [ ] **Step 3: Write minimal implementation**

In `post_flash_pe_recipe.py`, add three fields to `PeRecipe` **after** `note` (defaults so locked recipes stay valid):

```python
    layout: str = "chain"  # chain | mesh
    pe_i: int = 0  # 0 => use pe (chain)
    pe_j: int = 1
```

Do **not** change `_RECIPES["autosa_mm"]` or `autosa_mm_32x8`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_post_flash_pe_recipe.py tests/test_compact_pe_search.py tests/test_compact_pe_instantiate.py -q`

Expected: PASS (including the new test)

- [ ] **Step 5: Commit only if the user asks**

---

### Task 2: Mesh search space (no Vitis)

**Files:**
- Modify: `compact_pe_search.py`
- Test: `tests/test_compact_pe_mesh.py` (create)

I=J=K=64. Mesh choices: `PE_I, PE_J ∈ {4,8,16,32}`, `SIMD ∈ {2,4,8}`. Keep if `I%pe_i==0`, `J%pe_j==0`, `K%simd==0`, `pe_i*pe_j <= 128`, `pe_i*pe_j*simd*5 <= 0.85*9024`.

Ids: `mesh{pe_i}x{pe_j}_simd{simd}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_compact_pe_mesh.py`:

```python
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compact_pe_search import (
    candidate_id,
    candidate_id_mesh,
    enumerate_mm_mesh_recipes,
    enumerate_mm_recipes,
    search_candidate_id,
)


def test_mesh_grid_includes_autosa_analogs():
    recs = {candidate_id_mesh(r): r for r in enumerate_mm_mesh_recipes()}
    r = recs["mesh8x4_simd8"]
    assert r.layout == "mesh"
    assert r.pe_i == 8 and r.pe_j == 4 and r.simd == 8
    assert r.pe == 32 and r.pack_bits == 256
    assert r.expected_dsp == 1280
    assert r.pe_kj == (64 // 8) * (64 // 4)  # 16 * 16 = 256
    assert r.tile_loop == "inside_tasks"
    r5 = recs["mesh16x8_simd8"]
    assert r5.pe == 128 and r5.expected_dsp == 5120
    assert r5.pe_i == 16 and r5.pe_j == 8


def test_mesh_caps_drop_oversize():
    recs = enumerate_mm_mesh_recipes()
    for r in recs:
        assert r.pe_i * r.pe_j <= 128
        assert r.expected_dsp <= 0.85 * 9024
        assert 64 % r.pe_i == 0 and 64 % r.pe_j == 0 and 64 % r.simd == 0
        assert r.layout == "mesh"
    ids = {candidate_id_mesh(r) for r in recs}
    assert "mesh32x16_simd2" not in ids
    assert 20 <= len(recs) <= 40


def test_search_candidate_id_dispatches():
    chain = next(r for r in enumerate_mm_recipes() if candidate_id(r) == "pe16_simd4")
    mesh = next(r for r in enumerate_mm_mesh_recipes() if candidate_id_mesh(r) == "mesh8x4_simd8")
    assert search_candidate_id(chain) == "pe16_simd4"
    assert search_candidate_id(mesh) == "mesh8x4_simd8"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_compact_pe_mesh.py -v`

Expected: FAIL (`ImportError` for `enumerate_mm_mesh_recipes` / `candidate_id_mesh`)

- [ ] **Step 3: Write minimal implementation**

Append to `compact_pe_search.py` (keep existing `enumerate_mm_recipes` / `candidate_id` unchanged except imports if needed):

```python
MESH_PE_CHOICES = (4, 8, 16, 32)


def candidate_id_mesh(rec: PeRecipe) -> str:
    return f"mesh{rec.pe_i}x{rec.pe_j}_simd{rec.simd}"


def search_candidate_id(rec: PeRecipe) -> str:
    if rec.layout == "mesh":
        return candidate_id_mesh(rec)
    return candidate_id(rec)


def enumerate_mm_mesh_recipes(
    *, dsp_cap: float = 0.85, max_pe: int = 128
) -> list[PeRecipe]:
    out: list[PeRecipe] = []
    max_dsp = dsp_cap * U280_DSP
    for pe_i in MESH_PE_CHOICES:
        if I % pe_i:
            continue
        for pe_j in MESH_PE_CHOICES:
            if J % pe_j:
                continue
            if pe_i * pe_j > max_pe:
                continue
            for simd in SIMD_CHOICES:
                if K % simd:
                    continue
                expected = pe_i * pe_j * simd * 5
                if expected > max_dsp:
                    continue
                rec = PeRecipe(
                    bench="autosa_mm",
                    pe=pe_i * pe_j,
                    simd=simd,
                    data_kind="float",
                    i_macro="I",
                    j_macro="J",
                    k_macro="K",
                    pack_bits=simd * 32,
                    expected_dsp=expected,
                    min_dsp=max(1, int(expected * 0.6)),
                    pe_kj=(K // simd) * (J // pe_j),
                    i_tiles=I // pe_i,
                    tile_loop="inside_tasks",
                    note=f"mesh {pe_i}x{pe_j} simd{simd}",
                    layout="mesh",
                    pe_i=pe_i,
                    pe_j=pe_j,
                )
                out.append(rec)
    return out
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_compact_pe_mesh.py tests/test_compact_pe_search.py -v`

Expected: PASS

- [ ] **Step 5: Commit only if the user asks**

---

### Task 3: Instantiate 2-D mesh kernel

**Files:**
- Create: `compact_pe_mesh_instantiate.py`
- Modify: `compact_pe_instantiate.py` (`instantiate_mm` dispatch only)
- Test: `tests/test_compact_pe_mesh.py` (add instantiate tests)

1-D `mm_pe(` tests must stay valid: mesh uses **`mesh_pe`**, not `mm_pe`.

Wiring (functional GEMM, compact):

- `load_A` writes `fifo_A[i][j]` for every `(i,j)` with row `i0+i` (A per-PE, same row for all `j`)
- `load_B` writes `fifo_B[0][j]` for B rows owned by column `j` (`j0 = j * (J/PE_J)` …)
- `mesh_pe(i,j)` reads A from `fifo_A[i][j]`, B from `fifo_B[i][j]`, writes B to `fifo_B[i+1][j]`, writes C beats to `fifo_C[i][j]`
- `drain_B` reads `fifo_B[PE_I][j]`
- `store_C` reads `fifo_C[i][j]` into `C[i0+i][j_base+jj]`
- Function-scope DATAFLOW. I-tiles inside tasks: `for (int tile = 0; tile < I / PE_I; ++tile)`
- Crow: `data_t Crow[J / PE_J]`, ram_2p, no complete partition, no LOOP_FLATTEN off
- Inner loop labeled `pe_kj:` trip `(K/SIMD)*(J/PE_J)` per I-tile, PIPELINE II=1

Reuse `emit_pack` / `emit_unpack` / `_INTERFACE` from `compact_pe_instantiate.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_compact_pe_mesh.py`:

```python
from compact_pe_instantiate import instantiate_mm


def test_mesh_8x4_has_32_mesh_pe_and_256bit():
    rec = next(
        r for r in enumerate_mm_mesh_recipes() if candidate_id_mesh(r) == "mesh8x4_simd8"
    )
    code = instantiate_mm(rec)
    assert "#define PE_I 8" in code
    assert "#define PE_J 4" in code
    assert "ap_uint<256>" in code
    assert code.count("mesh_pe(") == 33  # 1 definition + 32 calls
    assert "mm_pe(" not in code
    assert "#pragma HLS DATAFLOW" in code
    assert "for (int i0 = 0; i0 < I; i0 += PE_I)" not in code
    assert "for (int i0 = 0; i0 < I; i0 += PE_NUM)" not in code


def test_mesh_16x8_has_128_mesh_pe():
    rec = next(
        r for r in enumerate_mm_mesh_recipes() if candidate_id_mesh(r) == "mesh16x8_simd8"
    )
    code = instantiate_mm(rec)
    assert code.count("mesh_pe(") == 129
    assert "#define PE_I 16" in code
    assert "#define PE_J 8" in code
```

- [ ] **Step 2: Run to verify FAIL**

Run: `.venv/bin/python -m pytest tests/test_compact_pe_mesh.py::test_mesh_8x4_has_32_mesh_pe_and_256bit -v`

Expected: FAIL (chain `instantiate_mm` has no `PE_I` / `mesh_pe`)

- [ ] **Step 3: Implement mesh emitter + dispatch**

Create `compact_pe_mesh_instantiate.py`. Required public function: `instantiate_mesh(rec: PeRecipe) -> str`.

Minimal structure (implement fully; this is the generator):

```python
from compact_pe_instantiate import _INTERFACE, _vec_name, emit_pack, emit_unpack


def emit_mesh_pe_calls(pe_i: int, pe_j: int, indent: str = "    ") -> str:
    lines = []
    for i in range(pe_i):
        for j in range(pe_j):
            lines.append(
                f"{indent}mesh_pe(fifo_A[{i}][{j}], fifo_B[{i}][{j}], "
                f"fifo_B[{i + 1}][{j}], fifo_C[{i}][{j}]);"
            )
    return "\n".join(lines)
```

`mesh_pe` definition (one function, DATAFLOW-unrolled calls): packed A/B, Crow `[J/PE_J]`, I-tile loop `for (int tile = 0; tile < I / PE_I; ++tile)`, inner `pe_kj: for (int t = 0; t < (K / SIMD) * (J / PE_J); ++t)` PIPELINE II=1, MAC of `simd` lanes, drain C on last k-tile to `fifo_C`.

`load_A`: nested `tile`, `k0 += SIMD`, `i < PE_I`, `j < PE_J`, write `packN(A[i0+i][k0+…])` to `fifo_A[i][j]`.

`load_B`: for each `tile`, `k0`, `j < PE_J`, `jj < J/PE_J`, write packed `B[j* (J/PE_J) + jj][k0+…]` to `fifo_B[0][j]` (one packed word per jj beat — match pe_kj trip). Align B beats with PE: one B pack per `t` of pe_kj, `jj = t % (J/PE_J)`.

`drain_B`: same trip as load_B, read `fifo_B[PE_I][j]`.

`store_C`: for `tile`, `i`, `j`, `jj`, `C[i0+i][j*(J/PE_J)+jj] = fifo_C[i][j].read()`.

Top: `#define PE_I` `#define PE_J` `#define SIMD`, streams `fifo_A[PE_I][PE_J]`, `fifo_B[PE_I+1][PE_J]`, `fifo_C[PE_I][PE_J]`, complete partition, function-scope DATAFLOW, then load_A, load_B, mesh_pe calls, drain_B, store_C.

In `compact_pe_instantiate.py` `instantiate_mm`, add at the top (after pack_bits check):

```python
    if getattr(rec, "layout", "chain") == "mesh":
        from compact_pe_mesh_instantiate import instantiate_mesh
        return instantiate_mesh(rec)
```

Do not change the rest of the 1-D emitter.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_compact_pe_mesh.py tests/test_compact_pe_instantiate.py tests/test_compact_pe_search.py -v`

Expected: PASS

- [ ] **Step 5: Commit only if the user asks**

---

### Task 4: Mesh `architecture_ok_for_recipe`

**Files:**
- Modify: `post_flash_stream.py` (`architecture_ok_for_recipe` only)
- Test: `tests/test_compact_pe_mesh.py`

Do **not** change `architecture_ok(bench=...)`.

- [ ] **Step 1: Write the failing test**

```python
def test_instantiated_mesh_8x4_passes_architecture_ok_for_recipe():
    import post_flash_stream as pfs
    rec = next(
        r for r in enumerate_mm_mesh_recipes() if candidate_id_mesh(r) == "mesh8x4_simd8"
    )
    code = instantiate_mm(rec)
    report = {
        "dsp": 1280,
        "latency_cycles": 1200,
        "interval": 1100,
        "feedback": {"scopes": [{"name": "pe_kj", "pipeline_ii": 1}]},
    }
    assert pfs.architecture_ok_for_recipe(code, report, rec) is True
```

- [ ] **Step 2: Run to see FAIL if the count/DATAFLOW checks are missing (or PASS if existing inside_tasks path already accepts it — then add a negative test)**

Also add:

```python
def test_mesh_architecture_ok_rejects_wrong_pe_count():
    import post_flash_stream as pfs
    rec = next(
        r for r in enumerate_mm_mesh_recipes() if candidate_id_mesh(r) == "mesh8x4_simd8"
    )
    code = instantiate_mm(rec).replace("mesh_pe(", "mm_pe(", 1)
    report = {
        "dsp": 1280,
        "latency_cycles": 1200,
        "interval": 1100,
        "feedback": {"scopes": [{"name": "pe_kj", "pipeline_ii": 1}]},
    }
    assert pfs.architecture_ok_for_recipe(code, report, rec) is False
```

Run: `.venv/bin/python -m pytest tests/test_compact_pe_mesh.py::test_mesh_architecture_ok_rejects_wrong_pe_count -v`

Expected: FAIL until the mesh count check exists.

- [ ] **Step 3: Implement mesh branch**

At the start of `architecture_ok_for_recipe` in `post_flash_stream.py`, after the function docstring:

```python
    if rec.layout == "mesh":
        pe_i = rec.pe_i if rec.pe_i else rec.pe
        pe_j = rec.pe_j
        if code.count("mesh_pe(") != 1 + pe_i * pe_j:
            return False
        if _i_tile_loop_around_dataflow(code):
            return False
```

Leave the rest of the function (streams, pack, Crow, kj flatten, dsp, II, overlap) as-is. Mesh `tile_loop` is `inside_tasks`, so the around_dataflow wrap gate does not apply.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_compact_pe_mesh.py tests/test_post_flash_stream.py -q`

Expected: PASS

- [ ] **Step 5: Commit only if the user asks**

---

### Task 5: Validate result.json ids + layout fields

**Files:**
- Modify: `compact_pe_validate.py`
- Test: `tests/test_compact_pe_validate.py` (add one mesh id test; keep existing pe16 mocks)

`validate_candidate` today uses `candidate_id(rec)`, which would name a 16×8 mesh `pe128_simd8`. Switch to `search_candidate_id`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_compact_pe_validate.py`:

```python
def test_validate_mesh_writes_mesh_id_dir(tmp_path, monkeypatch):
    from compact_pe_validate import validate_candidate
    from compact_pe_search import enumerate_mm_mesh_recipes, candidate_id_mesh

    rec = next(
        r for r in enumerate_mm_mesh_recipes() if candidate_id_mesh(r) == "mesh8x4_simd8"
    )

    def fake_run(**_kwargs):
        return {
            "synth": {
                "success": True,
                "report": {
                    "latency_cycles": 2000,
                    "interval": 1100,
                    "dsp": 1280,
                    "feedback": {"scopes": [{"name": "pe_kj", "pipeline_ii": 1}]},
                },
            },
            "csim": {"success": True},
            "cosim": None,
        }

    monkeypatch.setattr("compact_pe_validate.compile_check_cpp", lambda *a, **k: (True, ""))
    monkeypatch.setattr("compact_pe_validate._run_synth_csim_cosim", fake_run)
    validate_candidate(rec, tmp_path, header_code="//h", testbench_code="int main(){}")
    result = json.loads((tmp_path / "mesh8x4_simd8" / "result.json").read_text())
    assert result["cand_id"] == "mesh8x4_simd8"
    assert result["layout"] == "mesh"
    assert result["pe_i"] == 8 and result["pe_j"] == 4
    assert result["hls_csim_pass"] is True
    assert result["csynth_latency"] == 2000
```

- [ ] **Step 2: Run to FAIL** (directory still `pe32_simd8` until id switch)

Run: `.venv/bin/python -m pytest tests/test_compact_pe_validate.py::test_validate_mesh_writes_mesh_id_dir -v`

- [ ] **Step 3: Implement**

In `compact_pe_validate.py`:

- `from compact_pe_search import search_candidate_id` (replace `candidate_id` import)
- `cand_dir = out_root / search_candidate_id(rec)`
- In `_terminal_result`, `"cand_id": search_candidate_id(rec)`, plus:

```python
        "layout": rec.layout,
        "pe_i": rec.pe_i if rec.pe_i else rec.pe,
        "pe_j": rec.pe_j,
```

Existing pe16 test still looks at `tmp_path / "pe16_simd4"` — chain `search_candidate_id` stays `pe16_simd4`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_compact_pe_validate.py tests/test_compact_pe_mesh.py -v`

Expected: PASS

- [ ] **Step 5: Commit only if the user asks**

---

### Task 6: Driver + launcher (chain ∪ mesh)

**Files:**
- Modify: `compact_pe_search_main.py`
- Modify: `scripts/pc2/start_autosa_mm_pe_search.sh`
- Modify: `scripts/pc2/compact_pe_search.sbatch.sh`
- Modify: `tests/test_compact_pe_search_main.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_compact_pe_search_main.py`:

```python
def test_main_ranks_mesh_ahead_of_slower_chain(tmp_path, monkeypatch):
    from compact_pe_search import enumerate_mm_mesh_recipes, candidate_id_mesh

    chain = _recs("pe16_simd4")
    mesh = [
        r for r in enumerate_mm_mesh_recipes() if candidate_id_mesh(r) == "mesh8x4_simd8"
    ]
    monkeypatch.setattr("compact_pe_search_main.enumerate_mm_recipes", lambda: chain)
    monkeypatch.setattr("compact_pe_search_main.enumerate_mm_mesh_recipes", lambda: mesh)
    kernels = {
        "pe16_simd4": "// chain\n",
        "mesh8x4_simd8": "// kernel mesh\n",
    }
    results = {
        "pe16_simd4": _pass_row("pe16_simd4", 4292, 320),
        "mesh8x4_simd8": _pass_row("mesh8x4_simd8", 2000, 1280),
    }
    monkeypatch.setattr(
        "compact_pe_search_main.validate_candidate",
        _fake_validate(kernels, results),
    )
    header = tmp_path / "kernel.h"
    tb = tmp_path / "testbench.cpp"
    header.write_text("//h\n")
    tb.write_text("int main(){}\n")
    out = tmp_path / "out"
    rc = main(["--stamp", "mesh", "--out", str(out), "--header", str(header), "--testbench", str(tb)])
    assert rc == 0
    first = json.loads((out / "ranking.jsonl").read_text().splitlines()[0])
    assert first["cand_id"] == "mesh8x4_simd8"
    assert (out / "selected.cpp").read_text() == kernels["mesh8x4_simd8"]
```

Update `test_launcher_dry_run_prints_ids_and_exits_zero` to also assert `"mesh8x4_simd8" in stdout` and `"mesh16x8_simd8" in stdout`.

Update `_fake_validate` to use `search_candidate_id`:

```python
from compact_pe_search import search_candidate_id
# inside _validate:
cid = search_candidate_id(rec)
```

Existing tests that monkeypatch only `enumerate_mm_recipes` will also run mesh points unless we monkeypatch `enumerate_mm_mesh_recipes` to `lambda: []` in those two tests. **Do that** in `test_main_ranks_lower_latency_first_and_copies_selected` and `test_main_omits_csim_fail_from_ranking`:

```python
    monkeypatch.setattr("compact_pe_search_main.enumerate_mm_mesh_recipes", lambda: [])
```

- [ ] **Step 2: Run to FAIL** (driver does not call mesh enumerate; dry-run has no mesh ids)

- [ ] **Step 3: Implement**

`compact_pe_search_main.py`:

```python
from compact_pe_search import enumerate_mm_mesh_recipes, enumerate_mm_recipes
```

Replace the validate loop with:

```python
    recs = list(enumerate_mm_recipes()) + list(enumerate_mm_mesh_recipes())
    rows = []
    for rec in recs:
        rows.append(
            validate_candidate(
                rec,
                out_root,
                header_code,
                testbench_code,
                part=part,
                clock_ns=clock_ns,
            )
        )
```

`start_autosa_mm_pe_search.sh` candidate print:

```bash
"${PY}" - <<'PY'
from compact_pe_search import enumerate_mm_mesh_recipes, enumerate_mm_recipes, search_candidate_id
for rec in list(enumerate_mm_recipes()) + list(enumerate_mm_mesh_recipes()):
    print(search_candidate_id(rec))
PY
```

Change `--time=4:00:00` to `--time=12:00:00` in both `start_autosa_mm_pe_search.sh` and `scripts/pc2/compact_pe_search.sbatch.sh`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_compact_pe_search_main.py tests/test_compact_pe_mesh.py tests/test_compact_pe_validate.py tests/test_compact_pe_search.py tests/test_compact_pe_instantiate.py tests/test_compact_pe_rank.py tests/test_post_flash_pe_recipe.py -q`

Expected: PASS

- [ ] **Step 5: Commit only if the user asks**

---

### Task 7: First real HLS run (operator, after Tasks 1–6 green)

Not CI.

```bash
env -u C2HLS_TMP_RUN -u C2HLS_PE_RECIPE \
  C2HLS_SYNTH_TIMEOUT=3600 C2HLS_CSIM_TIMEOUT=1800 \
  ./scripts/pc2/start_autosa_mm_pe_search.sh --stamp 20260831_mesh
```

Expect:

- Dry-run lists `pe16_simd4` and `mesh8x4_simd8` / `mesh16x8_simd8`
- Artifacts under `artifacts/pc2/compact_pe_search_20260831_mesh` (not mmflow)
- `ranking.jsonl` mixes chain and mesh; quote it
- Do not update the Aug 18 slide unless rank-1 beats 4285 **and** the user says so
- Compact mesh will not match AutoSA 1846/2033 (`kernel0`)

---

## Out of this plan

- AutoSA binary / `kernel0` / host-serialize ABI
- Vivado impl / xclbin
- Other six GEMMs
- Changing success = 4228 for any mesh point until this ranking exists

---

## Spec coverage

| Requirement | Task |
|------------|------|
| PeRecipe layout/pe_i/pe_j defaults | 1 |
| Mesh grid + analog ids + caps | 2 |
| Compact 2-D instantiate, no kernel0 | 3 |
| architecture_ok_for_recipe mesh checks | 4 |
| result.json id/layout; no C2HLS_PE_RECIPE | 5 |
| Driver chain∪mesh, dry-run, 12h walltime | 6 |
| Operator HLS | 7 |
| 1-D tests unchanged | 3–6 regression commands |
| No AutoSA / no mmflow overwrite | all |

Placeholder scan: none. `search_candidate_id` is defined in Task 2 and used in Tasks 5–6. `instantiate_mesh` is defined in Task 3.
