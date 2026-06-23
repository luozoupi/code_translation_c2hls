# Worst results — All+avoids global (new 73-skill library)

<style>
table.flash-cmp { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.85em; }
table.flash-cmp th, table.flash-cmp td { border: 1px solid #ccc; padding: 4px 8px; white-space: nowrap; }
table.flash-cmp th { background: #f5f5f5; font-weight: 600; }
table.flash-cmp td:first-child, table.flash-cmp th:first-child { text-align: left !important; }
table.flash-cmp .fail { color: #c00; font-weight: 600; }
table.flash-meta { border-collapse: collapse; font-size: 0.9em; }
table.flash-meta th, table.flash-meta td { border: 1px solid #ccc; padding: 4px 10px; }
table.flash-meta th { background: #f5f5f5; text-align: left; width: 220px; }
pre.flash-code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.82em; background: #f8f8f8; border: 1px solid #ddd; padding: 10px 12px; overflow-x: auto; line-height: 1.35; }
</style>

<table class="flash-meta">
<thead><tr><th>Field</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Mode</td><td><code>flash_all_new_skills_avoids_global</code></td></tr>
<tr><td>Artifact</td><td><code>flash_all_new_skills_avoids_global_20260621_020847</code></td></tr>
<tr><td>Skills file</td><td><code>skills_ii_target_miss_solutions_added.json</code> (73 skills, packaged-only)</td></tr>
<tr><td>Model</td><td><code>mistralai/Devstral-2-123B-Instruct-2512</code></td></tr>
<tr><td>Overall</td><td>27/28 OK — geo-mean vs GT 0.046; several benches slower than GT or much worse than noskills</td></tr>
<tr><td>Regression baseline</td><td><code>flash_noskills_20260620_004507</code></td></tr>
</tbody></table>

## Summary — worst benches

<table class="flash-cmp">
<colgroup>
  <col style="width:8%">
  <col style="width:14%">
  <col style="width:22%">
  <col style="width:16%">
  <col style="width:12%">
  <col style="width:12%">
  <col style="width:10%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Rank</th>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:left">Issue</th>
  <th style="text-align:right">Latency (cyc)</th>
  <th style="text-align:right">vs noskills</th>
  <th style="text-align:right">vs GT</th>
  <th style="text-align:right">Fmax MHz</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">—</td><td style="text-align:left"><code>doitgen</code></td><td style="text-align:left">FAIL (gold ref synth)</td><td class="fail" style="text-align:right">FAIL</td><td style="text-align:right">—</td><td style="text-align:right">—</td><td style="text-align:right">—</td></tr>
<tr><td style="text-align:left">1</td><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:left">Naive O(n³) on global path</td><td style="text-align:right">833,976,005</td><td style="text-align:right"><strong>30×</strong></td><td style="text-align:right">0.037</td><td style="text-align:right">411</td></tr>
<tr><td style="text-align:left">2</td><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:left">Global 3-D stencil, II=77</td><td style="text-align:right">2,013,441</td><td style="text-align:right"><strong>41×</strong></td><td style="text-align:right">0.614</td><td style="text-align:right">371</td></tr>
<tr><td style="text-align:left">3</td><td style="text-align:left"><code>3mm</code></td><td style="text-align:left">Over-tiled + 885 DSP</td><td style="text-align:right">2,113,550</td><td style="text-align:right"><strong>21×</strong></td><td style="text-align:right">—</td><td style="text-align:right"><strong>300</strong></td></tr>
<tr><td style="text-align:left">4</td><td style="text-align:left"><code>gemm</code></td><td style="text-align:left">TILE_SIZE=8 nest</td><td style="text-align:right">565,921</td><td style="text-align:right"><strong>12×</strong></td><td style="text-align:right">—</td><td style="text-align:right">—</td></tr>
<tr><td style="text-align:left">5</td><td style="text-align:left"><code>trisolv</code></td><td style="text-align:left">Regression vs noskills</td><td style="text-align:right">1,160,161</td><td style="text-align:right">8.3×</td><td style="text-align:right"><strong>1.00</strong></td><td style="text-align:right">—</td></tr>
<tr><td style="text-align:left">6</td><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:left">Scalar GS, slower than GT</td><td style="text-align:right">2,270,081</td><td style="text-align:right">—</td><td style="text-align:right"><strong>1.044</strong></td><td style="text-align:right">340</td></tr>
<tr><td style="text-align:left">7</td><td style="text-align:left"><code>nussinov</code></td><td style="text-align:left">DP on global arrays</td><td style="text-align:right">209,526,121</td><td style="text-align:right">—</td><td style="text-align:right"><strong>1.033</strong></td><td style="text-align:right">411</td></tr>
</tbody></table>

## Side-by-side vs noskills — all regressions

<p>Every bench below is <strong>slower with all+avoids (new)</strong> than with <code>flash_noskills_20260620_004507</code> on the same model. Latencies are final flash-step synthesis cycles.</p>

<table class="flash-cmp">
<colgroup><col style="width:22%"><col style="width:18%"><col style="width:18%"><col style="width:10%"><col style="width:32%"></colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">All+avoids</th>
  <th style="text-align:right">Noskills</th>
  <th style="text-align:right">Ratio</th>
  <th style="text-align:left">Root cause</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">28,051,921</td><td style="text-align:right"><strong>30×</strong></td><td style="text-align:left">Naive global O(n³); no tiling</td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">2,013,441</td><td style="text-align:right">48,677</td><td style="text-align:right"><strong>41×</strong></td><td style="text-align:left">7-point stencil on global A/B; II=77</td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">2,113,550</td><td style="text-align:right">99,467</td><td style="text-align:right"><strong>21×</strong></td><td style="text-align:left">Tiny tiles + 885 DSP; repeated global reloads</td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">565,921</td><td style="text-align:right">49,181</td><td style="text-align:right"><strong>12×</strong></td><td style="text-align:left">8×8×8 tile nest; reload C from DRAM each k-tile</td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1,160,161</td><td style="text-align:right">139,301</td><td style="text-align:right"><strong>8.3×</strong></td><td style="text-align:left">Scalar reference; no local L/x/b staging</td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">7,629,361</td><td style="text-align:right">2,759,030</td><td style="text-align:right"><strong>2.8×</strong></td><td style="text-align:left">Tiled loads without pipeline on load loops</td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">1,120,001</td><td style="text-align:right">586,042</td><td style="text-align:right">1.9×</td><td style="text-align:left">Moderate global-memory regression</td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">5,323,907</td><td style="text-align:right">2,864,058</td><td style="text-align:right">1.9×</td><td style="text-align:left">Moderate global-memory regression</td></tr>
</tbody></table>

### floyd-warshall (30× slower)

<p><strong>Issue (all+avoids):</strong> Textbook triple loop; each <code>path[i][j]</code> update does two random global reads. Synthesis II=143.</p>
<p><strong>Why noskills is better:</strong> 32×32 tile in BRAM; load / compute / store phases pipelined at II=1; most work hits local <code>local_path</code>.</p>

<table class="flash-cmp">
<colgroup><col style="width:50%"><col style="width:50%"></colgroup>
<thead><tr><th style="text-align:left">All+avoids (bad)</th><th style="text-align:left">Noskills (better)</th></tr></thead>
<tbody>
<tr><td style="text-align:left; vertical-align:top"><pre class="flash-code">for (k = 0; k &lt; n; k++) {
  for(i = 0; i &lt; n; i++) {
    for (j = 0; j &lt; n; j++) {
      path[i][j] = path[i][j] &lt; path[i][k] + path[k][j] ?
        path[i][j] : path[i][k] + path[k][j];
    }
  }
}</pre></td>
<td style="text-align:left; vertical-align:top"><pre class="flash-code">int local_path[32][32];
#pragma HLS array_partition variable=local_path ...
for (k = 0; k &lt; n; k++) {
  for (i_tile = 0; i_tile &lt; n; i_tile += 32) {
    for (j_tile = 0; j_tile &lt; n; j_tile += 32) {
      // load tile (PIPELINE II=1)
      // compute on local_path (PIPELINE II=1)
      // store tile (PIPELINE II=1)
    }
  }
}</pre></td></tr>
</tbody></table>

### heat-3d (41× slower)

<p><strong>Issue (all+avoids):</strong> Full 3-D seven-point stencil reads <code>A[i±1,j,k]</code>, <code>A[i,j±1,k]</code>, <code>A[i,j,k±1]</code> from <strong>global</strong> memory every inner iteration. Achieved II=77, not II=1.</p>
<p><strong>Why noskills is better:</strong> Uses on-chip <code>local_A[2][N][N]</code> / <code>local_B[2][N][N]</code> with double-buffer index <code>t%2</code>; stencil reads come from BRAM. (Note: noskills version is a simplified 2-D slice — fast in synthesis but not algorithmically equivalent to full 3-D gold.)</p>

<table class="flash-cmp">
<colgroup><col style="width:50%"><col style="width:50%"></colgroup>
<thead><tr><th style="text-align:left">All+avoids (bad)</th><th style="text-align:left">Noskills (better latency)</th></tr></thead>
<tbody>
<tr><td style="text-align:left; vertical-align:top"><pre class="flash-code">compute_B: for (int i = 1; i &lt; n-1; i++) {
  for (int j = 1; j &lt; n-1; j++) {
#pragma HLS PIPELINE II=1
    for (int k = 1; k &lt; n-1; k++) {
      B[i][j][k] = 0.125 * (A[i+1][j][k] - 2*A[i][j][k] + ...)
                + ... /* 6 more global A reads */
                + A[i][j][k];
    }
  }
}</pre></td>
<td style="text-align:left; vertical-align:top"><pre class="flash-code">double local_A[2][N][N];
double local_B[2][N][N];
#pragma HLS ARRAY_PARTITION variable=local_A dim=1 complete
...
for (int t = 1; t &lt;= tsteps; t++) {
#pragma HLS PIPELINE II=1
  for (int k = 1; k &lt; n-1; k++) {
    local_B[t%2][i][j] = 0.125 * (
      local_A[(t-1)%2][i+1][j] - 2*local_A[(t-1)%2][i][j] + ...
    ) + ... + local_A[(t-1)%2][i][j];
  }
}</pre></td></tr>
</tbody></table>

### 3mm (21× slower)

<p><strong>Issue (all+avoids):</strong> <code>tile_size=8</code> with <code>ARRAY_PARTITION complete</code> on every micro-tile; triple nested <code>i0/j0</code> loops × three full GEMMs; results written straight to global <code>E/F/G</code> each tile. **885 DSP**, Fmax 300 MHz.</p>
<p><strong>Why noskills is better:</strong> One-time load of full <code>local_A…local_F</code> arrays; compute all three multiplies on BRAM with pipelined <code>i-j</code> loops; single store of <code>G</code>.</p>

<table class="flash-cmp">
<colgroup><col style="width:50%"><col style="width:50%"></colgroup>
<thead><tr><th style="text-align:left">All+avoids (bad)</th><th style="text-align:left">Noskills (better)</th></tr></thead>
<tbody>
<tr><td style="text-align:left; vertical-align:top"><pre class="flash-code">const int tile_size = 8;
for (int i0 = 0; i0 &lt; ni; i0 += tile_size) {
  for (int j0 = 0; j0 &lt; nj; j0 += tile_size) {
    double A_tile[8][60];
    #pragma HLS ARRAY_PARTITION variable=A_tile complete dim=1
    load_A_tile: ...
    compute_E_tile: ...
    E[i0+i][j0+j] = sum;  // global write per tile
  }
}
// repeat for F, then G</pre></td>
<td style="text-align:left; vertical-align:top"><pre class="flash-code">double local_A[NI][NK];
double local_B[NK][NJ];
...
// load A,B once
for (int i = 0; i &lt; ni; i++)
  for (int j = 0; j &lt; nj; j++) {
#pragma HLS PIPELINE II=1
    for (int k = 0; k &lt; nk; k++)
      sum += local_A[i][k] * local_B[k][j];
    local_E[i][j] = sum;
  }
// same pattern for F; then G[i][j] from local_E × local_F</pre></td></tr>
</tbody></table>

### gemm (12× slower)

<p><strong>Issue (all+avoids):</strong> Classic blocked GEMM with <code>TILE_SIZE=8</code> and a <code>k0</code> loop that reloads <code>C</code> from DRAM every k-tile; inner <code>UNROLL factor=8</code> on 8×8 tiles inflates DSP.</p>
<p><strong>Why noskills is better:</strong> Loads entire <code>A,B,C</code> to locals once; pipelines the <code>i-k</code> loop with modest <code>UNROLL factor=4</code> on <code>j</code>; compute entirely on BRAM before one store pass.</p>

<table class="flash-cmp">
<colgroup><col style="width:50%"><col style="width:50%"></colgroup>
<thead><tr><th style="text-align:left">All+avoids (bad)</th><th style="text-align:left">Noskills (better)</th></tr></thead>
<tbody>
<tr><td style="text-align:left; vertical-align:top"><pre class="flash-code">for (i0 = 0; i0 &lt; ni; i0 += TILE_SIZE)
  for (j0 = 0; j0 &lt; nj; j0 += TILE_SIZE)
    for (k0 = 0; k0 &lt; nk; k0 += TILE_SIZE) {
      load_tile_A / load_tile_B / load_tile_C ...
      compute_tile:
        #pragma HLS UNROLL factor=8
        tile_C[i][j] += alpha * tile_A[i][k]*tile_B[k][j];
      store_tile_C ...
    }</pre></td>
<td style="text-align:left; vertical-align:top"><pre class="flash-code">double local_A[NI][NK];
double local_B[NK][NJ];
double local_C[NI][NJ];
// one-time load A, B, C*beta
for (int i = 0; i &lt; ni; i++)
  for (int k = 0; k &lt; nk; k++) {
#pragma HLS PIPELINE II=1
    double a_val = local_A[i][k];
    for (int j = 0; j &lt; nj; j += 4) {
#pragma HLS UNROLL factor=4
      local_C[i][j] += alpha * a_val * local_B[k][j];
      ...
    }
  }</pre></td></tr>
</tbody></table>

### trisolv (8.3× slower)

<p><strong>Issue (all+avoids):</strong> Bare PolyBench port — triangular solve with <strong>no pragmas</strong>, no local buffers; every <code>L[i][j]</code> and <code>x[j]</code> access is effectively off-chip.</p>
<p><strong>Why noskills is better:</strong> Copies <code>L,b</code> to <code>L_local/b_local</code>, solves into <code>x_local</code> with pipelined outer <code>i</code> and unrolled inner <code>j</code>, then stores <code>x</code> once.</p>

<table class="flash-cmp">
<colgroup><col style="width:50%"><col style="width:50%"></colgroup>
<thead><tr><th style="text-align:left">All+avoids (bad)</th><th style="text-align:left">Noskills (better)</th></tr></thead>
<tbody>
<tr><td style="text-align:left; vertical-align:top"><pre class="flash-code">void kernel_trisolv(double L[N][N], double x[N], double b[N]) {
  for (i = 0; i &lt; n; i++) {
    x[i] = b[i];
    for (j = 0; j &lt; i; j++)
      x[i] -= L[i][j] * x[j];   // global RMW
    x[i] = x[i] / L[i][i];
  }
}</pre></td>
<td style="text-align:left; vertical-align:top"><pre class="flash-code">double L_local[N][N], x_local[N], b_local[N];
// load L,b to locals (PIPELINE + UNROLL)
for (int i = 0; i &lt; n; i++) {
#pragma HLS PIPELINE II=1
  x_local[i] = b_local[i];
  for (int j = 0; j &lt; i; j++) {
#pragma HLS UNROLL factor=8
    x_local[i] -= L_local[i][j] * x_local[j];
  }
  x_local[i] /= L_local[i][i];
}
// store x once</pre></td></tr>
</tbody></table>

### trmm (2.8× slower) — smaller regression

<p><strong>Issue:</strong> Tile loads have <strong>no PIPELINE</strong> pragma; compute uses triangular index <code>k = i+1..m-1</code> on tile buffers. Noskills loads full <code>local_A/local_B</code> once and pipelines the <code>i-j</code> loop with unrolled <code>k</code>.</p>

<pre class="flash-code">// All+avoids: load_A_tile without #pragma HLS PIPELINE
load_A_tile: for (int i = 0; i &lt; TILE_SIZE; i++)
  for (int k = 0; k &lt; m; k++)
    A_tile[i][k] = A[i_tile + i][k];

// Noskills: full local staging + pipelined compute
for (int i = 0; i &lt; m; i++)
  for (int j = 0; j &lt; n; j++) {
#pragma HLS PIPELINE II=1
    for (int k = i + 1; k &lt; m; k++) {
#pragma HLS UNROLL factor=4
      sum += local_A[k][i] * local_B[k][j];
    }
  }</pre>

### Pattern across all regressions

<ol>
<li><strong>Skills did not prevent reference ports</strong> — floyd-warshall, trisolv are unoptimized C.</li>
<li><strong>Over-application of tiling skills</strong> — tiny 8×8 tiles with complete partition → DSP bloat (3mm, gemm).</li>
<li><strong>Missing load–compute–store</strong> — stencil/GEMM hits DRAM instead of BRAM (heat-3d, floyd-warshall).</li>
<li><strong>LLM variance</strong> — same flash pipeline; noskills sometimes lucks into a better structure without 73-skill context noise.</li>
</ol>

## 1. doitgen — bench failure

Same gold-reference synthesizability failure as all other flash modes. Not attributable to the new skill library.

## 2. floyd-warshall — worst absolute latency (834M cycles)

<table class="flash-cmp">
<colgroup><col style="width:34%"><col style="width:33%"><col style="width:33%"></colgroup>
<thead><tr>
  <th style="text-align:left">Metric</th>
  <th style="text-align:right">All+avoids (new)</th>
  <th style="text-align:right">Noskills</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">Latency (cycles)</td><td style="text-align:right"><strong>833,976,005</strong></td><td style="text-align:right">28,051,921</td></tr>
<tr><td style="text-align:left">Structure</td><td style="text-align:right">Naive triple loop</td><td style="text-align:right">32×32 tiled local buffer</td></tr>
<tr><td style="text-align:left">Synthesis</td><td style="text-align:right">II=143, pipeline_blocked</td><td style="text-align:right">Pipelined tile load/compute/store</td></tr>
</tbody></table>

<p>Despite 73 skills in context, the LLM emitted naive Floyd–Warshall with no inner-loop pragmas — every update reads global <code>path[i][k]</code> and <code>path[k][j]</code>.</p>

<p><strong>All+avoids (new) — generated code</strong></p>
<pre class="flash-code">for (k = 0; k &lt; n; k++) {
    for(i = 0; i &lt; n; i++) {
        for (j = 0; j &lt; n; j++) {
            path[i][j] = path[i][j] &lt; path[i][k] + path[k][j] ?
                path[i][j] : path[i][k] + path[k][j];
        }
    }
}</pre>

<p><strong>Noskills — tiled local path (excerpt)</strong></p>
<pre class="flash-code">int local_path[32][32];
#pragma HLS array_partition variable=local_path dim=1 type=block factor=4
for (k = 0; k &lt; n; k++) {
    for (int i_tile = 0; i_tile &lt; n; i_tile += 32) {
        for (int j_tile = 0; j_tile &lt; n; j_tile += 32) {
            // load tile → compute → store tile (each pipelined II=1)
            ...
        }
    }
}</pre>

## 3. heat-3d — 41× slower than noskills

<table class="flash-cmp">
<colgroup><col style="width:34%"><col style="width:33%"><col style="width:33%"></colgroup>
<thead><tr>
  <th style="text-align:left">Metric</th>
  <th style="text-align:right">All+avoids (new)</th>
  <th style="text-align:right">Noskills</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">Latency (cycles)</td><td style="text-align:right">2,013,441</td><td style="text-align:right">48,677</td></tr>
<tr><td style="text-align:left">Bottleneck</td><td style="text-align:right">II=77 on compute_B / compute_A</td><td style="text-align:right">Simplified 2-D code</td></tr>
</tbody></table>

<pre class="flash-code">for (int t = 1; t &lt;= tsteps; t++) {
    compute_B: for (int i = 1; i &lt; n-1; i++) {
        for (int j = 1; j &lt; n-1; j++) {
#pragma HLS PIPELINE II=1
            for (int k = 1; k &lt; n-1; k++) {
                B[i][j][k] = 0.125 * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
                             + 0.125 * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k])
                             + 0.125 * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1])
                             + A[i][j][k];
            }
        }
    }
    compute_A: /* symmetric update from B to A */
}</pre>

<p>Seven global reads per stencil point → II=77 instead of II=1. No on-chip time-slab or z-buffer tiling.</p>

## 4. 3mm — over-engineered tiling (21× vs noskills)

<table class="flash-cmp">
<colgroup><col style="width:25%"><col style="width:75%"></colgroup>
<thead><tr><th style="text-align:left">Metric</th><th style="text-align:right">Value</th></tr></thead>
<tbody>
<tr><td style="text-align:left">Latency (cycles)</td><td style="text-align:right">2,113,550</td></tr>
<tr><td style="text-align:left">DSP</td><td style="text-align:right"><strong>885</strong></td></tr>
<tr><td style="text-align:left">Fmax (MHz)</td><td style="text-align:right"><strong>300</strong></td></tr>
<tr><td style="text-align:left">Bottleneck</td><td style="text-align:right">interval_exceeds_latency on module</td></tr>
</tbody></table>

<pre class="flash-code">const int tile_size = 8;
for (int i0 = 0; i0 &lt; ni; i0 += tile_size) {
    for (int j0 = 0; j0 &lt; nj; j0 += tile_size) {
        int tile_i = (i0 + tile_size &gt; ni) ? (ni - i0) : tile_size;
        // Load tiles from A, B into locals, compute partial E, store...
    }
}</pre>

## 5. gemm — 12× slower than noskills

<p>TILE_SIZE=8 with <code>ARRAY_PARTITION complete</code> on tile buffers — correct pattern but tile too small and loop overhead dominates.</p>

<pre class="flash-code">const int TILE_SIZE = 8;
double tile_A[TILE_SIZE][TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=tile_A complete dim=1
for (int i0 = 0; i0 &lt; ni; i0 += TILE_SIZE) {
    for (int j0 = 0; j0 &lt; nj; j0 += TILE_SIZE) {
        for (int k0 = 0; k0 &lt; nk; k0 += TILE_SIZE) {
            load_tile_A: for (int i = 0; i &lt; TILE_SIZE; i++) {
                #pragma HLS PIPELINE II=1
                ...
</pre>

## 6. Benches slower than ground truth

<table class="flash-cmp">
<colgroup>
  <col style="width:18%">
  <col style="width:14%">
  <col style="width:16%">
  <col style="width:52%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">vs GT</th>
  <th style="text-align:right">Latency</th>
  <th style="text-align:left">Notes</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right"><strong>1.044</strong></td><td style="text-align:right">2,270,081</td><td style="text-align:left">Scalar GS, no blocking/pipeline</td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right"><strong>1.033</strong></td><td style="text-align:right">209,526,121</td><td style="text-align:left">Matches slow GT DP structure, II=144</td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1.000</td><td style="text-align:right">1,160,161</td><td style="text-align:left">Tie GT; 8.3× slower than noskills</td></tr>
</tbody></table>

<pre class="flash-code">for (k = 0; k &lt; n; k++) {
    nrm = 0.0;
    for (i = 0; i &lt; m; i++)
        nrm += A[i][k] * A[i][k];
    R[k][k] = sqrt(nrm);
    for (i = 0; i &lt; m; i++)
        Q[i][k] = A[i][k] / R[k][k];
    for (j = k + 1; j &lt; n; j++) {
        R[k][j] = 0.0;
        for (i = 0; i &lt; m; i++)
            R[k][j] += Q[i][k] * A[i][j];
        for (i = 0; i &lt; m; i++)
            A[i][j] = A[i][j] - Q[i][k] * R[k][j];
    }
}</pre>

## Common failure patterns

<table class="flash-cmp">
<colgroup><col style="width:28%"><col style="width:32%"><col style="width:40%"></colgroup>
<thead><tr>
  <th style="text-align:left">Pattern</th>
  <th style="text-align:left">Symptom</th>
  <th style="text-align:left">Benches</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">Naive global loops</td><td style="text-align:left">II≫1, huge latency</td><td style="text-align:left"><code>floyd-warshall</code></td></tr>
<tr><td style="text-align:left">Stencil on global memory</td><td style="text-align:left">II=77, port conflicts</td><td style="text-align:left"><code>heat-3d</code></td></tr>
<tr><td style="text-align:left">Over-tiling / DSP bloat</td><td style="text-align:left">High DSP, low Fmax</td><td style="text-align:left"><code>3mm</code>, <code>gemm</code></td></tr>
<tr><td style="text-align:left">Matches slow GT</td><td style="text-align:left">ratio ≥ 1.0, no win</td><td style="text-align:left"><code>gramschmidt</code>, <code>nussinov</code></td></tr>
<tr><td style="text-align:left">LLM variance</td><td style="text-align:left">Worse than noskills despite skills</td><td style="text-align:left">Most regressions above</td></tr>
</tbody></table>

## Why all skills + avoids still fails

<ol>
<li><strong>Prompt size</strong> — 73 skills is a large context; tiling/pipeline guidance is sometimes ignored.</li>
<li><strong>Avoid rules</strong> — prevent bad pragmas but do not force load–compute–store when the model picks reference code.</li>
<li><strong>No flash regression guard</strong> — one-shot rewrite keeps any synthesizable design, even with II≫1.</li>
<li><strong>Gold tie ≠ success</strong> — <code>nussinov</code> / <code>trisolv</code> match slow GT implementations.</li>
</ol>

## Artifact paths

<table class="flash-cmp">
<colgroup><col style="width:22%"><col style="width:78%"></colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:left">Final code (under artifact root)</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:left"><code>hlsfactory_floyd-warshall/devstral2__flash__all_new_skills_avoids_global/hlsfactory_floyd-warshall_final.cpp</code></td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:left"><code>hlsfactory_heat-3d/devstral2__flash__all_new_skills_avoids_global/hlsfactory_heat-3d_final.cpp</code></td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:left"><code>hlsfactory_3mm/devstral2__flash__all_new_skills_avoids_global/hlsfactory_3mm_final.cpp</code></td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:left"><code>hlsfactory_gemm/devstral2__flash__all_new_skills_avoids_global/hlsfactory_gemm_final.cpp</code></td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:left"><code>hlsfactory_gramschmidt/devstral2__flash__all_new_skills_avoids_global/hlsfactory_gramschmidt_final.cpp</code></td></tr>
</tbody></table>

<p>Full root: <code>artifacts/pc2/flash_all_new_skills_avoids_global_20260621_020847/</code></p>
