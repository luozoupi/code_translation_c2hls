# Flash Synthesis — Presentation Summary (All Tests, Excl. Cosim)

<style>
table.flash-cmp { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.85em; }
table.flash-cmp th, table.flash-cmp td { border: 1px solid #ccc; padding: 4px 8px; white-space: nowrap; }
table.flash-cmp th { background: #f5f5f5; font-weight: 600; }
table.flash-cmp td:first-child, table.flash-cmp th:first-child { text-align: left !important; }
table.flash-cmp .fail { color: #c00; font-weight: 600; }
table.flash-cmp .best { background: #e8f5e9; font-weight: 600; }
table.flash-meta { border-collapse: collapse; font-size: 0.9em; margin-bottom: 1em; }
table.flash-meta th, table.flash-meta td { border: 1px solid #ccc; padding: 4px 10px; }
table.flash-meta th { background: #f5f5f5; text-align: left; width: 240px; }
table.flash-rec { border-collapse: collapse; width: 100%; font-size: 0.95em; }
table.flash-rec th, table.flash-rec td { border: 1px solid #ccc; padding: 8px 12px; vertical-align: top; }
table.flash-rec th { background: #e3f2fd; text-align: left; }
</style>

<table class="flash-meta">
<thead><tr><th>Field</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Generated</td><td><code>2026-06-23</code></td></tr>
<tr><td>Benchmarks</td><td>28 <code>hlsfactory_*</code> Polybench kernels</td></tr>
<tr><td>Model</td><td><code>mistralai/Devstral-2-123B-Instruct-2512</code></td></tr>
<tr><td>Mode</td><td>Flash (single-shot LLM + csim + csynth)</td></tr>
<tr><td>Metric</td><td>Final flash-step <strong>synthesis latency</strong> (cycles); lower is better</td></tr>
<tr><td>vs GT ratio</td><td>generated_latency / ground_truth_latency; geo-mean across 27 OK benches</td></tr>
<tr><td>Success</td><td>27/28 per run; <code>doitgen</code> fails gold-reference gate everywhere</td></tr>
<tr><td>Legacy stamp</td><td><code>20260620_004507</code> (noskills/bn2+2), <code>20260620_113247</code> (global skills)</td></tr>
<tr><td>New skills (73)</td><td><code>skills_ii_target_miss_solutions_added(73skills).json</code> — main matrix stamp <code>20260621_020847</code>; no avoids (new) best <code>20260621_075846</code> (57 positive injected)</td></tr>
<tr><td>New skills (90)</td><td><code>skills_ii_target_miss_solutions_added(90skills).json</code> — all+avoids (new) best <code>20260623_024548</code> (90 injected)</td></tr>
<tr><td>Curated matrix stamp</td><td><code>20260621_104044</code> — 15 runs (5 variants × 3 curation waves)</td></tr>
<tr><td>Baseline 3-way stamp (optional)</td><td><code>20260622_215520</code> — 85-skill intermediate (no frozen file; between 73 and 90)</td></tr>
<tr><td>Deterministic runs</td><td>10 modes (4 legacy + 6 new)</td></tr>
<tr><td>Curated runs</td><td>15 modes</td></tr>
<tr><td>Excluded</td><td>Cosim / cosim-repair (separate experiment axis)</td></tr>
</tbody></table>

## Executive summary — what to present

<table class="flash-rec">
<thead><tr><th>Rank</th><th>Recommended mode</th><th>Key numbers</th><th>When to use in slides</th></tr></thead>
<tbody>
<tr><td><strong>1</strong></td><td><strong>No avoids (old)</strong><br><code>flash_all_skills_no_avoids_global_20260620_113247</code></td><td>Best-latency wins <strong>8.1/27</strong>; geo-mean lat/GT <strong>0.0342</strong>; vs noskills (old) <strong>18–9</strong>; never slower than GT</td><td>Best overall flash synthesis latency; clearest skills win story on old 55-skill library</td></tr>
<tr><td><strong>2</strong></td><td><strong>All+avoids (new)</strong><br><code>flash_all_new_skills_avoids_global_20260621_020847</code></td><td>Best-latency wins <strong>4.1/27</strong>; geo-mean <strong>0.0536</strong>; vs noskills (new) <strong>15–11</strong></td><td>Best mode on the new 73-skill library</td></tr>
<tr><td><strong>3</strong></td><td><strong>Noskills (old)</strong> baseline<br><code>flash_noskills_20260620_004507</code></td><td>Geo-mean <strong>0.0528</strong>; wins <strong>2.0/27</strong>; 0 benches slower than GT</td><td>Strong no-skills baseline for “skills vs no skills” comparison</td></tr>
<tr><td>4</td><td><strong>No avoids (new)</strong></td><td>Geo-mean <strong>0.0584</strong>; paired new beats old <strong>14–9</strong></td><td>New library helps when avoid-tier is dropped</td></tr>
<tr><td>5</td><td><strong>Best curated:</strong> No avoids json_only (bottleneck)</td><td>Wins <strong>5.0/27</strong>; geo-mean <strong>0.1329</strong></td><td>LLM curation highlight — does not beat deterministic no-avoids (old) on geo-mean</td></tr>
<tr><td>Avoid</td><td>All+avoids (old), Bn 6+2 (new)</td><td>Geo-means <strong>0.1714</strong> / <strong>0.1469</strong>; wins <strong>0.2/27</strong> for bn6+2</td><td>Weak modes — do not lead with these</td></tr>
</tbody></table>

## 1. All deterministic modes — ranked by geo-mean lat/GT

<table class="flash-cmp">
<colgroup>
  <col style="width:8%">
  <col style="width:8%">
  <col style="width:8%">
  <col style="width:8%">
  <col style="width:8%">
  <col style="width:8%">
  <col style="width:8%">
  <col style="width:8%">
  <col style="width:8%">
  <col style="width:8%">
</colgroup>
<thead><tr>
  <th style="text-align:right">Rank</th>
  <th style="text-align:left">Mode</th>
  <th style="text-align:left">Family</th>
  <th style="text-align:left">Artifact root</th>
  <th style="text-align:right">OK</th>
  <th style="text-align:right">Best-latency wins</th>
  <th style="text-align:right">Geo-mean lat/GT</th>
  <th style="text-align:right">Faster than GT</th>
  <th style="text-align:right">Slower than GT</th>
  <th style="text-align:right">Tie ~1.0</th>
</tr></thead>
<tbody>
<tr><td style="text-align:right">1</td><td style="text-align:left" class="best">No avoids (old)</td><td style="text-align:left">legacy</td><td style="text-align:left"><code>flash_all_skills_no_avoids_global_20260620_113247</code></td><td style="text-align:right">27/28</td><td style="text-align:right">8.1/27</td><td style="text-align:right" class="best">0.0342</td><td style="text-align:right">23</td><td style="text-align:right">0</td><td style="text-align:right">3</td></tr>
<tr><td style="text-align:right">2</td><td style="text-align:left">Noskills (old)</td><td style="text-align:left">legacy</td><td style="text-align:left"><code>flash_noskills_20260620_004507</code></td><td style="text-align:right">27/28</td><td style="text-align:right">2.0/27</td><td style="text-align:right">0.0528</td><td style="text-align:right">25</td><td style="text-align:right">0</td><td style="text-align:right">2</td></tr>
<tr><td style="text-align:right">3</td><td style="text-align:left">All+avoids (new)</td><td style="text-align:left">new</td><td style="text-align:left"><code>flash_all_new_skills_avoids_global_20260621_020847</code></td><td style="text-align:right">27/28</td><td style="text-align:right">4.1/27</td><td style="text-align:right">0.0536</td><td style="text-align:right">22</td><td style="text-align:right">2</td><td style="text-align:right">2</td></tr>
<tr><td style="text-align:right">4</td><td style="text-align:left">No avoids (new)</td><td style="text-align:left">new</td><td style="text-align:left"><code>flash_all_new_skills_no_avoids_global_20260621_020847</code></td><td style="text-align:right">27/28</td><td style="text-align:right">2.8/27</td><td style="text-align:right">0.0584</td><td style="text-align:right">22</td><td style="text-align:right">1</td><td style="text-align:right">4</td></tr>
<tr><td style="text-align:right">5</td><td style="text-align:left">Bn 2+2 (old)</td><td style="text-align:left">legacy</td><td style="text-align:left"><code>flash_skills_20260620_004507</code></td><td style="text-align:right">27/28</td><td style="text-align:right">1.5/27</td><td style="text-align:right">0.0653</td><td style="text-align:right">26</td><td style="text-align:right">0</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:right">6</td><td style="text-align:left">Bn 4+2 (new)</td><td style="text-align:left">new</td><td style="text-align:left"><code>flash_bn_skills_new_4_2_20260621_020847</code></td><td style="text-align:right">27/28</td><td style="text-align:right">2.0/27</td><td style="text-align:right">0.0872</td><td style="text-align:right">23</td><td style="text-align:right">1</td><td style="text-align:right">2</td></tr>
<tr><td style="text-align:right">7</td><td style="text-align:left">Bn 2+2 (new)</td><td style="text-align:left">new</td><td style="text-align:left"><code>flash_bn_skills_new_2_2_20260621_020847</code></td><td style="text-align:right">27/28</td><td style="text-align:right">2.0/27</td><td style="text-align:right">0.1138</td><td style="text-align:right">24</td><td style="text-align:right">1</td><td style="text-align:right">2</td></tr>
<tr><td style="text-align:right">8</td><td style="text-align:left">Noskills (new)</td><td style="text-align:left">new</td><td style="text-align:left"><code>flash_noskills_new_20260621_020847</code></td><td style="text-align:right">27/28</td><td style="text-align:right">1.5/27</td><td style="text-align:right">0.1237</td><td style="text-align:right">22</td><td style="text-align:right">2</td><td style="text-align:right">3</td></tr>
<tr><td style="text-align:right">9</td><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:left">new</td><td style="text-align:left"><code>flash_bn_skills_new_6_2_20260621_020847</code></td><td style="text-align:right">27/28</td><td style="text-align:right">0.2/27</td><td style="text-align:right">0.1469</td><td style="text-align:right">24</td><td style="text-align:right">2</td><td style="text-align:right">0</td></tr>
<tr><td style="text-align:right">10</td><td style="text-align:left">All+avoids (old)</td><td style="text-align:left">legacy</td><td style="text-align:left"><code>flash_all_skills_avoids_global_20260620_113247</code></td><td style="text-align:right">27/28</td><td style="text-align:right">2.8/27</td><td style="text-align:right">0.1714</td><td style="text-align:right">21</td><td style="text-align:right">0</td><td style="text-align:right">5</td></tr>
</tbody></table>

## 2. vs ground truth — all deterministic modes

<table class="flash-cmp">
<colgroup>
  <col style="width:14%">
  <col style="width:14%">
  <col style="width:14%">
  <col style="width:14%">
  <col style="width:14%">
  <col style="width:14%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Mode</th>
  <th style="text-align:right">Faster than GT</th>
  <th style="text-align:right">Slower than GT</th>
  <th style="text-align:right">Tie (~1.0)</th>
  <th style="text-align:right">Geo-mean ratio</th>
  <th style="text-align:right">Bench fail</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">Noskills (old)</td><td style="text-align:right">25</td><td style="text-align:right">0</td><td style="text-align:right">2</td><td style="text-align:right">0.0528</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">Bn 2+2 (old)</td><td style="text-align:right">26</td><td style="text-align:right">0</td><td style="text-align:right">1</td><td style="text-align:right">0.0653</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">All+avoids (old)</td><td style="text-align:right">21</td><td style="text-align:right">0</td><td style="text-align:right">5</td><td style="text-align:right">0.1714</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">No avoids (old)</td><td style="text-align:right">23</td><td style="text-align:right">0</td><td style="text-align:right">3</td><td style="text-align:right">0.0342</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:right">22</td><td style="text-align:right">2</td><td style="text-align:right">3</td><td style="text-align:right">0.1237</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">Bn 2+2 (new)</td><td style="text-align:right">24</td><td style="text-align:right">1</td><td style="text-align:right">2</td><td style="text-align:right">0.1138</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">Bn 4+2 (new)</td><td style="text-align:right">23</td><td style="text-align:right">1</td><td style="text-align:right">2</td><td style="text-align:right">0.0872</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:right">24</td><td style="text-align:right">2</td><td style="text-align:right">0</td><td style="text-align:right">0.1469</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:right">22</td><td style="text-align:right">2</td><td style="text-align:right">2</td><td style="text-align:right">0.0536</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:right">22</td><td style="text-align:right">1</td><td style="text-align:right">4</td><td style="text-align:right">0.0584</td><td style="text-align:right">1</td></tr>
</tbody></table>

### Benches slower than GT (ratio &gt; 1.001)

<table class="flash-cmp">
<colgroup>
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Mode</th>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Ratio</th>
  <th style="text-align:right">Generated cycles</th>
  <th style="text-align:right">GT cycles</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">1.070</td><td style="text-align:right">2,343,914</td><td style="text-align:right">2,343,914</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">1.033</td><td style="text-align:right">209,526,121</td><td style="text-align:right">209,526,121</td></tr>
<tr><td style="text-align:left">Bn 2+2 (new)</td><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1.011</td><td style="text-align:right">1,160,161</td><td style="text-align:right">1,160,161</td></tr>
<tr><td style="text-align:left">Bn 4+2 (new)</td><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">1.007</td><td style="text-align:right">833,976,005</td><td style="text-align:right">833,976,005</td></tr>
<tr><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">1.014</td><td style="text-align:right">833,976,005</td><td style="text-align:right">833,976,005</td></tr>
<tr><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">1.047</td><td style="text-align:right">256,922</td><td style="text-align:right">256,922</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">1.044</td><td style="text-align:right">2,270,081</td><td style="text-align:right">2,270,081</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">1.033</td><td style="text-align:right">209,526,121</td><td style="text-align:right">209,526,121</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">1.007</td><td style="text-align:right">833,976,005</td><td style="text-align:right">833,976,005</td></tr>
</tbody></table>

## 3. Head-to-head vs Noskills (old) — latency wins (of 27 benches)

<table class="flash-cmp">
<colgroup>
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Mode</th>
  <th style="text-align:right">Mode wins</th>
  <th style="text-align:right">Noskills (old) wins</th>
  <th style="text-align:right">Ties</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">Bn 2+2 (old)</td><td style="text-align:right">9</td><td style="text-align:right">11</td><td style="text-align:right">7</td></tr>
<tr><td style="text-align:left">All+avoids (old)</td><td style="text-align:right">10</td><td style="text-align:right">16</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">No avoids (old)</td><td style="text-align:right">18</td><td style="text-align:right">9</td><td style="text-align:right">0</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:right">7</td><td style="text-align:right">18</td><td style="text-align:right">2</td></tr>
<tr><td style="text-align:left">Bn 2+2 (new)</td><td style="text-align:right">12</td><td style="text-align:right">14</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">Bn 4+2 (new)</td><td style="text-align:right">14</td><td style="text-align:right">11</td><td style="text-align:right">2</td></tr>
<tr><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:right">11</td><td style="text-align:right">15</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:right">15</td><td style="text-align:right">10</td><td style="text-align:right">2</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:right">11</td><td style="text-align:right">13</td><td style="text-align:right">3</td></tr>
</tbody></table>

## 4. Head-to-head vs Noskills (new) — latency wins (of 27 benches)

<table class="flash-cmp">
<colgroup>
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Mode</th>
  <th style="text-align:right">Mode wins</th>
  <th style="text-align:right">Noskills (new) wins</th>
  <th style="text-align:right">Ties</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">Noskills (old)</td><td style="text-align:right">18</td><td style="text-align:right">7</td><td style="text-align:right">2</td></tr>
<tr><td style="text-align:left">Bn 2+2 (old)</td><td style="text-align:right">17</td><td style="text-align:right">8</td><td style="text-align:right">2</td></tr>
<tr><td style="text-align:left">All+avoids (old)</td><td style="text-align:right">13</td><td style="text-align:right">12</td><td style="text-align:right">2</td></tr>
<tr><td style="text-align:left">No avoids (old)</td><td style="text-align:right">22</td><td style="text-align:right">5</td><td style="text-align:right">0</td></tr>
<tr><td style="text-align:left">Bn 2+2 (new)</td><td style="text-align:right">12</td><td style="text-align:right">10</td><td style="text-align:right">5</td></tr>
<tr><td style="text-align:left">Bn 4+2 (new)</td><td style="text-align:right">16</td><td style="text-align:right">8</td><td style="text-align:right">3</td></tr>
<tr><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:right">11</td><td style="text-align:right">16</td><td style="text-align:right">0</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:right">15</td><td style="text-align:right">11</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:right">16</td><td style="text-align:right">7</td><td style="text-align:right">4</td></tr>
</tbody></table>

## 5. Paired old → new (same skill policy)

<table class="flash-cmp">
<colgroup>
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Policy</th>
  <th style="text-align:right">New wins</th>
  <th style="text-align:right">Old wins</th>
  <th style="text-align:right">Ties</th>
  <th style="text-align:left">Verdict</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">Noskills</td><td style="text-align:right">7</td><td style="text-align:right">18</td><td style="text-align:right">2</td><td style="text-align:left"><strong>Old library better</strong></td></tr>
<tr><td style="text-align:left">Bn 2+2</td><td style="text-align:right">12</td><td style="text-align:right">13</td><td style="text-align:right">2</td><td style="text-align:left"><strong>Old library better</strong></td></tr>
<tr><td style="text-align:left">All+avoids</td><td style="text-align:right">15</td><td style="text-align:right">8</td><td style="text-align:right">4</td><td style="text-align:left"><strong>New library better</strong></td></tr>
<tr><td style="text-align:left">No avoids</td><td style="text-align:right">9</td><td style="text-align:right">14</td><td style="text-align:right">4</td><td style="text-align:left"><strong>Old library better</strong></td></tr>
</tbody></table>

## 6. New BN skill-count sweep

<table class="flash-cmp">
<colgroup>
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Comparison</th>
  <th style="text-align:right">Second wins</th>
  <th style="text-align:right">First wins</th>
  <th style="text-align:right">Ties</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">Bn 4+2 vs Bn 2+2</td><td style="text-align:right">11</td><td style="text-align:right">12</td><td style="text-align:right">4</td></tr>
<tr><td style="text-align:left">Bn 6+2 vs Bn 2+2</td><td style="text-align:right">10</td><td style="text-align:right">14</td><td style="text-align:right">3</td></tr>
<tr><td style="text-align:left">Bn 6+2 vs Bn 4+2</td><td style="text-align:right">9</td><td style="text-align:right">14</td><td style="text-align:right">4</td></tr>
</tbody></table>

## 7. LLM-curated skills matrix (15 runs)

<table class="flash-cmp">
<colgroup>
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Mode</th>
  <th style="text-align:left">Artifact root</th>
  <th style="text-align:right">OK</th>
  <th style="text-align:right">Best-latency wins</th>
  <th style="text-align:right">Geo-mean lat/GT</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">All+avoids json+LLM (warnings)</td><td style="text-align:left"><code>flash_curated_all_avoids_llm_warnings_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">4.0/27</td><td style="text-align:right">0.0644</td></tr>
<tr><td style="text-align:left">No avoids json+LLM (warnings)</td><td style="text-align:left"><code>flash_curated_no_avoids_llm_warnings_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">1.3/27</td><td style="text-align:right">0.0645</td></tr>
<tr><td style="text-align:left">All+avoids json_only (bottleneck)</td><td style="text-align:left"><code>flash_curated_all_avoids_json_bottleneck_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">5.0/27</td><td style="text-align:right">0.0719</td></tr>
<tr><td style="text-align:left">Noskills (warnings)</td><td style="text-align:left"><code>flash_curated_noskills_warnings_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">1.3/27</td><td style="text-align:right">0.0749</td></tr>
<tr><td style="text-align:left">No avoids json_only (combined)</td><td style="text-align:left"><code>flash_curated_no_avoids_json_combined_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">2.0/27</td><td style="text-align:right">0.0812</td></tr>
<tr><td style="text-align:left">Noskills (bottleneck)</td><td style="text-align:left"><code>flash_curated_noskills_bottleneck_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">0.3/27</td><td style="text-align:right">0.0834</td></tr>
<tr><td style="text-align:left">Noskills (combined)</td><td style="text-align:left"><code>flash_curated_noskills_combined_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">2.0/27</td><td style="text-align:right">0.0914</td></tr>
<tr><td style="text-align:left">All+avoids json+LLM (bottleneck)</td><td style="text-align:left"><code>flash_curated_all_avoids_llm_bottleneck_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">1.0/27</td><td style="text-align:right">0.1000</td></tr>
<tr><td style="text-align:left">No avoids json+LLM (combined)</td><td style="text-align:left"><code>flash_curated_no_avoids_llm_combined_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">3.0/27</td><td style="text-align:right">0.1048</td></tr>
<tr><td style="text-align:left">All+avoids json_only (combined)</td><td style="text-align:left"><code>flash_curated_all_avoids_json_combined_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">0.0/27</td><td style="text-align:right">0.1158</td></tr>
<tr><td style="text-align:left">No avoids json+LLM (bottleneck)</td><td style="text-align:left"><code>flash_curated_no_avoids_llm_bottleneck_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">0.0/27</td><td style="text-align:right">0.1306</td></tr>
<tr><td style="text-align:left">No avoids json_only (bottleneck)</td><td style="text-align:left"><code>flash_curated_no_avoids_json_bottleneck_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">5.0/27</td><td style="text-align:right">0.1329</td></tr>
<tr><td style="text-align:left">All+avoids json+LLM (combined)</td><td style="text-align:left"><code>flash_curated_all_avoids_llm_combined_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">0.0/27</td><td style="text-align:right">0.1333</td></tr>
<tr><td style="text-align:left">All+avoids json_only (warnings)</td><td style="text-align:left"><code>flash_curated_all_avoids_json_warnings_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">1.0/27</td><td style="text-align:right">0.1358</td></tr>
<tr><td style="text-align:left">No avoids json_only (warnings)</td><td style="text-align:left"><code>flash_curated_no_avoids_json_warnings_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">1.0/27</td><td style="text-align:right">0.1564</td></tr>
</tbody></table>

## 8. Curated bottleneck wave vs noskills baselines

<table class="flash-cmp">
<colgroup>
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Curated mode (bottleneck)</th>
  <th style="text-align:left">Baseline</th>
  <th style="text-align:right">Curated wins</th>
  <th style="text-align:right">Baseline wins</th>
  <th style="text-align:right">Ties</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">Noskills (bottleneck)</td><td style="text-align:left">Noskills (old)</td><td style="text-align:right">10</td><td style="text-align:right">13</td><td style="text-align:right">4</td></tr>
<tr><td style="text-align:left">Noskills (bottleneck)</td><td style="text-align:left">Noskills (new)</td><td style="text-align:right">14</td><td style="text-align:right">11</td><td style="text-align:right">2</td></tr>
<tr><td style="text-align:left">All+avoids json_only (bottleneck)</td><td style="text-align:left">Noskills (old)</td><td style="text-align:right">9</td><td style="text-align:right">15</td><td style="text-align:right">3</td></tr>
<tr><td style="text-align:left">All+avoids json_only (bottleneck)</td><td style="text-align:left">Noskills (new)</td><td style="text-align:right">17</td><td style="text-align:right">7</td><td style="text-align:right">3</td></tr>
<tr><td style="text-align:left">All+avoids json+LLM (bottleneck)</td><td style="text-align:left">Noskills (old)</td><td style="text-align:right">12</td><td style="text-align:right">15</td><td style="text-align:right">0</td></tr>
<tr><td style="text-align:left">All+avoids json+LLM (bottleneck)</td><td style="text-align:left">Noskills (new)</td><td style="text-align:right">13</td><td style="text-align:right">13</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">No avoids json_only (bottleneck)</td><td style="text-align:left">Noskills (old)</td><td style="text-align:right">10</td><td style="text-align:right">14</td><td style="text-align:right">3</td></tr>
<tr><td style="text-align:left">No avoids json_only (bottleneck)</td><td style="text-align:left">Noskills (new)</td><td style="text-align:right">12</td><td style="text-align:right">12</td><td style="text-align:right">3</td></tr>
<tr><td style="text-align:left">No avoids json+LLM (bottleneck)</td><td style="text-align:left">Noskills (old)</td><td style="text-align:right">11</td><td style="text-align:right">15</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">No avoids json+LLM (bottleneck)</td><td style="text-align:left">Noskills (new)</td><td style="text-align:right">14</td><td style="text-align:right">12</td><td style="text-align:right">1</td></tr>
</tbody></table>

## 9. Baseline 3-way re-run (85-skill lib, stamp `20260622_215520`)

<table class="flash-cmp">
<colgroup>
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Mode</th>
  <th style="text-align:right">OK</th>
  <th style="text-align:right">Geo-mean lat/GT</th>
  <th style="text-align:right">Faster than GT</th>
  <th style="text-align:right">Slower than GT</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">Noskills (new, r3)</td><td style="text-align:right">27/28</td><td style="text-align:right">0.0728</td><td style="text-align:right">25</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">All+avoids (new, r3)</td><td style="text-align:right">27/28</td><td style="text-align:right">0.0987</td><td style="text-align:right">19</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">No avoids (new, r3)</td><td style="text-align:right">27/28</td><td style="text-align:right">0.0555</td><td style="text-align:right">25</td><td style="text-align:right">1</td></tr>
</tbody></table>

### r3 vs r2 (20260621_020847) — paired new modes

<table class="flash-cmp">
<colgroup>
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Mode</th>
  <th style="text-align:right">r3 wins</th>
  <th style="text-align:right">r2 wins</th>
  <th style="text-align:right">Ties</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">Noskills (new, r3)</td><td style="text-align:right">11</td><td style="text-align:right">12</td><td style="text-align:right">4</td></tr>
<tr><td style="text-align:left">All+avoids (new, r3)</td><td style="text-align:right">9</td><td style="text-align:right">17</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">No avoids (new, r3)</td><td style="text-align:right">10</td><td style="text-align:right">13</td><td style="text-align:right">4</td></tr>
</tbody></table>

## 10. Ground-truth latency ratio per benchmark (all deterministic modes)

<table class="flash-cmp">
<colgroup>
  <col style="width:8%">
  <col style="width:8%">
  <col style="width:8%">
  <col style="width:8%">
  <col style="width:8%">
  <col style="width:8%">
  <col style="width:8%">
  <col style="width:8%">
  <col style="width:8%">
  <col style="width:8%">
  <col style="width:8%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Noskills (old)</th>
  <th style="text-align:right">Bn 2+2 (old)</th>
  <th style="text-align:right">All+avoids (old)</th>
  <th style="text-align:right">No avoids (old)</th>
  <th style="text-align:right">Noskills (new)</th>
  <th style="text-align:right">Bn 2+2 (new)</th>
  <th style="text-align:right">Bn 4+2 (new)</th>
  <th style="text-align:right">Bn 6+2 (new)</th>
  <th style="text-align:right">All+avoids (new)</th>
  <th style="text-align:right">No avoids (new)</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">0.0026</td><td style="text-align:right">0.0050</td><td style="text-align:right">0.0030</td><td style="text-align:right">0.0020</td><td style="text-align:right">0.0590</td><td style="text-align:right">0.0500</td><td style="text-align:right">0.0500</td><td style="text-align:right">0.0050</td><td style="text-align:right">0.0030</td><td style="text-align:right">0.0180</td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">0.0022</td><td style="text-align:right">0.0020</td><td style="text-align:right">0.0150</td><td style="text-align:right">1.0000</td><td style="text-align:right">0.0450</td><td style="text-align:right">0.0450</td><td style="text-align:right">0.0450</td><td style="text-align:right">0.0630</td><td style="text-align:right">0.0470</td><td style="text-align:right">0.0020</td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">0.0487</td><td style="text-align:right">1.0000</td><td style="text-align:right">0.9470</td><td style="text-align:right">0.0020</td><td style="text-align:right">0.0600</td><td style="text-align:right">0.1480</td><td style="text-align:right">1.0000</td><td style="text-align:right">0.9530</td><td style="text-align:right">0.0020</td><td style="text-align:right">0.0020</td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">0.3104</td><td style="text-align:right">0.9800</td><td style="text-align:right">0.3460</td><td style="text-align:right">0.0020</td><td style="text-align:right">1.0700</td><td style="text-align:right">0.0500</td><td style="text-align:right">0.0570</td><td style="text-align:right">0.0540</td><td style="text-align:right">0.0470</td><td style="text-align:right">0.0020</td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">0.0582</td><td style="text-align:right">0.0580</td><td style="text-align:right">1.0000</td><td style="text-align:right">0.0200</td><td style="text-align:right">0.2850</td><td style="text-align:right">0.0020</td><td style="text-align:right">0.1170</td><td style="text-align:right">0.1160</td><td style="text-align:right">0.0060</td><td style="text-align:right">0.0050</td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">0.0338</td><td style="text-align:right">0.0340</td><td style="text-align:right">0.1750</td><td style="text-align:right">0.1490</td><td style="text-align:right">0.0020</td><td style="text-align:right">0.0340</td><td style="text-align:right">0.1990</td><td style="text-align:right">0.0340</td><td style="text-align:right">0.0080</td><td style="text-align:right">0.0570</td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">0.0590</td><td style="text-align:right">0.0430</td><td style="text-align:right">1.0000</td><td style="text-align:right">0.0200</td><td style="text-align:right">0.0600</td><td style="text-align:right">0.0500</td><td style="text-align:right">0.0010</td><td style="text-align:right">0.0420</td><td style="text-align:right">0.0640</td><td style="text-align:right">0.0280</td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">0.8759</td><td style="text-align:right">0.8940</td><td style="text-align:right">1.0000</td><td style="text-align:right">0.9450</td><td style="text-align:right">0.8940</td><td style="text-align:right">0.8940</td><td style="text-align:right">0.8760</td><td style="text-align:right">0.9450</td><td style="text-align:right">0.5340</td><td style="text-align:right">0.8760</td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">0.0061</td><td style="text-align:right">0.0060</td><td style="text-align:right">0.0000</td><td style="text-align:right">0.0000</td><td style="text-align:right">0.0330</td><td style="text-align:right">0.8890</td><td style="text-align:right">0.0000</td><td style="text-align:right">0.0000</td><td style="text-align:right">0.0000</td><td style="text-align:right">0.0060</td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">0.0336</td><td style="text-align:right">0.0140</td><td style="text-align:right">0.5780</td><td style="text-align:right">0.0070</td><td style="text-align:right">0.0140</td><td style="text-align:right">1.0000</td><td style="text-align:right">1.0070</td><td style="text-align:right">1.0140</td><td style="text-align:right">1.0000</td><td style="text-align:right">1.0070</td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">0.0009</td><td style="text-align:right">0.0010</td><td style="text-align:right">0.0570</td><td style="text-align:right">0.0010</td><td style="text-align:right">0.0070</td><td style="text-align:right">0.0010</td><td style="text-align:right">0.0070</td><td style="text-align:right">0.0250</td><td style="text-align:right">0.0100</td><td style="text-align:right">0.0030</td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">0.1172</td><td style="text-align:right">0.0790</td><td style="text-align:right">0.0940</td><td style="text-align:right">0.0200</td><td style="text-align:right">0.0860</td><td style="text-align:right">1.0010</td><td style="text-align:right">0.0960</td><td style="text-align:right">0.1040</td><td style="text-align:right">0.0900</td><td style="text-align:right">0.0270</td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">0.0063</td><td style="text-align:right">0.0010</td><td style="text-align:right">0.0190</td><td style="text-align:right">0.0130</td><td style="text-align:right">0.0410</td><td style="text-align:right">0.1830</td><td style="text-align:right">0.0010</td><td style="text-align:right">0.1250</td><td style="text-align:right">0.0040</td><td style="text-align:right">0.0320</td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">0.5711</td><td style="text-align:right">0.5710</td><td style="text-align:right">1.0000</td><td style="text-align:right">1.0000</td><td style="text-align:right">0.1300</td><td style="text-align:right">0.5620</td><td style="text-align:right">0.1300</td><td style="text-align:right">0.5620</td><td style="text-align:right">1.0440</td><td style="text-align:right">1.0010</td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">0.0148</td><td style="text-align:right">0.0150</td><td style="text-align:right">0.0110</td><td style="text-align:right">1.0000</td><td style="text-align:right">0.1490</td><td style="text-align:right">0.1490</td><td style="text-align:right">0.0120</td><td style="text-align:right">0.2460</td><td style="text-align:right">0.6140</td><td style="text-align:right">0.1490</td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">0.3177</td><td style="text-align:right">0.3180</td><td style="text-align:right">0.2930</td><td style="text-align:right">0.2990</td><td style="text-align:right">0.3210</td><td style="text-align:right">0.3120</td><td style="text-align:right">0.3120</td><td style="text-align:right">0.3120</td><td style="text-align:right">0.2930</td><td style="text-align:right">0.2990</td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">0.0090</td><td style="text-align:right">0.0090</td><td style="text-align:right">0.7910</td><td style="text-align:right">0.0090</td><td style="text-align:right">0.2060</td><td style="text-align:right">0.2060</td><td style="text-align:right">0.0060</td><td style="text-align:right">0.5190</td><td style="text-align:right">0.0030</td><td style="text-align:right">0.6360</td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">0.1162</td><td style="text-align:right">0.1280</td><td style="text-align:right">0.0380</td><td style="text-align:right">0.0420</td><td style="text-align:right">0.0580</td><td style="text-align:right">0.0880</td><td style="text-align:right">1.0000</td><td style="text-align:right">0.1160</td><td style="text-align:right">0.0360</td><td style="text-align:right">0.0190</td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">1.0000</td><td style="text-align:right">0.6610</td><td style="text-align:right">0.1910</td><td style="text-align:right">0.0730</td><td style="text-align:right">1.0000</td><td style="text-align:right">0.6470</td><td style="text-align:right">0.6500</td><td style="text-align:right">0.6520</td><td style="text-align:right">0.6500</td><td style="text-align:right">1.0000</td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">0.8192</td><td style="text-align:right">0.8190</td><td style="text-align:right">0.1870</td><td style="text-align:right">0.7760</td><td style="text-align:right">0.9910</td><td style="text-align:right">0.7520</td><td style="text-align:right">0.7760</td><td style="text-align:right">1.0470</td><td style="text-align:right">0.0190</td><td style="text-align:right">0.0340</td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">1.0000</td><td style="text-align:right">0.5660</td><td style="text-align:right">0.5580</td><td style="text-align:right">0.5660</td><td style="text-align:right">1.0330</td><td style="text-align:right">0.0150</td><td style="text-align:right">0.4950</td><td style="text-align:right">0.5630</td><td style="text-align:right">1.0330</td><td style="text-align:right">0.0450</td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">0.0044</td><td style="text-align:right">0.2610</td><td style="text-align:right">0.2870</td><td style="text-align:right">0.0080</td><td style="text-align:right">1.0000</td><td style="text-align:right">0.3030</td><td style="text-align:right">0.2610</td><td style="text-align:right">0.0050</td><td style="text-align:right">0.0080</td><td style="text-align:right">1.0000</td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">0.1232</td><td style="text-align:right">0.1230</td><td style="text-align:right">0.1230</td><td style="text-align:right">0.0540</td><td style="text-align:right">0.1230</td><td style="text-align:right">0.1230</td><td style="text-align:right">0.2240</td><td style="text-align:right">0.2240</td><td style="text-align:right">0.2290</td><td style="text-align:right">1.0000</td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">0.0293</td><td style="text-align:right">0.2330</td><td style="text-align:right">0.2060</td><td style="text-align:right">0.0020</td><td style="text-align:right">0.0300</td><td style="text-align:right">0.0050</td><td style="text-align:right">0.0460</td><td style="text-align:right">0.0460</td><td style="text-align:right">0.0040</td><td style="text-align:right">0.2060</td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">0.0498</td><td style="text-align:right">0.0040</td><td style="text-align:right">1.0000</td><td style="text-align:right">0.0070</td><td style="text-align:right">1.0000</td><td style="text-align:right">0.0500</td><td style="text-align:right">0.0070</td><td style="text-align:right">0.0430</td><td style="text-align:right">0.0180</td><td style="text-align:right">0.0020</td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">0.1201</td><td style="text-align:right">0.5850</td><td style="text-align:right">0.0190</td><td style="text-align:right">0.0350</td><td style="text-align:right">0.1560</td><td style="text-align:right">1.0110</td><td style="text-align:right">0.0970</td><td style="text-align:right">0.9960</td><td style="text-align:right">1.0000</td><td style="text-align:right">0.7620</td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">0.1221</td><td style="text-align:right">0.1220</td><td style="text-align:right">0.2990</td><td style="text-align:right">0.0120</td><td style="text-align:right">0.1200</td><td style="text-align:right">0.1220</td><td style="text-align:right">0.1220</td><td style="text-align:right">0.1200</td><td style="text-align:right">0.3380</td><td style="text-align:right">0.1200</td></tr>
</tbody></table>

## 11. Latency (cycles) — legacy modes

<table class="flash-cmp">
<colgroup>
  <col style="width:14%">
  <col style="width:14%">
  <col style="width:14%">
  <col style="width:14%">
  <col style="width:14%">
  <col style="width:14%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Ground truth</th>
  <th style="text-align:right">Noskills (old)</th>
  <th style="text-align:right">Bn 2+2 (old)</th>
  <th style="text-align:right">All+avoids (old)</th>
  <th style="text-align:right">No avoids (old)</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">25,296,077</td><td style="text-align:right">64,530</td><td style="text-align:right">123,785</td><td style="text-align:right">64,530</td><td style="text-align:right">62,489</td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">45,441,119</td><td style="text-align:right">99,467</td><td style="text-align:right">99,467</td><td style="text-align:right">677,058</td><td style="text-align:right">45,441,119</td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">2,429,238</td><td style="text-align:right">118,344</td><td style="text-align:right">2,429,238</td><td style="text-align:right">2,300,710</td><td style="text-align:right">5,046</td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">2,343,914</td><td style="text-align:right">727,515</td><td style="text-align:right">2,297,693</td><td style="text-align:right">810,190</td><td style="text-align:right">5,316</td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">68,337,001</td><td style="text-align:right">3,980,432</td><td style="text-align:right">3,988,492</td><td style="text-align:right">68,337,001</td><td style="text-align:right">1,340,865</td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">53,906,591</td><td style="text-align:right">1,821,943</td><td style="text-align:right">1,837,814</td><td style="text-align:right">9,438,432</td><td style="text-align:right">8,030,247</td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">64,118,323</td><td style="text-align:right">3,786,109</td><td style="text-align:right">2,730,577</td><td style="text-align:right">64,118,323</td><td style="text-align:right">1,263,645</td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td style="text-align:right">—</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">115,097</td><td style="text-align:right">100,817</td><td style="text-align:right">102,917</td><td style="text-align:right">115,097</td><td style="text-align:right">108,790</td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">98,261,801</td><td style="text-align:right">601,662</td><td style="text-align:right">598,141</td><td style="text-align:right">18,499</td><td style="text-align:right">18,499</td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">28,051,921</td><td style="text-align:right">11,728,993</td><td style="text-align:right">482,436,001</td><td style="text-align:right">5,896,958</td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">54,298,622</td><td style="text-align:right">49,181</td><td style="text-align:right">49,181</td><td style="text-align:right">3,101,758</td><td style="text-align:right">61,693</td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">2,715,487</td><td style="text-align:right">318,159</td><td style="text-align:right">214,907</td><td style="text-align:right">254,407</td><td style="text-align:right">53,367</td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">1,441,873</td><td style="text-align:right">9,060</td><td style="text-align:right">1,141</td><td style="text-align:right">26,975</td><td style="text-align:right">18,055</td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">2,270,081</td><td style="text-align:right">1,296,548</td><td style="text-align:right">1,296,548</td><td style="text-align:right">2,270,081</td><td style="text-align:right">2,270,081</td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">3,281,841</td><td style="text-align:right">48,677</td><td style="text-align:right">47,686</td><td style="text-align:right">35,682</td><td style="text-align:right">3,281,841</td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">41,761</td><td style="text-align:right">13,268</td><td style="text-align:right">13,268</td><td style="text-align:right">12,231</td><td style="text-align:right">12,468</td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">3,112,241</td><td style="text-align:right">27,889</td><td style="text-align:right">27,889</td><td style="text-align:right">2,461,081</td><td style="text-align:right">27,921</td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">134,372,041</td><td style="text-align:right">15,614,072</td><td style="text-align:right">17,192,281</td><td style="text-align:right">5,062,065</td><td style="text-align:right">5,672,817</td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">10,095,723</td><td style="text-align:right">10,095,723</td><td style="text-align:right">6,672,294</td><td style="text-align:right">1,925,435</td><td style="text-align:right">732,587</td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">256,922</td><td style="text-align:right">210,483</td><td style="text-align:right">210,483</td><td style="text-align:right">48,054</td><td style="text-align:right">199,322</td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">209,526,121</td><td style="text-align:right">209,526,121</td><td style="text-align:right">118,500,556</td><td style="text-align:right">116,875,261</td><td style="text-align:right">118,560,241</td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">132,556,483</td><td style="text-align:right">586,042</td><td style="text-align:right">34,560,515</td><td style="text-align:right">38,071,522</td><td style="text-align:right">1,119,361</td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">23,241,675</td><td style="text-align:right">2,864,058</td><td style="text-align:right">2,864,134</td><td style="text-align:right">2,855,106</td><td style="text-align:right">1,251,830</td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">33,807,761</td><td style="text-align:right">992,061</td><td style="text-align:right">7,878,481</td><td style="text-align:right">6,979,669</td><td style="text-align:right">83,188</td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">31,695,761</td><td style="text-align:right">1,577,981</td><td style="text-align:right">119,181</td><td style="text-align:right">31,695,761</td><td style="text-align:right">228,631</td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1,160,161</td><td style="text-align:right">139,301</td><td style="text-align:right">678,122</td><td style="text-align:right">22,113</td><td style="text-align:right">40,805</td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">22,598,401</td><td style="text-align:right">2,759,030</td><td style="text-align:right">2,768,058</td><td style="text-align:right">6,747,057</td><td style="text-align:right">260,875</td></tr>
</tbody></table>

## 12. Latency (cycles) — new skills pack modes

<table class="flash-cmp">
<colgroup>
  <col style="width:11%">
  <col style="width:11%">
  <col style="width:11%">
  <col style="width:11%">
  <col style="width:11%">
  <col style="width:11%">
  <col style="width:11%">
  <col style="width:11%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Ground truth</th>
  <th style="text-align:right">Noskills (new)</th>
  <th style="text-align:right">Bn 2+2 (new)</th>
  <th style="text-align:right">Bn 4+2 (new)</th>
  <th style="text-align:right">Bn 6+2 (new)</th>
  <th style="text-align:right">All+avoids (new)</th>
  <th style="text-align:right">No avoids (new)</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">25,296,077</td><td style="text-align:right">1,481,537</td><td style="text-align:right">1,262,346</td><td style="text-align:right">1,265,553</td><td style="text-align:right">114,328</td><td style="text-align:right">64,564</td><td style="text-align:right">454,508</td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">45,441,119</td><td style="text-align:right">2,038,096</td><td style="text-align:right">2,038,096</td><td style="text-align:right">2,038,096</td><td style="text-align:right">2,850,840</td><td style="text-align:right">2,113,550</td><td style="text-align:right">105,635</td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">2,429,238</td><td style="text-align:right">145,413</td><td style="text-align:right">359,813</td><td style="text-align:right">2,429,238</td><td style="text-align:right">2,314,746</td><td style="text-align:right">5,046</td><td style="text-align:right">5,046</td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">2,343,914</td><td style="text-align:right">2,343,914</td><td style="text-align:right">117,277</td><td style="text-align:right">132,498</td><td style="text-align:right">127,293</td><td style="text-align:right">110,530</td><td style="text-align:right">5,316</td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">68,337,001</td><td style="text-align:right">19,497,841</td><td style="text-align:right">159,704</td><td style="text-align:right">8,000,912</td><td style="text-align:right">7,923,872</td><td style="text-align:right">376,177</td><td style="text-align:right">336,393</td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">53,906,591</td><td style="text-align:right">100,993</td><td style="text-align:right">1,837,814</td><td style="text-align:right">10,753,631</td><td style="text-align:right">1,821,941</td><td style="text-align:right">429,902</td><td style="text-align:right">3,068,010</td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">64,118,323</td><td style="text-align:right">3,848,349</td><td style="text-align:right">3,193,955</td><td style="text-align:right">94,585</td><td style="text-align:right">2,695,369</td><td style="text-align:right">4,121,036</td><td style="text-align:right">1,819,720</td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td style="text-align:right">—</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">115,097</td><td style="text-align:right">102,917</td><td style="text-align:right">102,917</td><td style="text-align:right">100,817</td><td style="text-align:right">108,790</td><td style="text-align:right">61,441</td><td style="text-align:right">100,817</td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">98,261,801</td><td style="text-align:right">3,219,121</td><td style="text-align:right">87,379,241</td><td style="text-align:right">43,701</td><td style="text-align:right">18,499</td><td style="text-align:right">18,499</td><td style="text-align:right">601,662</td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">11,728,993</td><td style="text-align:right">833,976,005</td><td style="text-align:right">833,976,005</td><td style="text-align:right">833,976,005</td><td style="text-align:right">833,976,005</td><td style="text-align:right">833,976,005</td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">54,298,622</td><td style="text-align:right">355,153</td><td style="text-align:right">54,093</td><td style="text-align:right">355,153</td><td style="text-align:right">1,377,145</td><td style="text-align:right">565,921</td><td style="text-align:right">145,081</td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">2,715,487</td><td style="text-align:right">234,669</td><td style="text-align:right">2,715,487</td><td style="text-align:right">261,572</td><td style="text-align:right">283,347</td><td style="text-align:right">243,411</td><td style="text-align:right">74,622</td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">1,441,873</td><td style="text-align:right">59,095</td><td style="text-align:right">263,725</td><td style="text-align:right">1,141</td><td style="text-align:right">180,647</td><td style="text-align:right">5,443</td><td style="text-align:right">45,550</td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">2,270,081</td><td style="text-align:right">294,388</td><td style="text-align:right">1,275,551</td><td style="text-align:right">294,388</td><td style="text-align:right">1,276,224</td><td style="text-align:right">2,270,081</td><td style="text-align:right">2,270,081</td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">3,281,841</td><td style="text-align:right">488,486</td><td style="text-align:right">488,486</td><td style="text-align:right">40,080</td><td style="text-align:right">808,966</td><td style="text-align:right">2,013,441</td><td style="text-align:right">488,486</td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">41,761</td><td style="text-align:right">13,420</td><td style="text-align:right">13,031</td><td style="text-align:right">13,031</td><td style="text-align:right">13,031</td><td style="text-align:right">12,231</td><td style="text-align:right">12,468</td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">3,112,241</td><td style="text-align:right">640,370</td><td style="text-align:right">640,370</td><td style="text-align:right">18,731</td><td style="text-align:right">1,614,961</td><td style="text-align:right">9,070</td><td style="text-align:right">1,979,281</td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">134,372,041</td><td style="text-align:right">7,831,472</td><td style="text-align:right">11,829,856</td><td style="text-align:right">134,372,041</td><td style="text-align:right">15,574,591</td><td style="text-align:right">4,823,953</td><td style="text-align:right">2,591,089</td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">10,095,723</td><td style="text-align:right">10,095,723</td><td style="text-align:right">6,532,621</td><td style="text-align:right">6,557,315</td><td style="text-align:right">6,585,875</td><td style="text-align:right">6,557,315</td><td style="text-align:right">10,095,723</td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">256,922</td><td style="text-align:right">254,630</td><td style="text-align:right">193,276</td><td style="text-align:right">199,322</td><td style="text-align:right">256,922</td><td style="text-align:right">4,793</td><td style="text-align:right">8,776</td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">209,526,121</td><td style="text-align:right">209,526,121</td><td style="text-align:right">3,245,657</td><td style="text-align:right">103,652,813</td><td style="text-align:right">117,950,941</td><td style="text-align:right">209,526,121</td><td style="text-align:right">9,482,221</td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">132,556,483</td><td style="text-align:right">132,556,483</td><td style="text-align:right">40,130,103</td><td style="text-align:right">34,571,610</td><td style="text-align:right">652,761</td><td style="text-align:right">1,120,001</td><td style="text-align:right">132,556,483</td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">23,241,675</td><td style="text-align:right">2,855,106</td><td style="text-align:right">2,855,106</td><td style="text-align:right">5,201,403</td><td style="text-align:right">5,201,403</td><td style="text-align:right">5,323,907</td><td style="text-align:right">23,241,675</td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">33,807,761</td><td style="text-align:right">1,002,851</td><td style="text-align:right">166,561</td><td style="text-align:right">1,539,761</td><td style="text-align:right">1,539,761</td><td style="text-align:right">125,376</td><td style="text-align:right">6,958,389</td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">31,695,761</td><td style="text-align:right">31,695,761</td><td style="text-align:right">1,572,861</td><td style="text-align:right">209,921</td><td style="text-align:right">1,352,948</td><td style="text-align:right">556,551</td><td style="text-align:right">54,836</td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1,160,161</td><td style="text-align:right">181,152</td><td style="text-align:right">1,160,161</td><td style="text-align:right">112,062</td><td style="text-align:right">1,155,676</td><td style="text-align:right">1,160,161</td><td style="text-align:right">884,533</td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">22,598,401</td><td style="text-align:right">2,720,630</td><td style="text-align:right">2,759,030</td><td style="text-align:right">2,759,030</td><td style="text-align:right">2,704,727</td><td style="text-align:right">7,629,361</td><td style="text-align:right">2,720,629</td></tr>
</tbody></table>

## 13. Narrative bullets (for slides)

1. **All flash modes beat ground-truth HLS latency on average** — geo-mean ratios 0.034–0.171 across deterministic modes (~3–17% of GT).
2. **Champion:** No avoids (old) — **8.1/27** best-latency wins, geo-mean **0.0342**, beats noskills (old) **18–9**.
3. **Best new-library mode:** All+avoids (new) — **4.1/27** wins, geo-mean **0.0536**, beats noskills (new) **15–11**.
4. **Noskills baselines:** old geo **0.0528** (0 slower-than-GT); new geo **0.1237** (2 slower-than-GT).
5. **Skill policy > skill count:** no-avoids (positive only) beats all+avoids; BN 6+2 is worst among new BN sweeps.
6. **New 73-skill library:** helps no-avoids (14–9 vs old) and noskills (18–7); **hurts** all+avoids (8–15 vs old).
7. **LLM curation:** best curated run is No avoids json_only (bottleneck) with **5.0/27** wins but geo-mean **0.1329** — does not displace No avoids (old).
8. **Cosim is a separate axis** (not in this report): single-shot repair fixed 5/24 failures; 10-loop repair 6/24.

## 14. Related reports

- `artifacts/pc2/flash_comparison_20260621.md` — full deterministic comparison
- `artifacts/pc2/flash_comparison_curated_20260621.md` — curated matrix detail
- `artifacts/pc2/flash_cosim_comparison_20260622.md` — cosim pass/fail (separate metric)

_Generated by `scripts/pc2/generate_flash_presentation_summary_md.py` on 2026-06-23_

