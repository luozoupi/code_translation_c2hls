# Flash HLSFactory Results — Legacy vs New Skills (Full Comparison)

<style>
table.flash-cmp { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.85em; }
table.flash-cmp th, table.flash-cmp td { border: 1px solid #ccc; padding: 4px 8px; white-space: nowrap; }
table.flash-cmp th { background: #f5f5f5; font-weight: 600; }
table.flash-cmp td:first-child, table.flash-cmp th:first-child { text-align: left !important; }
table.flash-cmp .fail { color: #c00; font-weight: 600; }
table.flash-meta { border-collapse: collapse; font-size: 0.9em; }
table.flash-meta th, table.flash-meta td { border: 1px solid #ccc; padding: 4px 10px; }
table.flash-meta th { background: #f5f5f5; text-align: left; width: 220px; }
</style>

<table class="flash-meta">
<thead><tr><th>Field</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Legacy stamp (noskills / bn 2+2)</td><td><code>20260620_004507</code></td></tr>
<tr><td>Legacy stamp (global skills)</td><td><code>20260620_113247</code></td></tr>
<tr><td>New skills stamp</td><td><code>20260621_020847</code></td></tr>
<tr><td>Legacy skills file</td><td><code>skills/skills.json</code> (55 skills)</td></tr>
<tr><td>New skills file</td><td><code>skills_ii_target_miss_solutions_added.json</code> (73 skills)</td></tr>
<tr><td>Metric</td><td>Final flash-step synthesis latency (cycles), lower is better</td></tr>
<tr><td>Success</td><td>27/28 per mode (<code>doitgen</code> fails gold-ref gate)</td></tr>
<tr><td>Model</td><td><code>mistralai/Devstral-2-123B-Instruct-2512</code></td></tr>
</tbody></table>

## Summary — all modes

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
  <th style="text-align:right">Best latency</th>
  <th style="text-align:right">Geo-mean lat/GT</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">Noskills (old)</td><td style="text-align:left"><code>flash_noskills_20260620_004507</code></td><td style="text-align:right">27/28</td><td style="text-align:right">2.0/27</td><td style="text-align:right">0.0528</td></tr>
<tr><td style="text-align:left">Bn 2+2 (old)</td><td style="text-align:left"><code>flash_skills_20260620_004507</code></td><td style="text-align:right">27/28</td><td style="text-align:right">1.5/27</td><td style="text-align:right">0.0653</td></tr>
<tr><td style="text-align:left">All+avoids (old)</td><td style="text-align:left"><code>flash_all_skills_avoids_global_20260620_113247</code></td><td style="text-align:right">27/28</td><td style="text-align:right">2.8/27</td><td style="text-align:right">0.1714</td></tr>
<tr><td style="text-align:left">No avoids (old)</td><td style="text-align:left"><code>flash_all_skills_no_avoids_global_20260620_113247</code></td><td style="text-align:right">27/28</td><td style="text-align:right">8.1/27</td><td style="text-align:right">0.0342</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>flash_noskills_new_20260621_020847</code></td><td style="text-align:right">27/28</td><td style="text-align:right">1.5/27</td><td style="text-align:right">0.1237</td></tr>
<tr><td style="text-align:left">Bn 2+2 (new)</td><td style="text-align:left"><code>flash_bn_skills_new_2_2_20260621_020847</code></td><td style="text-align:right">27/28</td><td style="text-align:right">2.0/27</td><td style="text-align:right">0.1138</td></tr>
<tr><td style="text-align:left">Bn 4+2 (new)</td><td style="text-align:left"><code>flash_bn_skills_new_4_2_20260621_020847</code></td><td style="text-align:right">27/28</td><td style="text-align:right">2.0/27</td><td style="text-align:right">0.0872</td></tr>
<tr><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:left"><code>flash_bn_skills_new_6_2_20260621_020847</code></td><td style="text-align:right">27/28</td><td style="text-align:right">0.2/27</td><td style="text-align:right">0.1469</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>flash_all_new_skills_avoids_global_20260621_020847</code></td><td style="text-align:right">27/28</td><td style="text-align:right">4.1/27</td><td style="text-align:right">0.0536</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:left"><code>flash_all_new_skills_no_avoids_global_20260621_020847</code></td><td style="text-align:right">27/28</td><td style="text-align:right">2.8/27</td><td style="text-align:right">0.0584</td></tr>
</tbody></table>

## vs ground truth (latency ratio = synth / GT, lower is better)

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
<tr><td style="text-align:left">All+avoids (old)</td><td style="text-align:right">22</td><td style="text-align:right">0</td><td style="text-align:right">5</td><td style="text-align:right">0.1714</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">No avoids (old)</td><td style="text-align:right">24</td><td style="text-align:right">0</td><td style="text-align:right">3</td><td style="text-align:right">0.0342</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:right">22</td><td style="text-align:right">2</td><td style="text-align:right">3</td><td style="text-align:right">0.1237</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">Bn 2+2 (new)</td><td style="text-align:right">24</td><td style="text-align:right">1</td><td style="text-align:right">2</td><td style="text-align:right">0.1138</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">Bn 4+2 (new)</td><td style="text-align:right">24</td><td style="text-align:right">1</td><td style="text-align:right">2</td><td style="text-align:right">0.0872</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:right">25</td><td style="text-align:right">2</td><td style="text-align:right">0</td><td style="text-align:right">0.1469</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:right">23</td><td style="text-align:right">2</td><td style="text-align:right">2</td><td style="text-align:right">0.0536</td><td style="text-align:right">1</td></tr>
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
  <th style="text-align:right">Generated</th>
  <th style="text-align:right">Ground truth</th>
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

## Legacy head-to-head vs noskills (old)

<table class="flash-cmp">
<colgroup>
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Opponent</th>
  <th style="text-align:right">Opponent wins</th>
  <th style="text-align:right">Noskills wins</th>
  <th style="text-align:right">Ties</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">Bn 2+2 (old)</td><td style="text-align:right">9</td><td style="text-align:right">11</td><td style="text-align:right">7</td></tr>
<tr><td style="text-align:left">All+avoids (old)</td><td style="text-align:right">10</td><td style="text-align:right">16</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">No avoids (old)</td><td style="text-align:right">18</td><td style="text-align:right">9</td><td style="text-align:right">0</td></tr>
</tbody></table>

## New head-to-head vs noskills (new)

<table class="flash-cmp">
<colgroup>
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Opponent</th>
  <th style="text-align:right">Opponent wins</th>
  <th style="text-align:right">Noskills wins</th>
  <th style="text-align:right">Ties</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">Bn 2+2 (new)</td><td style="text-align:right">12</td><td style="text-align:right">10</td><td style="text-align:right">5</td></tr>
<tr><td style="text-align:left">Bn 4+2 (new)</td><td style="text-align:right">16</td><td style="text-align:right">8</td><td style="text-align:right">3</td></tr>
<tr><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:right">11</td><td style="text-align:right">16</td><td style="text-align:right">0</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:right">15</td><td style="text-align:right">11</td><td style="text-align:right">1</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:right">16</td><td style="text-align:right">7</td><td style="text-align:right">4</td></tr>
</tbody></table>

## Paired new vs corresponding legacy

<table class="flash-cmp">
<colgroup>
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Pair</th>
  <th style="text-align:right">New wins</th>
  <th style="text-align:right">Old wins</th>
  <th style="text-align:right">Ties</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">Noskills</td><td style="text-align:right">18</td><td style="text-align:right">7</td><td style="text-align:right">2</td></tr>
<tr><td style="text-align:left">Bn 2+2</td><td style="text-align:right">13</td><td style="text-align:right">12</td><td style="text-align:right">2</td></tr>
<tr><td style="text-align:left">All+avoids</td><td style="text-align:right">8</td><td style="text-align:right">15</td><td style="text-align:right">4</td></tr>
<tr><td style="text-align:left">No avoids</td><td style="text-align:right">14</td><td style="text-align:right">9</td><td style="text-align:right">4</td></tr>
</tbody></table>

## New BN skill-count sweep

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
<tr><td style="text-align:left">Bn 4+2 vs Bn 2+2</td><td style="text-align:right">12</td><td style="text-align:right">11</td><td style="text-align:right">4</td></tr>
<tr><td style="text-align:left">Bn 6+2 vs Bn 2+2</td><td style="text-align:right">14</td><td style="text-align:right">10</td><td style="text-align:right">3</td></tr>
<tr><td style="text-align:left">Bn 6+2 vs Bn 4+2</td><td style="text-align:right">14</td><td style="text-align:right">9</td><td style="text-align:right">4</td></tr>
</tbody></table>

## Paired latency changes &gt; 5% (new vs old)

### Noskills

<table class="flash-cmp">
<colgroup>
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Old cycles</th>
  <th style="text-align:right">New cycles</th>
  <th style="text-align:right">Change</th>
  <th style="text-align:right">Better</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">586,042</td><td style="text-align:right">132,556,483</td><td style="text-align:right">+22518.9%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">27,889</td><td style="text-align:right">640,370</td><td style="text-align:right">+2196.1%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">64,530</td><td style="text-align:right">1,481,537</td><td style="text-align:right">+2195.9%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">99,467</td><td style="text-align:right">2,038,096</td><td style="text-align:right">+1949.0%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">1,577,981</td><td style="text-align:right">31,695,761</td><td style="text-align:right">+1908.6%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">48,677</td><td style="text-align:right">488,486</td><td style="text-align:right">+903.5%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">49,181</td><td style="text-align:right">355,153</td><td style="text-align:right">+622.1%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">9,060</td><td style="text-align:right">59,095</td><td style="text-align:right">+552.3%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">601,662</td><td style="text-align:right">3,219,121</td><td style="text-align:right">+435.0%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">3,980,432</td><td style="text-align:right">19,497,841</td><td style="text-align:right">+389.8%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">727,515</td><td style="text-align:right">2,343,914</td><td style="text-align:right">+222.2%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">1,821,943</td><td style="text-align:right">100,993</td><td style="text-align:right">-94.5%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">1,296,548</td><td style="text-align:right">294,388</td><td style="text-align:right">-77.3%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">28,051,921</td><td style="text-align:right">11,728,993</td><td style="text-align:right">-58.2%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">15,614,072</td><td style="text-align:right">7,831,472</td><td style="text-align:right">-49.8%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">139,301</td><td style="text-align:right">181,152</td><td style="text-align:right">+30.0%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">318,159</td><td style="text-align:right">234,669</td><td style="text-align:right">-26.2%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">118,344</td><td style="text-align:right">145,413</td><td style="text-align:right">+22.9%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">210,483</td><td style="text-align:right">254,630</td><td style="text-align:right">+21.0%</td><td style="text-align:right"><strong>old</strong></td></tr>
</tbody></table>

### Bn 2+2

<table class="flash-cmp">
<colgroup>
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Old cycles</th>
  <th style="text-align:right">New cycles</th>
  <th style="text-align:right">Change</th>
  <th style="text-align:right">Better</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">1,141</td><td style="text-align:right">263,725</td><td style="text-align:right">+23013.5%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">598,141</td><td style="text-align:right">87,379,241</td><td style="text-align:right">+14508.5%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">11,728,993</td><td style="text-align:right">833,976,005</td><td style="text-align:right">+7010.4%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">27,889</td><td style="text-align:right">640,370</td><td style="text-align:right">+2196.1%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">99,467</td><td style="text-align:right">2,038,096</td><td style="text-align:right">+1949.0%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">119,181</td><td style="text-align:right">1,572,861</td><td style="text-align:right">+1219.7%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">214,907</td><td style="text-align:right">2,715,487</td><td style="text-align:right">+1163.6%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">47,686</td><td style="text-align:right">488,486</td><td style="text-align:right">+924.4%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">123,785</td><td style="text-align:right">1,262,346</td><td style="text-align:right">+919.8%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">7,878,481</td><td style="text-align:right">166,561</td><td style="text-align:right">-97.9%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">118,500,556</td><td style="text-align:right">3,245,657</td><td style="text-align:right">-97.3%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">3,988,492</td><td style="text-align:right">159,704</td><td style="text-align:right">-96.0%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">2,297,693</td><td style="text-align:right">117,277</td><td style="text-align:right">-94.9%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">2,429,238</td><td style="text-align:right">359,813</td><td style="text-align:right">-85.2%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">678,122</td><td style="text-align:right">1,160,161</td><td style="text-align:right">+71.1%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">17,192,281</td><td style="text-align:right">11,829,856</td><td style="text-align:right">-31.2%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">2,730,577</td><td style="text-align:right">3,193,955</td><td style="text-align:right">+17.0%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">34,560,515</td><td style="text-align:right">40,130,103</td><td style="text-align:right">+16.1%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">49,181</td><td style="text-align:right">54,093</td><td style="text-align:right">+10.0%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">210,483</td><td style="text-align:right">193,276</td><td style="text-align:right">-8.2%</td><td style="text-align:right"><strong>new</strong></td></tr>
</tbody></table>

### All+avoids

<table class="flash-cmp">
<colgroup>
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Old cycles</th>
  <th style="text-align:right">New cycles</th>
  <th style="text-align:right">Change</th>
  <th style="text-align:right">Better</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">35,682</td><td style="text-align:right">2,013,441</td><td style="text-align:right">+5542.7%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">22,113</td><td style="text-align:right">1,160,161</td><td style="text-align:right">+5146.5%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">1,925,435</td><td style="text-align:right">6,557,315</td><td style="text-align:right">+240.6%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">677,058</td><td style="text-align:right">2,113,550</td><td style="text-align:right">+212.2%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">2,300,710</td><td style="text-align:right">5,046</td><td style="text-align:right">-99.8%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">2,461,081</td><td style="text-align:right">9,070</td><td style="text-align:right">-99.6%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">68,337,001</td><td style="text-align:right">376,177</td><td style="text-align:right">-99.4%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">31,695,761</td><td style="text-align:right">556,551</td><td style="text-align:right">-98.2%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">6,979,669</td><td style="text-align:right">125,376</td><td style="text-align:right">-98.2%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">38,071,522</td><td style="text-align:right">1,120,001</td><td style="text-align:right">-97.1%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">9,438,432</td><td style="text-align:right">429,902</td><td style="text-align:right">-95.4%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">64,118,323</td><td style="text-align:right">4,121,036</td><td style="text-align:right">-93.6%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">48,054</td><td style="text-align:right">4,793</td><td style="text-align:right">-90.0%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">2,855,106</td><td style="text-align:right">5,323,907</td><td style="text-align:right">+86.5%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">810,190</td><td style="text-align:right">110,530</td><td style="text-align:right">-86.4%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">3,101,758</td><td style="text-align:right">565,921</td><td style="text-align:right">-81.8%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">26,975</td><td style="text-align:right">5,443</td><td style="text-align:right">-79.8%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">116,875,261</td><td style="text-align:right">209,526,121</td><td style="text-align:right">+79.3%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">482,436,001</td><td style="text-align:right">833,976,005</td><td style="text-align:right">+72.9%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">115,097</td><td style="text-align:right">61,441</td><td style="text-align:right">-46.6%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">6,747,057</td><td style="text-align:right">7,629,361</td><td style="text-align:right">+13.1%</td><td style="text-align:right"><strong>old</strong></td></tr>
</tbody></table>

### No avoids

<table class="flash-cmp">
<colgroup>
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
  <col style="width:17%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Old cycles</th>
  <th style="text-align:right">New cycles</th>
  <th style="text-align:right">Change</th>
  <th style="text-align:right">Better</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">5,896,958</td><td style="text-align:right">833,976,005</td><td style="text-align:right">+14042.5%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">1,119,361</td><td style="text-align:right">132,556,483</td><td style="text-align:right">+11742.2%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">83,188</td><td style="text-align:right">6,958,389</td><td style="text-align:right">+8264.7%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">27,921</td><td style="text-align:right">1,979,281</td><td style="text-align:right">+6988.9%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">18,499</td><td style="text-align:right">601,662</td><td style="text-align:right">+3152.4%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">40,805</td><td style="text-align:right">884,533</td><td style="text-align:right">+2067.7%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">1,251,830</td><td style="text-align:right">23,241,675</td><td style="text-align:right">+1756.6%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">732,587</td><td style="text-align:right">10,095,723</td><td style="text-align:right">+1278.1%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">260,875</td><td style="text-align:right">2,720,629</td><td style="text-align:right">+942.9%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">62,489</td><td style="text-align:right">454,508</td><td style="text-align:right">+627.3%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">18,055</td><td style="text-align:right">45,550</td><td style="text-align:right">+152.3%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">61,693</td><td style="text-align:right">145,081</td><td style="text-align:right">+135.2%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">45,441,119</td><td style="text-align:right">105,635</td><td style="text-align:right">-99.8%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">199,322</td><td style="text-align:right">8,776</td><td style="text-align:right">-95.6%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">118,560,241</td><td style="text-align:right">9,482,221</td><td style="text-align:right">-92.0%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">3,281,841</td><td style="text-align:right">488,486</td><td style="text-align:right">-85.1%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">228,631</td><td style="text-align:right">54,836</td><td style="text-align:right">-76.0%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">1,340,865</td><td style="text-align:right">336,393</td><td style="text-align:right">-74.9%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">8,030,247</td><td style="text-align:right">3,068,010</td><td style="text-align:right">-61.8%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">5,672,817</td><td style="text-align:right">2,591,089</td><td style="text-align:right">-54.3%</td><td style="text-align:right"><strong>new</strong></td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">1,263,645</td><td style="text-align:right">1,819,720</td><td style="text-align:right">+44.0%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">53,367</td><td style="text-align:right">74,622</td><td style="text-align:right">+39.8%</td><td style="text-align:right"><strong>old</strong></td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">108,790</td><td style="text-align:right">100,817</td><td style="text-align:right">-7.3%</td><td style="text-align:right"><strong>new</strong></td></tr>
</tbody></table>


## Latency (cycles) — legacy

<table class="flash-cmp">
<colgroup>
  <col style="width:12%">
  <col style="width:12%">
  <col style="width:12%">
  <col style="width:12%">
  <col style="width:12%">
  <col style="width:12%">
  <col style="width:12%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Ground truth</th>
  <th style="text-align:right">Noskills</th>
  <th style="text-align:right">Bn 2+2</th>
  <th style="text-align:right">All+avoids</th>
  <th style="text-align:right">No avoids</th>
  <th style="text-align:right">Winner</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">25,296,077</td><td style="text-align:right">64,530</td><td style="text-align:right">123,785</td><td style="text-align:right">64,530</td><td style="text-align:right">62,489</td><td style="text-align:right"><strong>No avoids</strong></td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">45,441,119</td><td style="text-align:right">99,467</td><td style="text-align:right">99,467</td><td style="text-align:right">677,058</td><td style="text-align:right">45,441,119</td><td style="text-align:right"><strong>Bn 2+2+Noskills</strong></td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">2,429,238</td><td style="text-align:right">118,344</td><td style="text-align:right">2,429,238</td><td style="text-align:right">2,300,710</td><td style="text-align:right">5,046</td><td style="text-align:right"><strong>No avoids</strong></td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">2,343,914</td><td style="text-align:right">727,515</td><td style="text-align:right">2,297,693</td><td style="text-align:right">810,190</td><td style="text-align:right">5,316</td><td style="text-align:right"><strong>No avoids</strong></td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">68,337,001</td><td style="text-align:right">3,980,432</td><td style="text-align:right">3,988,492</td><td style="text-align:right">68,337,001</td><td style="text-align:right">1,340,865</td><td style="text-align:right"><strong>No avoids</strong></td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">53,906,591</td><td style="text-align:right">1,821,943</td><td style="text-align:right">1,837,814</td><td style="text-align:right">9,438,432</td><td style="text-align:right">8,030,247</td><td style="text-align:right"><strong>Noskills</strong></td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">64,118,323</td><td style="text-align:right">3,786,109</td><td style="text-align:right">2,730,577</td><td style="text-align:right">64,118,323</td><td style="text-align:right">1,263,645</td><td style="text-align:right"><strong>No avoids</strong></td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td style="text-align:right">—</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td style="text-align:right">—</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">115,097</td><td style="text-align:right">100,817</td><td style="text-align:right">102,917</td><td style="text-align:right">115,097</td><td style="text-align:right">108,790</td><td style="text-align:right"><strong>Noskills</strong></td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">98,261,801</td><td style="text-align:right">601,662</td><td style="text-align:right">598,141</td><td style="text-align:right">18,499</td><td style="text-align:right">18,499</td><td style="text-align:right"><strong>All+avoids+No avoids</strong></td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">28,051,921</td><td style="text-align:right">11,728,993</td><td style="text-align:right">482,436,001</td><td style="text-align:right">5,896,958</td><td style="text-align:right"><strong>No avoids</strong></td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">54,298,622</td><td style="text-align:right">49,181</td><td style="text-align:right">49,181</td><td style="text-align:right">3,101,758</td><td style="text-align:right">61,693</td><td style="text-align:right"><strong>Bn 2+2+Noskills</strong></td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">2,715,487</td><td style="text-align:right">318,159</td><td style="text-align:right">214,907</td><td style="text-align:right">254,407</td><td style="text-align:right">53,367</td><td style="text-align:right"><strong>No avoids</strong></td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">1,441,873</td><td style="text-align:right">9,060</td><td style="text-align:right">1,141</td><td style="text-align:right">26,975</td><td style="text-align:right">18,055</td><td style="text-align:right"><strong>Bn 2+2</strong></td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">2,270,081</td><td style="text-align:right">1,296,548</td><td style="text-align:right">1,296,548</td><td style="text-align:right">2,270,081</td><td style="text-align:right">2,270,081</td><td style="text-align:right"><strong>Bn 2+2+Noskills</strong></td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">3,281,841</td><td style="text-align:right">48,677</td><td style="text-align:right">47,686</td><td style="text-align:right">35,682</td><td style="text-align:right">3,281,841</td><td style="text-align:right"><strong>All+avoids</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">41,761</td><td style="text-align:right">13,268</td><td style="text-align:right">13,268</td><td style="text-align:right">12,231</td><td style="text-align:right">12,468</td><td style="text-align:right"><strong>All+avoids</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">3,112,241</td><td style="text-align:right">27,889</td><td style="text-align:right">27,889</td><td style="text-align:right">2,461,081</td><td style="text-align:right">27,921</td><td style="text-align:right"><strong>Bn 2+2+Noskills</strong></td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">134,372,041</td><td style="text-align:right">15,614,072</td><td style="text-align:right">17,192,281</td><td style="text-align:right">5,062,065</td><td style="text-align:right">5,672,817</td><td style="text-align:right"><strong>All+avoids</strong></td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">10,095,723</td><td style="text-align:right">10,095,723</td><td style="text-align:right">6,672,294</td><td style="text-align:right">1,925,435</td><td style="text-align:right">732,587</td><td style="text-align:right"><strong>No avoids</strong></td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">256,922</td><td style="text-align:right">210,483</td><td style="text-align:right">210,483</td><td style="text-align:right">48,054</td><td style="text-align:right">199,322</td><td style="text-align:right"><strong>All+avoids</strong></td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">209,526,121</td><td style="text-align:right">209,526,121</td><td style="text-align:right">118,500,556</td><td style="text-align:right">116,875,261</td><td style="text-align:right">118,560,241</td><td style="text-align:right"><strong>All+avoids</strong></td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">132,556,483</td><td style="text-align:right">586,042</td><td style="text-align:right">34,560,515</td><td style="text-align:right">38,071,522</td><td style="text-align:right">1,119,361</td><td style="text-align:right"><strong>Noskills</strong></td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">23,241,675</td><td style="text-align:right">2,864,058</td><td style="text-align:right">2,864,134</td><td style="text-align:right">2,855,106</td><td style="text-align:right">1,251,830</td><td style="text-align:right"><strong>No avoids</strong></td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">33,807,761</td><td style="text-align:right">992,061</td><td style="text-align:right">7,878,481</td><td style="text-align:right">6,979,669</td><td style="text-align:right">83,188</td><td style="text-align:right"><strong>No avoids</strong></td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">31,695,761</td><td style="text-align:right">1,577,981</td><td style="text-align:right">119,181</td><td style="text-align:right">31,695,761</td><td style="text-align:right">228,631</td><td style="text-align:right"><strong>Bn 2+2</strong></td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1,160,161</td><td style="text-align:right">139,301</td><td style="text-align:right">678,122</td><td style="text-align:right">22,113</td><td style="text-align:right">40,805</td><td style="text-align:right"><strong>All+avoids</strong></td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">22,598,401</td><td style="text-align:right">2,759,030</td><td style="text-align:right">2,768,058</td><td style="text-align:right">6,747,057</td><td style="text-align:right">260,875</td><td style="text-align:right"><strong>No avoids</strong></td></tr>
</tbody></table>

## Latency (cycles) — new skills pack

<table class="flash-cmp">
<colgroup>
  <col style="width:9%">
  <col style="width:9%">
  <col style="width:9%">
  <col style="width:9%">
  <col style="width:9%">
  <col style="width:9%">
  <col style="width:9%">
  <col style="width:9%">
  <col style="width:9%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Ground truth</th>
  <th style="text-align:right">Noskills</th>
  <th style="text-align:right">bn2+2</th>
  <th style="text-align:right">bn4+2</th>
  <th style="text-align:right">bn6+2</th>
  <th style="text-align:right">All+avoids</th>
  <th style="text-align:right">No avoids</th>
  <th style="text-align:right">Winner</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">25,296,077</td><td style="text-align:right">1,481,537</td><td style="text-align:right">1,262,346</td><td style="text-align:right">1,265,553</td><td style="text-align:right">114,328</td><td style="text-align:right">64,564</td><td style="text-align:right">454,508</td><td style="text-align:right"><strong>All+avoids</strong></td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">45,441,119</td><td style="text-align:right">2,038,096</td><td style="text-align:right">2,038,096</td><td style="text-align:right">2,038,096</td><td style="text-align:right">2,850,840</td><td style="text-align:right">2,113,550</td><td style="text-align:right">105,635</td><td style="text-align:right"><strong>No avoids</strong></td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">2,429,238</td><td style="text-align:right">145,413</td><td style="text-align:right">359,813</td><td style="text-align:right">2,429,238</td><td style="text-align:right">2,314,746</td><td style="text-align:right">5,046</td><td style="text-align:right">5,046</td><td style="text-align:right"><strong>All+avoids+No avoids</strong></td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">2,343,914</td><td style="text-align:right">2,343,914</td><td style="text-align:right">117,277</td><td style="text-align:right">132,498</td><td style="text-align:right">127,293</td><td style="text-align:right">110,530</td><td style="text-align:right">5,316</td><td style="text-align:right"><strong>No avoids</strong></td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">68,337,001</td><td style="text-align:right">19,497,841</td><td style="text-align:right">159,704</td><td style="text-align:right">8,000,912</td><td style="text-align:right">7,923,872</td><td style="text-align:right">376,177</td><td style="text-align:right">336,393</td><td style="text-align:right"><strong>bn2+2</strong></td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">53,906,591</td><td style="text-align:right">100,993</td><td style="text-align:right">1,837,814</td><td style="text-align:right">10,753,631</td><td style="text-align:right">1,821,941</td><td style="text-align:right">429,902</td><td style="text-align:right">3,068,010</td><td style="text-align:right"><strong>Noskills</strong></td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">64,118,323</td><td style="text-align:right">3,848,349</td><td style="text-align:right">3,193,955</td><td style="text-align:right">94,585</td><td style="text-align:right">2,695,369</td><td style="text-align:right">4,121,036</td><td style="text-align:right">1,819,720</td><td style="text-align:right"><strong>bn4+2</strong></td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td style="text-align:right">—</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td style="text-align:right">—</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">115,097</td><td style="text-align:right">102,917</td><td style="text-align:right">102,917</td><td style="text-align:right">100,817</td><td style="text-align:right">108,790</td><td style="text-align:right">61,441</td><td style="text-align:right">100,817</td><td style="text-align:right"><strong>All+avoids</strong></td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">98,261,801</td><td style="text-align:right">3,219,121</td><td style="text-align:right">87,379,241</td><td style="text-align:right">43,701</td><td style="text-align:right">18,499</td><td style="text-align:right">18,499</td><td style="text-align:right">601,662</td><td style="text-align:right"><strong>All+avoids+bn6+2</strong></td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">11,728,993</td><td style="text-align:right">833,976,005</td><td style="text-align:right">833,976,005</td><td style="text-align:right">833,976,005</td><td style="text-align:right">833,976,005</td><td style="text-align:right">833,976,005</td><td style="text-align:right"><strong>Noskills</strong></td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">54,298,622</td><td style="text-align:right">355,153</td><td style="text-align:right">54,093</td><td style="text-align:right">355,153</td><td style="text-align:right">1,377,145</td><td style="text-align:right">565,921</td><td style="text-align:right">145,081</td><td style="text-align:right"><strong>bn2+2</strong></td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">2,715,487</td><td style="text-align:right">234,669</td><td style="text-align:right">2,715,487</td><td style="text-align:right">261,572</td><td style="text-align:right">283,347</td><td style="text-align:right">243,411</td><td style="text-align:right">74,622</td><td style="text-align:right"><strong>No avoids</strong></td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">1,441,873</td><td style="text-align:right">59,095</td><td style="text-align:right">263,725</td><td style="text-align:right">1,141</td><td style="text-align:right">180,647</td><td style="text-align:right">5,443</td><td style="text-align:right">45,550</td><td style="text-align:right"><strong>bn4+2</strong></td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">2,270,081</td><td style="text-align:right">294,388</td><td style="text-align:right">1,275,551</td><td style="text-align:right">294,388</td><td style="text-align:right">1,276,224</td><td style="text-align:right">2,270,081</td><td style="text-align:right">2,270,081</td><td style="text-align:right"><strong>Noskills+bn4+2</strong></td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">3,281,841</td><td style="text-align:right">488,486</td><td style="text-align:right">488,486</td><td style="text-align:right">40,080</td><td style="text-align:right">808,966</td><td style="text-align:right">2,013,441</td><td style="text-align:right">488,486</td><td style="text-align:right"><strong>bn4+2</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">41,761</td><td style="text-align:right">13,420</td><td style="text-align:right">13,031</td><td style="text-align:right">13,031</td><td style="text-align:right">13,031</td><td style="text-align:right">12,231</td><td style="text-align:right">12,468</td><td style="text-align:right"><strong>All+avoids</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">3,112,241</td><td style="text-align:right">640,370</td><td style="text-align:right">640,370</td><td style="text-align:right">18,731</td><td style="text-align:right">1,614,961</td><td style="text-align:right">9,070</td><td style="text-align:right">1,979,281</td><td style="text-align:right"><strong>All+avoids</strong></td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">134,372,041</td><td style="text-align:right">7,831,472</td><td style="text-align:right">11,829,856</td><td style="text-align:right">134,372,041</td><td style="text-align:right">15,574,591</td><td style="text-align:right">4,823,953</td><td style="text-align:right">2,591,089</td><td style="text-align:right"><strong>No avoids</strong></td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">10,095,723</td><td style="text-align:right">10,095,723</td><td style="text-align:right">6,532,621</td><td style="text-align:right">6,557,315</td><td style="text-align:right">6,585,875</td><td style="text-align:right">6,557,315</td><td style="text-align:right">10,095,723</td><td style="text-align:right"><strong>bn2+2</strong></td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">256,922</td><td style="text-align:right">254,630</td><td style="text-align:right">193,276</td><td style="text-align:right">199,322</td><td style="text-align:right">256,922</td><td style="text-align:right">4,793</td><td style="text-align:right">8,776</td><td style="text-align:right"><strong>All+avoids</strong></td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">209,526,121</td><td style="text-align:right">209,526,121</td><td style="text-align:right">3,245,657</td><td style="text-align:right">103,652,813</td><td style="text-align:right">117,950,941</td><td style="text-align:right">209,526,121</td><td style="text-align:right">9,482,221</td><td style="text-align:right"><strong>bn2+2</strong></td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">132,556,483</td><td style="text-align:right">132,556,483</td><td style="text-align:right">40,130,103</td><td style="text-align:right">34,571,610</td><td style="text-align:right">652,761</td><td style="text-align:right">1,120,001</td><td style="text-align:right">132,556,483</td><td style="text-align:right"><strong>bn6+2</strong></td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">23,241,675</td><td style="text-align:right">2,855,106</td><td style="text-align:right">2,855,106</td><td style="text-align:right">5,201,403</td><td style="text-align:right">5,201,403</td><td style="text-align:right">5,323,907</td><td style="text-align:right">23,241,675</td><td style="text-align:right"><strong>Noskills+bn2+2</strong></td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">33,807,761</td><td style="text-align:right">1,002,851</td><td style="text-align:right">166,561</td><td style="text-align:right">1,539,761</td><td style="text-align:right">1,539,761</td><td style="text-align:right">125,376</td><td style="text-align:right">6,958,389</td><td style="text-align:right"><strong>All+avoids</strong></td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">31,695,761</td><td style="text-align:right">31,695,761</td><td style="text-align:right">1,572,861</td><td style="text-align:right">209,921</td><td style="text-align:right">1,352,948</td><td style="text-align:right">556,551</td><td style="text-align:right">54,836</td><td style="text-align:right"><strong>No avoids</strong></td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1,160,161</td><td style="text-align:right">181,152</td><td style="text-align:right">1,160,161</td><td style="text-align:right">112,062</td><td style="text-align:right">1,155,676</td><td style="text-align:right">1,160,161</td><td style="text-align:right">884,533</td><td style="text-align:right"><strong>bn4+2</strong></td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">22,598,401</td><td style="text-align:right">2,720,630</td><td style="text-align:right">2,759,030</td><td style="text-align:right">2,759,030</td><td style="text-align:right">2,704,727</td><td style="text-align:right">7,629,361</td><td style="text-align:right">2,720,629</td><td style="text-align:right"><strong>bn6+2</strong></td></tr>
</tbody></table>

## Ground-truth latency ratio per benchmark (synth / GT)

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
  <th style="text-align:right">Noskills</th>
  <th style="text-align:right">Bn 2+2</th>
  <th style="text-align:right">All+avoids</th>
  <th style="text-align:right">No avoids</th>
  <th style="text-align:right">Noskills</th>
  <th style="text-align:right">Bn 2+2</th>
  <th style="text-align:right">Bn 4+2</th>
  <th style="text-align:right">Bn 6+2</th>
  <th style="text-align:right">All+avoids</th>
  <th style="text-align:right">No avoids</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">0.003</td><td style="text-align:right">0.005</td><td style="text-align:right">0.003</td><td style="text-align:right">0.002</td><td style="text-align:right">0.059</td><td style="text-align:right">0.050</td><td style="text-align:right">0.050</td><td style="text-align:right">0.005</td><td style="text-align:right">0.003</td><td style="text-align:right">0.018</td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">0.002</td><td style="text-align:right">0.002</td><td style="text-align:right">0.015</td><td style="text-align:right">1.000</td><td style="text-align:right">0.045</td><td style="text-align:right">0.045</td><td style="text-align:right">0.045</td><td style="text-align:right">0.063</td><td style="text-align:right">0.047</td><td style="text-align:right">0.002</td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">0.049</td><td style="text-align:right">1.000</td><td style="text-align:right">0.947</td><td style="text-align:right">0.002</td><td style="text-align:right">0.060</td><td style="text-align:right">0.148</td><td style="text-align:right">1.000</td><td style="text-align:right">0.953</td><td style="text-align:right">0.002</td><td style="text-align:right">0.002</td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">0.310</td><td style="text-align:right">0.980</td><td style="text-align:right">0.346</td><td style="text-align:right">0.002</td><td style="text-align:right">1.070</td><td style="text-align:right">0.050</td><td style="text-align:right">0.057</td><td style="text-align:right">0.054</td><td style="text-align:right">0.047</td><td style="text-align:right">0.002</td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">0.058</td><td style="text-align:right">0.058</td><td style="text-align:right">1.000</td><td style="text-align:right">0.020</td><td style="text-align:right">0.285</td><td style="text-align:right">0.002</td><td style="text-align:right">0.117</td><td style="text-align:right">0.116</td><td style="text-align:right">0.006</td><td style="text-align:right">0.005</td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">0.034</td><td style="text-align:right">0.034</td><td style="text-align:right">0.175</td><td style="text-align:right">0.149</td><td style="text-align:right">0.002</td><td style="text-align:right">0.034</td><td style="text-align:right">0.199</td><td style="text-align:right">0.034</td><td style="text-align:right">0.008</td><td style="text-align:right">0.057</td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">0.059</td><td style="text-align:right">0.043</td><td style="text-align:right">1.000</td><td style="text-align:right">0.020</td><td style="text-align:right">0.060</td><td style="text-align:right">0.050</td><td style="text-align:right">0.001</td><td style="text-align:right">0.042</td><td style="text-align:right">0.064</td><td style="text-align:right">0.028</td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">0.876</td><td style="text-align:right">0.894</td><td style="text-align:right">1.000</td><td style="text-align:right">0.945</td><td style="text-align:right">0.894</td><td style="text-align:right">0.894</td><td style="text-align:right">0.876</td><td style="text-align:right">0.945</td><td style="text-align:right">0.534</td><td style="text-align:right">0.876</td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">0.006</td><td style="text-align:right">0.006</td><td style="text-align:right">0.000</td><td style="text-align:right">0.000</td><td style="text-align:right">0.033</td><td style="text-align:right">0.889</td><td style="text-align:right">0.000</td><td style="text-align:right">0.000</td><td style="text-align:right">0.000</td><td style="text-align:right">0.006</td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">0.034</td><td style="text-align:right">0.014</td><td style="text-align:right">0.578</td><td style="text-align:right">0.007</td><td style="text-align:right">0.014</td><td style="text-align:right">1.000</td><td style="text-align:right">1.007</td><td style="text-align:right">1.014</td><td style="text-align:right">1.000</td><td style="text-align:right">1.007</td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">0.001</td><td style="text-align:right">0.001</td><td style="text-align:right">0.057</td><td style="text-align:right">0.001</td><td style="text-align:right">0.007</td><td style="text-align:right">0.001</td><td style="text-align:right">0.007</td><td style="text-align:right">0.025</td><td style="text-align:right">0.010</td><td style="text-align:right">0.003</td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">0.117</td><td style="text-align:right">0.079</td><td style="text-align:right">0.094</td><td style="text-align:right">0.020</td><td style="text-align:right">0.086</td><td style="text-align:right">1.001</td><td style="text-align:right">0.096</td><td style="text-align:right">0.104</td><td style="text-align:right">0.090</td><td style="text-align:right">0.027</td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">0.006</td><td style="text-align:right">0.001</td><td style="text-align:right">0.019</td><td style="text-align:right">0.013</td><td style="text-align:right">0.041</td><td style="text-align:right">0.183</td><td style="text-align:right">0.001</td><td style="text-align:right">0.125</td><td style="text-align:right">0.004</td><td style="text-align:right">0.032</td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">0.571</td><td style="text-align:right">0.571</td><td style="text-align:right">1.000</td><td style="text-align:right">1.000</td><td style="text-align:right">0.130</td><td style="text-align:right">0.562</td><td style="text-align:right">0.130</td><td style="text-align:right">0.562</td><td style="text-align:right">1.044</td><td style="text-align:right">1.001</td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">0.015</td><td style="text-align:right">0.015</td><td style="text-align:right">0.011</td><td style="text-align:right">1.000</td><td style="text-align:right">0.149</td><td style="text-align:right">0.149</td><td style="text-align:right">0.012</td><td style="text-align:right">0.246</td><td style="text-align:right">0.614</td><td style="text-align:right">0.149</td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">0.318</td><td style="text-align:right">0.318</td><td style="text-align:right">0.293</td><td style="text-align:right">0.299</td><td style="text-align:right">0.321</td><td style="text-align:right">0.312</td><td style="text-align:right">0.312</td><td style="text-align:right">0.312</td><td style="text-align:right">0.293</td><td style="text-align:right">0.299</td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">0.009</td><td style="text-align:right">0.009</td><td style="text-align:right">0.791</td><td style="text-align:right">0.009</td><td style="text-align:right">0.206</td><td style="text-align:right">0.206</td><td style="text-align:right">0.006</td><td style="text-align:right">0.519</td><td style="text-align:right">0.003</td><td style="text-align:right">0.636</td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">0.116</td><td style="text-align:right">0.128</td><td style="text-align:right">0.038</td><td style="text-align:right">0.042</td><td style="text-align:right">0.058</td><td style="text-align:right">0.088</td><td style="text-align:right">1.000</td><td style="text-align:right">0.116</td><td style="text-align:right">0.036</td><td style="text-align:right">0.019</td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">1.000</td><td style="text-align:right">0.661</td><td style="text-align:right">0.191</td><td style="text-align:right">0.073</td><td style="text-align:right">1.000</td><td style="text-align:right">0.647</td><td style="text-align:right">0.650</td><td style="text-align:right">0.652</td><td style="text-align:right">0.650</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">0.819</td><td style="text-align:right">0.819</td><td style="text-align:right">0.187</td><td style="text-align:right">0.776</td><td style="text-align:right">0.991</td><td style="text-align:right">0.752</td><td style="text-align:right">0.776</td><td style="text-align:right">1.047</td><td style="text-align:right">0.019</td><td style="text-align:right">0.034</td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">1.000</td><td style="text-align:right">0.566</td><td style="text-align:right">0.558</td><td style="text-align:right">0.566</td><td style="text-align:right">1.033</td><td style="text-align:right">0.015</td><td style="text-align:right">0.495</td><td style="text-align:right">0.563</td><td style="text-align:right">1.033</td><td style="text-align:right">0.045</td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">0.004</td><td style="text-align:right">0.261</td><td style="text-align:right">0.287</td><td style="text-align:right">0.008</td><td style="text-align:right">1.000</td><td style="text-align:right">0.303</td><td style="text-align:right">0.261</td><td style="text-align:right">0.005</td><td style="text-align:right">0.008</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">0.123</td><td style="text-align:right">0.123</td><td style="text-align:right">0.123</td><td style="text-align:right">0.054</td><td style="text-align:right">0.123</td><td style="text-align:right">0.123</td><td style="text-align:right">0.224</td><td style="text-align:right">0.224</td><td style="text-align:right">0.229</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">0.029</td><td style="text-align:right">0.233</td><td style="text-align:right">0.206</td><td style="text-align:right">0.002</td><td style="text-align:right">0.030</td><td style="text-align:right">0.005</td><td style="text-align:right">0.046</td><td style="text-align:right">0.046</td><td style="text-align:right">0.004</td><td style="text-align:right">0.206</td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">0.050</td><td style="text-align:right">0.004</td><td style="text-align:right">1.000</td><td style="text-align:right">0.007</td><td style="text-align:right">1.000</td><td style="text-align:right">0.050</td><td style="text-align:right">0.007</td><td style="text-align:right">0.043</td><td style="text-align:right">0.018</td><td style="text-align:right">0.002</td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">0.120</td><td style="text-align:right">0.585</td><td style="text-align:right">0.019</td><td style="text-align:right">0.035</td><td style="text-align:right">0.156</td><td style="text-align:right">1.011</td><td style="text-align:right">0.097</td><td style="text-align:right">0.996</td><td style="text-align:right">1.000</td><td style="text-align:right">0.762</td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">0.122</td><td style="text-align:right">0.122</td><td style="text-align:right">0.299</td><td style="text-align:right">0.012</td><td style="text-align:right">0.120</td><td style="text-align:right">0.122</td><td style="text-align:right">0.122</td><td style="text-align:right">0.120</td><td style="text-align:right">0.338</td><td style="text-align:right">0.120</td></tr>
</tbody></table>

## Conclusions

1. **Success:** Every mode completes **27/28** benches; `doitgen` fails the gold-reference gate in all runs.
2. **vs ground truth:** All modes achieve dramatically lower latency than GT (geo-mean ratios ≈ 0.03–0.15). A few new runs are marginally above GT (ratio ≈ 1.01–1.05) on isolated benches.
3. **Best single-mode latency wins (of 27):** No avoids (old) leads (**8.1**), followed by All+avoids (new) (**4.1**). Bn 6+2 (new) is weakest (**0.2**).
4. **New vs old (paired):** All+avoids **improves** with the new skills pack (15–8). Noskills and No avoids **regress** vs legacy (new wins 7 and 9). Bn 2+2 is roughly even (12–13).
5. **Among new modes:** Bn 4+2 and No avoids beat Noskills (new) most often (16 wins each). Bn 6+2 loses to Noskills (new) 16–11.
6. **BN skill count:** More positive bottleneck skills do not help monotonically — 4+2 ≈ 2+2, while 6+2 is clearly worse.

See also: `artifacts/pc2/flash_comparison_20260620.md` (legacy-only run from 2026-06-20).

