# Flash HLSFactory — Round 1 vs Round 2

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
<thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>
<tr><td>Round 1 legacy stamp (noskills / bn)</td><td><code>20260620_004507</code></td></tr>
<tr><td>Round 1 legacy stamp (global)</td><td><code>20260620_113247</code></td></tr>
<tr><td>Round 1 new skills stamp</td><td><code>20260621_020847</code></td></tr>
<tr><td>Round 2 stamp (all 10 modes)</td><td><code>20260621_075846</code></td></tr>
<tr><td>Metric</td><td>Final flash-step synthesis latency (cycles), lower is better</td></tr>
<tr><td>Success</td><td>27/28 per mode per round (<code>doitgen</code> fails gold-ref gate)</td></tr>
<tr><td>Round 2 settings</td><td>watch 60s, compute walltime 12h, auto-stop on success</td></tr>
</tbody></table>

## Summary — paired modes (Round 2 vs Round 1)

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
  <col style="width:9%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Mode</th>
  <th style="text-align:left">Round 1 artifact</th>
  <th style="text-align:left">Round 2 artifact</th>
  <th style="text-align:right">OK</th>
  <th style="text-align:right">R1 geo/GT</th>
  <th style="text-align:right">R2 geo/GT</th>
  <th style="text-align:right">R2 wins</th>
  <th style="text-align:right">R1 wins</th>
  <th style="text-align:right">Ties</th>
  <th style="text-align:right">&gt;50% Δ</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">Noskills (legacy)</td><td style="text-align:left"><code>flash_noskills_20260620_004507</code></td><td style="text-align:left"><code>flash_noskills_20260621_075846</code></td><td style="text-align:right">27/27</td><td style="text-align:right">0.0528</td><td style="text-align:right">0.0667</td><td style="text-align:right">14</td><td style="text-align:right">10</td><td style="text-align:right">3</td><td style="text-align:right">12</td></tr>
<tr><td style="text-align:left">Bn 2+2 (legacy)</td><td style="text-align:left"><code>flash_skills_20260620_004507</code></td><td style="text-align:left"><code>flash_skills_20260621_075846</code></td><td style="text-align:right">27/27</td><td style="text-align:right">0.0653</td><td style="text-align:right">0.1348</td><td style="text-align:right">8</td><td style="text-align:right">16</td><td style="text-align:right">3</td><td style="text-align:right">17</td></tr>
<tr><td style="text-align:left">All+avoids (legacy)</td><td style="text-align:left"><code>flash_all_skills_avoids_global_20260620_113247</code></td><td style="text-align:left"><code>flash_all_skills_avoids_global_20260621_075846</code></td><td style="text-align:right">27/27</td><td style="text-align:right">0.1714</td><td style="text-align:right">0.0620</td><td style="text-align:right">18</td><td style="text-align:right">6</td><td style="text-align:right">3</td><td style="text-align:right">19</td></tr>
<tr><td style="text-align:left">No avoids (legacy)</td><td style="text-align:left"><code>flash_all_skills_no_avoids_global_20260620_113247</code></td><td style="text-align:left"><code>flash_all_skills_no_avoids_global_20260621_075846</code></td><td style="text-align:right">27/27</td><td style="text-align:right">0.0342</td><td style="text-align:right">0.0815</td><td style="text-align:right">6</td><td style="text-align:right">18</td><td style="text-align:right">3</td><td style="text-align:right">18</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>flash_noskills_new_20260621_020847</code></td><td style="text-align:left"><code>flash_noskills_new_20260621_075846</code></td><td style="text-align:right">27/27</td><td style="text-align:right">0.1237</td><td style="text-align:right">0.0712</td><td style="text-align:right">16</td><td style="text-align:right">10</td><td style="text-align:right">1</td><td style="text-align:right">18</td></tr>
<tr><td style="text-align:left">Bn 2+2 (new)</td><td style="text-align:left"><code>flash_bn_skills_new_2_2_20260621_020847</code></td><td style="text-align:left"><code>flash_bn_skills_new_2_2_20260621_075846</code></td><td style="text-align:right">27/27</td><td style="text-align:right">0.1138</td><td style="text-align:right">0.0843</td><td style="text-align:right">10</td><td style="text-align:right">9</td><td style="text-align:right">8</td><td style="text-align:right">13</td></tr>
<tr><td style="text-align:left">Bn 4+2 (new)</td><td style="text-align:left"><code>flash_bn_skills_new_4_2_20260621_020847</code></td><td style="text-align:left"><code>flash_bn_skills_new_4_2_20260621_075846</code></td><td style="text-align:right">27/27</td><td style="text-align:right">0.0872</td><td style="text-align:right">0.0767</td><td style="text-align:right">10</td><td style="text-align:right">11</td><td style="text-align:right">6</td><td style="text-align:right">10</td></tr>
<tr><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:left"><code>flash_bn_skills_new_6_2_20260621_020847</code></td><td style="text-align:left"><code>flash_bn_skills_new_6_2_20260621_075846</code></td><td style="text-align:right">27/27</td><td style="text-align:right">0.1469</td><td style="text-align:right">0.1199</td><td style="text-align:right">15</td><td style="text-align:right">6</td><td style="text-align:right">6</td><td style="text-align:right">13</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>flash_all_new_skills_avoids_global_20260621_020847</code></td><td style="text-align:left"><code>flash_all_new_skills_avoids_global_20260621_075846</code></td><td style="text-align:right">27/27</td><td style="text-align:right">0.0536</td><td style="text-align:right">0.0517</td><td style="text-align:right">12</td><td style="text-align:right">13</td><td style="text-align:right">2</td><td style="text-align:right">21</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:left"><code>flash_all_new_skills_no_avoids_global_20260621_020847</code></td><td style="text-align:left"><code>flash_all_new_skills_no_avoids_global_20260621_075846</code></td><td style="text-align:right">27/27</td><td style="text-align:right">0.0584</td><td style="text-align:right">0.0474</td><td style="text-align:right">13</td><td style="text-align:right">11</td><td style="text-align:right">3</td><td style="text-align:right">15</td></tr>
<tr><td style="text-align:left"><strong>Total</strong></td><td colspan="5"></td><td style="text-align:right"><strong>122</strong></td><td style="text-align:right"><strong>110</strong></td><td style="text-align:right"><strong>38</strong></td><td style="text-align:right"><strong>156</strong></td></tr>
</tbody></table>

## Latency stability (same mode, two runs)

<table class="flash-cmp">
<colgroup>
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Mode</th>
  <th style="text-align:right">Within 1%</th>
  <th style="text-align:right">Within 50%</th>
  <th style="text-align:right">&gt;50% apart</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">Noskills (legacy)</td><td style="text-align:right">6/27</td><td style="text-align:right">9/27</td><td style="text-align:right">12/27</td></tr>
<tr><td style="text-align:left">Bn 2+2 (legacy)</td><td style="text-align:right">5/27</td><td style="text-align:right">5/27</td><td style="text-align:right">17/27</td></tr>
<tr><td style="text-align:left">All+avoids (legacy)</td><td style="text-align:right">3/27</td><td style="text-align:right">5/27</td><td style="text-align:right">19/27</td></tr>
<tr><td style="text-align:left">No avoids (legacy)</td><td style="text-align:right">3/27</td><td style="text-align:right">6/27</td><td style="text-align:right">18/27</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:right">2/27</td><td style="text-align:right">7/27</td><td style="text-align:right">18/27</td></tr>
<tr><td style="text-align:left">Bn 2+2 (new)</td><td style="text-align:right">9/27</td><td style="text-align:right">5/27</td><td style="text-align:right">13/27</td></tr>
<tr><td style="text-align:left">Bn 4+2 (new)</td><td style="text-align:right">8/27</td><td style="text-align:right">9/27</td><td style="text-align:right">10/27</td></tr>
<tr><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:right">7/27</td><td style="text-align:right">7/27</td><td style="text-align:right">13/27</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:right">2/27</td><td style="text-align:right">4/27</td><td style="text-align:right">21/27</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:right">3/27</td><td style="text-align:right">9/27</td><td style="text-align:right">15/27</td></tr>
</tbody></table>

## Noskills (legacy) — latency (cycles)

<table class="flash-cmp">
<colgroup>
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Ground truth</th>
  <th style="text-align:right">Round 1</th>
  <th style="text-align:right">Round 2</th>
  <th style="text-align:right">Δ R2 vs R1</th>
  <th style="text-align:right">Better</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">25,296,077</td><td style="text-align:right">64,530</td><td style="text-align:right">1,481,537</td><td style="text-align:right">+2195.9%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">45,441,119</td><td style="text-align:right">99,467</td><td style="text-align:right">22,748</td><td style="text-align:right">-77.1%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">2,429,238</td><td style="text-align:right">118,344</td><td style="text-align:right">2,429,238</td><td style="text-align:right">+1952.7%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">2,343,914</td><td style="text-align:right">727,515</td><td style="text-align:right">92,192</td><td style="text-align:right">-87.3%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">68,337,001</td><td style="text-align:right">3,980,432</td><td style="text-align:right">3,980,432</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">53,906,591</td><td style="text-align:right">1,821,943</td><td style="text-align:right">1,837,814</td><td style="text-align:right">+0.9%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">64,118,323</td><td style="text-align:right">3,786,109</td><td style="text-align:right">1,859,454</td><td style="text-align:right">-50.9%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td style="text-align:right">—</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td style="text-align:right">—</td><td style="text-align:right">—</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">115,097</td><td style="text-align:right">100,817</td><td style="text-align:right">61,583</td><td style="text-align:right">-38.9%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">98,261,801</td><td style="text-align:right">601,662</td><td style="text-align:right">601,662</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">28,051,921</td><td style="text-align:right">17,496,145</td><td style="text-align:right">-37.6%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">54,298,622</td><td style="text-align:right">49,181</td><td style="text-align:right">54,093</td><td style="text-align:right">+10.0%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">2,715,487</td><td style="text-align:right">318,159</td><td style="text-align:right">261,323</td><td style="text-align:right">-17.9%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">1,441,873</td><td style="text-align:right">9,060</td><td style="text-align:right">1,394</td><td style="text-align:right">-84.6%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">2,270,081</td><td style="text-align:right">1,296,548</td><td style="text-align:right">294,388</td><td style="text-align:right">-77.3%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">3,281,841</td><td style="text-align:right">48,677</td><td style="text-align:right">48,078</td><td style="text-align:right">-1.2%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">41,761</td><td style="text-align:right">13,268</td><td style="text-align:right">13,031</td><td style="text-align:right">-1.8%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">3,112,241</td><td style="text-align:right">27,889</td><td style="text-align:right">972,921</td><td style="text-align:right">+3388.5%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">134,372,041</td><td style="text-align:right">15,614,072</td><td style="text-align:right">15,757,441</td><td style="text-align:right">+0.9%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">10,095,723</td><td style="text-align:right">10,095,723</td><td style="text-align:right">6,643,734</td><td style="text-align:right">-34.2%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">256,922</td><td style="text-align:right">210,483</td><td style="text-align:right">256,922</td><td style="text-align:right">+22.1%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">209,526,121</td><td style="text-align:right">209,526,121</td><td style="text-align:right">209,526,121</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">132,556,483</td><td style="text-align:right">586,042</td><td style="text-align:right">55,195,677</td><td style="text-align:right">+9318.4%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">23,241,675</td><td style="text-align:right">2,864,058</td><td style="text-align:right">2,850,230</td><td style="text-align:right">-0.5%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">33,807,761</td><td style="text-align:right">992,061</td><td style="text-align:right">1,572,501</td><td style="text-align:right">+58.5%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">31,695,761</td><td style="text-align:right">1,577,981</td><td style="text-align:right">569,371</td><td style="text-align:right">-63.9%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1,160,161</td><td style="text-align:right">139,301</td><td style="text-align:right">674,142</td><td style="text-align:right">+383.9%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">22,598,401</td><td style="text-align:right">2,759,030</td><td style="text-align:right">2,720,630</td><td style="text-align:right">-1.4%</td><td style="text-align:right"><strong>R2</strong></td></tr>
</tbody></table>

### Noskills (legacy) — ground-truth latency ratio (synth / GT)

<table class="flash-cmp">
<colgroup>
  <col style="width:30%">
  <col style="width:30%">
  <col style="width:30%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Round 1</th>
  <th style="text-align:right">Round 2</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">0.003</td><td style="text-align:right">0.059</td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">0.002</td><td style="text-align:right">0.001</td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">0.049</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">0.310</td><td style="text-align:right">0.039</td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">0.058</td><td style="text-align:right">0.058</td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">0.034</td><td style="text-align:right">0.034</td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">0.059</td><td style="text-align:right">0.029</td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">0.876</td><td style="text-align:right">0.535</td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">0.006</td><td style="text-align:right">0.006</td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">0.034</td><td style="text-align:right">0.021</td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">0.001</td><td style="text-align:right">0.001</td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">0.117</td><td style="text-align:right">0.096</td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">0.006</td><td style="text-align:right">0.001</td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">0.571</td><td style="text-align:right">0.130</td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">0.015</td><td style="text-align:right">0.015</td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">0.318</td><td style="text-align:right">0.312</td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">0.009</td><td style="text-align:right">0.313</td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">0.116</td><td style="text-align:right">0.117</td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">1.000</td><td style="text-align:right">0.658</td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">0.819</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">1.000</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">0.004</td><td style="text-align:right">0.416</td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">0.123</td><td style="text-align:right">0.123</td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">0.029</td><td style="text-align:right">0.047</td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">0.050</td><td style="text-align:right">0.018</td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">0.120</td><td style="text-align:right">0.581</td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">0.122</td><td style="text-align:right">0.120</td></tr>
</tbody></table>

## Bn 2+2 (legacy) — latency (cycles)

<table class="flash-cmp">
<colgroup>
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Ground truth</th>
  <th style="text-align:right">Round 1</th>
  <th style="text-align:right">Round 2</th>
  <th style="text-align:right">Δ R2 vs R1</th>
  <th style="text-align:right">Better</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">25,296,077</td><td style="text-align:right">123,785</td><td style="text-align:right">1,262,346</td><td style="text-align:right">+919.8%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">45,441,119</td><td style="text-align:right">99,467</td><td style="text-align:right">2,047,248</td><td style="text-align:right">+1958.2%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">2,429,238</td><td style="text-align:right">2,429,238</td><td style="text-align:right">2,429,238</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">2,343,914</td><td style="text-align:right">2,297,693</td><td style="text-align:right">191,293</td><td style="text-align:right">-91.7%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">68,337,001</td><td style="text-align:right">3,988,492</td><td style="text-align:right">68,265,001</td><td style="text-align:right">+1611.5%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">53,906,591</td><td style="text-align:right">1,837,814</td><td style="text-align:right">10,753,631</td><td style="text-align:right">+485.1%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">64,118,323</td><td style="text-align:right">2,730,577</td><td style="text-align:right">3,848,196</td><td style="text-align:right">+40.9%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td style="text-align:right">—</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td style="text-align:right">—</td><td style="text-align:right">—</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">115,097</td><td style="text-align:right">102,917</td><td style="text-align:right">102,084</td><td style="text-align:right">-0.8%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">98,261,801</td><td style="text-align:right">598,141</td><td style="text-align:right">4,530,321</td><td style="text-align:right">+657.4%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">11,728,993</td><td style="text-align:right">833,976,005</td><td style="text-align:right">+7010.4%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">54,298,622</td><td style="text-align:right">49,181</td><td style="text-align:right">988,345</td><td style="text-align:right">+1909.6%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">2,715,487</td><td style="text-align:right">214,907</td><td style="text-align:right">226,048</td><td style="text-align:right">+5.2%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">1,441,873</td><td style="text-align:right">1,141</td><td style="text-align:right">89,000</td><td style="text-align:right">+7700.2%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">2,270,081</td><td style="text-align:right">1,296,548</td><td style="text-align:right">294,388</td><td style="text-align:right">-77.3%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">3,281,841</td><td style="text-align:right">47,686</td><td style="text-align:right">488,486</td><td style="text-align:right">+924.4%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">41,761</td><td style="text-align:right">13,268</td><td style="text-align:right">13,269</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">3,112,241</td><td style="text-align:right">27,889</td><td style="text-align:right">27,889</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">134,372,041</td><td style="text-align:right">17,192,281</td><td style="text-align:right">134,372,041</td><td style="text-align:right">+681.6%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">10,095,723</td><td style="text-align:right">6,672,294</td><td style="text-align:right">4,882,187</td><td style="text-align:right">-26.8%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">256,922</td><td style="text-align:right">210,483</td><td style="text-align:right">256,922</td><td style="text-align:right">+22.1%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">209,526,121</td><td style="text-align:right">118,500,556</td><td style="text-align:right">6,473,311</td><td style="text-align:right">-94.5%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">132,556,483</td><td style="text-align:right">34,560,515</td><td style="text-align:right">42,784,641</td><td style="text-align:right">+23.8%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">23,241,675</td><td style="text-align:right">2,864,134</td><td style="text-align:right">1,256,706</td><td style="text-align:right">-56.1%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">33,807,761</td><td style="text-align:right">7,878,481</td><td style="text-align:right">149,271</td><td style="text-align:right">-98.1%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">31,695,761</td><td style="text-align:right">119,181</td><td style="text-align:right">1,572,861</td><td style="text-align:right">+1219.7%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1,160,161</td><td style="text-align:right">678,122</td><td style="text-align:right">1,160,161</td><td style="text-align:right">+71.1%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">22,598,401</td><td style="text-align:right">2,768,058</td><td style="text-align:right">2,759,030</td><td style="text-align:right">-0.3%</td><td style="text-align:right"><strong>R2</strong></td></tr>
</tbody></table>

### Bn 2+2 (legacy) — ground-truth latency ratio (synth / GT)

<table class="flash-cmp">
<colgroup>
  <col style="width:30%">
  <col style="width:30%">
  <col style="width:30%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Round 1</th>
  <th style="text-align:right">Round 2</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">0.005</td><td style="text-align:right">0.050</td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">0.002</td><td style="text-align:right">0.045</td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">1.000</td><td style="text-align:right">1.034</td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">0.980</td><td style="text-align:right">0.082</td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">0.058</td><td style="text-align:right">0.999</td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">0.034</td><td style="text-align:right">0.199</td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">0.043</td><td style="text-align:right">0.060</td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">0.894</td><td style="text-align:right">0.887</td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">0.006</td><td style="text-align:right">0.046</td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">0.014</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">0.001</td><td style="text-align:right">0.018</td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">0.079</td><td style="text-align:right">0.083</td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">0.001</td><td style="text-align:right">0.062</td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">0.571</td><td style="text-align:right">0.130</td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">0.015</td><td style="text-align:right">0.149</td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">0.318</td><td style="text-align:right">0.318</td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">0.009</td><td style="text-align:right">0.009</td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">0.128</td><td style="text-align:right">1.006</td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">0.661</td><td style="text-align:right">0.484</td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">0.819</td><td style="text-align:right">1.028</td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">0.566</td><td style="text-align:right">0.031</td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">0.261</td><td style="text-align:right">0.323</td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">0.123</td><td style="text-align:right">0.054</td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">0.233</td><td style="text-align:right">0.004</td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">0.004</td><td style="text-align:right">0.050</td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">0.585</td><td style="text-align:right">1.007</td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">0.122</td><td style="text-align:right">0.122</td></tr>
</tbody></table>

## All+avoids (legacy) — latency (cycles)

<table class="flash-cmp">
<colgroup>
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Ground truth</th>
  <th style="text-align:right">Round 1</th>
  <th style="text-align:right">Round 2</th>
  <th style="text-align:right">Δ R2 vs R1</th>
  <th style="text-align:right">Better</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">25,296,077</td><td style="text-align:right">64,530</td><td style="text-align:right">66,847</td><td style="text-align:right">+3.6%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">45,441,119</td><td style="text-align:right">677,058</td><td style="text-align:right">45,441,119</td><td style="text-align:right">+6611.6%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">2,429,238</td><td style="text-align:right">2,300,710</td><td style="text-align:right">135,333</td><td style="text-align:right">-94.1%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">2,343,914</td><td style="text-align:right">810,190</td><td style="text-align:right">110,231</td><td style="text-align:right">-86.4%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">68,337,001</td><td style="text-align:right">68,337,001</td><td style="text-align:right">9,484,353</td><td style="text-align:right">-86.1%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">53,906,591</td><td style="text-align:right">9,438,432</td><td style="text-align:right">3,068,010</td><td style="text-align:right">-67.5%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">64,118,323</td><td style="text-align:right">64,118,323</td><td style="text-align:right">3,187,785</td><td style="text-align:right">-95.0%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td style="text-align:right">—</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td style="text-align:right">—</td><td style="text-align:right">—</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">115,097</td><td style="text-align:right">115,097</td><td style="text-align:right">115,097</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">98,261,801</td><td style="text-align:right">18,499</td><td style="text-align:right">18,499</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">482,436,001</td><td style="text-align:right">1,043,281</td><td style="text-align:right">-99.8%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">54,298,622</td><td style="text-align:right">3,101,758</td><td style="text-align:right">1,226,881</td><td style="text-align:right">-60.4%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">2,715,487</td><td style="text-align:right">254,407</td><td style="text-align:right">74,622</td><td style="text-align:right">-70.7%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">1,441,873</td><td style="text-align:right">26,975</td><td style="text-align:right">8,521</td><td style="text-align:right">-68.4%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">2,270,081</td><td style="text-align:right">2,270,081</td><td style="text-align:right">2,270,081</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">3,281,841</td><td style="text-align:right">35,682</td><td style="text-align:right">3,281,841</td><td style="text-align:right">+9097.5%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">41,761</td><td style="text-align:right">12,231</td><td style="text-align:right">12,468</td><td style="text-align:right">+1.9%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">3,112,241</td><td style="text-align:right">2,461,081</td><td style="text-align:right">27,072</td><td style="text-align:right">-98.9%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">134,372,041</td><td style="text-align:right">5,062,065</td><td style="text-align:right">4,853,617</td><td style="text-align:right">-4.1%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">10,095,723</td><td style="text-align:right">1,925,435</td><td style="text-align:right">312,199</td><td style="text-align:right">-83.8%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">256,922</td><td style="text-align:right">48,054</td><td style="text-align:right">199,322</td><td style="text-align:right">+314.8%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">209,526,121</td><td style="text-align:right">116,875,261</td><td style="text-align:right">7,221,241</td><td style="text-align:right">-93.8%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">132,556,483</td><td style="text-align:right">38,071,522</td><td style="text-align:right">23,951,361</td><td style="text-align:right">-37.1%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">23,241,675</td><td style="text-align:right">2,855,106</td><td style="text-align:right">819,521</td><td style="text-align:right">-71.3%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">33,807,761</td><td style="text-align:right">6,979,669</td><td style="text-align:right">721,711</td><td style="text-align:right">-89.7%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">31,695,761</td><td style="text-align:right">31,695,761</td><td style="text-align:right">3,734,388</td><td style="text-align:right">-88.2%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1,160,161</td><td style="text-align:right">22,113</td><td style="text-align:right">26,125</td><td style="text-align:right">+18.1%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">22,598,401</td><td style="text-align:right">6,747,057</td><td style="text-align:right">2,699,987</td><td style="text-align:right">-60.0%</td><td style="text-align:right"><strong>R2</strong></td></tr>
</tbody></table>

### All+avoids (legacy) — ground-truth latency ratio (synth / GT)

<table class="flash-cmp">
<colgroup>
  <col style="width:30%">
  <col style="width:30%">
  <col style="width:30%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Round 1</th>
  <th style="text-align:right">Round 2</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">0.003</td><td style="text-align:right">0.003</td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">0.015</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">0.947</td><td style="text-align:right">0.056</td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">0.346</td><td style="text-align:right">0.047</td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">1.000</td><td style="text-align:right">0.139</td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">0.175</td><td style="text-align:right">0.057</td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">1.000</td><td style="text-align:right">0.050</td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">1.000</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">0.000</td><td style="text-align:right">0.000</td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">0.578</td><td style="text-align:right">0.001</td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">0.057</td><td style="text-align:right">0.023</td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">0.094</td><td style="text-align:right">0.027</td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">0.019</td><td style="text-align:right">0.006</td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">1.000</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">0.011</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">0.293</td><td style="text-align:right">0.299</td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">0.791</td><td style="text-align:right">0.009</td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">0.038</td><td style="text-align:right">0.036</td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">0.191</td><td style="text-align:right">0.031</td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">0.187</td><td style="text-align:right">0.776</td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">0.558</td><td style="text-align:right">0.034</td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">0.287</td><td style="text-align:right">0.181</td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">0.123</td><td style="text-align:right">0.035</td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">0.206</td><td style="text-align:right">0.021</td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">1.000</td><td style="text-align:right">0.118</td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">0.019</td><td style="text-align:right">0.023</td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">0.299</td><td style="text-align:right">0.119</td></tr>
</tbody></table>

## No avoids (legacy) — latency (cycles)

<table class="flash-cmp">
<colgroup>
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Ground truth</th>
  <th style="text-align:right">Round 1</th>
  <th style="text-align:right">Round 2</th>
  <th style="text-align:right">Δ R2 vs R1</th>
  <th style="text-align:right">Better</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">25,296,077</td><td style="text-align:right">62,489</td><td style="text-align:right">66,847</td><td style="text-align:right">+7.0%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">45,441,119</td><td style="text-align:right">45,441,119</td><td style="text-align:right">45,441,119</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">2,429,238</td><td style="text-align:right">5,046</td><td style="text-align:right">36,478</td><td style="text-align:right">+622.9%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">2,343,914</td><td style="text-align:right">5,316</td><td style="text-align:right">110,231</td><td style="text-align:right">+1973.6%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">68,337,001</td><td style="text-align:right">1,340,865</td><td style="text-align:right">1,340,865</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">53,906,591</td><td style="text-align:right">8,030,247</td><td style="text-align:right">3,564,760</td><td style="text-align:right">-55.6%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">64,118,323</td><td style="text-align:right">1,263,645</td><td style="text-align:right">3,187,785</td><td style="text-align:right">+152.3%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td style="text-align:right">—</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td style="text-align:right">—</td><td style="text-align:right">—</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">115,097</td><td style="text-align:right">108,790</td><td style="text-align:right">100,817</td><td style="text-align:right">-7.3%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">98,261,801</td><td style="text-align:right">18,499</td><td style="text-align:right">43,701</td><td style="text-align:right">+136.2%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">5,896,958</td><td style="text-align:right">27,540,001</td><td style="text-align:right">+367.0%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">54,298,622</td><td style="text-align:right">61,693</td><td style="text-align:right">120,241</td><td style="text-align:right">+94.9%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">2,715,487</td><td style="text-align:right">53,367</td><td style="text-align:right">351,519</td><td style="text-align:right">+558.7%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">1,441,873</td><td style="text-align:right">18,055</td><td style="text-align:right">66,673</td><td style="text-align:right">+269.3%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">2,270,081</td><td style="text-align:right">2,270,081</td><td style="text-align:right">2,270,081</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">3,281,841</td><td style="text-align:right">3,281,841</td><td style="text-align:right">488,488</td><td style="text-align:right">-85.1%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">41,761</td><td style="text-align:right">12,468</td><td style="text-align:right">3,701</td><td style="text-align:right">-70.3%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">3,112,241</td><td style="text-align:right">27,921</td><td style="text-align:right">1,110,086</td><td style="text-align:right">+3875.8%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">134,372,041</td><td style="text-align:right">5,672,817</td><td style="text-align:right">1,229,169</td><td style="text-align:right">-78.3%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">10,095,723</td><td style="text-align:right">732,587</td><td style="text-align:right">642,523</td><td style="text-align:right">-12.3%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">256,922</td><td style="text-align:right">199,322</td><td style="text-align:right">220,738</td><td style="text-align:right">+10.7%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">209,526,121</td><td style="text-align:right">118,560,241</td><td style="text-align:right">209,526,121</td><td style="text-align:right">+76.7%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">132,556,483</td><td style="text-align:right">1,119,361</td><td style="text-align:right">132,556,483</td><td style="text-align:right">+11742.2%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">23,241,675</td><td style="text-align:right">1,251,830</td><td style="text-align:right">2,855,106</td><td style="text-align:right">+128.1%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">33,807,761</td><td style="text-align:right">83,188</td><td style="text-align:right">6,958,389</td><td style="text-align:right">+8264.7%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">31,695,761</td><td style="text-align:right">228,631</td><td style="text-align:right">281,816</td><td style="text-align:right">+23.3%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1,160,161</td><td style="text-align:right">40,805</td><td style="text-align:right">127,781</td><td style="text-align:right">+213.2%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">22,598,401</td><td style="text-align:right">260,875</td><td style="text-align:right">376,641</td><td style="text-align:right">+44.4%</td><td style="text-align:right"><strong>R1</strong></td></tr>
</tbody></table>

### No avoids (legacy) — ground-truth latency ratio (synth / GT)

<table class="flash-cmp">
<colgroup>
  <col style="width:30%">
  <col style="width:30%">
  <col style="width:30%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Round 1</th>
  <th style="text-align:right">Round 2</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">0.002</td><td style="text-align:right">0.003</td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">1.000</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">0.002</td><td style="text-align:right">0.015</td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">0.002</td><td style="text-align:right">0.047</td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">0.020</td><td style="text-align:right">0.020</td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">0.149</td><td style="text-align:right">0.066</td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">0.020</td><td style="text-align:right">0.050</td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">0.945</td><td style="text-align:right">0.876</td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">0.000</td><td style="text-align:right">0.000</td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">0.007</td><td style="text-align:right">0.033</td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">0.001</td><td style="text-align:right">0.002</td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">0.020</td><td style="text-align:right">0.129</td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">0.013</td><td style="text-align:right">0.046</td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">1.000</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">1.000</td><td style="text-align:right">0.149</td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">0.299</td><td style="text-align:right">0.089</td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">0.009</td><td style="text-align:right">0.357</td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">0.042</td><td style="text-align:right">0.009</td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">0.073</td><td style="text-align:right">0.064</td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">0.776</td><td style="text-align:right">0.859</td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">0.566</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">0.008</td><td style="text-align:right">1.032</td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">0.054</td><td style="text-align:right">0.123</td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">0.002</td><td style="text-align:right">0.206</td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">0.007</td><td style="text-align:right">0.009</td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">0.035</td><td style="text-align:right">0.110</td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">0.012</td><td style="text-align:right">0.017</td></tr>
</tbody></table>

## Noskills (new) — latency (cycles)

<table class="flash-cmp">
<colgroup>
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Ground truth</th>
  <th style="text-align:right">Round 1</th>
  <th style="text-align:right">Round 2</th>
  <th style="text-align:right">Δ R2 vs R1</th>
  <th style="text-align:right">Better</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">25,296,077</td><td style="text-align:right">1,481,537</td><td style="text-align:right">836,292</td><td style="text-align:right">-43.6%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">45,441,119</td><td style="text-align:right">2,038,096</td><td style="text-align:right">22,748</td><td style="text-align:right">-98.9%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">2,429,238</td><td style="text-align:right">145,413</td><td style="text-align:right">2,429,238</td><td style="text-align:right">+1570.6%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">2,343,914</td><td style="text-align:right">2,343,914</td><td style="text-align:right">412,468</td><td style="text-align:right">-82.4%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">68,337,001</td><td style="text-align:right">19,497,841</td><td style="text-align:right">68,337,001</td><td style="text-align:right">+250.5%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">53,906,591</td><td style="text-align:right">100,993</td><td style="text-align:right">1,896,766</td><td style="text-align:right">+1778.1%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">64,118,323</td><td style="text-align:right">3,848,349</td><td style="text-align:right">2,140,200</td><td style="text-align:right">-44.4%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td style="text-align:right">—</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td style="text-align:right">—</td><td style="text-align:right">—</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">115,097</td><td style="text-align:right">102,917</td><td style="text-align:right">56,465</td><td style="text-align:right">-45.1%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">98,261,801</td><td style="text-align:right">3,219,121</td><td style="text-align:right">601,662</td><td style="text-align:right">-81.3%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">11,728,993</td><td style="text-align:right">17,496,145</td><td style="text-align:right">+49.2%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">54,298,622</td><td style="text-align:right">355,153</td><td style="text-align:right">194,054</td><td style="text-align:right">-45.4%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">2,715,487</td><td style="text-align:right">234,669</td><td style="text-align:right">19,913</td><td style="text-align:right">-91.5%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">1,441,873</td><td style="text-align:right">59,095</td><td style="text-align:right">150,946</td><td style="text-align:right">+155.4%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">2,270,081</td><td style="text-align:right">294,388</td><td style="text-align:right">1,296,548</td><td style="text-align:right">+340.4%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">3,281,841</td><td style="text-align:right">488,486</td><td style="text-align:right">40,080</td><td style="text-align:right">-91.8%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">41,761</td><td style="text-align:right">13,420</td><td style="text-align:right">13,031</td><td style="text-align:right">-2.9%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">3,112,241</td><td style="text-align:right">640,370</td><td style="text-align:right">22,780</td><td style="text-align:right">-96.4%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">134,372,041</td><td style="text-align:right">7,831,472</td><td style="text-align:right">134,372,041</td><td style="text-align:right">+1615.8%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">10,095,723</td><td style="text-align:right">10,095,723</td><td style="text-align:right">675,602</td><td style="text-align:right">-93.3%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">256,922</td><td style="text-align:right">254,630</td><td style="text-align:right">256,922</td><td style="text-align:right">+0.9%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">209,526,121</td><td style="text-align:right">209,526,121</td><td style="text-align:right">106,835,758</td><td style="text-align:right">-49.0%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">132,556,483</td><td style="text-align:right">132,556,483</td><td style="text-align:right">559,298</td><td style="text-align:right">-99.6%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">23,241,675</td><td style="text-align:right">2,855,106</td><td style="text-align:right">1,256,706</td><td style="text-align:right">-56.0%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">33,807,761</td><td style="text-align:right">1,002,851</td><td style="text-align:right">4,354,901</td><td style="text-align:right">+334.3%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">31,695,761</td><td style="text-align:right">31,695,761</td><td style="text-align:right">1,572,861</td><td style="text-align:right">-95.0%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1,160,161</td><td style="text-align:right">181,152</td><td style="text-align:right">1,155,753</td><td style="text-align:right">+538.0%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">22,598,401</td><td style="text-align:right">2,720,630</td><td style="text-align:right">2,720,630</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
</tbody></table>

### Noskills (new) — ground-truth latency ratio (synth / GT)

<table class="flash-cmp">
<colgroup>
  <col style="width:30%">
  <col style="width:30%">
  <col style="width:30%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Round 1</th>
  <th style="text-align:right">Round 2</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">0.059</td><td style="text-align:right">0.033</td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">0.045</td><td style="text-align:right">0.001</td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">0.060</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">1.070</td><td style="text-align:right">0.176</td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">0.285</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">0.002</td><td style="text-align:right">0.035</td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">0.060</td><td style="text-align:right">0.033</td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">0.894</td><td style="text-align:right">0.491</td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">0.033</td><td style="text-align:right">0.006</td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">0.014</td><td style="text-align:right">0.021</td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">0.007</td><td style="text-align:right">0.004</td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">0.086</td><td style="text-align:right">0.007</td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">0.041</td><td style="text-align:right">0.105</td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">0.130</td><td style="text-align:right">0.571</td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">0.149</td><td style="text-align:right">0.012</td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">0.321</td><td style="text-align:right">0.312</td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">0.206</td><td style="text-align:right">0.007</td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">0.058</td><td style="text-align:right">1.006</td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">1.000</td><td style="text-align:right">0.067</td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">0.991</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">1.033</td><td style="text-align:right">0.510</td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">1.000</td><td style="text-align:right">0.004</td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">0.123</td><td style="text-align:right">0.054</td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">0.030</td><td style="text-align:right">0.129</td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">1.000</td><td style="text-align:right">0.050</td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">0.156</td><td style="text-align:right">0.996</td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">0.120</td><td style="text-align:right">0.120</td></tr>
</tbody></table>

## Bn 2+2 (new) — latency (cycles)

<table class="flash-cmp">
<colgroup>
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Ground truth</th>
  <th style="text-align:right">Round 1</th>
  <th style="text-align:right">Round 2</th>
  <th style="text-align:right">Δ R2 vs R1</th>
  <th style="text-align:right">Better</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">25,296,077</td><td style="text-align:right">1,262,346</td><td style="text-align:right">1,262,346</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">45,441,119</td><td style="text-align:right">2,038,096</td><td style="text-align:right">2,047,248</td><td style="text-align:right">+0.4%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">2,429,238</td><td style="text-align:right">359,813</td><td style="text-align:right">635,987</td><td style="text-align:right">+76.8%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">2,343,914</td><td style="text-align:right">117,277</td><td style="text-align:right">50,503</td><td style="text-align:right">-56.9%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">68,337,001</td><td style="text-align:right">159,704</td><td style="text-align:right">7,950,677</td><td style="text-align:right">+4878.4%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">53,906,591</td><td style="text-align:right">1,837,814</td><td style="text-align:right">480,510</td><td style="text-align:right">-73.9%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">64,118,323</td><td style="text-align:right">3,193,955</td><td style="text-align:right">2,798,447</td><td style="text-align:right">-12.4%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td style="text-align:right">—</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td style="text-align:right">—</td><td style="text-align:right">—</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">115,097</td><td style="text-align:right">102,917</td><td style="text-align:right">100,817</td><td style="text-align:right">-2.0%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">98,261,801</td><td style="text-align:right">87,379,241</td><td style="text-align:right">2,884,561</td><td style="text-align:right">-96.7%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">833,976,005</td><td style="text-align:right">11,728,993</td><td style="text-align:right">-98.6%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">54,298,622</td><td style="text-align:right">54,093</td><td style="text-align:right">54,093</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">2,715,487</td><td style="text-align:right">2,715,487</td><td style="text-align:right">351,538</td><td style="text-align:right">-87.1%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">1,441,873</td><td style="text-align:right">263,725</td><td style="text-align:right">360,000</td><td style="text-align:right">+36.5%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">2,270,081</td><td style="text-align:right">1,275,551</td><td style="text-align:right">1,276,224</td><td style="text-align:right">+0.1%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">3,281,841</td><td style="text-align:right">488,486</td><td style="text-align:right">34,080</td><td style="text-align:right">-93.0%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">41,761</td><td style="text-align:right">13,031</td><td style="text-align:right">13,031</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">3,112,241</td><td style="text-align:right">640,370</td><td style="text-align:right">19,649</td><td style="text-align:right">-96.9%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">134,372,041</td><td style="text-align:right">11,829,856</td><td style="text-align:right">17,998,592</td><td style="text-align:right">+52.1%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">10,095,723</td><td style="text-align:right">6,532,621</td><td style="text-align:right">6,643,734</td><td style="text-align:right">+1.7%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">256,922</td><td style="text-align:right">193,276</td><td style="text-align:right">256,922</td><td style="text-align:right">+32.9%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">209,526,121</td><td style="text-align:right">3,245,657</td><td style="text-align:right">103,652,813</td><td style="text-align:right">+3093.6%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">132,556,483</td><td style="text-align:right">40,130,103</td><td style="text-align:right">586,042</td><td style="text-align:right">-98.5%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">23,241,675</td><td style="text-align:right">2,855,106</td><td style="text-align:right">2,855,106</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">33,807,761</td><td style="text-align:right">166,561</td><td style="text-align:right">33,807,761</td><td style="text-align:right">+20197.5%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">31,695,761</td><td style="text-align:right">1,572,861</td><td style="text-align:right">1,574,381</td><td style="text-align:right">+0.1%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1,160,161</td><td style="text-align:right">1,160,161</td><td style="text-align:right">1,160,161</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">22,598,401</td><td style="text-align:right">2,759,030</td><td style="text-align:right">2,759,030</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
</tbody></table>

### Bn 2+2 (new) — ground-truth latency ratio (synth / GT)

<table class="flash-cmp">
<colgroup>
  <col style="width:30%">
  <col style="width:30%">
  <col style="width:30%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Round 1</th>
  <th style="text-align:right">Round 2</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">0.050</td><td style="text-align:right">0.050</td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">0.045</td><td style="text-align:right">0.045</td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">0.148</td><td style="text-align:right">0.262</td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">0.050</td><td style="text-align:right">0.022</td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">0.002</td><td style="text-align:right">0.116</td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">0.034</td><td style="text-align:right">0.009</td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">0.050</td><td style="text-align:right">0.044</td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">0.894</td><td style="text-align:right">0.876</td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">0.889</td><td style="text-align:right">0.029</td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">1.000</td><td style="text-align:right">0.014</td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">0.001</td><td style="text-align:right">0.001</td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">1.001</td><td style="text-align:right">0.129</td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">0.183</td><td style="text-align:right">0.250</td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">0.562</td><td style="text-align:right">0.562</td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">0.149</td><td style="text-align:right">0.010</td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">0.312</td><td style="text-align:right">0.312</td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">0.206</td><td style="text-align:right">0.006</td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">0.088</td><td style="text-align:right">0.134</td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">0.647</td><td style="text-align:right">0.658</td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">0.752</td><td style="text-align:right">1.028</td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">0.015</td><td style="text-align:right">0.495</td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">0.303</td><td style="text-align:right">0.004</td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">0.123</td><td style="text-align:right">0.123</td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">0.005</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">0.050</td><td style="text-align:right">0.050</td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1.011</td><td style="text-align:right">1.013</td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">0.122</td><td style="text-align:right">0.122</td></tr>
</tbody></table>

## Bn 4+2 (new) — latency (cycles)

<table class="flash-cmp">
<colgroup>
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Ground truth</th>
  <th style="text-align:right">Round 1</th>
  <th style="text-align:right">Round 2</th>
  <th style="text-align:right">Δ R2 vs R1</th>
  <th style="text-align:right">Better</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">25,296,077</td><td style="text-align:right">1,265,553</td><td style="text-align:right">1,262,346</td><td style="text-align:right">-0.3%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">45,441,119</td><td style="text-align:right">2,038,096</td><td style="text-align:right">2,038,096</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">2,429,238</td><td style="text-align:right">2,429,238</td><td style="text-align:right">2,383,503</td><td style="text-align:right">-1.9%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">2,343,914</td><td style="text-align:right">132,498</td><td style="text-align:right">83,202</td><td style="text-align:right">-37.2%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">68,337,001</td><td style="text-align:right">8,000,912</td><td style="text-align:right">8,752,471</td><td style="text-align:right">+9.4%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">53,906,591</td><td style="text-align:right">10,753,631</td><td style="text-align:right">480,510</td><td style="text-align:right">-95.5%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">64,118,323</td><td style="text-align:right">94,585</td><td style="text-align:right">2,798,519</td><td style="text-align:right">+2858.7%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td style="text-align:right">—</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td style="text-align:right">—</td><td style="text-align:right">—</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">115,097</td><td style="text-align:right">100,817</td><td style="text-align:right">100,817</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">98,261,801</td><td style="text-align:right">43,701</td><td style="text-align:right">3,406,321</td><td style="text-align:right">+7694.6%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">833,976,005</td><td style="text-align:right">833,976,005</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">54,298,622</td><td style="text-align:right">355,153</td><td style="text-align:right">54,093</td><td style="text-align:right">-84.8%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">2,715,487</td><td style="text-align:right">261,572</td><td style="text-align:right">163,404</td><td style="text-align:right">-37.5%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">1,441,873</td><td style="text-align:right">1,141</td><td style="text-align:right">43,573</td><td style="text-align:right">+3718.8%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">2,270,081</td><td style="text-align:right">294,388</td><td style="text-align:right">1,296,548</td><td style="text-align:right">+340.4%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">3,281,841</td><td style="text-align:right">40,080</td><td style="text-align:right">40,760</td><td style="text-align:right">+1.7%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">41,761</td><td style="text-align:right">13,031</td><td style="text-align:right">13,031</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">3,112,241</td><td style="text-align:right">18,731</td><td style="text-align:right">18,771</td><td style="text-align:right">+0.2%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">134,372,041</td><td style="text-align:right">134,372,041</td><td style="text-align:right">166,593</td><td style="text-align:right">-99.9%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">10,095,723</td><td style="text-align:right">6,557,315</td><td style="text-align:right">6,672,294</td><td style="text-align:right">+1.8%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">256,922</td><td style="text-align:right">199,322</td><td style="text-align:right">213,919</td><td style="text-align:right">+7.3%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">209,526,121</td><td style="text-align:right">103,652,813</td><td style="text-align:right">103,652,813</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">132,556,483</td><td style="text-align:right">34,571,610</td><td style="text-align:right">332,366</td><td style="text-align:right">-99.0%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">23,241,675</td><td style="text-align:right">5,201,403</td><td style="text-align:right">2,855,106</td><td style="text-align:right">-45.1%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">33,807,761</td><td style="text-align:right">1,539,761</td><td style="text-align:right">1,375,901</td><td style="text-align:right">-10.6%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">31,695,761</td><td style="text-align:right">209,921</td><td style="text-align:right">17,549,101</td><td style="text-align:right">+8259.9%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1,160,161</td><td style="text-align:right">112,062</td><td style="text-align:right">1,160,161</td><td style="text-align:right">+935.3%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">22,598,401</td><td style="text-align:right">2,759,030</td><td style="text-align:right">2,759,030</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
</tbody></table>

### Bn 4+2 (new) — ground-truth latency ratio (synth / GT)

<table class="flash-cmp">
<colgroup>
  <col style="width:30%">
  <col style="width:30%">
  <col style="width:30%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Round 1</th>
  <th style="text-align:right">Round 2</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">0.050</td><td style="text-align:right">0.050</td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">0.045</td><td style="text-align:right">0.045</td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">1.000</td><td style="text-align:right">0.981</td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">0.057</td><td style="text-align:right">0.035</td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">0.117</td><td style="text-align:right">0.128</td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">0.199</td><td style="text-align:right">0.009</td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">0.001</td><td style="text-align:right">0.044</td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">0.876</td><td style="text-align:right">0.876</td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">0.000</td><td style="text-align:right">0.035</td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">1.007</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">0.007</td><td style="text-align:right">0.001</td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">0.096</td><td style="text-align:right">0.060</td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">0.001</td><td style="text-align:right">0.030</td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">0.130</td><td style="text-align:right">0.571</td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">0.012</td><td style="text-align:right">0.012</td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">0.312</td><td style="text-align:right">0.312</td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">0.006</td><td style="text-align:right">0.006</td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">1.000</td><td style="text-align:right">0.001</td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">0.650</td><td style="text-align:right">0.661</td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">0.776</td><td style="text-align:right">0.833</td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">0.495</td><td style="text-align:right">0.495</td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">0.261</td><td style="text-align:right">0.003</td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">0.224</td><td style="text-align:right">0.123</td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">0.046</td><td style="text-align:right">0.041</td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">0.007</td><td style="text-align:right">0.554</td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">0.097</td><td style="text-align:right">1.011</td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">0.122</td><td style="text-align:right">0.122</td></tr>
</tbody></table>

## Bn 6+2 (new) — latency (cycles)

<table class="flash-cmp">
<colgroup>
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Ground truth</th>
  <th style="text-align:right">Round 1</th>
  <th style="text-align:right">Round 2</th>
  <th style="text-align:right">Δ R2 vs R1</th>
  <th style="text-align:right">Better</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">25,296,077</td><td style="text-align:right">114,328</td><td style="text-align:right">2,003,313</td><td style="text-align:right">+1652.3%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">45,441,119</td><td style="text-align:right">2,850,840</td><td style="text-align:right">280,431</td><td style="text-align:right">-90.2%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">2,429,238</td><td style="text-align:right">2,314,746</td><td style="text-align:right">2,429,238</td><td style="text-align:right">+4.9%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">2,343,914</td><td style="text-align:right">127,293</td><td style="text-align:right">58,814</td><td style="text-align:right">-53.8%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">68,337,001</td><td style="text-align:right">7,923,872</td><td style="text-align:right">7,907,672</td><td style="text-align:right">-0.2%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">53,906,591</td><td style="text-align:right">1,821,941</td><td style="text-align:right">3,355,921</td><td style="text-align:right">+84.2%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">64,118,323</td><td style="text-align:right">2,695,369</td><td style="text-align:right">4,121,036</td><td style="text-align:right">+52.9%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td style="text-align:right">—</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td style="text-align:right">—</td><td style="text-align:right">—</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">115,097</td><td style="text-align:right">108,790</td><td style="text-align:right">102,084</td><td style="text-align:right">-6.2%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">98,261,801</td><td style="text-align:right">18,499</td><td style="text-align:right">18,499</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">833,976,005</td><td style="text-align:right">16,667,641</td><td style="text-align:right">-98.0%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">54,298,622</td><td style="text-align:right">1,377,145</td><td style="text-align:right">477,937</td><td style="text-align:right">-65.3%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">2,715,487</td><td style="text-align:right">283,347</td><td style="text-align:right">274,640</td><td style="text-align:right">-3.1%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">1,441,873</td><td style="text-align:right">180,647</td><td style="text-align:right">90,362</td><td style="text-align:right">-50.0%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">2,270,081</td><td style="text-align:right">1,276,224</td><td style="text-align:right">1,277,100</td><td style="text-align:right">+0.1%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">3,281,841</td><td style="text-align:right">808,966</td><td style="text-align:right">368,166</td><td style="text-align:right">-54.5%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">41,761</td><td style="text-align:right">13,031</td><td style="text-align:right">13,031</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">3,112,241</td><td style="text-align:right">1,614,961</td><td style="text-align:right">958,681</td><td style="text-align:right">-40.6%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">134,372,041</td><td style="text-align:right">15,574,591</td><td style="text-align:right">15,588,877</td><td style="text-align:right">+0.1%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">10,095,723</td><td style="text-align:right">6,585,875</td><td style="text-align:right">6,483,870</td><td style="text-align:right">-1.5%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">256,922</td><td style="text-align:right">256,922</td><td style="text-align:right">256,922</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">209,526,121</td><td style="text-align:right">117,950,941</td><td style="text-align:right">7,621,741</td><td style="text-align:right">-93.5%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">132,556,483</td><td style="text-align:right">652,761</td><td style="text-align:right">40,130,103</td><td style="text-align:right">+6047.7%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">23,241,675</td><td style="text-align:right">5,201,403</td><td style="text-align:right">5,200,727</td><td style="text-align:right">-0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">33,807,761</td><td style="text-align:right">1,539,761</td><td style="text-align:right">33,807,761</td><td style="text-align:right">+2095.6%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">31,695,761</td><td style="text-align:right">1,352,948</td><td style="text-align:right">201,476</td><td style="text-align:right">-85.1%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1,160,161</td><td style="text-align:right">1,155,676</td><td style="text-align:right">217,525</td><td style="text-align:right">-81.2%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">22,598,401</td><td style="text-align:right">2,704,727</td><td style="text-align:right">2,603,927</td><td style="text-align:right">-3.7%</td><td style="text-align:right"><strong>R2</strong></td></tr>
</tbody></table>

### Bn 6+2 (new) — ground-truth latency ratio (synth / GT)

<table class="flash-cmp">
<colgroup>
  <col style="width:30%">
  <col style="width:30%">
  <col style="width:30%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Round 1</th>
  <th style="text-align:right">Round 2</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">0.005</td><td style="text-align:right">0.079</td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">0.063</td><td style="text-align:right">0.006</td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">0.953</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">0.054</td><td style="text-align:right">0.025</td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">0.116</td><td style="text-align:right">0.116</td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">0.034</td><td style="text-align:right">0.062</td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">0.042</td><td style="text-align:right">0.064</td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">0.945</td><td style="text-align:right">0.887</td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">0.000</td><td style="text-align:right">0.000</td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">1.014</td><td style="text-align:right">0.020</td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">0.025</td><td style="text-align:right">0.009</td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">0.104</td><td style="text-align:right">0.101</td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">0.125</td><td style="text-align:right">0.063</td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">0.562</td><td style="text-align:right">0.563</td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">0.246</td><td style="text-align:right">0.112</td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">0.312</td><td style="text-align:right">0.312</td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">0.519</td><td style="text-align:right">0.308</td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">0.116</td><td style="text-align:right">0.116</td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">0.652</td><td style="text-align:right">0.642</td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">1.047</td><td style="text-align:right">1.427</td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">0.563</td><td style="text-align:right">0.036</td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">0.005</td><td style="text-align:right">0.303</td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">0.224</td><td style="text-align:right">0.224</td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">0.046</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">0.043</td><td style="text-align:right">0.006</td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">0.996</td><td style="text-align:right">0.187</td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">0.120</td><td style="text-align:right">0.115</td></tr>
</tbody></table>

## All+avoids (new) — latency (cycles)

<table class="flash-cmp">
<colgroup>
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Ground truth</th>
  <th style="text-align:right">Round 1</th>
  <th style="text-align:right">Round 2</th>
  <th style="text-align:right">Δ R2 vs R1</th>
  <th style="text-align:right">Better</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">25,296,077</td><td style="text-align:right">64,564</td><td style="text-align:right">123,753</td><td style="text-align:right">+91.7%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">45,441,119</td><td style="text-align:right">2,113,550</td><td style="text-align:right">105,236</td><td style="text-align:right">-95.0%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">2,429,238</td><td style="text-align:right">5,046</td><td style="text-align:right">238,548</td><td style="text-align:right">+4627.5%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">2,343,914</td><td style="text-align:right">110,530</td><td style="text-align:right">32,482</td><td style="text-align:right">-70.6%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">68,337,001</td><td style="text-align:right">376,177</td><td style="text-align:right">1,307,057</td><td style="text-align:right">+247.5%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">53,906,591</td><td style="text-align:right">429,902</td><td style="text-align:right">2,574,066</td><td style="text-align:right">+498.8%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">64,118,323</td><td style="text-align:right">4,121,036</td><td style="text-align:right">3,187,785</td><td style="text-align:right">-22.6%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td style="text-align:right">—</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td style="text-align:right">—</td><td style="text-align:right">—</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">115,097</td><td style="text-align:right">61,441</td><td style="text-align:right">100,817</td><td style="text-align:right">+64.1%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">98,261,801</td><td style="text-align:right">18,499</td><td style="text-align:right">43,701</td><td style="text-align:right">+136.2%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">833,976,005</td><td style="text-align:right">5,896,992</td><td style="text-align:right">-99.3%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">54,298,622</td><td style="text-align:right">565,921</td><td style="text-align:right">121,461</td><td style="text-align:right">-78.5%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">2,715,487</td><td style="text-align:right">243,411</td><td style="text-align:right">2,288,829</td><td style="text-align:right">+840.3%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">1,441,873</td><td style="text-align:right">5,443</td><td style="text-align:right">71,291</td><td style="text-align:right">+1209.8%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">2,270,081</td><td style="text-align:right">2,270,081</td><td style="text-align:right">2,270,081</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">3,281,841</td><td style="text-align:right">2,013,441</td><td style="text-align:right">63,850</td><td style="text-align:right">-96.8%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">41,761</td><td style="text-align:right">12,231</td><td style="text-align:right">12,620</td><td style="text-align:right">+3.2%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">3,112,241</td><td style="text-align:right">9,070</td><td style="text-align:right">1,574,241</td><td style="text-align:right">+17256.6%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">134,372,041</td><td style="text-align:right">4,823,953</td><td style="text-align:right">772,465</td><td style="text-align:right">-84.0%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">10,095,723</td><td style="text-align:right">6,557,315</td><td style="text-align:right">189,651</td><td style="text-align:right">-97.1%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">256,922</td><td style="text-align:right">4,793</td><td style="text-align:right">63,182</td><td style="text-align:right">+1218.2%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">209,526,121</td><td style="text-align:right">209,526,121</td><td style="text-align:right">117,487,334</td><td style="text-align:right">-43.9%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">132,556,483</td><td style="text-align:right">1,120,001</td><td style="text-align:right">2,217,601</td><td style="text-align:right">+98.0%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">23,241,675</td><td style="text-align:right">5,323,907</td><td style="text-align:right">2,558,081</td><td style="text-align:right">-52.0%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">33,807,761</td><td style="text-align:right">125,376</td><td style="text-align:right">1,909,201</td><td style="text-align:right">+1422.8%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">31,695,761</td><td style="text-align:right">556,551</td><td style="text-align:right">42,169</td><td style="text-align:right">-92.4%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1,160,161</td><td style="text-align:right">1,160,161</td><td style="text-align:right">1,160,161</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">22,598,401</td><td style="text-align:right">7,629,361</td><td style="text-align:right">5,165,089</td><td style="text-align:right">-32.3%</td><td style="text-align:right"><strong>R2</strong></td></tr>
</tbody></table>

### All+avoids (new) — ground-truth latency ratio (synth / GT)

<table class="flash-cmp">
<colgroup>
  <col style="width:30%">
  <col style="width:30%">
  <col style="width:30%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Round 1</th>
  <th style="text-align:right">Round 2</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">0.003</td><td style="text-align:right">0.005</td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">0.047</td><td style="text-align:right">0.002</td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">0.002</td><td style="text-align:right">0.098</td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">0.047</td><td style="text-align:right">0.014</td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">0.006</td><td style="text-align:right">0.019</td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">0.008</td><td style="text-align:right">0.048</td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">0.064</td><td style="text-align:right">0.050</td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">0.534</td><td style="text-align:right">0.876</td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">0.000</td><td style="text-align:right">0.000</td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">1.000</td><td style="text-align:right">0.007</td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">0.010</td><td style="text-align:right">0.002</td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">0.090</td><td style="text-align:right">0.843</td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">0.004</td><td style="text-align:right">0.049</td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">1.044</td><td style="text-align:right">1.001</td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">0.614</td><td style="text-align:right">0.019</td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">0.293</td><td style="text-align:right">0.302</td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">0.003</td><td style="text-align:right">0.506</td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">0.036</td><td style="text-align:right">0.006</td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">0.650</td><td style="text-align:right">0.019</td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">0.019</td><td style="text-align:right">0.246</td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">1.033</td><td style="text-align:right">0.561</td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">0.008</td><td style="text-align:right">0.017</td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">0.229</td><td style="text-align:right">0.110</td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">0.004</td><td style="text-align:right">0.056</td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">0.018</td><td style="text-align:right">0.001</td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1.000</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">0.338</td><td style="text-align:right">0.229</td></tr>
</tbody></table>

## No avoids (new) — latency (cycles)

<table class="flash-cmp">
<colgroup>
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
  <col style="width:15%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Ground truth</th>
  <th style="text-align:right">Round 1</th>
  <th style="text-align:right">Round 2</th>
  <th style="text-align:right">Δ R2 vs R1</th>
  <th style="text-align:right">Better</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">25,296,077</td><td style="text-align:right">454,508</td><td style="text-align:right">64,770</td><td style="text-align:right">-85.7%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">45,441,119</td><td style="text-align:right">105,635</td><td style="text-align:right">677,058</td><td style="text-align:right">+540.9%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">2,429,238</td><td style="text-align:right">5,046</td><td style="text-align:right">107,225</td><td style="text-align:right">+2025.0%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">2,343,914</td><td style="text-align:right">5,316</td><td style="text-align:right">5,120</td><td style="text-align:right">-3.7%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">68,337,001</td><td style="text-align:right">336,393</td><td style="text-align:right">1,320,305</td><td style="text-align:right">+292.5%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">53,906,591</td><td style="text-align:right">3,068,010</td><td style="text-align:right">1,837,814</td><td style="text-align:right">-40.1%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">64,118,323</td><td style="text-align:right">1,819,720</td><td style="text-align:right">64,118,323</td><td style="text-align:right">+3423.5%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td style="text-align:right">—</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td style="text-align:right">—</td><td style="text-align:right">—</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">115,097</td><td style="text-align:right">100,817</td><td style="text-align:right">115,097</td><td style="text-align:right">+14.2%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">98,261,801</td><td style="text-align:right">601,662</td><td style="text-align:right">1,215,001</td><td style="text-align:right">+101.9%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">833,976,005</td><td style="text-align:right">5,896,992</td><td style="text-align:right">-99.3%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">54,298,622</td><td style="text-align:right">145,081</td><td style="text-align:right">337,609</td><td style="text-align:right">+132.7%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">2,715,487</td><td style="text-align:right">74,622</td><td style="text-align:right">73,369</td><td style="text-align:right">-1.7%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">1,441,873</td><td style="text-align:right">45,550</td><td style="text-align:right">27,987</td><td style="text-align:right">-38.6%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">2,270,081</td><td style="text-align:right">2,270,081</td><td style="text-align:right">2,270,081</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">3,281,841</td><td style="text-align:right">488,486</td><td style="text-align:right">488,486</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">41,761</td><td style="text-align:right">12,468</td><td style="text-align:right">12,269</td><td style="text-align:right">-1.6%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">3,112,241</td><td style="text-align:right">1,979,281</td><td style="text-align:right">27,072</td><td style="text-align:right">-98.6%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">134,372,041</td><td style="text-align:right">2,591,089</td><td style="text-align:right">2,591,089</td><td style="text-align:right">+0.0%</td><td style="text-align:right"><strong>tie</strong></td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">10,095,723</td><td style="text-align:right">10,095,723</td><td style="text-align:right">8,280,235</td><td style="text-align:right">-18.0%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">256,922</td><td style="text-align:right">8,776</td><td style="text-align:right">256,922</td><td style="text-align:right">+2827.6%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">209,526,121</td><td style="text-align:right">9,482,221</td><td style="text-align:right">10,035,145</td><td style="text-align:right">+5.8%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">132,556,483</td><td style="text-align:right">132,556,483</td><td style="text-align:right">2,147,841</td><td style="text-align:right">-98.4%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">23,241,675</td><td style="text-align:right">23,241,675</td><td style="text-align:right">12,289,017</td><td style="text-align:right">-47.1%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">33,807,761</td><td style="text-align:right">6,958,389</td><td style="text-align:right">294,771</td><td style="text-align:right">-95.8%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">31,695,761</td><td style="text-align:right">54,836</td><td style="text-align:right">129,971</td><td style="text-align:right">+137.0%</td><td style="text-align:right"><strong>R1</strong></td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1,160,161</td><td style="text-align:right">884,533</td><td style="text-align:right">31,117</td><td style="text-align:right">-96.5%</td><td style="text-align:right"><strong>R2</strong></td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">22,598,401</td><td style="text-align:right">2,720,629</td><td style="text-align:right">22,598,401</td><td style="text-align:right">+730.6%</td><td style="text-align:right"><strong>R1</strong></td></tr>
</tbody></table>

### No avoids (new) — ground-truth latency ratio (synth / GT)

<table class="flash-cmp">
<colgroup>
  <col style="width:30%">
  <col style="width:30%">
  <col style="width:30%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Round 1</th>
  <th style="text-align:right">Round 2</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">0.018</td><td style="text-align:right">0.003</td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">0.002</td><td style="text-align:right">0.015</td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">0.002</td><td style="text-align:right">0.044</td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">0.002</td><td style="text-align:right">0.002</td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">0.005</td><td style="text-align:right">0.019</td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">0.057</td><td style="text-align:right">0.034</td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">0.028</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">0.876</td><td style="text-align:right">1.096</td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">0.006</td><td style="text-align:right">0.012</td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">1.007</td><td style="text-align:right">0.007</td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">0.003</td><td style="text-align:right">0.006</td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">0.027</td><td style="text-align:right">0.027</td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">0.032</td><td style="text-align:right">0.019</td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">1.001</td><td style="text-align:right">1.001</td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">0.149</td><td style="text-align:right">0.149</td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">0.299</td><td style="text-align:right">0.294</td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">0.636</td><td style="text-align:right">0.009</td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">0.019</td><td style="text-align:right">0.019</td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">1.000</td><td style="text-align:right">0.820</td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">0.034</td><td style="text-align:right">1.000</td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">0.045</td><td style="text-align:right">0.048</td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">1.000</td><td style="text-align:right">0.016</td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">1.000</td><td style="text-align:right">0.529</td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">0.206</td><td style="text-align:right">0.009</td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">0.002</td><td style="text-align:right">0.004</td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">0.762</td><td style="text-align:right">0.027</td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">0.120</td><td style="text-align:right">1.000</td></tr>
</tbody></table>

## Combined latency — legacy modes (Round 1 vs Round 2)

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
  <col style="width:9%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">GT</th>
  <th style="text-align:right">Noskills (legacy) R1</th>
  <th style="text-align:right">Bn 2+2 (legacy) R1</th>
  <th style="text-align:right">All+avoids (legacy) R1</th>
  <th style="text-align:right">No avoids (legacy) R1</th>
  <th style="text-align:right">Noskills (legacy) R2</th>
  <th style="text-align:right">Bn 2+2 (legacy) R2</th>
  <th style="text-align:right">All+avoids (legacy) R2</th>
  <th style="text-align:right">No avoids (legacy) R2</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">25,296,077</td><td style="text-align:right">64,530</td><td style="text-align:right">123,785</td><td style="text-align:right">64,530</td><td style="text-align:right">62,489</td><td style="text-align:right">1,481,537</td><td style="text-align:right">1,262,346</td><td style="text-align:right">66,847</td><td style="text-align:right">66,847</td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">45,441,119</td><td style="text-align:right">99,467</td><td style="text-align:right">99,467</td><td style="text-align:right">677,058</td><td style="text-align:right">45,441,119</td><td style="text-align:right">22,748</td><td style="text-align:right">2,047,248</td><td style="text-align:right">45,441,119</td><td style="text-align:right">45,441,119</td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">2,429,238</td><td style="text-align:right">118,344</td><td style="text-align:right">2,429,238</td><td style="text-align:right">2,300,710</td><td style="text-align:right">5,046</td><td style="text-align:right">2,429,238</td><td style="text-align:right">2,429,238</td><td style="text-align:right">135,333</td><td style="text-align:right">36,478</td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">2,343,914</td><td style="text-align:right">727,515</td><td style="text-align:right">2,297,693</td><td style="text-align:right">810,190</td><td style="text-align:right">5,316</td><td style="text-align:right">92,192</td><td style="text-align:right">191,293</td><td style="text-align:right">110,231</td><td style="text-align:right">110,231</td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">68,337,001</td><td style="text-align:right">3,980,432</td><td style="text-align:right">3,988,492</td><td style="text-align:right">68,337,001</td><td style="text-align:right">1,340,865</td><td style="text-align:right">3,980,432</td><td style="text-align:right">68,265,001</td><td style="text-align:right">9,484,353</td><td style="text-align:right">1,340,865</td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">53,906,591</td><td style="text-align:right">1,821,943</td><td style="text-align:right">1,837,814</td><td style="text-align:right">9,438,432</td><td style="text-align:right">8,030,247</td><td style="text-align:right">1,837,814</td><td style="text-align:right">10,753,631</td><td style="text-align:right">3,068,010</td><td style="text-align:right">3,564,760</td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">64,118,323</td><td style="text-align:right">3,786,109</td><td style="text-align:right">2,730,577</td><td style="text-align:right">64,118,323</td><td style="text-align:right">1,263,645</td><td style="text-align:right">1,859,454</td><td style="text-align:right">3,848,196</td><td style="text-align:right">3,187,785</td><td style="text-align:right">3,187,785</td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td style="text-align:right">—</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">115,097</td><td style="text-align:right">100,817</td><td style="text-align:right">102,917</td><td style="text-align:right">115,097</td><td style="text-align:right">108,790</td><td style="text-align:right">61,583</td><td style="text-align:right">102,084</td><td style="text-align:right">115,097</td><td style="text-align:right">100,817</td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">98,261,801</td><td style="text-align:right">601,662</td><td style="text-align:right">598,141</td><td style="text-align:right">18,499</td><td style="text-align:right">18,499</td><td style="text-align:right">601,662</td><td style="text-align:right">4,530,321</td><td style="text-align:right">18,499</td><td style="text-align:right">43,701</td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">28,051,921</td><td style="text-align:right">11,728,993</td><td style="text-align:right">482,436,001</td><td style="text-align:right">5,896,958</td><td style="text-align:right">17,496,145</td><td style="text-align:right">833,976,005</td><td style="text-align:right">1,043,281</td><td style="text-align:right">27,540,001</td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">54,298,622</td><td style="text-align:right">49,181</td><td style="text-align:right">49,181</td><td style="text-align:right">3,101,758</td><td style="text-align:right">61,693</td><td style="text-align:right">54,093</td><td style="text-align:right">988,345</td><td style="text-align:right">1,226,881</td><td style="text-align:right">120,241</td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">2,715,487</td><td style="text-align:right">318,159</td><td style="text-align:right">214,907</td><td style="text-align:right">254,407</td><td style="text-align:right">53,367</td><td style="text-align:right">261,323</td><td style="text-align:right">226,048</td><td style="text-align:right">74,622</td><td style="text-align:right">351,519</td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">1,441,873</td><td style="text-align:right">9,060</td><td style="text-align:right">1,141</td><td style="text-align:right">26,975</td><td style="text-align:right">18,055</td><td style="text-align:right">1,394</td><td style="text-align:right">89,000</td><td style="text-align:right">8,521</td><td style="text-align:right">66,673</td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">2,270,081</td><td style="text-align:right">1,296,548</td><td style="text-align:right">1,296,548</td><td style="text-align:right">2,270,081</td><td style="text-align:right">2,270,081</td><td style="text-align:right">294,388</td><td style="text-align:right">294,388</td><td style="text-align:right">2,270,081</td><td style="text-align:right">2,270,081</td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">3,281,841</td><td style="text-align:right">48,677</td><td style="text-align:right">47,686</td><td style="text-align:right">35,682</td><td style="text-align:right">3,281,841</td><td style="text-align:right">48,078</td><td style="text-align:right">488,486</td><td style="text-align:right">3,281,841</td><td style="text-align:right">488,488</td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">41,761</td><td style="text-align:right">13,268</td><td style="text-align:right">13,268</td><td style="text-align:right">12,231</td><td style="text-align:right">12,468</td><td style="text-align:right">13,031</td><td style="text-align:right">13,269</td><td style="text-align:right">12,468</td><td style="text-align:right">3,701</td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">3,112,241</td><td style="text-align:right">27,889</td><td style="text-align:right">27,889</td><td style="text-align:right">2,461,081</td><td style="text-align:right">27,921</td><td style="text-align:right">972,921</td><td style="text-align:right">27,889</td><td style="text-align:right">27,072</td><td style="text-align:right">1,110,086</td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">134,372,041</td><td style="text-align:right">15,614,072</td><td style="text-align:right">17,192,281</td><td style="text-align:right">5,062,065</td><td style="text-align:right">5,672,817</td><td style="text-align:right">15,757,441</td><td style="text-align:right">134,372,041</td><td style="text-align:right">4,853,617</td><td style="text-align:right">1,229,169</td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">10,095,723</td><td style="text-align:right">10,095,723</td><td style="text-align:right">6,672,294</td><td style="text-align:right">1,925,435</td><td style="text-align:right">732,587</td><td style="text-align:right">6,643,734</td><td style="text-align:right">4,882,187</td><td style="text-align:right">312,199</td><td style="text-align:right">642,523</td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">256,922</td><td style="text-align:right">210,483</td><td style="text-align:right">210,483</td><td style="text-align:right">48,054</td><td style="text-align:right">199,322</td><td style="text-align:right">256,922</td><td style="text-align:right">256,922</td><td style="text-align:right">199,322</td><td style="text-align:right">220,738</td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">209,526,121</td><td style="text-align:right">209,526,121</td><td style="text-align:right">118,500,556</td><td style="text-align:right">116,875,261</td><td style="text-align:right">118,560,241</td><td style="text-align:right">209,526,121</td><td style="text-align:right">6,473,311</td><td style="text-align:right">7,221,241</td><td style="text-align:right">209,526,121</td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">132,556,483</td><td style="text-align:right">586,042</td><td style="text-align:right">34,560,515</td><td style="text-align:right">38,071,522</td><td style="text-align:right">1,119,361</td><td style="text-align:right">55,195,677</td><td style="text-align:right">42,784,641</td><td style="text-align:right">23,951,361</td><td style="text-align:right">132,556,483</td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">23,241,675</td><td style="text-align:right">2,864,058</td><td style="text-align:right">2,864,134</td><td style="text-align:right">2,855,106</td><td style="text-align:right">1,251,830</td><td style="text-align:right">2,850,230</td><td style="text-align:right">1,256,706</td><td style="text-align:right">819,521</td><td style="text-align:right">2,855,106</td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">33,807,761</td><td style="text-align:right">992,061</td><td style="text-align:right">7,878,481</td><td style="text-align:right">6,979,669</td><td style="text-align:right">83,188</td><td style="text-align:right">1,572,501</td><td style="text-align:right">149,271</td><td style="text-align:right">721,711</td><td style="text-align:right">6,958,389</td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">31,695,761</td><td style="text-align:right">1,577,981</td><td style="text-align:right">119,181</td><td style="text-align:right">31,695,761</td><td style="text-align:right">228,631</td><td style="text-align:right">569,371</td><td style="text-align:right">1,572,861</td><td style="text-align:right">3,734,388</td><td style="text-align:right">281,816</td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1,160,161</td><td style="text-align:right">139,301</td><td style="text-align:right">678,122</td><td style="text-align:right">22,113</td><td style="text-align:right">40,805</td><td style="text-align:right">674,142</td><td style="text-align:right">1,160,161</td><td style="text-align:right">26,125</td><td style="text-align:right">127,781</td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">22,598,401</td><td style="text-align:right">2,759,030</td><td style="text-align:right">2,768,058</td><td style="text-align:right">6,747,057</td><td style="text-align:right">260,875</td><td style="text-align:right">2,720,630</td><td style="text-align:right">2,759,030</td><td style="text-align:right">2,699,987</td><td style="text-align:right">376,641</td></tr>
</tbody></table>

## Combined latency — new skills modes (Round 1 vs Round 2)

<table class="flash-cmp">
<colgroup>
  <col style="width:7%">
  <col style="width:7%">
  <col style="width:7%">
  <col style="width:7%">
  <col style="width:7%">
  <col style="width:7%">
  <col style="width:7%">
  <col style="width:7%">
  <col style="width:7%">
  <col style="width:7%">
  <col style="width:7%">
  <col style="width:7%">
  <col style="width:7%">
  <col style="width:7%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">GT</th>
  <th style="text-align:right">Noskills R1</th>
  <th style="text-align:right">Bn 2+2 R1</th>
  <th style="text-align:right">Bn 4+2 R1</th>
  <th style="text-align:right">Bn 6+2 R1</th>
  <th style="text-align:right">All+avoids R1</th>
  <th style="text-align:right">No avoids R1</th>
  <th style="text-align:right">Noskills R2</th>
  <th style="text-align:right">Bn 2+2 R2</th>
  <th style="text-align:right">Bn 4+2 R2</th>
  <th style="text-align:right">Bn 6+2 R2</th>
  <th style="text-align:right">All+avoids R2</th>
  <th style="text-align:right">No avoids R2</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">25,296,077</td><td style="text-align:right">1,481,537</td><td style="text-align:right">1,262,346</td><td style="text-align:right">1,265,553</td><td style="text-align:right">114,328</td><td style="text-align:right">64,564</td><td style="text-align:right">454,508</td><td style="text-align:right">836,292</td><td style="text-align:right">1,262,346</td><td style="text-align:right">1,262,346</td><td style="text-align:right">2,003,313</td><td style="text-align:right">123,753</td><td style="text-align:right">64,770</td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">45,441,119</td><td style="text-align:right">2,038,096</td><td style="text-align:right">2,038,096</td><td style="text-align:right">2,038,096</td><td style="text-align:right">2,850,840</td><td style="text-align:right">2,113,550</td><td style="text-align:right">105,635</td><td style="text-align:right">22,748</td><td style="text-align:right">2,047,248</td><td style="text-align:right">2,038,096</td><td style="text-align:right">280,431</td><td style="text-align:right">105,236</td><td style="text-align:right">677,058</td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">2,429,238</td><td style="text-align:right">145,413</td><td style="text-align:right">359,813</td><td style="text-align:right">2,429,238</td><td style="text-align:right">2,314,746</td><td style="text-align:right">5,046</td><td style="text-align:right">5,046</td><td style="text-align:right">2,429,238</td><td style="text-align:right">635,987</td><td style="text-align:right">2,383,503</td><td style="text-align:right">2,429,238</td><td style="text-align:right">238,548</td><td style="text-align:right">107,225</td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">2,343,914</td><td style="text-align:right">2,343,914</td><td style="text-align:right">117,277</td><td style="text-align:right">132,498</td><td style="text-align:right">127,293</td><td style="text-align:right">110,530</td><td style="text-align:right">5,316</td><td style="text-align:right">412,468</td><td style="text-align:right">50,503</td><td style="text-align:right">83,202</td><td style="text-align:right">58,814</td><td style="text-align:right">32,482</td><td style="text-align:right">5,120</td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">68,337,001</td><td style="text-align:right">19,497,841</td><td style="text-align:right">159,704</td><td style="text-align:right">8,000,912</td><td style="text-align:right">7,923,872</td><td style="text-align:right">376,177</td><td style="text-align:right">336,393</td><td style="text-align:right">68,337,001</td><td style="text-align:right">7,950,677</td><td style="text-align:right">8,752,471</td><td style="text-align:right">7,907,672</td><td style="text-align:right">1,307,057</td><td style="text-align:right">1,320,305</td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">53,906,591</td><td style="text-align:right">100,993</td><td style="text-align:right">1,837,814</td><td style="text-align:right">10,753,631</td><td style="text-align:right">1,821,941</td><td style="text-align:right">429,902</td><td style="text-align:right">3,068,010</td><td style="text-align:right">1,896,766</td><td style="text-align:right">480,510</td><td style="text-align:right">480,510</td><td style="text-align:right">3,355,921</td><td style="text-align:right">2,574,066</td><td style="text-align:right">1,837,814</td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">64,118,323</td><td style="text-align:right">3,848,349</td><td style="text-align:right">3,193,955</td><td style="text-align:right">94,585</td><td style="text-align:right">2,695,369</td><td style="text-align:right">4,121,036</td><td style="text-align:right">1,819,720</td><td style="text-align:right">2,140,200</td><td style="text-align:right">2,798,447</td><td style="text-align:right">2,798,519</td><td style="text-align:right">4,121,036</td><td style="text-align:right">3,187,785</td><td style="text-align:right">64,118,323</td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td style="text-align:right">—</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">115,097</td><td style="text-align:right">102,917</td><td style="text-align:right">102,917</td><td style="text-align:right">100,817</td><td style="text-align:right">108,790</td><td style="text-align:right">61,441</td><td style="text-align:right">100,817</td><td style="text-align:right">56,465</td><td style="text-align:right">100,817</td><td style="text-align:right">100,817</td><td style="text-align:right">102,084</td><td style="text-align:right">100,817</td><td style="text-align:right">115,097</td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">98,261,801</td><td style="text-align:right">3,219,121</td><td style="text-align:right">87,379,241</td><td style="text-align:right">43,701</td><td style="text-align:right">18,499</td><td style="text-align:right">18,499</td><td style="text-align:right">601,662</td><td style="text-align:right">601,662</td><td style="text-align:right">2,884,561</td><td style="text-align:right">3,406,321</td><td style="text-align:right">18,499</td><td style="text-align:right">43,701</td><td style="text-align:right">1,215,001</td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">11,728,993</td><td style="text-align:right">833,976,005</td><td style="text-align:right">833,976,005</td><td style="text-align:right">833,976,005</td><td style="text-align:right">833,976,005</td><td style="text-align:right">833,976,005</td><td style="text-align:right">17,496,145</td><td style="text-align:right">11,728,993</td><td style="text-align:right">833,976,005</td><td style="text-align:right">16,667,641</td><td style="text-align:right">5,896,992</td><td style="text-align:right">5,896,992</td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">54,298,622</td><td style="text-align:right">355,153</td><td style="text-align:right">54,093</td><td style="text-align:right">355,153</td><td style="text-align:right">1,377,145</td><td style="text-align:right">565,921</td><td style="text-align:right">145,081</td><td style="text-align:right">194,054</td><td style="text-align:right">54,093</td><td style="text-align:right">54,093</td><td style="text-align:right">477,937</td><td style="text-align:right">121,461</td><td style="text-align:right">337,609</td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">2,715,487</td><td style="text-align:right">234,669</td><td style="text-align:right">2,715,487</td><td style="text-align:right">261,572</td><td style="text-align:right">283,347</td><td style="text-align:right">243,411</td><td style="text-align:right">74,622</td><td style="text-align:right">19,913</td><td style="text-align:right">351,538</td><td style="text-align:right">163,404</td><td style="text-align:right">274,640</td><td style="text-align:right">2,288,829</td><td style="text-align:right">73,369</td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">1,441,873</td><td style="text-align:right">59,095</td><td style="text-align:right">263,725</td><td style="text-align:right">1,141</td><td style="text-align:right">180,647</td><td style="text-align:right">5,443</td><td style="text-align:right">45,550</td><td style="text-align:right">150,946</td><td style="text-align:right">360,000</td><td style="text-align:right">43,573</td><td style="text-align:right">90,362</td><td style="text-align:right">71,291</td><td style="text-align:right">27,987</td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">2,270,081</td><td style="text-align:right">294,388</td><td style="text-align:right">1,275,551</td><td style="text-align:right">294,388</td><td style="text-align:right">1,276,224</td><td style="text-align:right">2,270,081</td><td style="text-align:right">2,270,081</td><td style="text-align:right">1,296,548</td><td style="text-align:right">1,276,224</td><td style="text-align:right">1,296,548</td><td style="text-align:right">1,277,100</td><td style="text-align:right">2,270,081</td><td style="text-align:right">2,270,081</td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">3,281,841</td><td style="text-align:right">488,486</td><td style="text-align:right">488,486</td><td style="text-align:right">40,080</td><td style="text-align:right">808,966</td><td style="text-align:right">2,013,441</td><td style="text-align:right">488,486</td><td style="text-align:right">40,080</td><td style="text-align:right">34,080</td><td style="text-align:right">40,760</td><td style="text-align:right">368,166</td><td style="text-align:right">63,850</td><td style="text-align:right">488,486</td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">41,761</td><td style="text-align:right">13,420</td><td style="text-align:right">13,031</td><td style="text-align:right">13,031</td><td style="text-align:right">13,031</td><td style="text-align:right">12,231</td><td style="text-align:right">12,468</td><td style="text-align:right">13,031</td><td style="text-align:right">13,031</td><td style="text-align:right">13,031</td><td style="text-align:right">13,031</td><td style="text-align:right">12,620</td><td style="text-align:right">12,269</td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">3,112,241</td><td style="text-align:right">640,370</td><td style="text-align:right">640,370</td><td style="text-align:right">18,731</td><td style="text-align:right">1,614,961</td><td style="text-align:right">9,070</td><td style="text-align:right">1,979,281</td><td style="text-align:right">22,780</td><td style="text-align:right">19,649</td><td style="text-align:right">18,771</td><td style="text-align:right">958,681</td><td style="text-align:right">1,574,241</td><td style="text-align:right">27,072</td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">134,372,041</td><td style="text-align:right">7,831,472</td><td style="text-align:right">11,829,856</td><td style="text-align:right">134,372,041</td><td style="text-align:right">15,574,591</td><td style="text-align:right">4,823,953</td><td style="text-align:right">2,591,089</td><td style="text-align:right">134,372,041</td><td style="text-align:right">17,998,592</td><td style="text-align:right">166,593</td><td style="text-align:right">15,588,877</td><td style="text-align:right">772,465</td><td style="text-align:right">2,591,089</td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">10,095,723</td><td style="text-align:right">10,095,723</td><td style="text-align:right">6,532,621</td><td style="text-align:right">6,557,315</td><td style="text-align:right">6,585,875</td><td style="text-align:right">6,557,315</td><td style="text-align:right">10,095,723</td><td style="text-align:right">675,602</td><td style="text-align:right">6,643,734</td><td style="text-align:right">6,672,294</td><td style="text-align:right">6,483,870</td><td style="text-align:right">189,651</td><td style="text-align:right">8,280,235</td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">256,922</td><td style="text-align:right">254,630</td><td style="text-align:right">193,276</td><td style="text-align:right">199,322</td><td style="text-align:right">256,922</td><td style="text-align:right">4,793</td><td style="text-align:right">8,776</td><td style="text-align:right">256,922</td><td style="text-align:right">256,922</td><td style="text-align:right">213,919</td><td style="text-align:right">256,922</td><td style="text-align:right">63,182</td><td style="text-align:right">256,922</td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">209,526,121</td><td style="text-align:right">209,526,121</td><td style="text-align:right">3,245,657</td><td style="text-align:right">103,652,813</td><td style="text-align:right">117,950,941</td><td style="text-align:right">209,526,121</td><td style="text-align:right">9,482,221</td><td style="text-align:right">106,835,758</td><td style="text-align:right">103,652,813</td><td style="text-align:right">103,652,813</td><td style="text-align:right">7,621,741</td><td style="text-align:right">117,487,334</td><td style="text-align:right">10,035,145</td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">132,556,483</td><td style="text-align:right">132,556,483</td><td style="text-align:right">40,130,103</td><td style="text-align:right">34,571,610</td><td style="text-align:right">652,761</td><td style="text-align:right">1,120,001</td><td style="text-align:right">132,556,483</td><td style="text-align:right">559,298</td><td style="text-align:right">586,042</td><td style="text-align:right">332,366</td><td style="text-align:right">40,130,103</td><td style="text-align:right">2,217,601</td><td style="text-align:right">2,147,841</td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">23,241,675</td><td style="text-align:right">2,855,106</td><td style="text-align:right">2,855,106</td><td style="text-align:right">5,201,403</td><td style="text-align:right">5,201,403</td><td style="text-align:right">5,323,907</td><td style="text-align:right">23,241,675</td><td style="text-align:right">1,256,706</td><td style="text-align:right">2,855,106</td><td style="text-align:right">2,855,106</td><td style="text-align:right">5,200,727</td><td style="text-align:right">2,558,081</td><td style="text-align:right">12,289,017</td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">33,807,761</td><td style="text-align:right">1,002,851</td><td style="text-align:right">166,561</td><td style="text-align:right">1,539,761</td><td style="text-align:right">1,539,761</td><td style="text-align:right">125,376</td><td style="text-align:right">6,958,389</td><td style="text-align:right">4,354,901</td><td style="text-align:right">33,807,761</td><td style="text-align:right">1,375,901</td><td style="text-align:right">33,807,761</td><td style="text-align:right">1,909,201</td><td style="text-align:right">294,771</td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">31,695,761</td><td style="text-align:right">31,695,761</td><td style="text-align:right">1,572,861</td><td style="text-align:right">209,921</td><td style="text-align:right">1,352,948</td><td style="text-align:right">556,551</td><td style="text-align:right">54,836</td><td style="text-align:right">1,572,861</td><td style="text-align:right">1,574,381</td><td style="text-align:right">17,549,101</td><td style="text-align:right">201,476</td><td style="text-align:right">42,169</td><td style="text-align:right">129,971</td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1,160,161</td><td style="text-align:right">181,152</td><td style="text-align:right">1,160,161</td><td style="text-align:right">112,062</td><td style="text-align:right">1,155,676</td><td style="text-align:right">1,160,161</td><td style="text-align:right">884,533</td><td style="text-align:right">1,155,753</td><td style="text-align:right">1,160,161</td><td style="text-align:right">1,160,161</td><td style="text-align:right">217,525</td><td style="text-align:right">1,160,161</td><td style="text-align:right">31,117</td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">22,598,401</td><td style="text-align:right">2,720,630</td><td style="text-align:right">2,759,030</td><td style="text-align:right">2,759,030</td><td style="text-align:right">2,704,727</td><td style="text-align:right">7,629,361</td><td style="text-align:right">2,720,629</td><td style="text-align:right">2,720,630</td><td style="text-align:right">2,759,030</td><td style="text-align:right">2,759,030</td><td style="text-align:right">2,603,927</td><td style="text-align:right">5,165,089</td><td style="text-align:right">22,598,401</td></tr>
</tbody></table>

## Large swings (&gt;2× Round 2 / Round 1)

<table class="flash-cmp">
<colgroup>
  <col style="width:18%">
  <col style="width:18%">
  <col style="width:18%">
  <col style="width:18%">
  <col style="width:18%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Mode</th>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Round 1</th>
  <th style="text-align:right">Round 2</th>
  <th style="text-align:right">R2/R1</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">Noskills (legacy)</td><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">586,042</td><td style="text-align:right">55,195,677</td><td style="text-align:right">94.18×</td></tr>
<tr><td style="text-align:left">Noskills (legacy)</td><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">27,889</td><td style="text-align:right">972,921</td><td style="text-align:right">34.89×</td></tr>
<tr><td style="text-align:left">Noskills (legacy)</td><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">64,530</td><td style="text-align:right">1,481,537</td><td style="text-align:right">22.96×</td></tr>
<tr><td style="text-align:left">Noskills (legacy)</td><td style="text-align:left"><code>atax</code></td><td style="text-align:right">118,344</td><td style="text-align:right">2,429,238</td><td style="text-align:right">20.53×</td></tr>
<tr><td style="text-align:left">Noskills (legacy)</td><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">727,515</td><td style="text-align:right">92,192</td><td style="text-align:right">0.13×</td></tr>
<tr><td style="text-align:left">Noskills (legacy)</td><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">9,060</td><td style="text-align:right">1,394</td><td style="text-align:right">0.15×</td></tr>
<tr><td style="text-align:left">Noskills (legacy)</td><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">139,301</td><td style="text-align:right">674,142</td><td style="text-align:right">4.84×</td></tr>
<tr><td style="text-align:left">Noskills (legacy)</td><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">1,296,548</td><td style="text-align:right">294,388</td><td style="text-align:right">0.23×</td></tr>
<tr><td style="text-align:left">Noskills (legacy)</td><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">99,467</td><td style="text-align:right">22,748</td><td style="text-align:right">0.23×</td></tr>
<tr><td style="text-align:left">Noskills (legacy)</td><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">1,577,981</td><td style="text-align:right">569,371</td><td style="text-align:right">0.36×</td></tr>
<tr><td style="text-align:left">Noskills (legacy)</td><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">3,786,109</td><td style="text-align:right">1,859,454</td><td style="text-align:right">0.49×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (legacy)</td><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">1,141</td><td style="text-align:right">89,000</td><td style="text-align:right">78.00×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (legacy)</td><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">11,728,993</td><td style="text-align:right">833,976,005</td><td style="text-align:right">71.10×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (legacy)</td><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">7,878,481</td><td style="text-align:right">149,271</td><td style="text-align:right">0.02×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (legacy)</td><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">99,467</td><td style="text-align:right">2,047,248</td><td style="text-align:right">20.58×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (legacy)</td><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">49,181</td><td style="text-align:right">988,345</td><td style="text-align:right">20.10×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (legacy)</td><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">118,500,556</td><td style="text-align:right">6,473,311</td><td style="text-align:right">0.05×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (legacy)</td><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">3,988,492</td><td style="text-align:right">68,265,001</td><td style="text-align:right">17.12×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (legacy)</td><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">119,181</td><td style="text-align:right">1,572,861</td><td style="text-align:right">13.20×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (legacy)</td><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">2,297,693</td><td style="text-align:right">191,293</td><td style="text-align:right">0.08×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (legacy)</td><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">47,686</td><td style="text-align:right">488,486</td><td style="text-align:right">10.24×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (legacy)</td><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">123,785</td><td style="text-align:right">1,262,346</td><td style="text-align:right">10.20×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (legacy)</td><td style="text-align:left"><code>lu</code></td><td style="text-align:right">17,192,281</td><td style="text-align:right">134,372,041</td><td style="text-align:right">7.82×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (legacy)</td><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">598,141</td><td style="text-align:right">4,530,321</td><td style="text-align:right">7.57×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (legacy)</td><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">1,837,814</td><td style="text-align:right">10,753,631</td><td style="text-align:right">5.85×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (legacy)</td><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">1,296,548</td><td style="text-align:right">294,388</td><td style="text-align:right">0.23×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (legacy)</td><td style="text-align:left"><code>symm</code></td><td style="text-align:right">2,864,134</td><td style="text-align:right">1,256,706</td><td style="text-align:right">0.44×</td></tr>
<tr><td style="text-align:left">All+avoids (legacy)</td><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">482,436,001</td><td style="text-align:right">1,043,281</td><td style="text-align:right">0.00×</td></tr>
<tr><td style="text-align:left">All+avoids (legacy)</td><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">35,682</td><td style="text-align:right">3,281,841</td><td style="text-align:right">91.97×</td></tr>
<tr><td style="text-align:left">All+avoids (legacy)</td><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">2,461,081</td><td style="text-align:right">27,072</td><td style="text-align:right">0.01×</td></tr>
<tr><td style="text-align:left">All+avoids (legacy)</td><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">677,058</td><td style="text-align:right">45,441,119</td><td style="text-align:right">67.12×</td></tr>
<tr><td style="text-align:left">All+avoids (legacy)</td><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">64,118,323</td><td style="text-align:right">3,187,785</td><td style="text-align:right">0.05×</td></tr>
<tr><td style="text-align:left">All+avoids (legacy)</td><td style="text-align:left"><code>atax</code></td><td style="text-align:right">2,300,710</td><td style="text-align:right">135,333</td><td style="text-align:right">0.06×</td></tr>
<tr><td style="text-align:left">All+avoids (legacy)</td><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">116,875,261</td><td style="text-align:right">7,221,241</td><td style="text-align:right">0.06×</td></tr>
<tr><td style="text-align:left">All+avoids (legacy)</td><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">6,979,669</td><td style="text-align:right">721,711</td><td style="text-align:right">0.10×</td></tr>
<tr><td style="text-align:left">All+avoids (legacy)</td><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">31,695,761</td><td style="text-align:right">3,734,388</td><td style="text-align:right">0.12×</td></tr>
<tr><td style="text-align:left">All+avoids (legacy)</td><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">810,190</td><td style="text-align:right">110,231</td><td style="text-align:right">0.14×</td></tr>
<tr><td style="text-align:left">All+avoids (legacy)</td><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">68,337,001</td><td style="text-align:right">9,484,353</td><td style="text-align:right">0.14×</td></tr>
<tr><td style="text-align:left">All+avoids (legacy)</td><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">1,925,435</td><td style="text-align:right">312,199</td><td style="text-align:right">0.16×</td></tr>
<tr><td style="text-align:left">All+avoids (legacy)</td><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">48,054</td><td style="text-align:right">199,322</td><td style="text-align:right">4.15×</td></tr>
<tr><td style="text-align:left">All+avoids (legacy)</td><td style="text-align:left"><code>symm</code></td><td style="text-align:right">2,855,106</td><td style="text-align:right">819,521</td><td style="text-align:right">0.29×</td></tr>
<tr><td style="text-align:left">All+avoids (legacy)</td><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">254,407</td><td style="text-align:right">74,622</td><td style="text-align:right">0.29×</td></tr>
<tr><td style="text-align:left">All+avoids (legacy)</td><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">26,975</td><td style="text-align:right">8,521</td><td style="text-align:right">0.32×</td></tr>
<tr><td style="text-align:left">All+avoids (legacy)</td><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">9,438,432</td><td style="text-align:right">3,068,010</td><td style="text-align:right">0.33×</td></tr>
<tr><td style="text-align:left">All+avoids (legacy)</td><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">3,101,758</td><td style="text-align:right">1,226,881</td><td style="text-align:right">0.40×</td></tr>
<tr><td style="text-align:left">All+avoids (legacy)</td><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">6,747,057</td><td style="text-align:right">2,699,987</td><td style="text-align:right">0.40×</td></tr>
<tr><td style="text-align:left">No avoids (legacy)</td><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">1,119,361</td><td style="text-align:right">132,556,483</td><td style="text-align:right">118.42×</td></tr>
<tr><td style="text-align:left">No avoids (legacy)</td><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">83,188</td><td style="text-align:right">6,958,389</td><td style="text-align:right">83.65×</td></tr>
<tr><td style="text-align:left">No avoids (legacy)</td><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">27,921</td><td style="text-align:right">1,110,086</td><td style="text-align:right">39.76×</td></tr>
<tr><td style="text-align:left">No avoids (legacy)</td><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">5,316</td><td style="text-align:right">110,231</td><td style="text-align:right">20.74×</td></tr>
<tr><td style="text-align:left">No avoids (legacy)</td><td style="text-align:left"><code>atax</code></td><td style="text-align:right">5,046</td><td style="text-align:right">36,478</td><td style="text-align:right">7.23×</td></tr>
<tr><td style="text-align:left">No avoids (legacy)</td><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">3,281,841</td><td style="text-align:right">488,488</td><td style="text-align:right">0.15×</td></tr>
<tr><td style="text-align:left">No avoids (legacy)</td><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">53,367</td><td style="text-align:right">351,519</td><td style="text-align:right">6.59×</td></tr>
<tr><td style="text-align:left">No avoids (legacy)</td><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">5,896,958</td><td style="text-align:right">27,540,001</td><td style="text-align:right">4.67×</td></tr>
<tr><td style="text-align:left">No avoids (legacy)</td><td style="text-align:left"><code>lu</code></td><td style="text-align:right">5,672,817</td><td style="text-align:right">1,229,169</td><td style="text-align:right">0.22×</td></tr>
<tr><td style="text-align:left">No avoids (legacy)</td><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">18,055</td><td style="text-align:right">66,673</td><td style="text-align:right">3.69×</td></tr>
<tr><td style="text-align:left">No avoids (legacy)</td><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">12,468</td><td style="text-align:right">3,701</td><td style="text-align:right">0.30×</td></tr>
<tr><td style="text-align:left">No avoids (legacy)</td><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">40,805</td><td style="text-align:right">127,781</td><td style="text-align:right">3.13×</td></tr>
<tr><td style="text-align:left">No avoids (legacy)</td><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">1,263,645</td><td style="text-align:right">3,187,785</td><td style="text-align:right">2.52×</td></tr>
<tr><td style="text-align:left">No avoids (legacy)</td><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">18,499</td><td style="text-align:right">43,701</td><td style="text-align:right">2.36×</td></tr>
<tr><td style="text-align:left">No avoids (legacy)</td><td style="text-align:left"><code>symm</code></td><td style="text-align:right">1,251,830</td><td style="text-align:right">2,855,106</td><td style="text-align:right">2.28×</td></tr>
<tr><td style="text-align:left">No avoids (legacy)</td><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">8,030,247</td><td style="text-align:right">3,564,760</td><td style="text-align:right">0.44×</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">132,556,483</td><td style="text-align:right">559,298</td><td style="text-align:right">0.00×</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">2,038,096</td><td style="text-align:right">22,748</td><td style="text-align:right">0.01×</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">640,370</td><td style="text-align:right">22,780</td><td style="text-align:right">0.04×</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">31,695,761</td><td style="text-align:right">1,572,861</td><td style="text-align:right">0.05×</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">100,993</td><td style="text-align:right">1,896,766</td><td style="text-align:right">18.78×</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>lu</code></td><td style="text-align:right">7,831,472</td><td style="text-align:right">134,372,041</td><td style="text-align:right">17.16×</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>atax</code></td><td style="text-align:right">145,413</td><td style="text-align:right">2,429,238</td><td style="text-align:right">16.71×</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">10,095,723</td><td style="text-align:right">675,602</td><td style="text-align:right">0.07×</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">488,486</td><td style="text-align:right">40,080</td><td style="text-align:right">0.08×</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">234,669</td><td style="text-align:right">19,913</td><td style="text-align:right">0.08×</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">181,152</td><td style="text-align:right">1,155,753</td><td style="text-align:right">6.38×</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">2,343,914</td><td style="text-align:right">412,468</td><td style="text-align:right">0.18×</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">3,219,121</td><td style="text-align:right">601,662</td><td style="text-align:right">0.19×</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">294,388</td><td style="text-align:right">1,296,548</td><td style="text-align:right">4.40×</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">1,002,851</td><td style="text-align:right">4,354,901</td><td style="text-align:right">4.34×</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">19,497,841</td><td style="text-align:right">68,337,001</td><td style="text-align:right">3.50×</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">59,095</td><td style="text-align:right">150,946</td><td style="text-align:right">2.55×</td></tr>
<tr><td style="text-align:left">Noskills (new)</td><td style="text-align:left"><code>symm</code></td><td style="text-align:right">2,855,106</td><td style="text-align:right">1,256,706</td><td style="text-align:right">0.44×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (new)</td><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">166,561</td><td style="text-align:right">33,807,761</td><td style="text-align:right">202.98×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (new)</td><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">11,728,993</td><td style="text-align:right">0.01×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (new)</td><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">40,130,103</td><td style="text-align:right">586,042</td><td style="text-align:right">0.01×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (new)</td><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">159,704</td><td style="text-align:right">7,950,677</td><td style="text-align:right">49.78×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (new)</td><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">640,370</td><td style="text-align:right">19,649</td><td style="text-align:right">0.03×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (new)</td><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">3,245,657</td><td style="text-align:right">103,652,813</td><td style="text-align:right">31.94×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (new)</td><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">87,379,241</td><td style="text-align:right">2,884,561</td><td style="text-align:right">0.03×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (new)</td><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">488,486</td><td style="text-align:right">34,080</td><td style="text-align:right">0.07×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (new)</td><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">2,715,487</td><td style="text-align:right">351,538</td><td style="text-align:right">0.13×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (new)</td><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">1,837,814</td><td style="text-align:right">480,510</td><td style="text-align:right">0.26×</td></tr>
<tr><td style="text-align:left">Bn 2+2 (new)</td><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">117,277</td><td style="text-align:right">50,503</td><td style="text-align:right">0.43×</td></tr>
<tr><td style="text-align:left">Bn 4+2 (new)</td><td style="text-align:left"><code>lu</code></td><td style="text-align:right">134,372,041</td><td style="text-align:right">166,593</td><td style="text-align:right">0.00×</td></tr>
<tr><td style="text-align:left">Bn 4+2 (new)</td><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">34,571,610</td><td style="text-align:right">332,366</td><td style="text-align:right">0.01×</td></tr>
<tr><td style="text-align:left">Bn 4+2 (new)</td><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">209,921</td><td style="text-align:right">17,549,101</td><td style="text-align:right">83.60×</td></tr>
<tr><td style="text-align:left">Bn 4+2 (new)</td><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">43,701</td><td style="text-align:right">3,406,321</td><td style="text-align:right">77.95×</td></tr>
<tr><td style="text-align:left">Bn 4+2 (new)</td><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">1,141</td><td style="text-align:right">43,573</td><td style="text-align:right">38.19×</td></tr>
<tr><td style="text-align:left">Bn 4+2 (new)</td><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">94,585</td><td style="text-align:right">2,798,519</td><td style="text-align:right">29.59×</td></tr>
<tr><td style="text-align:left">Bn 4+2 (new)</td><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">10,753,631</td><td style="text-align:right">480,510</td><td style="text-align:right">0.04×</td></tr>
<tr><td style="text-align:left">Bn 4+2 (new)</td><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">112,062</td><td style="text-align:right">1,160,161</td><td style="text-align:right">10.35×</td></tr>
<tr><td style="text-align:left">Bn 4+2 (new)</td><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">355,153</td><td style="text-align:right">54,093</td><td style="text-align:right">0.15×</td></tr>
<tr><td style="text-align:left">Bn 4+2 (new)</td><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">294,388</td><td style="text-align:right">1,296,548</td><td style="text-align:right">4.40×</td></tr>
<tr><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">652,761</td><td style="text-align:right">40,130,103</td><td style="text-align:right">61.48×</td></tr>
<tr><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">16,667,641</td><td style="text-align:right">0.02×</td></tr>
<tr><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">1,539,761</td><td style="text-align:right">33,807,761</td><td style="text-align:right">21.96×</td></tr>
<tr><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">114,328</td><td style="text-align:right">2,003,313</td><td style="text-align:right">17.52×</td></tr>
<tr><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">117,950,941</td><td style="text-align:right">7,621,741</td><td style="text-align:right">0.06×</td></tr>
<tr><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">2,850,840</td><td style="text-align:right">280,431</td><td style="text-align:right">0.10×</td></tr>
<tr><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">1,352,948</td><td style="text-align:right">201,476</td><td style="text-align:right">0.15×</td></tr>
<tr><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1,155,676</td><td style="text-align:right">217,525</td><td style="text-align:right">0.19×</td></tr>
<tr><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">1,377,145</td><td style="text-align:right">477,937</td><td style="text-align:right">0.35×</td></tr>
<tr><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">808,966</td><td style="text-align:right">368,166</td><td style="text-align:right">0.46×</td></tr>
<tr><td style="text-align:left">Bn 6+2 (new)</td><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">127,293</td><td style="text-align:right">58,814</td><td style="text-align:right">0.46×</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">9,070</td><td style="text-align:right">1,574,241</td><td style="text-align:right">173.57×</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">5,896,992</td><td style="text-align:right">0.01×</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>atax</code></td><td style="text-align:right">5,046</td><td style="text-align:right">238,548</td><td style="text-align:right">47.27×</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">6,557,315</td><td style="text-align:right">189,651</td><td style="text-align:right">0.03×</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">2,013,441</td><td style="text-align:right">63,850</td><td style="text-align:right">0.03×</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">2,113,550</td><td style="text-align:right">105,236</td><td style="text-align:right">0.05×</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">125,376</td><td style="text-align:right">1,909,201</td><td style="text-align:right">15.23×</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">556,551</td><td style="text-align:right">42,169</td><td style="text-align:right">0.08×</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">4,793</td><td style="text-align:right">63,182</td><td style="text-align:right">13.18×</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">5,443</td><td style="text-align:right">71,291</td><td style="text-align:right">13.10×</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">243,411</td><td style="text-align:right">2,288,829</td><td style="text-align:right">9.40×</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>lu</code></td><td style="text-align:right">4,823,953</td><td style="text-align:right">772,465</td><td style="text-align:right">0.16×</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">429,902</td><td style="text-align:right">2,574,066</td><td style="text-align:right">5.99×</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">565,921</td><td style="text-align:right">121,461</td><td style="text-align:right">0.21×</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">376,177</td><td style="text-align:right">1,307,057</td><td style="text-align:right">3.47×</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">110,530</td><td style="text-align:right">32,482</td><td style="text-align:right">0.29×</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">18,499</td><td style="text-align:right">43,701</td><td style="text-align:right">2.36×</td></tr>
<tr><td style="text-align:left">All+avoids (new)</td><td style="text-align:left"><code>symm</code></td><td style="text-align:right">5,323,907</td><td style="text-align:right">2,558,081</td><td style="text-align:right">0.48×</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">5,896,992</td><td style="text-align:right">0.01×</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">1,979,281</td><td style="text-align:right">27,072</td><td style="text-align:right">0.01×</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">132,556,483</td><td style="text-align:right">2,147,841</td><td style="text-align:right">0.02×</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">1,819,720</td><td style="text-align:right">64,118,323</td><td style="text-align:right">35.24×</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">8,776</td><td style="text-align:right">256,922</td><td style="text-align:right">29.28×</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">884,533</td><td style="text-align:right">31,117</td><td style="text-align:right">0.04×</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">6,958,389</td><td style="text-align:right">294,771</td><td style="text-align:right">0.04×</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:left"><code>atax</code></td><td style="text-align:right">5,046</td><td style="text-align:right">107,225</td><td style="text-align:right">21.25×</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">2,720,629</td><td style="text-align:right">22,598,401</td><td style="text-align:right">8.31×</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">454,508</td><td style="text-align:right">64,770</td><td style="text-align:right">0.14×</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">105,635</td><td style="text-align:right">677,058</td><td style="text-align:right">6.41×</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">336,393</td><td style="text-align:right">1,320,305</td><td style="text-align:right">3.92×</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">54,836</td><td style="text-align:right">129,971</td><td style="text-align:right">2.37×</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">145,081</td><td style="text-align:right">337,609</td><td style="text-align:right">2.33×</td></tr>
<tr><td style="text-align:left">No avoids (new)</td><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">601,662</td><td style="text-align:right">1,215,001</td><td style="text-align:right">2.02×</td></tr>
</tbody></table>

## Conclusions

1. **Success is stable:** every mode scores **27/28** in both rounds (`doitgen` fails consistently).
2. **Head-to-head latency:** Round 2 wins **122** bench comparisons, Round 1 wins **110**, ties **38** — roughly even overall.
3. **High variance:** **156** of 270 bench×mode pairs differ by **&gt;50%** between rounds on the same configuration. Only a handful of benches are within **1%** across repeats.
4. **vs ground truth:** geo-mean rankings shift between rounds (e.g. No avoids legacy was best in R1; No avoids **new** is best in R2). All modes remain far below GT on most benches.
5. **Interpretation:** Round-to-round differences are dominated by **LLM sampling noise**, not by skills-file changes (each paired comparison holds skills constant). Use **multiple runs** or aggregation before drawing strong conclusions about skills.

See also: `artifacts/pc2/flash_comparison_20260621.md` (Round 1 legacy vs new), `artifacts/pc2/flash_comparison_20260620.md` (Round 1 legacy only).

