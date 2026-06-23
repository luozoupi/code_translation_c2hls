# Flash HLSFactory Results — Full Comparison

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
<colgroup>
  <col style="width:28%">
  <col style="width:72%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Field</th>
  <th style="text-align:left">Value</th>
</tr></thead>
<tbody>
<tr>
  <td style="text-align:left">Noskills / Bn skills stamp</td>
  <td style="text-align:left"><code>20260620_004507</code></td>
</tr>
<tr>
  <td style="text-align:left">Global skills stamp</td>
  <td style="text-align:left"><code>20260620_113247</code></td>
</tr>
<tr>
  <td style="text-align:left">Metric</td>
  <td style="text-align:left">Final flash-step synthesis latency (cycles), lower is better</td>
</tr>
<tr>
  <td style="text-align:left">Success</td>
  <td style="text-align:left">27/28 per mode (<code>doitgen</code> fails gold ref gate)</td>
</tr>
<tr>
  <td style="text-align:left">Session wall time (avoids)</td>
  <td style="text-align:left">~2h37m</td>
</tr>
<tr>
  <td style="text-align:left">Session wall time (no avoids)</td>
  <td style="text-align:left">~1h26m</td>
</tr>
</tbody></table>

## Summary

<table class="flash-cmp">
<colgroup>
  <col style="width:14%">
  <col style="width:52%">
  <col style="width:8%">
  <col style="width:12%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Mode</th>
  <th style="text-align:left">Artifact root</th>
  <th style="text-align:right">OK</th>
  <th style="text-align:right">Best latency</th>
</tr></thead>
<tbody>
<tr>
  <td style="text-align:left">Noskills</td>
  <td style="text-align:left"><code>flash_noskills_20260620_004507</code></td>
  <td style="text-align:right">27/28</td>
  <td style="text-align:right">7/27</td>
</tr>
<tr>
  <td style="text-align:left">Bn skills</td>
  <td style="text-align:left"><code>flash_skills_20260620_004507</code></td>
  <td style="text-align:right">27/28</td>
  <td style="text-align:right">6/27</td>
</tr>
<tr>
  <td style="text-align:left">All+avoids</td>
  <td style="text-align:left"><code>flash_all_skills_avoids_global_20260620_113247</code></td>
  <td style="text-align:right">27/28</td>
  <td style="text-align:right">7/27</td>
</tr>
<tr>
  <td style="text-align:left">No avoids</td>
  <td style="text-align:left"><code>flash_all_skills_no_avoids_global_20260620_113247</code></td>
  <td style="text-align:right">27/28</td>
  <td style="text-align:right">12/27</td>
</tr>
</tbody></table>

### Head-to-head vs noskills

<table class="flash-cmp">
<colgroup>
  <col style="width:30%">
  <col style="width:20%">
  <col style="width:22%">
  <col style="width:12%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Opponent</th>
  <th style="text-align:right">Opponent wins</th>
  <th style="text-align:right">Noskills wins</th>
  <th style="text-align:right">Ties</th>
</tr></thead>
<tbody>
<tr>
  <td style="text-align:left">Bn skills</td>
  <td style="text-align:right">9</td>
  <td style="text-align:right">12</td>
  <td style="text-align:right">6</td>
</tr>
<tr>
  <td style="text-align:left">All+avoids</td>
  <td style="text-align:right">10</td>
  <td style="text-align:right">16</td>
  <td style="text-align:right">1</td>
</tr>
<tr>
  <td style="text-align:left">No avoids</td>
  <td style="text-align:right">18</td>
  <td style="text-align:right">9</td>
  <td style="text-align:right">0</td>
</tr>
</tbody></table>

## Latency (cycles)

<table class="flash-cmp">
<colgroup>
  <col style="width:11%">
  <col style="width:13%">
  <col style="width:13%">
  <col style="width:13%">
  <col style="width:13%">
  <col style="width:13%">
  <col style="width:14%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Baseline</th>
  <th style="text-align:right">Noskills</th>
  <th style="text-align:right">Bn skills</th>
  <th style="text-align:right">All+avoids</th>
  <th style="text-align:right">No avoids</th>
  <th style="text-align:right">Winner</th>
</tr></thead>
<tbody>
<tr>
  <td style="text-align:left"><code>2mm</code></td>
  <td style="text-align:right">25,296,077</td>
  <td style="text-align:right">64,530</td>
  <td style="text-align:right">123,785</td>
  <td style="text-align:right">64,530</td>
  <td style="text-align:right">62,489</td>
  <td style="text-align:right"><strong>no_av</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>3mm</code></td>
  <td style="text-align:right">45,441,119</td>
  <td style="text-align:right">99,467</td>
  <td style="text-align:right">99,467</td>
  <td style="text-align:right">677,058</td>
  <td style="text-align:right">45,441,119</td>
  <td style="text-align:right"><strong>nosk+bn_sk</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>atax</code></td>
  <td style="text-align:right">2,429,238</td>
  <td style="text-align:right">118,344</td>
  <td style="text-align:right">2,429,238</td>
  <td style="text-align:right">2,300,710</td>
  <td style="text-align:right">5,046</td>
  <td style="text-align:right"><strong>no_av</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>bicg</code></td>
  <td style="text-align:right">2,343,914</td>
  <td style="text-align:right">727,515</td>
  <td style="text-align:right">2,297,693</td>
  <td style="text-align:right">810,190</td>
  <td style="text-align:right">5,316</td>
  <td style="text-align:right"><strong>no_av</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>cholesky</code></td>
  <td style="text-align:right">68,337,001</td>
  <td style="text-align:right">3,980,432</td>
  <td style="text-align:right">3,988,492</td>
  <td style="text-align:right">68,337,001</td>
  <td style="text-align:right">1,340,865</td>
  <td style="text-align:right"><strong>no_av</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>correlation</code></td>
  <td style="text-align:right">53,906,591</td>
  <td style="text-align:right">1,821,943</td>
  <td style="text-align:right">1,837,814</td>
  <td style="text-align:right">9,438,432</td>
  <td style="text-align:right">8,030,247</td>
  <td style="text-align:right"><strong>nosk</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>covariance</code></td>
  <td style="text-align:right">64,118,323</td>
  <td style="text-align:right">3,786,109</td>
  <td style="text-align:right">2,730,577</td>
  <td style="text-align:right">64,118,323</td>
  <td style="text-align:right">1,263,645</td>
  <td style="text-align:right"><strong>no_av</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>doitgen</code></td>
  <td style="text-align:right">—</td>
  <td style="text-align:right">FAIL</td>
  <td style="text-align:right">FAIL</td>
  <td style="text-align:right">FAIL</td>
  <td style="text-align:right">FAIL</td>
  <td style="text-align:right">—</td>
</tr>
<tr>
  <td style="text-align:left"><code>durbin</code></td>
  <td style="text-align:right">115,097</td>
  <td style="text-align:right">100,817</td>
  <td style="text-align:right">102,917</td>
  <td style="text-align:right">115,097</td>
  <td style="text-align:right">108,790</td>
  <td style="text-align:right"><strong>nosk</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>fdtd-2d</code></td>
  <td style="text-align:right">98,261,801</td>
  <td style="text-align:right">601,662</td>
  <td style="text-align:right">598,141</td>
  <td style="text-align:right">18,499</td>
  <td style="text-align:right">18,499</td>
  <td style="text-align:right"><strong>all_av+no_av</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>floyd-warshall</code></td>
  <td style="text-align:right">833,976,005</td>
  <td style="text-align:right">28,051,921</td>
  <td style="text-align:right">11,728,993</td>
  <td style="text-align:right">482,436,001</td>
  <td style="text-align:right">5,896,958</td>
  <td style="text-align:right"><strong>no_av</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>gemm</code></td>
  <td style="text-align:right">54,298,622</td>
  <td style="text-align:right">49,181</td>
  <td style="text-align:right">49,181</td>
  <td style="text-align:right">3,101,758</td>
  <td style="text-align:right">61,693</td>
  <td style="text-align:right"><strong>nosk+bn_sk</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>gemver</code></td>
  <td style="text-align:right">2,715,487</td>
  <td style="text-align:right">318,159</td>
  <td style="text-align:right">214,907</td>
  <td style="text-align:right">254,407</td>
  <td style="text-align:right">53,367</td>
  <td style="text-align:right"><strong>no_av</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>gesummv</code></td>
  <td style="text-align:right">1,441,873</td>
  <td style="text-align:right">9,060</td>
  <td style="text-align:right">1,141</td>
  <td style="text-align:right">26,975</td>
  <td style="text-align:right">18,055</td>
  <td style="text-align:right"><strong>bn_sk</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>gramschmidt</code></td>
  <td style="text-align:right">2,270,081</td>
  <td style="text-align:right">1,296,548</td>
  <td style="text-align:right">1,296,548</td>
  <td style="text-align:right">2,270,081</td>
  <td style="text-align:right">2,270,081</td>
  <td style="text-align:right"><strong>nosk+bn_sk</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>heat-3d</code></td>
  <td style="text-align:right">3,281,841</td>
  <td style="text-align:right">48,677</td>
  <td style="text-align:right">47,686</td>
  <td style="text-align:right">35,682</td>
  <td style="text-align:right">3,281,841</td>
  <td style="text-align:right"><strong>all_av</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>jacobi-1d</code></td>
  <td style="text-align:right">41,761</td>
  <td style="text-align:right">13,268</td>
  <td style="text-align:right">13,268</td>
  <td style="text-align:right">12,231</td>
  <td style="text-align:right">12,468</td>
  <td style="text-align:right"><strong>all_av</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>jacobi-2d</code></td>
  <td style="text-align:right">3,112,241</td>
  <td style="text-align:right">27,889</td>
  <td style="text-align:right">27,889</td>
  <td style="text-align:right">2,461,081</td>
  <td style="text-align:right">27,921</td>
  <td style="text-align:right"><strong>nosk+bn_sk</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>lu</code></td>
  <td style="text-align:right">134,372,041</td>
  <td style="text-align:right">15,614,072</td>
  <td style="text-align:right">17,192,281</td>
  <td style="text-align:right">5,062,065</td>
  <td style="text-align:right">5,672,817</td>
  <td style="text-align:right"><strong>all_av</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>ludcmp</code></td>
  <td style="text-align:right">10,095,723</td>
  <td style="text-align:right">10,095,723</td>
  <td style="text-align:right">6,672,294</td>
  <td style="text-align:right">1,925,435</td>
  <td style="text-align:right">732,587</td>
  <td style="text-align:right"><strong>no_av</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>mvt</code></td>
  <td style="text-align:right">256,922</td>
  <td style="text-align:right">210,483</td>
  <td style="text-align:right">210,483</td>
  <td style="text-align:right">48,054</td>
  <td style="text-align:right">199,322</td>
  <td style="text-align:right"><strong>all_av</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>nussinov</code></td>
  <td style="text-align:right">209,526,121</td>
  <td style="text-align:right">209,526,121</td>
  <td style="text-align:right">118,500,556</td>
  <td style="text-align:right">116,875,261</td>
  <td style="text-align:right">118,560,241</td>
  <td style="text-align:right"><strong>all_av</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>seidel-2d</code></td>
  <td style="text-align:right">132,556,483</td>
  <td style="text-align:right">586,042</td>
  <td style="text-align:right">34,560,515</td>
  <td style="text-align:right">38,071,522</td>
  <td style="text-align:right">1,119,361</td>
  <td style="text-align:right"><strong>nosk</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>symm</code></td>
  <td style="text-align:right">23,241,675</td>
  <td style="text-align:right">2,864,058</td>
  <td style="text-align:right">2,864,134</td>
  <td style="text-align:right">2,855,106</td>
  <td style="text-align:right">1,251,830</td>
  <td style="text-align:right"><strong>no_av</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>syr2k</code></td>
  <td style="text-align:right">33,807,761</td>
  <td style="text-align:right">992,061</td>
  <td style="text-align:right">7,878,481</td>
  <td style="text-align:right">6,979,669</td>
  <td style="text-align:right">83,188</td>
  <td style="text-align:right"><strong>no_av</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>syrk</code></td>
  <td style="text-align:right">31,695,761</td>
  <td style="text-align:right">1,577,981</td>
  <td style="text-align:right">119,181</td>
  <td style="text-align:right">31,695,761</td>
  <td style="text-align:right">228,631</td>
  <td style="text-align:right"><strong>bn_sk</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>trisolv</code></td>
  <td style="text-align:right">1,160,161</td>
  <td style="text-align:right">139,301</td>
  <td style="text-align:right">678,122</td>
  <td style="text-align:right">22,113</td>
  <td style="text-align:right">40,805</td>
  <td style="text-align:right"><strong>all_av</strong></td>
</tr>
<tr>
  <td style="text-align:left"><code>trmm</code></td>
  <td style="text-align:right">22,598,401</td>
  <td style="text-align:right">2,759,030</td>
  <td style="text-align:right">2,768,058</td>
  <td style="text-align:right">6,747,057</td>
  <td style="text-align:right">260,875</td>
  <td style="text-align:right"><strong>no_av</strong></td>
</tr>
</tbody></table>

## Latency ratio vs noskills (opponent ÷ noskills; &lt;1 = faster)

<table class="flash-cmp">
<colgroup>
  <col style="width:22%">
  <col style="width:26%">
  <col style="width:26%">
  <col style="width:26%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Bn skills</th>
  <th style="text-align:right">All+avoids</th>
  <th style="text-align:right">No avoids</th>
</tr></thead>
<tbody>
<tr>
  <td style="text-align:left"><code>2mm</code></td>
  <td style="text-align:right">1.918</td>
  <td style="text-align:right">1.000</td>
  <td style="text-align:right">0.968</td>
</tr>
<tr>
  <td style="text-align:left"><code>3mm</code></td>
  <td style="text-align:right">1.000</td>
  <td style="text-align:right">6.807</td>
  <td style="text-align:right">456.846</td>
</tr>
<tr>
  <td style="text-align:left"><code>atax</code></td>
  <td style="text-align:right">20.527</td>
  <td style="text-align:right">19.441</td>
  <td style="text-align:right">0.043</td>
</tr>
<tr>
  <td style="text-align:left"><code>bicg</code></td>
  <td style="text-align:right">3.158</td>
  <td style="text-align:right">1.114</td>
  <td style="text-align:right">0.007</td>
</tr>
<tr>
  <td style="text-align:left"><code>cholesky</code></td>
  <td style="text-align:right">1.002</td>
  <td style="text-align:right">17.168</td>
  <td style="text-align:right">0.337</td>
</tr>
<tr>
  <td style="text-align:left"><code>correlation</code></td>
  <td style="text-align:right">1.009</td>
  <td style="text-align:right">5.180</td>
  <td style="text-align:right">4.408</td>
</tr>
<tr>
  <td style="text-align:left"><code>covariance</code></td>
  <td style="text-align:right">0.721</td>
  <td style="text-align:right">16.935</td>
  <td style="text-align:right">0.334</td>
</tr>
<tr>
  <td style="text-align:left"><code>durbin</code></td>
  <td style="text-align:right">1.021</td>
  <td style="text-align:right">1.142</td>
  <td style="text-align:right">1.079</td>
</tr>
<tr>
  <td style="text-align:left"><code>fdtd-2d</code></td>
  <td style="text-align:right">0.994</td>
  <td style="text-align:right">0.031</td>
  <td style="text-align:right">0.031</td>
</tr>
<tr>
  <td style="text-align:left"><code>floyd-warshall</code></td>
  <td style="text-align:right">0.418</td>
  <td style="text-align:right">17.198</td>
  <td style="text-align:right">0.210</td>
</tr>
<tr>
  <td style="text-align:left"><code>gemm</code></td>
  <td style="text-align:right">1.000</td>
  <td style="text-align:right">63.068</td>
  <td style="text-align:right">1.254</td>
</tr>
<tr>
  <td style="text-align:left"><code>gemver</code></td>
  <td style="text-align:right">0.675</td>
  <td style="text-align:right">0.800</td>
  <td style="text-align:right">0.168</td>
</tr>
<tr>
  <td style="text-align:left"><code>gesummv</code></td>
  <td style="text-align:right">0.126</td>
  <td style="text-align:right">2.977</td>
  <td style="text-align:right">1.993</td>
</tr>
<tr>
  <td style="text-align:left"><code>gramschmidt</code></td>
  <td style="text-align:right">1.000</td>
  <td style="text-align:right">1.751</td>
  <td style="text-align:right">1.751</td>
</tr>
<tr>
  <td style="text-align:left"><code>heat-3d</code></td>
  <td style="text-align:right">0.980</td>
  <td style="text-align:right">0.733</td>
  <td style="text-align:right">67.421</td>
</tr>
<tr>
  <td style="text-align:left"><code>jacobi-1d</code></td>
  <td style="text-align:right">1.000</td>
  <td style="text-align:right">0.922</td>
  <td style="text-align:right">0.940</td>
</tr>
<tr>
  <td style="text-align:left"><code>jacobi-2d</code></td>
  <td style="text-align:right">1.000</td>
  <td style="text-align:right">88.246</td>
  <td style="text-align:right">1.001</td>
</tr>
<tr>
  <td style="text-align:left"><code>lu</code></td>
  <td style="text-align:right">1.101</td>
  <td style="text-align:right">0.324</td>
  <td style="text-align:right">0.363</td>
</tr>
<tr>
  <td style="text-align:left"><code>ludcmp</code></td>
  <td style="text-align:right">0.661</td>
  <td style="text-align:right">0.191</td>
  <td style="text-align:right">0.073</td>
</tr>
<tr>
  <td style="text-align:left"><code>mvt</code></td>
  <td style="text-align:right">1.000</td>
  <td style="text-align:right">0.228</td>
  <td style="text-align:right">0.947</td>
</tr>
<tr>
  <td style="text-align:left"><code>nussinov</code></td>
  <td style="text-align:right">0.566</td>
  <td style="text-align:right">0.558</td>
  <td style="text-align:right">0.566</td>
</tr>
<tr>
  <td style="text-align:left"><code>seidel-2d</code></td>
  <td style="text-align:right">58.973</td>
  <td style="text-align:right">64.964</td>
  <td style="text-align:right">1.910</td>
</tr>
<tr>
  <td style="text-align:left"><code>symm</code></td>
  <td style="text-align:right">1.000</td>
  <td style="text-align:right">0.997</td>
  <td style="text-align:right">0.437</td>
</tr>
<tr>
  <td style="text-align:left"><code>syr2k</code></td>
  <td style="text-align:right">7.942</td>
  <td style="text-align:right">7.036</td>
  <td style="text-align:right">0.084</td>
</tr>
<tr>
  <td style="text-align:left"><code>syrk</code></td>
  <td style="text-align:right">0.076</td>
  <td style="text-align:right">20.086</td>
  <td style="text-align:right">0.145</td>
</tr>
<tr>
  <td style="text-align:left"><code>trisolv</code></td>
  <td style="text-align:right">4.868</td>
  <td style="text-align:right">0.159</td>
  <td style="text-align:right">0.293</td>
</tr>
<tr>
  <td style="text-align:left"><code>trmm</code></td>
  <td style="text-align:right">1.003</td>
  <td style="text-align:right">2.445</td>
  <td style="text-align:right">0.095</td>
</tr>
</tbody></table>

## Fmax (MHz)

<table class="flash-cmp">
<colgroup>
  <col style="width:18%">
  <col style="width:20.5%">
  <col style="width:20.5%">
  <col style="width:20.5%">
  <col style="width:20.5%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Noskills</th>
  <th style="text-align:right">Bn skills</th>
  <th style="text-align:right">All+avoids</th>
  <th style="text-align:right">No avoids</th>
</tr></thead>
<tbody>
<tr>
  <td style="text-align:left"><code>2mm</code></td>
  <td style="text-align:right">360.1</td>
  <td style="text-align:right">388.4</td>
  <td style="text-align:right">360.1</td>
  <td style="text-align:right">401.0</td>
</tr>
<tr>
  <td style="text-align:left"><code>3mm</code></td>
  <td style="text-align:right">401.0</td>
  <td style="text-align:right">401.0</td>
  <td style="text-align:right">341.9</td>
  <td style="text-align:right">341.9</td>
</tr>
<tr>
  <td style="text-align:left"><code>atax</code></td>
  <td style="text-align:right">232.8</td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">344.1</td>
</tr>
<tr>
  <td style="text-align:left"><code>bicg</code></td>
  <td style="text-align:right">384.5</td>
  <td style="text-align:right">355.6</td>
  <td style="text-align:right">385.4</td>
  <td style="text-align:right">348.9</td>
</tr>
<tr>
  <td style="text-align:left"><code>cholesky</code></td>
  <td style="text-align:right">401.0</td>
  <td style="text-align:right">382.4</td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">403.2</td>
</tr>
<tr>
  <td style="text-align:left"><code>correlation</code></td>
  <td style="text-align:right">170.0</td>
  <td style="text-align:right">170.0</td>
  <td style="text-align:right">170.0</td>
  <td style="text-align:right">169.9</td>
</tr>
<tr>
  <td style="text-align:left"><code>covariance</code></td>
  <td style="text-align:right">330.1</td>
  <td style="text-align:right">382.4</td>
  <td style="text-align:right">405.2</td>
  <td style="text-align:right">340.8</td>
</tr>
<tr>
  <td style="text-align:left"><code>doitgen</code></td>
  <td style="text-align:right">—</td>
  <td style="text-align:right">—</td>
  <td style="text-align:right">—</td>
  <td style="text-align:right">—</td>
</tr>
<tr>
  <td style="text-align:left"><code>durbin</code></td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">411.4</td>
</tr>
<tr>
  <td style="text-align:left"><code>fdtd-2d</code></td>
  <td style="text-align:right">401.0</td>
  <td style="text-align:right">384.5</td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">411.4</td>
</tr>
<tr>
  <td style="text-align:left"><code>floyd-warshall</code></td>
  <td style="text-align:right">403.2</td>
  <td style="text-align:right">188.8</td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">385.9</td>
</tr>
<tr>
  <td style="text-align:left"><code>gemm</code></td>
  <td style="text-align:right">382.6</td>
  <td style="text-align:right">382.6</td>
  <td style="text-align:right">361.4</td>
  <td style="text-align:right">401.0</td>
</tr>
<tr>
  <td style="text-align:left"><code>gemver</code></td>
  <td style="text-align:right">171.0</td>
  <td style="text-align:right">324.7</td>
  <td style="text-align:right">181.8</td>
  <td style="text-align:right">346.5</td>
</tr>
<tr>
  <td style="text-align:left"><code>gesummv</code></td>
  <td style="text-align:right">341.8</td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">329.3</td>
  <td style="text-align:right">411.4</td>
</tr>
<tr>
  <td style="text-align:left"><code>gramschmidt</code></td>
  <td style="text-align:right">401.0</td>
  <td style="text-align:right">401.0</td>
  <td style="text-align:right">339.6</td>
  <td style="text-align:right">339.6</td>
</tr>
<tr>
  <td style="text-align:left"><code>heat-3d</code></td>
  <td style="text-align:right">405.5</td>
  <td style="text-align:right">378.5</td>
  <td style="text-align:right">378.5</td>
  <td style="text-align:right">411.4</td>
</tr>
<tr>
  <td style="text-align:left"><code>jacobi-1d</code></td>
  <td style="text-align:right">391.7</td>
  <td style="text-align:right">391.7</td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">411.4</td>
</tr>
<tr>
  <td style="text-align:left"><code>jacobi-2d</code></td>
  <td style="text-align:right">382.4</td>
  <td style="text-align:right">382.4</td>
  <td style="text-align:right">403.2</td>
  <td style="text-align:right">376.8</td>
</tr>
<tr>
  <td style="text-align:left"><code>lu</code></td>
  <td style="text-align:right">384.5</td>
  <td style="text-align:right">358.8</td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">403.2</td>
</tr>
<tr>
  <td style="text-align:left"><code>ludcmp</code></td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">382.4</td>
  <td style="text-align:right">403.2</td>
  <td style="text-align:right">403.2</td>
</tr>
<tr>
  <td style="text-align:left"><code>mvt</code></td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">232.8</td>
  <td style="text-align:right">411.4</td>
</tr>
<tr>
  <td style="text-align:left"><code>nussinov</code></td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">402.1</td>
</tr>
<tr>
  <td style="text-align:left"><code>seidel-2d</code></td>
  <td style="text-align:right">382.4</td>
  <td style="text-align:right">205.4</td>
  <td style="text-align:right">333.6</td>
  <td style="text-align:right">363.9</td>
</tr>
<tr>
  <td style="text-align:left"><code>symm</code></td>
  <td style="text-align:right">379.6</td>
  <td style="text-align:right">379.6</td>
  <td style="text-align:right">379.6</td>
  <td style="text-align:right">360.1</td>
</tr>
<tr>
  <td style="text-align:left"><code>syr2k</code></td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">379.1</td>
  <td style="text-align:right">403.2</td>
</tr>
<tr>
  <td style="text-align:left"><code>syrk</code></td>
  <td style="text-align:right">401.0</td>
  <td style="text-align:right">401.0</td>
  <td style="text-align:right">392.5</td>
  <td style="text-align:right">401.0</td>
</tr>
<tr>
  <td style="text-align:left"><code>trisolv</code></td>
  <td style="text-align:right">384.3</td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">411.4</td>
</tr>
<tr>
  <td style="text-align:left"><code>trmm</code></td>
  <td style="text-align:right">401.0</td>
  <td style="text-align:right">401.0</td>
  <td style="text-align:right">411.4</td>
  <td style="text-align:right">368.7</td>
</tr>
</tbody></table>

## BRAM

<table class="flash-cmp">
<colgroup>
  <col style="width:18%">
  <col style="width:20.5%">
  <col style="width:20.5%">
  <col style="width:20.5%">
  <col style="width:20.5%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Noskills</th>
  <th style="text-align:right">Bn skills</th>
  <th style="text-align:right">All+avoids</th>
  <th style="text-align:right">No avoids</th>
</tr></thead>
<tbody>
<tr>
  <td style="text-align:left"><code>2mm</code></td>
  <td style="text-align:right">130</td>
  <td style="text-align:right">130</td>
  <td style="text-align:right">130</td>
  <td style="text-align:right">78</td>
</tr>
<tr>
  <td style="text-align:left"><code>3mm</code></td>
  <td style="text-align:right">290</td>
  <td style="text-align:right">290</td>
  <td style="text-align:right">30</td>
  <td style="text-align:right">30</td>
</tr>
<tr>
  <td style="text-align:left"><code>atax</code></td>
  <td style="text-align:right">16</td>
  <td style="text-align:right">16</td>
  <td style="text-align:right">16</td>
  <td style="text-align:right">16</td>
</tr>
<tr>
  <td style="text-align:left"><code>bicg</code></td>
  <td style="text-align:right">26</td>
  <td style="text-align:right">34</td>
  <td style="text-align:right">16</td>
  <td style="text-align:right">16</td>
</tr>
<tr>
  <td style="text-align:left"><code>cholesky</code></td>
  <td style="text-align:right">30</td>
  <td style="text-align:right">158</td>
  <td style="text-align:right">4</td>
  <td style="text-align:right">4</td>
</tr>
<tr>
  <td style="text-align:left"><code>correlation</code></td>
  <td style="text-align:right">34</td>
  <td style="text-align:right">30</td>
  <td style="text-align:right">30</td>
  <td style="text-align:right">30</td>
</tr>
<tr>
  <td style="text-align:left"><code>covariance</code></td>
  <td style="text-align:right">80</td>
  <td style="text-align:right">62</td>
  <td style="text-align:right">30</td>
  <td style="text-align:right">30</td>
</tr>
<tr>
  <td style="text-align:left"><code>doitgen</code></td>
  <td style="text-align:right">—</td>
  <td style="text-align:right">—</td>
  <td style="text-align:right">—</td>
  <td style="text-align:right">—</td>
</tr>
<tr>
  <td style="text-align:left"><code>durbin</code></td>
  <td style="text-align:right">6</td>
  <td style="text-align:right">34</td>
  <td style="text-align:right">6</td>
  <td style="text-align:right">6</td>
</tr>
<tr>
  <td style="text-align:left"><code>fdtd-2d</code></td>
  <td style="text-align:right">126</td>
  <td style="text-align:right">390</td>
  <td style="text-align:right">30</td>
  <td style="text-align:right">30</td>
</tr>
<tr>
  <td style="text-align:left"><code>floyd-warshall</code></td>
  <td style="text-align:right">2</td>
  <td style="text-align:right">72</td>
  <td style="text-align:right">8</td>
  <td style="text-align:right">368</td>
</tr>
<tr>
  <td style="text-align:left"><code>gemm</code></td>
  <td style="text-align:right">542</td>
  <td style="text-align:right">542</td>
  <td style="text-align:right">30</td>
  <td style="text-align:right">254</td>
</tr>
<tr>
  <td style="text-align:left"><code>gemver</code></td>
  <td style="text-align:right">270</td>
  <td style="text-align:right">102</td>
  <td style="text-align:right">30</td>
  <td style="text-align:right">30</td>
</tr>
<tr>
  <td style="text-align:left"><code>gesummv</code></td>
  <td style="text-align:right">8</td>
  <td style="text-align:right">720</td>
  <td style="text-align:right">36</td>
  <td style="text-align:right">8</td>
</tr>
<tr>
  <td style="text-align:left"><code>gramschmidt</code></td>
  <td style="text-align:right">126</td>
  <td style="text-align:right">126</td>
  <td style="text-align:right">4</td>
  <td style="text-align:right">4</td>
</tr>
<tr>
  <td style="text-align:left"><code>heat-3d</code></td>
  <td style="text-align:right">20</td>
  <td style="text-align:right">16</td>
  <td style="text-align:right">16</td>
  <td style="text-align:right">4</td>
</tr>
<tr>
  <td style="text-align:left"><code>jacobi-1d</code></td>
  <td style="text-align:right">30</td>
  <td style="text-align:right">30</td>
  <td style="text-align:right">30</td>
  <td style="text-align:right">30</td>
</tr>
<tr>
  <td style="text-align:left"><code>jacobi-2d</code></td>
  <td style="text-align:right">8</td>
  <td style="text-align:right">8</td>
  <td style="text-align:right">52</td>
  <td style="text-align:right">12</td>
</tr>
<tr>
  <td style="text-align:left"><code>lu</code></td>
  <td style="text-align:right">270</td>
  <td style="text-align:right">30</td>
  <td style="text-align:right">30</td>
  <td style="text-align:right">4</td>
</tr>
<tr>
  <td style="text-align:left"><code>ludcmp</code></td>
  <td style="text-align:right">30</td>
  <td style="text-align:right">158</td>
  <td style="text-align:right">30</td>
  <td style="text-align:right">4</td>
</tr>
<tr>
  <td style="text-align:left"><code>mvt</code></td>
  <td style="text-align:right">90</td>
  <td style="text-align:right">90</td>
  <td style="text-align:right">4</td>
  <td style="text-align:right">30</td>
</tr>
<tr>
  <td style="text-align:left"><code>nussinov</code></td>
  <td style="text-align:right">2</td>
  <td style="text-align:right">3</td>
  <td style="text-align:right">9</td>
  <td style="text-align:right">8</td>
</tr>
<tr>
  <td style="text-align:left"><code>seidel-2d</code></td>
  <td style="text-align:right">30</td>
  <td style="text-align:right">30</td>
  <td style="text-align:right">30</td>
  <td style="text-align:right">30</td>
</tr>
<tr>
  <td style="text-align:left"><code>symm</code></td>
  <td style="text-align:right">94</td>
  <td style="text-align:right">94</td>
  <td style="text-align:right">94</td>
  <td style="text-align:right">94</td>
</tr>
<tr>
  <td style="text-align:left"><code>syr2k</code></td>
  <td style="text-align:right">24</td>
  <td style="text-align:right">36</td>
  <td style="text-align:right">126</td>
  <td style="text-align:right">4</td>
</tr>
<tr>
  <td style="text-align:left"><code>syrk</code></td>
  <td style="text-align:right">36</td>
  <td style="text-align:right">36</td>
  <td style="text-align:right">4</td>
  <td style="text-align:right">126</td>
</tr>
<tr>
  <td style="text-align:left"><code>trisolv</code></td>
  <td style="text-align:right">3844</td>
  <td style="text-align:right">88</td>
  <td style="text-align:right">34</td>
  <td style="text-align:right">8</td>
</tr>
<tr>
  <td style="text-align:left"><code>trmm</code></td>
  <td style="text-align:right">62</td>
  <td style="text-align:right">62</td>
  <td style="text-align:right">46</td>
  <td style="text-align:right">4</td>
</tr>
</tbody></table>

## LUT

<table class="flash-cmp">
<colgroup>
  <col style="width:18%">
  <col style="width:20.5%">
  <col style="width:20.5%">
  <col style="width:20.5%">
  <col style="width:20.5%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Noskills</th>
  <th style="text-align:right">Bn skills</th>
  <th style="text-align:right">All+avoids</th>
  <th style="text-align:right">No avoids</th>
</tr></thead>
<tbody>
<tr>
  <td style="text-align:left"><code>2mm</code></td>
  <td style="text-align:right">84,437</td>
  <td style="text-align:right">143,088</td>
  <td style="text-align:right">84,437</td>
  <td style="text-align:right">179,720</td>
</tr>
<tr>
  <td style="text-align:left"><code>3mm</code></td>
  <td style="text-align:right">100,039</td>
  <td style="text-align:right">100,039</td>
  <td style="text-align:right">562,375</td>
  <td style="text-align:right">361,516</td>
</tr>
<tr>
  <td style="text-align:left"><code>atax</code></td>
  <td style="text-align:right">8,031</td>
  <td style="text-align:right">9,417</td>
  <td style="text-align:right">9,369</td>
  <td style="text-align:right">24,251</td>
</tr>
<tr>
  <td style="text-align:left"><code>bicg</code></td>
  <td style="text-align:right">8,077</td>
  <td style="text-align:right">22,128</td>
  <td style="text-align:right">25,296</td>
  <td style="text-align:right">23,609</td>
</tr>
<tr>
  <td style="text-align:left"><code>cholesky</code></td>
  <td style="text-align:right">89,570</td>
  <td style="text-align:right">25,904</td>
  <td style="text-align:right">6,864</td>
  <td style="text-align:right">13,431</td>
</tr>
<tr>
  <td style="text-align:left"><code>correlation</code></td>
  <td style="text-align:right">521,296</td>
  <td style="text-align:right">526,998</td>
  <td style="text-align:right">111,703</td>
  <td style="text-align:right">136,742</td>
</tr>
<tr>
  <td style="text-align:left"><code>covariance</code></td>
  <td style="text-align:right">23,484</td>
  <td style="text-align:right">18,932</td>
  <td style="text-align:right">28,663</td>
  <td style="text-align:right">365,560</td>
</tr>
<tr>
  <td style="text-align:left"><code>doitgen</code></td>
  <td style="text-align:right">—</td>
  <td style="text-align:right">—</td>
  <td style="text-align:right">—</td>
  <td style="text-align:right">—</td>
</tr>
<tr>
  <td style="text-align:left"><code>durbin</code></td>
  <td style="text-align:right">6,593</td>
  <td style="text-align:right">33,890</td>
  <td style="text-align:right">5,836</td>
  <td style="text-align:right">8,061</td>
</tr>
<tr>
  <td style="text-align:left"><code>fdtd-2d</code></td>
  <td style="text-align:right">13,242</td>
  <td style="text-align:right">42,762</td>
  <td style="text-align:right">463,270</td>
  <td style="text-align:right">463,270</td>
</tr>
<tr>
  <td style="text-align:left"><code>floyd-warshall</code></td>
  <td style="text-align:right">5,221</td>
  <td style="text-align:right">7,215</td>
  <td style="text-align:right">52,431</td>
  <td style="text-align:right">29,150</td>
</tr>
<tr>
  <td style="text-align:left"><code>gemm</code></td>
  <td style="text-align:right">34,489</td>
  <td style="text-align:right">34,489</td>
  <td style="text-align:right">14,993</td>
  <td style="text-align:right">82,281</td>
</tr>
<tr>
  <td style="text-align:left"><code>gemver</code></td>
  <td style="text-align:right">42,313</td>
  <td style="text-align:right">12,596</td>
  <td style="text-align:right">40,167</td>
  <td style="text-align:right">411,882</td>
</tr>
<tr>
  <td style="text-align:left"><code>gesummv</code></td>
  <td style="text-align:right">20,359</td>
  <td style="text-align:right">263,842</td>
  <td style="text-align:right">643,276</td>
  <td style="text-align:right">91,295</td>
</tr>
<tr>
  <td style="text-align:left"><code>gramschmidt</code></td>
  <td style="text-align:right">10,113</td>
  <td style="text-align:right">10,113</td>
  <td style="text-align:right">66,797</td>
  <td style="text-align:right">66,797</td>
</tr>
<tr>
  <td style="text-align:left"><code>heat-3d</code></td>
  <td style="text-align:right">13,854</td>
  <td style="text-align:right">263,375</td>
  <td style="text-align:right">262,919</td>
  <td style="text-align:right">7,944</td>
</tr>
<tr>
  <td style="text-align:left"><code>jacobi-1d</code></td>
  <td style="text-align:right">8,890</td>
  <td style="text-align:right">8,890</td>
  <td style="text-align:right">7,954</td>
  <td style="text-align:right">8,674</td>
</tr>
<tr>
  <td style="text-align:left"><code>jacobi-2d</code></td>
  <td style="text-align:right">360,705</td>
  <td style="text-align:right">467,128</td>
  <td style="text-align:right">12,664</td>
  <td style="text-align:right">79,896</td>
</tr>
<tr>
  <td style="text-align:left"><code>lu</code></td>
  <td style="text-align:right">23,592</td>
  <td style="text-align:right">9,219</td>
  <td style="text-align:right">83,541</td>
  <td style="text-align:right">24,795</td>
</tr>
<tr>
  <td style="text-align:left"><code>ludcmp</code></td>
  <td style="text-align:right">48,030</td>
  <td style="text-align:right">38,809</td>
  <td style="text-align:right">36,441</td>
  <td style="text-align:right">13,748</td>
</tr>
<tr>
  <td style="text-align:left"><code>mvt</code></td>
  <td style="text-align:right">16,763</td>
  <td style="text-align:right">16,763</td>
  <td style="text-align:right">36,736</td>
  <td style="text-align:right">18,670</td>
</tr>
<tr>
  <td style="text-align:left"><code>nussinov</code></td>
  <td style="text-align:right">5,879</td>
  <td style="text-align:right">5,250</td>
  <td style="text-align:right">7,995</td>
  <td style="text-align:right">17,964</td>
</tr>
<tr>
  <td style="text-align:left"><code>seidel-2d</code></td>
  <td style="text-align:right">16,294</td>
  <td style="text-align:right">12,258</td>
  <td style="text-align:right">282,829</td>
  <td style="text-align:right">108,410</td>
</tr>
<tr>
  <td style="text-align:left"><code>symm</code></td>
  <td style="text-align:right">10,255</td>
  <td style="text-align:right">11,027</td>
  <td style="text-align:right">11,588</td>
  <td style="text-align:right">10,775</td>
</tr>
<tr>
  <td style="text-align:left"><code>syr2k</code></td>
  <td style="text-align:right">8,524</td>
  <td style="text-align:right">374,943</td>
  <td style="text-align:right">43,287</td>
  <td style="text-align:right">109,327</td>
</tr>
<tr>
  <td style="text-align:left"><code>syrk</code></td>
  <td style="text-align:right">12,610</td>
  <td style="text-align:right">10,775</td>
  <td style="text-align:right">4,539</td>
  <td style="text-align:right">22,517</td>
</tr>
<tr>
  <td style="text-align:left"><code>trisolv</code></td>
  <td style="text-align:right">405,943</td>
  <td style="text-align:right">10,833</td>
  <td style="text-align:right">29,916</td>
  <td style="text-align:right">19,353</td>
</tr>
<tr>
  <td style="text-align:left"><code>trmm</code></td>
  <td style="text-align:right">10,199</td>
  <td style="text-align:right">9,633</td>
  <td style="text-align:right">51,155</td>
  <td style="text-align:right">12,872</td>
</tr>
</tbody></table>

## vs ground truth — latency ratio (generated ÷ gold)

<table class="flash-cmp">
<colgroup>
  <col style="width:18%">
  <col style="width:20.5%">
  <col style="width:20.5%">
  <col style="width:20.5%">
  <col style="width:20.5%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Benchmark</th>
  <th style="text-align:right">Noskills</th>
  <th style="text-align:right">Bn skills</th>
  <th style="text-align:right">All+avoids</th>
  <th style="text-align:right">No avoids</th>
</tr></thead>
<tbody>
<tr>
  <td style="text-align:left"><code>2mm</code></td>
  <td style="text-align:right">0.003</td>
  <td style="text-align:right">0.005</td>
  <td style="text-align:right">0.003</td>
  <td style="text-align:right">0.002</td>
</tr>
<tr>
  <td style="text-align:left"><code>3mm</code></td>
  <td style="text-align:right">0.002</td>
  <td style="text-align:right">0.002</td>
  <td style="text-align:right">0.015</td>
  <td style="text-align:right">—</td>
</tr>
<tr>
  <td style="text-align:left"><code>atax</code></td>
  <td style="text-align:right">0.049</td>
  <td style="text-align:right">—</td>
  <td style="text-align:right">0.947</td>
  <td style="text-align:right">0.002</td>
</tr>
<tr>
  <td style="text-align:left"><code>bicg</code></td>
  <td style="text-align:right">0.310</td>
  <td style="text-align:right">0.980</td>
  <td style="text-align:right">0.346</td>
  <td style="text-align:right">0.002</td>
</tr>
<tr>
  <td style="text-align:left"><code>cholesky</code></td>
  <td style="text-align:right">0.058</td>
  <td style="text-align:right">0.058</td>
  <td style="text-align:right">—</td>
  <td style="text-align:right">0.020</td>
</tr>
<tr>
  <td style="text-align:left"><code>correlation</code></td>
  <td style="text-align:right">0.034</td>
  <td style="text-align:right">0.034</td>
  <td style="text-align:right">0.175</td>
  <td style="text-align:right">0.149</td>
</tr>
<tr>
  <td style="text-align:left"><code>covariance</code></td>
  <td style="text-align:right">0.059</td>
  <td style="text-align:right">0.043</td>
  <td style="text-align:right">—</td>
  <td style="text-align:right">0.020</td>
</tr>
<tr>
  <td style="text-align:left"><code>durbin</code></td>
  <td style="text-align:right">0.876</td>
  <td style="text-align:right">0.894</td>
  <td style="text-align:right">—</td>
  <td style="text-align:right">0.945</td>
</tr>
<tr>
  <td style="text-align:left"><code>fdtd-2d</code></td>
  <td style="text-align:right">0.006</td>
  <td style="text-align:right">0.006</td>
  <td style="text-align:right">0.000</td>
  <td style="text-align:right">0.000</td>
</tr>
<tr>
  <td style="text-align:left"><code>floyd-warshall</code></td>
  <td style="text-align:right">0.034</td>
  <td style="text-align:right">0.014</td>
  <td style="text-align:right">0.578</td>
  <td style="text-align:right">0.007</td>
</tr>
<tr>
  <td style="text-align:left"><code>gemm</code></td>
  <td style="text-align:right">0.001</td>
  <td style="text-align:right">0.001</td>
  <td style="text-align:right">0.057</td>
  <td style="text-align:right">0.001</td>
</tr>
<tr>
  <td style="text-align:left"><code>gemver</code></td>
  <td style="text-align:right">0.117</td>
  <td style="text-align:right">0.079</td>
  <td style="text-align:right">0.094</td>
  <td style="text-align:right">0.020</td>
</tr>
<tr>
  <td style="text-align:left"><code>gesummv</code></td>
  <td style="text-align:right">0.006</td>
  <td style="text-align:right">0.001</td>
  <td style="text-align:right">0.019</td>
  <td style="text-align:right">0.013</td>
</tr>
<tr>
  <td style="text-align:left"><code>gramschmidt</code></td>
  <td style="text-align:right">0.571</td>
  <td style="text-align:right">0.571</td>
  <td style="text-align:right">—</td>
  <td style="text-align:right">—</td>
</tr>
<tr>
  <td style="text-align:left"><code>heat-3d</code></td>
  <td style="text-align:right">0.015</td>
  <td style="text-align:right">0.015</td>
  <td style="text-align:right">0.011</td>
  <td style="text-align:right">1.000</td>
</tr>
<tr>
  <td style="text-align:left"><code>jacobi-1d</code></td>
  <td style="text-align:right">0.318</td>
  <td style="text-align:right">0.318</td>
  <td style="text-align:right">0.293</td>
  <td style="text-align:right">0.299</td>
</tr>
<tr>
  <td style="text-align:left"><code>jacobi-2d</code></td>
  <td style="text-align:right">0.009</td>
  <td style="text-align:right">0.009</td>
  <td style="text-align:right">0.791</td>
  <td style="text-align:right">0.009</td>
</tr>
<tr>
  <td style="text-align:left"><code>lu</code></td>
  <td style="text-align:right">0.116</td>
  <td style="text-align:right">0.128</td>
  <td style="text-align:right">0.038</td>
  <td style="text-align:right">0.042</td>
</tr>
<tr>
  <td style="text-align:left"><code>ludcmp</code></td>
  <td style="text-align:right">—</td>
  <td style="text-align:right">0.661</td>
  <td style="text-align:right">0.191</td>
  <td style="text-align:right">0.073</td>
</tr>
<tr>
  <td style="text-align:left"><code>mvt</code></td>
  <td style="text-align:right">0.819</td>
  <td style="text-align:right">0.819</td>
  <td style="text-align:right">0.187</td>
  <td style="text-align:right">0.776</td>
</tr>
<tr>
  <td style="text-align:left"><code>nussinov</code></td>
  <td style="text-align:right">1.025</td>
  <td style="text-align:right">0.566</td>
  <td style="text-align:right">0.558</td>
  <td style="text-align:right">0.566</td>
</tr>
<tr>
  <td style="text-align:left"><code>seidel-2d</code></td>
  <td style="text-align:right">0.004</td>
  <td style="text-align:right">0.261</td>
  <td style="text-align:right">0.287</td>
  <td style="text-align:right">0.008</td>
</tr>
<tr>
  <td style="text-align:left"><code>symm</code></td>
  <td style="text-align:right">0.123</td>
  <td style="text-align:right">0.123</td>
  <td style="text-align:right">0.123</td>
  <td style="text-align:right">0.054</td>
</tr>
<tr>
  <td style="text-align:left"><code>syr2k</code></td>
  <td style="text-align:right">0.029</td>
  <td style="text-align:right">0.233</td>
  <td style="text-align:right">0.206</td>
  <td style="text-align:right">0.002</td>
</tr>
<tr>
  <td style="text-align:left"><code>syrk</code></td>
  <td style="text-align:right">0.050</td>
  <td style="text-align:right">0.004</td>
  <td style="text-align:right">—</td>
  <td style="text-align:right">0.007</td>
</tr>
<tr>
  <td style="text-align:left"><code>trisolv</code></td>
  <td style="text-align:right">0.120</td>
  <td style="text-align:right">0.585</td>
  <td style="text-align:right">0.019</td>
  <td style="text-align:right">0.035</td>
</tr>
<tr>
  <td style="text-align:left"><code>trmm</code></td>
  <td style="text-align:right">0.122</td>
  <td style="text-align:right">0.122</td>
  <td style="text-align:right">0.299</td>
  <td style="text-align:right">0.012</td>
</tr>
</tbody></table>

## Failures

All modes fail **`doitgen`**: gold HLS SYNCHK 200-43 (non-static pointer).

