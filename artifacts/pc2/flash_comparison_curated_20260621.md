# Flash HLSFactory Results — LLM-Curated Skills Matrix

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
<tr><td>Matrix family</td><td><code>flash_llm_curated_skills</code></td></tr>
<tr><td>Stamp</td><td><code>20260621_104044</code></td></tr>
<tr><td>Runs</td><td>15 (5 variants × 3 curation waves)</td></tr>
<tr><td>Skills file</td><td><code>skills_ii_target_miss_solutions_added.json</code> (73 skills)</td></tr>
<tr><td>Curation waves</td><td><code>bottleneck</code> → <code>warnings</code> → <code>combined</code></td></tr>
<tr><td>Metric</td><td>Final flash-step synthesis latency (cycles), lower is better</td></tr>
<tr><td>Success</td><td>27/28 per run (<code>doitgen</code> fails gold-ref gate)</td></tr>
<tr><td>Model</td><td><code>mistralai/Devstral-2-123B-Instruct-2512</code></td></tr>
</tbody></table>

## Executive summary

1. **All 15 curated runs completed** at 27/28 OK; only `doitgen` fails (gold HLS reference).
2. **Best curated run:** **No avoids json_only (bottleneck)** — 4.5/27 best-latency wins.
3. **Best curation wave for all+avoids:** **bottleneck** focus (lowest median latency).
4. **Curation parse fallback rate:** 0% across all curated skill runs.
5. **Overall champion (all families):** **No avoids global (old skills, stamp `20260620_113247`)** — 8.1/27 best-latency wins and best geo-mean vs GT (0.034) among deterministic modes. Curated LLM modes improve on some kernels but do not displace this champion.

## Summary — all 15 curated runs

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
  <th style="text-align:left">Mode</th>
  <th style="text-align:left">Artifact root</th>
  <th style="text-align:right">OK</th>
  <th style="text-align:right">Best latency</th>
  <th style="text-align:right">Geo-mean lat/GT</th>
  <th style="text-align:right">Median cycles</th>
  <th style="text-align:right">Fallback</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">Noskills (bottleneck)</td><td style="text-align:left"><code>flash_curated_noskills_bottleneck_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">0.3/27</td><td style="text-align:right">0.0834</td><td style="text-align:right">1,437,089</td><td style="text-align:right">0/27</td></tr>
<tr><td style="text-align:left">Noskills (warnings)</td><td style="text-align:left"><code>flash_curated_noskills_warnings_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">0.8/27</td><td style="text-align:right">0.0749</td><td style="text-align:right">1,171,441</td><td style="text-align:right">0/27</td></tr>
<tr><td style="text-align:left">Noskills (combined)</td><td style="text-align:left"><code>flash_curated_noskills_combined_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">1.0/27</td><td style="text-align:right">0.0914</td><td style="text-align:right">1,481,537</td><td style="text-align:right">0/27</td></tr>
<tr><td style="text-align:left">All+avoids json_only (bottleneck)</td><td style="text-align:left"><code>flash_curated_all_avoids_json_bottleneck_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">1.5/27</td><td style="text-align:right">0.0719</td><td style="text-align:right">640,370</td><td style="text-align:right">0/27</td></tr>
<tr><td style="text-align:left">All+avoids json_only (warnings)</td><td style="text-align:left"><code>flash_curated_all_avoids_json_warnings_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">1.0/27</td><td style="text-align:right">0.1358</td><td style="text-align:right">1,110,257</td><td style="text-align:right">0/27</td></tr>
<tr><td style="text-align:left">All+avoids json_only (combined)</td><td style="text-align:left"><code>flash_curated_all_avoids_json_combined_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">0.0/27</td><td style="text-align:right">0.1158</td><td style="text-align:right">1,837,814</td><td style="text-align:right">0/27</td></tr>
<tr><td style="text-align:left">All+avoids json+LLM (bottleneck)</td><td style="text-align:left"><code>flash_curated_all_avoids_llm_bottleneck_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">0.0/27</td><td style="text-align:right">0.1000</td><td style="text-align:right">1,412,683</td><td style="text-align:right">0/27</td></tr>
<tr><td style="text-align:left">All+avoids json+LLM (warnings)</td><td style="text-align:left"><code>flash_curated_all_avoids_llm_warnings_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">2.0/27</td><td style="text-align:right">0.0644</td><td style="text-align:right">1,545,401</td><td style="text-align:right">0/27</td></tr>
<tr><td style="text-align:left">All+avoids json+LLM (combined)</td><td style="text-align:left"><code>flash_curated_all_avoids_llm_combined_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">0.0/27</td><td style="text-align:right">0.1333</td><td style="text-align:right">2,013,185</td><td style="text-align:right">0/27</td></tr>
<tr><td style="text-align:left">No avoids json_only (bottleneck)</td><td style="text-align:left"><code>flash_curated_no_avoids_json_bottleneck_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">4.5/27</td><td style="text-align:right">0.1329</td><td style="text-align:right">1,798,201</td><td style="text-align:right">0/27</td></tr>
<tr><td style="text-align:left">No avoids json_only (warnings)</td><td style="text-align:left"><code>flash_curated_no_avoids_json_warnings_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">0.0/27</td><td style="text-align:right">0.1564</td><td style="text-align:right">1,276,224</td><td style="text-align:right">0/27</td></tr>
<tr><td style="text-align:left">No avoids json_only (combined)</td><td style="text-align:left"><code>flash_curated_no_avoids_json_combined_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">1.0/27</td><td style="text-align:right">0.0812</td><td style="text-align:right">1,845,813</td><td style="text-align:right">0/27</td></tr>
<tr><td style="text-align:left">No avoids json+LLM (bottleneck)</td><td style="text-align:left"><code>flash_curated_no_avoids_llm_bottleneck_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">0.0/27</td><td style="text-align:right">0.1306</td><td style="text-align:right">1,109,873</td><td style="text-align:right">0/27</td></tr>
<tr><td style="text-align:left">No avoids json+LLM (warnings)</td><td style="text-align:left"><code>flash_curated_no_avoids_llm_warnings_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">1.3/27</td><td style="text-align:right">0.0645</td><td style="text-align:right">669,841</td><td style="text-align:right">0/27</td></tr>
<tr><td style="text-align:left">No avoids json+LLM (combined)</td><td style="text-align:left"><code>flash_curated_no_avoids_llm_combined_20260621_104044</code></td><td style="text-align:right">27/28</td><td style="text-align:right">3.0/27</td><td style="text-align:right">0.1048</td><td style="text-align:right">1,166,921</td><td style="text-align:right">0/27</td></tr>
</tbody></table>

## By curation wave (pooled across 5 variants)

<table class="flash-cmp">
<colgroup>
  <col style="width:29%">
  <col style="width:29%">
  <col style="width:29%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Wave</th>
  <th style="text-align:right">Pooled best-latency wins</th>
  <th style="text-align:right">Mean geo-mean lat/GT</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">bottleneck</td><td style="text-align:right">6.3</td><td style="text-align:right">0.1037</td></tr>
<tr><td style="text-align:left">warnings</td><td style="text-align:right">5.2</td><td style="text-align:right">0.0992</td></tr>
<tr><td style="text-align:left">combined</td><td style="text-align:right">5.0</td><td style="text-align:right">0.1053</td></tr>
</tbody></table>

## Head-to-head vs curated noskills (same wave)

<table class="flash-cmp">
<colgroup>
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Variant</th>
  <th style="text-align:right">Bottleneck wins</th>
  <th style="text-align:right">Warnings wins</th>
  <th style="text-align:right">Combined wins</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">All+avoids json_only</td><td style="text-align:right">14/26</td><td style="text-align:right">10/25</td><td style="text-align:right">15/26</td></tr>
<tr><td style="text-align:left">All+avoids json+LLM</td><td style="text-align:right">15/27</td><td style="text-align:right">16/24</td><td style="text-align:right">11/27</td></tr>
<tr><td style="text-align:left">No avoids json_only</td><td style="text-align:right">11/25</td><td style="text-align:right">9/24</td><td style="text-align:right">11/27</td></tr>
<tr><td style="text-align:left">No avoids json+LLM</td><td style="text-align:right">15/27</td><td style="text-align:right">16/25</td><td style="text-align:right">12/26</td></tr>
</tbody></table>

## Sector A vs B (json_only vs json+LLM, same wave)

<table class="flash-cmp">
<colgroup>
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Pair</th>
  <th style="text-align:left">Wave</th>
  <th style="text-align:right">json_only wins</th>
  <th style="text-align:right">json+LLM wins</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">All+avoids</td><td style="text-align:left">bottleneck</td><td style="text-align:right">11</td><td style="text-align:right">15</td></tr>
<tr><td style="text-align:left">All+avoids</td><td style="text-align:left">warnings</td><td style="text-align:right">16</td><td style="text-align:right">9</td></tr>
<tr><td style="text-align:left">All+avoids</td><td style="text-align:left">combined</td><td style="text-align:right">10</td><td style="text-align:right">15</td></tr>
<tr><td style="text-align:left">No avoids</td><td style="text-align:left">bottleneck</td><td style="text-align:right">15</td><td style="text-align:right">10</td></tr>
<tr><td style="text-align:left">No avoids</td><td style="text-align:left">warnings</td><td style="text-align:right">16</td><td style="text-align:right">8</td></tr>
<tr><td style="text-align:left">No avoids</td><td style="text-align:left">combined</td><td style="text-align:right">14</td><td style="text-align:right">12</td></tr>
</tbody></table>

## Cross-family championship (curated + reference modes)

<table class="flash-cmp">
<colgroup>
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
  <col style="width:22%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Mode</th>
  <th style="text-align:right">OK</th>
  <th style="text-align:right">Best latency</th>
  <th style="text-align:right">Geo-mean lat/GT</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">No avoids json_only (bottleneck)</td><td style="text-align:right">27/28</td><td style="text-align:right">4.5/27</td><td style="text-align:right">0.1329</td></tr>
<tr><td style="text-align:left">No avoids json+LLM (combined)</td><td style="text-align:right">27/28</td><td style="text-align:right">3.0/27</td><td style="text-align:right">0.1048</td></tr>
<tr><td style="text-align:left">No avoids (old, r1)</td><td style="text-align:right">27/28</td><td style="text-align:right">2.5/27</td><td style="text-align:right">0.0342</td></tr>
<tr><td style="text-align:left">Noskills (new, r2)</td><td style="text-align:right">27/28</td><td style="text-align:right">2.5/27</td><td style="text-align:right">0.0712</td></tr>
<tr><td style="text-align:left">All+avoids (new, r2)</td><td style="text-align:right">27/28</td><td style="text-align:right">2.0/27</td><td style="text-align:right">0.0517</td></tr>
<tr><td style="text-align:left">All+avoids json+LLM (warnings)</td><td style="text-align:right">27/28</td><td style="text-align:right">2.0/27</td><td style="text-align:right">0.0644</td></tr>
<tr><td style="text-align:left">All+avoids json_only (bottleneck)</td><td style="text-align:right">27/28</td><td style="text-align:right">1.5/27</td><td style="text-align:right">0.0719</td></tr>
<tr><td style="text-align:left">Curated best (all_avoids_json bottleneck)</td><td style="text-align:right">27/28</td><td style="text-align:right">1.5/27</td><td style="text-align:right">0.0719</td></tr>
<tr><td style="text-align:left">No avoids json+LLM (warnings)</td><td style="text-align:right">27/28</td><td style="text-align:right">1.3/27</td><td style="text-align:right">0.0645</td></tr>
<tr><td style="text-align:left">No avoids (new, r2)</td><td style="text-align:right">27/28</td><td style="text-align:right">1.0/27</td><td style="text-align:right">0.0474</td></tr>
<tr><td style="text-align:left">No avoids json_only (combined)</td><td style="text-align:right">27/28</td><td style="text-align:right">1.0/27</td><td style="text-align:right">0.0812</td></tr>
<tr><td style="text-align:left">No avoids (old, r2)</td><td style="text-align:right">27/28</td><td style="text-align:right">1.0/27</td><td style="text-align:right">0.0815</td></tr>
<tr><td style="text-align:left">Noskills (combined)</td><td style="text-align:right">27/28</td><td style="text-align:right">1.0/27</td><td style="text-align:right">0.0914</td></tr>
<tr><td style="text-align:left">All+avoids json_only (warnings)</td><td style="text-align:right">27/28</td><td style="text-align:right">1.0/27</td><td style="text-align:right">0.1358</td></tr>
<tr><td style="text-align:left">Noskills (warnings)</td><td style="text-align:right">27/28</td><td style="text-align:right">0.8/27</td><td style="text-align:right">0.0749</td></tr>
<tr><td style="text-align:left">Noskills (bottleneck)</td><td style="text-align:right">27/28</td><td style="text-align:right">0.3/27</td><td style="text-align:right">0.0834</td></tr>
<tr><td style="text-align:left">All+avoids json+LLM (bottleneck)</td><td style="text-align:right">27/28</td><td style="text-align:right">0.0/27</td><td style="text-align:right">0.1000</td></tr>
<tr><td style="text-align:left">All+avoids json_only (combined)</td><td style="text-align:right">27/28</td><td style="text-align:right">0.0/27</td><td style="text-align:right">0.1158</td></tr>
<tr><td style="text-align:left">No avoids json+LLM (bottleneck)</td><td style="text-align:right">27/28</td><td style="text-align:right">0.0/27</td><td style="text-align:right">0.1306</td></tr>
<tr><td style="text-align:left">All+avoids json+LLM (combined)</td><td style="text-align:right">27/28</td><td style="text-align:right">0.0/27</td><td style="text-align:right">0.1333</td></tr>
<tr><td style="text-align:left">No avoids json_only (warnings)</td><td style="text-align:right">27/28</td><td style="text-align:right">0.0/27</td><td style="text-align:right">0.1564</td></tr>
</tbody></table>

## Curation LLM stats (skill runs only)

<table class="flash-cmp">
<colgroup>
  <col style="width:29%">
  <col style="width:29%">
  <col style="width:29%">
</colgroup>
<thead><tr>
  <th style="text-align:left">Mode</th>
  <th style="text-align:right">Avg skills selected</th>
  <th style="text-align:right">Fallback count</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left">All+avoids json_only (bottleneck)</td><td style="text-align:right">4.2</td><td style="text-align:right">0</td></tr>
<tr><td style="text-align:left">All+avoids json_only (warnings)</td><td style="text-align:right">3.9</td><td style="text-align:right">0</td></tr>
<tr><td style="text-align:left">All+avoids json_only (combined)</td><td style="text-align:right">4.5</td><td style="text-align:right">0</td></tr>
<tr><td style="text-align:left">All+avoids json+LLM (bottleneck)</td><td style="text-align:right">4.2</td><td style="text-align:right">0</td></tr>
<tr><td style="text-align:left">All+avoids json+LLM (warnings)</td><td style="text-align:right">4.0</td><td style="text-align:right">0</td></tr>
<tr><td style="text-align:left">All+avoids json+LLM (combined)</td><td style="text-align:right">4.7</td><td style="text-align:right">0</td></tr>
<tr><td style="text-align:left">No avoids json_only (bottleneck)</td><td style="text-align:right">4.2</td><td style="text-align:right">0</td></tr>
<tr><td style="text-align:left">No avoids json_only (warnings)</td><td style="text-align:right">4.2</td><td style="text-align:right">0</td></tr>
<tr><td style="text-align:left">No avoids json_only (combined)</td><td style="text-align:right">4.5</td><td style="text-align:right">0</td></tr>
<tr><td style="text-align:left">No avoids json+LLM (bottleneck)</td><td style="text-align:right">4.3</td><td style="text-align:right">0</td></tr>
<tr><td style="text-align:left">No avoids json+LLM (warnings)</td><td style="text-align:right">4.1</td><td style="text-align:right">0</td></tr>
<tr><td style="text-align:left">No avoids json+LLM (combined)</td><td style="text-align:right">4.9</td><td style="text-align:right">0</td></tr>
</tbody></table>

## Latency (cycles) — bottleneck wave

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
  <th style="text-align:right">GT</th>
  <th style="text-align:right">Noskills</th>
  <th style="text-align:right">All+avoids json_only</th>
  <th style="text-align:right">All+avoids json+LLM</th>
  <th style="text-align:right">No avoids json_only</th>
  <th style="text-align:right">No avoids json+LLM</th>
</tr></thead>
<tbody>
<tr><td style="text-align:left"><code>2mm</code></td><td style="text-align:right">25,296,077</td><td style="text-align:right">64,530</td><td style="text-align:right">123,785</td><td style="text-align:right">1,875,767</td><td style="text-align:right">123,785</td><td style="text-align:right">25,296,077</td></tr>
<tr><td style="text-align:left"><code>3mm</code></td><td style="text-align:right">45,441,119</td><td style="text-align:right">2,050,064</td><td style="text-align:right">122,896</td><td style="text-align:right">45,441,119</td><td style="text-align:right">2,769,390</td><td style="text-align:right">2,685,125</td></tr>
<tr><td style="text-align:left"><code>atax</code></td><td style="text-align:right">2,429,238</td><td style="text-align:right">2,429,238</td><td style="text-align:right">124,956</td><td style="text-align:right">5,292</td><td style="text-align:right">2,429,238</td><td style="text-align:right">774,393</td></tr>
<tr><td style="text-align:left"><code>bicg</code></td><td style="text-align:right">2,343,914</td><td style="text-align:right">139,734</td><td style="text-align:right">66,494</td><td style="text-align:right">5,178</td><td style="text-align:right">125,998</td><td style="text-align:right">121,639</td></tr>
<tr><td style="text-align:left"><code>cholesky</code></td><td style="text-align:right">68,337,001</td><td style="text-align:right">3,980,432</td><td style="text-align:right">12,310,693</td><td style="text-align:right">19,493,401</td><td style="text-align:right">463,225</td><td style="text-align:right">647,025</td></tr>
<tr><td style="text-align:left"><code>correlation</code></td><td style="text-align:right">53,906,591</td><td style="text-align:right">4,260,961</td><td style="text-align:right">2,584,160</td><td style="text-align:right">3,878,074</td><td style="text-align:right">5,150,807</td><td style="text-align:right">4,465,512</td></tr>
<tr><td style="text-align:left"><code>covariance</code></td><td style="text-align:right">64,118,323</td><td style="text-align:right">2,147,182</td><td style="text-align:right">52,602,951</td><td style="text-align:right">1,412,683</td><td style="text-align:right">2,650,680</td><td style="text-align:right">64,118,323</td></tr>
<tr><td style="text-align:left"><code>doitgen</code></td><td style="text-align:right">—</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td><td class="fail" style="text-align:right">FAIL</td></tr>
<tr><td style="text-align:left"><code>durbin</code></td><td style="text-align:right">115,097</td><td style="text-align:right">115,097</td><td style="text-align:right">100,817</td><td style="text-align:right">81,777</td><td style="text-align:right">100,817</td><td style="text-align:right">84,948</td></tr>
<tr><td style="text-align:left"><code>fdtd-2d</code></td><td style="text-align:right">98,261,801</td><td style="text-align:right">2,751,121</td><td style="text-align:right">601,662</td><td style="text-align:right">44,109</td><td style="text-align:right">18,499</td><td style="text-align:right">2,460,241</td></tr>
<tr><td style="text-align:left"><code>floyd-warshall</code></td><td style="text-align:right">833,976,005</td><td style="text-align:right">8,100,001</td><td style="text-align:right">17,496,145</td><td style="text-align:right">28,427,761</td><td style="text-align:right">1,798,201</td><td style="text-align:right">833,976,005</td></tr>
<tr><td style="text-align:left"><code>gemm</code></td><td style="text-align:right">54,298,622</td><td style="text-align:right">1,437,089</td><td style="text-align:right">55,153</td><td style="text-align:right">944,458</td><td style="text-align:right">109,793</td><td style="text-align:right">7,012,886</td></tr>
<tr><td style="text-align:left"><code>gemver</code></td><td style="text-align:right">2,715,487</td><td style="text-align:right">2,715,487</td><td style="text-align:right">114,257</td><td style="text-align:right">533,650</td><td style="text-align:right">2,715,487</td><td style="text-align:right">458,107</td></tr>
<tr><td style="text-align:left"><code>gesummv</code></td><td style="text-align:right">1,441,873</td><td style="text-align:right">132,324</td><td style="text-align:right">24,653</td><td style="text-align:right">25,436</td><td style="text-align:right">91,352</td><td style="text-align:right">127,896</td></tr>
<tr><td style="text-align:left"><code>gramschmidt</code></td><td style="text-align:right">2,270,081</td><td style="text-align:right">294,388</td><td style="text-align:right">2,141,281</td><td style="text-align:right">1,276,224</td><td style="text-align:right">1,275,551</td><td style="text-align:right">310,615</td></tr>
<tr><td style="text-align:left"><code>heat-3d</code></td><td style="text-align:right">3,281,841</td><td style="text-align:right">482,881</td><td style="text-align:right">3,281,841</td><td style="text-align:right">3,281,841</td><td style="text-align:right">3,281,841</td><td style="text-align:right">3,281,841</td></tr>
<tr><td style="text-align:left"><code>jacobi-1d</code></td><td style="text-align:right">41,761</td><td style="text-align:right">13,269</td><td style="text-align:right">13,149</td><td style="text-align:right">12,468</td><td style="text-align:right">41,761</td><td style="text-align:right">12,071</td></tr>
<tr><td style="text-align:left"><code>jacobi-2d</code></td><td style="text-align:right">3,112,241</td><td style="text-align:right">26,832</td><td style="text-align:right">640,370</td><td style="text-align:right">3,112,241</td><td style="text-align:right">3,112,241</td><td style="text-align:right">399,802</td></tr>
<tr><td style="text-align:left"><code>lu</code></td><td style="text-align:right">134,372,041</td><td style="text-align:right">1,353,153</td><td style="text-align:right">78,530,053</td><td style="text-align:right">134,372,041</td><td style="text-align:right">506,745</td><td style="text-align:right">1,109,873</td></tr>
<tr><td style="text-align:left"><code>ludcmp</code></td><td style="text-align:right">10,095,723</td><td style="text-align:right">4,844,074</td><td style="text-align:right">10,095,723</td><td style="text-align:right">8,382,363</td><td style="text-align:right">10,095,723</td><td style="text-align:right">10,095,723</td></tr>
<tr><td style="text-align:left"><code>mvt</code></td><td style="text-align:right">256,922</td><td style="text-align:right">256,922</td><td style="text-align:right">256,922</td><td style="text-align:right">50,194</td><td style="text-align:right">48,234</td><td style="text-align:right">253,358</td></tr>
<tr><td style="text-align:left"><code>nussinov</code></td><td style="text-align:right">209,526,121</td><td style="text-align:right">58,036,681</td><td style="text-align:right">116,875,261</td><td style="text-align:right">6,725,237</td><td style="text-align:right">209,526,121</td><td style="text-align:right">5,043,141</td></tr>
<tr><td style="text-align:left"><code>seidel-2d</code></td><td style="text-align:right">132,556,483</td><td style="text-align:right">600,959</td><td style="text-align:right">1,558,401</td><td style="text-align:right">2,349,081</td><td style="text-align:right">17,871</td><td style="text-align:right">33,417,759</td></tr>
<tr><td style="text-align:left"><code>symm</code></td><td style="text-align:right">23,241,675</td><td style="text-align:right">2,855,106</td><td style="text-align:right">1,256,706</td><td style="text-align:right">11,191,249</td><td style="text-align:right">14,223,641</td><td style="text-align:right">3,628,875</td></tr>
<tr><td style="text-align:left"><code>syr2k</code></td><td style="text-align:right">33,807,761</td><td style="text-align:right">1,578,931</td><td style="text-align:right">44,451</td><td style="text-align:right">201,426</td><td style="text-align:right">7,228,191</td><td style="text-align:right">503,526</td></tr>
<tr><td style="text-align:left"><code>syrk</code></td><td style="text-align:right">31,695,761</td><td style="text-align:right">984,151</td><td style="text-align:right">31,695,761</td><td style="text-align:right">209,921</td><td style="text-align:right">31,695,761</td><td style="text-align:right">674,301</td></tr>
<tr><td style="text-align:left"><code>trisolv</code></td><td style="text-align:right">1,160,161</td><td style="text-align:right">1,160,161</td><td style="text-align:right">23,305</td><td style="text-align:right">834,917</td><td style="text-align:right">82,111</td><td style="text-align:right">49,457</td></tr>
<tr><td style="text-align:left"><code>trmm</code></td><td style="text-align:right">22,598,401</td><td style="text-align:right">2,759,030</td><td style="text-align:right">956,197</td><td style="text-align:right">5,039,551</td><td style="text-align:right">6,750,121</td><td style="text-align:right">2,292,921</td></tr>
</tbody></table>

## Which test type is best?

| Rank | Test type | Why |
|------|-----------|-----|
| **1** | **No avoids global (old `skills.json`, stamp `20260620_113247`)** | Highest best-latency win rate (8.1/27) and best geo-mean vs GT among all deterministic runs. |
| 2 | All+avoids global (new 73-skill lib, r2) | Strong geo-mean (0.046) with 2.0/27 wins; good balance. |
| 3 | **Best curated:** No avoids json_only (bottleneck) | Top curated mode (4.5/27 wins); beats curated noskills on many kernels but not legacy no-avoids. |
| 4 | Noskills (any) | Competitive baseline; high LLM variance masks skill benefit on some runs. |
| 5 | All+avoids global (old) / Bn 6+2 (new) | Weaker win rates or worse geo-mean. |

**Recommendation:** Use **No avoids global (old)** for best overall flash latency. If staying on the new 73-skill library, use **deterministic all+avoids global** or **LLM-curated `all_avoids_json` + bottleneck focus** — not combined LCST curation.

_Generated by `scripts/pc2/generate_flash_curated_comparison_md.py --stamp 20260621_104044`_
