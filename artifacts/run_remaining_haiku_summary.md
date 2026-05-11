# Remaining-14 agentic-workflow run — claude-haiku-4-5-20251001

Vitis 2023.2 / xcu280-fsvh2892-2L-e / 3.33 ns / flow_target=vitis

| bench | phase | gen_lat_ns | gt_lat_ns | gen/gt | csim gen | csim gt | sec |
|---|---|---:|---:|---:|:---:|:---:|---:|
| StreamCluster | complete | 4051000.0 | 476000.0 | 8.51× | ✓ | ✓ | 638.1 |
| aes | complete | 3570.0 | 3573.0 | 1.00× | ✓ | ✓ | 52.5 |
| fft | reference | — | — | — | — | ✗ | 3.0 |
| gemm_ncubed | complete | 984000.0 | 984000.0 | 1.00× | ✓ | ✓ | 93.3 |
| hotspot | reference | — | 6920000.0 | — | — | ✗ | 296.5 |
| kmeans | complete | 23189000.0 | 49219000.0 | 0.47× | ✓ | ✓ | 281.8 |
| lavaMD | complete | 51429000.0 | 1547000.0 | 33.24× | ✓ | ✓ | 2880.3 |
| lud | complete | 168000000.0 | 75999000.0 | 2.21× | ✓ | ✓ | 2500.6 |
| md_knn | complete | 34252.0 | 18388.0 | 1.86× | ✓ | ✓ | 130.3 |
| sort_merge | reference | — | — | — | — | ✗ | 2.8 |
| spmv_crs | complete | — | — | — | ✓ | ✓ | 291.5 |
| srad | reference | — | 126000.0 | — | — | ✗ | 130.9 |
| stencil2D | complete | 81382.0 | 81435.0 | 1.00× | ✓ | ✓ | 32.9 |
| viterbi | reference | — | — | — | — | ✗ | 2.8 |
