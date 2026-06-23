# Flash Cosim Results Comparison

- Cosim run: `artifacts/pc2/flash_cosim/20260622_110920`
- Generated: 2026-06-22 19:49 UTC
- Scope: **973** generated kernels across all flash matrix artifact dirs

## Overall

| Metric | Value |
|---|---:|
| **Cosim PASS** | 640 (65.8%) |
| **Cosim FAIL** | 333 (34.2%) |

### Failure breakdown

- **TB functional mismatch**: 240
- **SIGSEGV / crash**: 72
- **Other**: 21

## Primary modes — cosim pass rate & median latency

| Mode | PASS | FAIL | Pass% | Median cosim cycles |
|---|---:|---:|---:|---:|
| Noskills (old) | 20 | 8 | 71.4% | 3,210 |
| Bn 2+2 (old) | 18 | 10 | 64.3% | 2,826 |
| All+avoids (old) | 21 | 7 | 75.0% | 10,988 |
| No avoids (old) | 20 | 8 | 71.4% | 3,699 |
| Noskills (new) | 20 | 8 | 71.4% | 3,816 |
| Bn 2+2 (new) | 14 | 14 | 50.0% | 5,706 |
| Bn 4+2 (new) | 20 | 8 | 71.4% | 3,280 |
| Bn 6+2 (new) | 16 | 12 | 57.1% | 3,937 |
| All+avoids (new) | 17 | 11 | 60.7% | 4,869 |
| No avoids (new) | 21 | 7 | 75.0% | 1,944 |
| Cur nosk/bn | 16 | 12 | 57.1% | 4,322 |
| Cur all/json/bn | 21 | 7 | 75.0% | 4,051 |
| Cur no/json/bn | 19 | 9 | 67.9% | 5,730 |
| Cur all/json/warn | 16 | 12 | 57.1% | 7,164 |
| Cur no/json/warn | 20 | 8 | 71.4% | 3,937 |

## Per-bench cosim (primary modes)

| Bench | Noskills ( | Bn 2+2 (ol | All+avoids | No avoids  | Noskills ( | Bn 2+2 (ne | Bn 4+2 (ne | Bn 6+2 (ne | All+avoids | No avoids  | Cur nosk/b | Cur all/js | Cur no/jso | Cur all/js | Cur no/jso |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2mm | ✓2,283 | ✓1,827 | ✓2,283 | FAIL | ✓14,211 | ✓1,099 | ✓1,112 | ✓969 | FAIL | ✓1,944 | ✓2,283 | ✓1,827 | ✓1,827 | ✓969 | ✓956 |
| 3mm | ✓2,826 | ✓2,826 | ✓2,633 | ✓43,154 | ✓846 | ✓1,088 | ✓1,088 | ✓8,216 | ✓1,632 | ✓1,476 | FAIL | ✓3,296 | ✓1,320 | ✓3,296 | ✓1,320 |
| atax | ✓3,420 | ✓805 | ✓805 | ✓515 | FAIL | FAIL | ✓805 | ✓805 | ✓515 | ✓515 | ✓805 | ✓1,245 | ✓805 | ✓11,185 | ✓3,672 |
| bicg | ✓505 | ✓2,199 | FAIL | ✓428 | ✓791 | FAIL | ✓3,233 | FAIL | ✓3,255 | ✓428 | ✓461 | FAIL | ✓4,476 | FAIL | ✓3,519 |
| cholesky | ✓3,210 | ✓3,182 | ✓21,193 | ✓125,437 | ✓13,066 | ✓2,979 | ✓3,280 | ✓2,986 | ✓4,869 | ✓3,945 | ✓3,210 | ✓3,462 | ✓4,491 | ✓4,967 | ✓3,721 |
| correlation | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| covariance | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| durbin | ✓3,123 | ✓3,339 | ✓3,179 | ✓3,699 | ✓3,339 | ✓3,339 | ✓3,123 | ✓3,706 | ✓1,035 | ✓3,123 | ✓3,179 | ✓3,123 | ✓3,123 | ✓3,179 | ✓3,123 |
| fdtd-2d | ✓2,885 | ✓2,795 | ✓2,005 | ✓2,005 | FAIL | ✓215,795 | ✓2,005 | ✓1,505 | ✓1,505 | ✓1,015 | FAIL | ✓1,015 | FAIL | FAIL | FAIL |
| floyd-warshall | ✓35,025 | ✓914 | ✓82,842 | ✓907 | ✓914 | ✓81,449 | ✓81,449 | ✓81,449 | ✓81,449 | ✓81,449 | ✓9,048 | ✓6,962 | ✓2,111 | ✓81,449 | ✓81,449 |
| gemm | ✓791 | ✓791 | ✓14,849 | ✓1,418 | ✓1,209 | FAIL | ✓4,047 | ✓2,155 | ✓780 | ✓494 | ✓901 | ✓340 | ✓2,100 | ✓5,059 | ✓967 |
| gemver | FAIL | ✓1,230 | ✓5,752 | ✓3,712 | ✓1,740 | ✓16,768 | ✓5,871 | ✓3,916 | ✓9,679 | ✓4,579 | ✓16,768 | ✓4,647 | ✓16,768 | ✓16,768 | ✓6,891 |
| gesummv | ✓449 | ✓819 | ✓371 | ✓1,203 | ✓2,369 | FAIL | ✓819 | ✓1,073 | ✓1,216 | ✓1,372 | FAIL | ✓7,683 | ✓1,047 | ✓683 | ✓2,880 |
| gramschmidt | FAIL | FAIL | ✓48,141 | ✓48,141 | FAIL | FAIL | FAIL | FAIL | FAIL | ✓48,141 | FAIL | ✓12,204 | FAIL | FAIL | FAIL |
| heat-3d | ✓2,971 | FAIL | ✓1,995 | ✓551,491 | FAIL | ✓4,907 | FAIL | FAIL | ✓67,619 | ✓5,787 | FAIL | ✓551,491 | ✓551,491 | FAIL | ✓551,491 |
| jacobi-1d | ✓827 | ✓827 | FAIL | ✓731 | ✓827 | FAIL | FAIL | FAIL | FAIL | ✓731 | ✓827 | FAIL | ✓11,115 | ✓827 | FAIL |
| jacobi-2d | FAIL | FAIL | ✓54,747 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| lu | ✓4,638 | FAIL | ✓228,029 | FAIL | ✓4,813 | FAIL | ✓29,474 | ✓4,533 | ✓217,116 | FAIL | ✓4,862 | ✓4,498 | ✓5,730 | ✓5,793 | ✓29,474 |
| ludcmp | ✓24,445 | ✓7,035 | ✓13,875 | FAIL | ✓24,445 | ✓4,155 | ✓7,075 | ✓7,035 | ✓7,075 | ✓24,445 | ✓6,925 | ✓24,445 | ✓24,445 | ✓10,335 | ✓24,445 |
| mvt | ✓3,530 | ✓3,530 | FAIL | ✓3,574 | ✓3,816 | ✓3,552 | ✓3,574 | ✓3,937 | FAIL | ✓670 | ✓3,937 | ✓3,937 | FAIL | FAIL | ✓3,937 |
| nussinov | ✓17,523 | ✓10,899 | ✓4,051 | ✓9,931 | ✓17,523 | FAIL | ✓2,163 | FAIL | ✓17,523 | FAIL | FAIL | ✓4,051 | ✓17,523 | ✓17,523 | ✓17,523 |
| seidel-2d | FAIL | FAIL | ✓27,836 | ✓5,835 | ✓71,271 | ✓18,211 | ✓18,491 | FAIL | ✓4,911 | ✓71,271 | FAIL | ✓19,555 | FAIL | ✓71,271 | ✓18,211 |
| symm | ✓7,644 | ✓10,922 | ✓10,999 | FAIL | ✓10,999 | ✓10,999 | ✓10,999 | ✓10,999 | FAIL | ✓56,374 | ✓7,347 | ✓5,268 | ✓49,598 | ✓56,374 | ✓10,999 |
| syr2k | ✓3,277 | FAIL | ✓10,988 | ✓5,851 | ✓3,442 | FAIL | FAIL | FAIL | ✓1,968 | FAIL | ✓4,322 | ✓2,287 | ✓16,510 | FAIL | ✓4,454 |
| syrk | FAIL | FAIL | ✓58,505 | ✓1,485 | ✓58,505 | FAIL | ✓965 | FAIL | FAIL | ✓1,625 | FAIL | ✓58,505 | ✓58,505 | FAIL | ✓2,065 |
| trisolv | ✓1,335 | ✓3,789 | ✓1,656 | ✓1,656 | ✓1,089 | ✓7,164 | ✓999 | ✓6,507 | ✓7,164 | ✓1,179 | ✓7,164 | ✓1,476 | FAIL | ✓7,164 | ✓3,771 |
| trmm | ✓5,706 | ✓5,625 | FAIL | ✓19,665 | ✓5,130 | ✓5,706 | ✓5,706 | ✓5,643 | FAIL | ✓5,409 | ✓5,805 | FAIL | ✓10,809 | FAIL | FAIL |

## Cosim latency championship (5 stamps)

| Bench | GT synth | No avoids (old) | All+avoids (old) | All+avoids (new) | Cur all/json/bn | Noskills (old) | Winner |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2mm | 25,296,077 | FAIL | 2,283 | FAIL | 1,827 | 2,283 | **Cur all/json/bn** |
| 3mm | 45,441,119 | 43,154 | 2,633 | 1,632 | 3,296 | 2,826 | **All+avoids (new)** |
| atax | 2,429,238 | 515 | 805 | 515 | 1,245 | 3,420 | **No avoids (old)** |
| bicg | 2,343,914 | 428 | FAIL | 3,255 | FAIL | 505 | **No avoids (old)** |
| cholesky | 68,337,001 | 125,437 | 21,193 | 4,869 | 3,462 | 3,210 | **Noskills (old)** |
| durbin | 115,097 | 3,699 | 3,179 | 1,035 | 3,123 | 3,123 | **All+avoids (new)** |
| fdtd-2d | 98,261,801 | 2,005 | 2,005 | 1,505 | 1,015 | 2,885 | **Cur all/json/bn** |
| floyd-warshall | 833,976,005 | 907 | 82,842 | 81,449 | 6,962 | 35,025 | **No avoids (old)** |
| gemm | 54,298,622 | 1,418 | 14,849 | 780 | 340 | 791 | **Cur all/json/bn** |
| gemver | 2,715,487 | 3,712 | 5,752 | 9,679 | 4,647 | FAIL | **No avoids (old)** |
| gesummv | 1,441,873 | 1,203 | 371 | 1,216 | 7,683 | 449 | **All+avoids (old)** |
| gramschmidt | 2,270,081 | 48,141 | 48,141 | FAIL | 12,204 | FAIL | **Cur all/json/bn** |
| heat-3d | 3,281,841 | 551,491 | 1,995 | 67,619 | 551,491 | 2,971 | **All+avoids (old)** |
| jacobi-1d | 41,761 | 731 | FAIL | FAIL | FAIL | 827 | **No avoids (old)** |
| jacobi-2d | 3,112,241 | FAIL | 54,747 | FAIL | FAIL | FAIL | **All+avoids (old)** |
| lu | 134,372,041 | FAIL | 228,029 | 217,116 | 4,498 | 4,638 | **Cur all/json/bn** |
| ludcmp | 10,095,723 | FAIL | 13,875 | 7,075 | 24,445 | 24,445 | **All+avoids (new)** |
| mvt | 256,922 | 3,574 | FAIL | FAIL | 3,937 | 3,530 | **Noskills (old)** |
| nussinov | 209,526,121 | 9,931 | 4,051 | 17,523 | 4,051 | 17,523 | **All+avoids (old)** |
| seidel-2d | 132,556,483 | 5,835 | 27,836 | 4,911 | 19,555 | FAIL | **All+avoids (new)** |
| symm | 23,241,675 | FAIL | 10,999 | FAIL | 5,268 | 7,644 | **Cur all/json/bn** |
| syr2k | 33,807,761 | 5,851 | 10,988 | 1,968 | 2,287 | 3,277 | **All+avoids (new)** |
| syrk | 31,695,761 | 1,485 | 58,505 | FAIL | 58,505 | FAIL | **No avoids (old)** |
| trisolv | 1,160,161 | 1,656 | 1,656 | 7,164 | 1,476 | 1,335 | **Noskills (old)** |
| trmm | 22,598,401 | 19,665 | FAIL | FAIL | FAIL | 5,706 | **Noskills (old)** |

### Latency wins

- **Cur all/json/bn**: 6 benches
- **No avoids (old)**: 6 benches
- **All+avoids (new)**: 5 benches
- **Noskills (old)**: 4 benches
- **All+avoids (old)**: 4 benches

## All families (latest stamp) — cosim pass%

| Family | PASS | Total | Pass% |
|---|---:|---:|---:|
| flash_all_skills_avoids_global | 22 | 28 | 78.6% |
| flash_curated_all_avoids_json_bottleneck | 21 | 28 | 75.0% |
| flash_curated_no_avoids_json_warnings | 20 | 28 | 71.4% |
| flash_curated_noskills_combined | 19 | 28 | 67.9% |
| flash_curated_no_avoids_json_bottleneck | 19 | 28 | 67.9% |
| flash_all_new_skills_no_avoids_global | 19 | 28 | 67.9% |
| flash_noskills | 18 | 28 | 64.3% |
| flash_skills | 18 | 28 | 64.3% |
| flash_all_skills_no_avoids_global | 18 | 28 | 64.3% |
| flash_bn_skills_new_2_2 | 17 | 28 | 60.7% |
| flash_curated_no_avoids_llm_warnings | 17 | 28 | 60.7% |
| flash_all_new_skills_avoids_global | 17 | 28 | 60.7% |
| flash_curated_no_avoids_llm_bottleneck | 17 | 28 | 60.7% |
| flash_curated_noskills_warnings | 17 | 28 | 60.7% |
| flash_noskills_new | 16 | 28 | 57.1% |
| flash_curated_noskills_bottleneck | 16 | 28 | 57.1% |
| flash_curated_no_avoids_json_combined | 16 | 28 | 57.1% |
| flash_bn_skills_new_6_2 | 16 | 28 | 57.1% |
| flash_curated_all_avoids_json_warnings | 16 | 28 | 57.1% |
| flash_curated_all_avoids_json_combined | 16 | 28 | 57.1% |
| flash_curated_all_avoids_llm_bottleneck | 16 | 28 | 57.1% |
| flash_curated_all_avoids_llm_warnings | 16 | 28 | 57.1% |
| flash_curated_no_avoids_llm_combined | 15 | 28 | 53.6% |
| flash_bn_skills_new_4_2 | 14 | 28 | 50.0% |
| flash_curated_all_avoids_llm_combined | 13 | 28 | 46.4% |
