| name | cosim | dut_done | v2021_end | v2021_pred | v2021_err% | v2023_end | v2023_pred | v2023_err% |
|---|---|---|---|---|---|---|---|---|
| self_good_bundles_n16 | 68679 | 68614 | 66836 | 66897 | -2.59 | 68616 | 68679 | +0.00 |
| self_good_n16 | 70884 | 70817 | 70328 | 70389 | -0.70 | 70812 | 70875 | -0.01 |

- `v2021`: 2 configs, mean |err| 1.65%, max |err| 2.59% (self_good_bundles_n16), within ±1% or ±5 cycles: 1/2; vs DUT ap_done: mean |err| 1133.5 cycles, max 1778, within ±5 cycles 0/2
- `v2023`: 2 configs, mean |err| 0.01%, max |err| 0.01% (self_good_n16), within ±1% or ±5 cycles: 2/2; vs DUT ap_done: mean |err| 3.5 cycles, max 5, within ±5 cycles 2/2
