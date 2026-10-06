# AutoSA skill holdout

Rule, taken from `avoid-autosa-shape-is-one-config`: a row is near the model when `abs(HLS kernel min - model) / model <= 0.10`. The comparison is per `sa_sizes`.

Held-out 128x8 candidates near: [17, 20, 23, 26].
Held-out 128x8 candidates not near: [15, 25, 28, 31, 32].
Space-time 3 rows within 10% of kernel min: 2 of 10. DSP above 9024: [1, 2, 3, 16, 18].
Space-time 4 rows within 10% of kernel min: 8 of 10.

The U280 searches 3464449, 3464452, and 3464455 completed. HLS jobs 3464450, 3464453, and 3464456 failed after writing 0 candidates, because validation looked under `default_cap` while the search wrote `u280_paper`. `mm1024_u280_st{0,3,4}.csv` each have a header and no data rows. Module-max versus the model is unmeasured, so `autosa-latency-max-of-modules` and `autosa-st4-kernel-min-band` stay at confidence medium.

## Checks

- pass: nine 128x8 rows
- pass: near set is 17,20,23,26
- pass: far set is 15,25,28,31,32
- pass: skill names every near candidate
- pass: skill names every far candidate
- pass: skill does not call every 128x8 within 10%
- pass: skill states the 0.10 rule
- pass: space-time 3 is not all within 10%
- pass: space-time 3 skill notes 13x16x8 absent
- pass: space-time 4 kernel-min skill stays medium
- pass: space-time 4 within 10% count is 8 of 10
- pass: paper module-max skill stays medium
- pass: U280 module-max CSV has no data rows
- pass: model DSP is 0 on these rows
- pass: resource skill forbids calling DSP 0 the paper search
- pass: DSP above 9024 is named
- pass: five space-time 3 rows exceed 9024 DSP

Failed checks: 0.
