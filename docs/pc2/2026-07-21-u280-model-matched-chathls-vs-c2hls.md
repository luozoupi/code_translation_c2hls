# U280 ChatHLS vs c2hls — **size-matched**, best/avg/worst, csynth ⊥ cosim

Generated: 2026-07-21 12:19 UTC

## Rules

1. **Never mix cosim with csynth** in one ratio.
2. **Same model only** (DeepSeek↔DeepSeek).
3. ChatHLS latency from **top-function** `*_csynth.xml` (best / avg / worst).
4. **Size-gate:** only pair benches with the same problem-size family.
5. Separate tables for **best**, **avg**, and **worst** csynth latency.

### Size families (and what “poly_mini” actually is)

**c2hls was not designed around PolyBench MINI.** Its own poly suite is `benchmarks/hlsfactory_*` (gemm **NI=60,NJ=70,NK=80**).

The `batch_parallel_chathls_*` campaigns run corpus `chathls_ready` — ChatHLS kernels ingested from `ChatHLS-ACL-26/benchmark/benchmark_optimization` via `prepare_chathls_ready.py`. That ChatHLS corpus uses Mini dims for gemm/atax/… (gemm **20×25×30**). So those runs are **c2hls-on-ChatHLS-imported-benches**, not “c2hls poly_mini”.

| Family | Example | Meaning | Fair partner |
|--------|---------|---------|--------------|
| `chathls_imported` (was labeled poly_mini) | gemm **20×25×30** → CH **902** | ChatHLS `benchmark_optimization` sizes | only vs `chathls_ready` / native ChatHLS |
| `hlsfactory` | gemm **60×70×80** → CH **342,901** | c2hls + ChatHLS-c2hlsport real poly | `hlsfactory_*` ↔ c2hlsport |
| `machsuite_64` | gemm_ncubed **64×64** → CH **4,627** (DSP 1408) | MachSuite N=64 | same-N only |

**Primary poly head-to-head = Section B (hlsfactory).** Section A is same-size but ChatHLS-corpus Mini, not c2hls’s HLSFactory poly.

CSV: [`2026-07-21-u280-model-matched-chathls-vs-c2hls.csv`](2026-07-21-u280-model-matched-chathls-vs-c2hls.csv).

## A. Csynth: ChatHLS **native** vs DeepSeek c2hls on **`chathls_ready`** (size-matched)

Same ChatHLS `benchmark_optimization` corpus on both sides (Mini gemm/atax/…; hlsfactory-sized `kernel_*`; machsuite 64). **Not** c2hls’s `hlsfactory_*` poly suite.

### vs `RAG2+skills`

**csynth worst**

- n=13; CH wins=9; C2 wins=4; geomean(c2/ch)=2.905

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemm_ncubed | machsuite_64 | 4,627 | 1,319,080 | 285.08 | chathls |
| 3mm | hlsfactory | 7,018 | 90,644 | 12.92 | chathls |
| 2mm | hlsfactory | 7,582 | 54,794 | 7.23 | chathls |
| matmul | poly_mini | 660 | 3,767 | 5.71 | chathls |
| gesummv | poly_mini | 482 | 2,344 | 4.86 | chathls |
| mvt | poly_mini | 1,234 | 5,007 | 4.06 | chathls |
| bicg | poly_mini | 843 | 1,934 | 2.29 | chathls |
| atax | poly_mini | 915 | 1,921 | 2.10 | chathls |
| syr2k | hlsfactory | 394,721 | 587,234 | 1.49 | chathls |
| syrk | hlsfactory | 619,281 | 558,495 | 0.90 | c2hls |
| gemm_blocked | machsuite_64 | 2,097,166 | 1,319,078 | 0.63 | c2hls |
| covariance | poly_mini | 17,022 | 9,066 | 0.53 | c2hls |
| symm | hlsfactory | 311,596 | 50,432 | 0.16 | c2hls |

**csynth avg**

- n=13; CH wins=8; C2 wins=5; geomean(c2/ch)=2.916

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemm_ncubed | machsuite_64 | 4,627 | 1,319,080 | 285.08 | chathls |
| 3mm | hlsfactory | 7,018 | 90,644 | 12.92 | chathls |
| 2mm | hlsfactory | 7,582 | 54,794 | 7.23 | chathls |
| matmul | poly_mini | 660 | 3,767 | 5.71 | chathls |
| gesummv | poly_mini | 482 | 2,344 | 4.86 | chathls |
| mvt | poly_mini | 1,234 | 5,007 | 4.06 | chathls |
| bicg | poly_mini | 843 | 1,934 | 2.29 | chathls |
| atax | poly_mini | 915 | 1,921 | 2.10 | chathls |
| syr2k | hlsfactory | 394,721 | 392,034 | 0.99 | c2hls |
| syrk | hlsfactory | 388,881 | 363,295 | 0.93 | c2hls |
| covariance | poly_mini | 10,750 | 8,674 | 0.81 | c2hls |
| gemm_blocked | machsuite_64 | 2,097,166 | 1,319,078 | 0.63 | c2hls |
| symm | hlsfactory | 311,596 | 50,432 | 0.16 | c2hls |

**csynth best**

- n=13; CH wins=10; C2 wins=3; geomean(c2/ch)=2.959

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemm_ncubed | machsuite_64 | 4,627 | 1,319,080 | 285.08 | chathls |
| 3mm | hlsfactory | 7,018 | 90,644 | 12.92 | chathls |
| 2mm | hlsfactory | 7,582 | 54,794 | 7.23 | chathls |
| matmul | poly_mini | 660 | 3,767 | 5.71 | chathls |
| gesummv | poly_mini | 482 | 2,344 | 4.86 | chathls |
| mvt | poly_mini | 1,234 | 5,007 | 4.06 | chathls |
| bicg | poly_mini | 843 | 1,934 | 2.29 | chathls |
| atax | poly_mini | 915 | 1,921 | 2.10 | chathls |
| covariance | poly_mini | 4,926 | 8,310 | 1.69 | chathls |
| syrk | hlsfactory | 164,241 | 172,975 | 1.05 | chathls |
| gemm_blocked | machsuite_64 | 2,097,166 | 1,319,078 | 0.63 | c2hls |
| syr2k | hlsfactory | 394,721 | 201,714 | 0.51 | c2hls |
| symm | hlsfactory | 311,596 | 50,432 | 0.16 | c2hls |

**cosim** (both sides measured)

- n=8; CH wins=4; C2 wins=4; geomean(c2/ch)=0.591

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| mvt | poly_mini | 2,979 | 5,103 | 1.71 | chathls |
| atax | poly_mini | 1,265 | 2,015 | 1.59 | chathls |
| 2mm | hlsfactory | 38,695 | 54,763 | 1.42 | chathls |
| gemm_blocked | machsuite_64 | 1,187,910 | 1,319,202 | 1.11 | chathls |
| gesummv | poly_mini | 4,339 | 3,497 | 0.81 | c2hls |
| bicg | poly_mini | 4,113 | 2,034 | 0.49 | c2hls |
| covariance | poly_mini | 74,630 | 8,870 | 0.12 | c2hls |
| syr2k | hlsfactory | 5,232,710 | 384,713 | 0.07 | c2hls |

### vs `RAG2+noskills`

**csynth worst**

- n=14; CH wins=12; C2 wins=2; geomean(c2/ch)=4.344

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| symm | hlsfactory | 311,596 | 42,404,657 | 136.09 | chathls |
| matmul | poly_mini | 660 | 53,217 | 80.63 | chathls |
| gemm_ncubed | machsuite_64 | 4,627 | 307,404 | 66.44 | chathls |
| 3mm | hlsfactory | 7,018 | 90,518 | 12.90 | chathls |
| syr2k | hlsfactory | 394,721 | 1,990,797 | 5.04 | chathls |
| syrk | hlsfactory | 619,281 | 2,699,191 | 4.36 | chathls |
| 2mm | hlsfactory | 7,582 | 25,322 | 3.34 | chathls |
| atax | poly_mini | 915 | 1,859 | 2.03 | chathls |
| gesummv | poly_mini | 482 | 878 | 1.82 | chathls |
| mvt | poly_mini | 1,234 | 2,149 | 1.74 | chathls |
| gemm | poly_mini | 902 | 1,419 | 1.57 | chathls |
| bicg | poly_mini | 843 | 982 | 1.16 | chathls |
| covariance | poly_mini | 17,022 | 6,290 | 0.37 | c2hls |
| gemm_blocked | machsuite_64 | 2,097,166 | 593,986 | 0.28 | c2hls |

**csynth avg**

- n=14; CH wins=12; C2 wins=2; geomean(c2/ch)=4.150

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| matmul | poly_mini | 660 | 53,217 | 80.63 | chathls |
| symm | hlsfactory | 311,596 | 21,731,057 | 69.74 | chathls |
| gemm_ncubed | machsuite_64 | 4,627 | 307,404 | 66.44 | chathls |
| 3mm | hlsfactory | 7,018 | 90,518 | 12.90 | chathls |
| syr2k | hlsfactory | 394,721 | 1,990,797 | 5.04 | chathls |
| syrk | hlsfactory | 388,881 | 1,351,991 | 3.48 | chathls |
| 2mm | hlsfactory | 7,582 | 25,322 | 3.34 | chathls |
| atax | poly_mini | 915 | 1,859 | 2.03 | chathls |
| gesummv | poly_mini | 482 | 878 | 1.82 | chathls |
| mvt | poly_mini | 1,234 | 2,149 | 1.74 | chathls |
| gemm | poly_mini | 902 | 1,419 | 1.57 | chathls |
| bicg | poly_mini | 843 | 982 | 1.16 | chathls |
| covariance | poly_mini | 10,750 | 5,114 | 0.48 | c2hls |
| gemm_blocked | machsuite_64 | 2,097,166 | 593,986 | 0.28 | c2hls |

**csynth best**

- n=14; CH wins=11; C2 wins=3; geomean(c2/ch)=2.803

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| matmul | poly_mini | 660 | 53,217 | 80.63 | chathls |
| gemm_ncubed | machsuite_64 | 4,627 | 307,404 | 66.44 | chathls |
| 3mm | hlsfactory | 7,018 | 90,518 | 12.90 | chathls |
| syr2k | hlsfactory | 394,721 | 1,990,797 | 5.04 | chathls |
| symm | hlsfactory | 311,596 | 1,057,457 | 3.39 | chathls |
| 2mm | hlsfactory | 7,582 | 25,322 | 3.34 | chathls |
| atax | poly_mini | 915 | 1,859 | 2.03 | chathls |
| gesummv | poly_mini | 482 | 878 | 1.82 | chathls |
| mvt | poly_mini | 1,234 | 2,149 | 1.74 | chathls |
| gemm | poly_mini | 902 | 1,419 | 1.57 | chathls |
| bicg | poly_mini | 843 | 982 | 1.16 | chathls |
| covariance | poly_mini | 4,926 | 2,937 | 0.60 | c2hls |
| gemm_blocked | machsuite_64 | 2,097,166 | 593,986 | 0.28 | c2hls |
| syrk | hlsfactory | 164,241 | 38,471 | 0.23 | c2hls |

**cosim** (both sides measured)

*no paired benches*

### vs `scrape+skills`

**csynth worst**

- n=13; CH wins=10; C2 wins=3; geomean(c2/ch)=4.162

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemm_ncubed | machsuite_64 | 4,627 | 1,319,077 | 285.08 | chathls |
| symm | hlsfactory | 311,596 | 48,249,601 | 154.85 | chathls |
| gemm | poly_mini | 902 | 25,405 | 28.17 | chathls |
| 3mm | hlsfactory | 7,018 | 90,644 | 12.92 | chathls |
| 2mm | hlsfactory | 7,582 | 70,402 | 9.29 | chathls |
| gesummv | poly_mini | 482 | 1,115 | 2.31 | chathls |
| mvt | poly_mini | 1,234 | 2,605 | 2.11 | chathls |
| atax | poly_mini | 915 | 1,897 | 2.07 | chathls |
| syr2k | hlsfactory | 394,721 | 800,481 | 2.03 | chathls |
| bicg | poly_mini | 843 | 1,135 | 1.35 | chathls |
| matmul | poly_mini | 660 | 333 | 0.50 | c2hls |
| covariance | poly_mini | 17,022 | 7,525 | 0.44 | c2hls |
| gemm_blocked | machsuite_64 | 2,097,166 | 256,425 | 0.12 | c2hls |

**csynth avg**

- n=13; CH wins=10; C2 wins=3; geomean(c2/ch)=4.080

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemm_ncubed | machsuite_64 | 4,627 | 1,319,077 | 285.08 | chathls |
| symm | hlsfactory | 311,596 | 24,859,201 | 79.78 | chathls |
| gemm | poly_mini | 902 | 25,405 | 28.17 | chathls |
| 3mm | hlsfactory | 7,018 | 90,644 | 12.92 | chathls |
| 2mm | hlsfactory | 7,582 | 70,402 | 9.29 | chathls |
| gesummv | poly_mini | 482 | 1,115 | 2.31 | chathls |
| mvt | poly_mini | 1,234 | 2,605 | 2.11 | chathls |
| atax | poly_mini | 915 | 1,897 | 2.07 | chathls |
| syr2k | hlsfactory | 394,721 | 800,481 | 2.03 | chathls |
| bicg | poly_mini | 843 | 1,135 | 1.35 | chathls |
| covariance | poly_mini | 10,750 | 7,133 | 0.66 | c2hls |
| matmul | poly_mini | 660 | 333 | 0.50 | c2hls |
| gemm_blocked | machsuite_64 | 2,097,166 | 256,425 | 0.12 | c2hls |

**csynth best**

- n=13; CH wins=11; C2 wins=2; geomean(c2/ch)=3.471

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemm_ncubed | machsuite_64 | 4,627 | 1,319,077 | 285.08 | chathls |
| gemm | poly_mini | 902 | 25,405 | 28.17 | chathls |
| 3mm | hlsfactory | 7,018 | 90,644 | 12.92 | chathls |
| 2mm | hlsfactory | 7,582 | 70,402 | 9.29 | chathls |
| symm | hlsfactory | 311,596 | 1,464,001 | 4.70 | chathls |
| gesummv | poly_mini | 482 | 1,115 | 2.31 | chathls |
| mvt | poly_mini | 1,234 | 2,605 | 2.11 | chathls |
| atax | poly_mini | 915 | 1,897 | 2.07 | chathls |
| syr2k | hlsfactory | 394,721 | 800,481 | 2.03 | chathls |
| covariance | poly_mini | 4,926 | 6,769 | 1.37 | chathls |
| bicg | poly_mini | 843 | 1,135 | 1.35 | chathls |
| matmul | poly_mini | 660 | 333 | 0.50 | c2hls |
| gemm_blocked | machsuite_64 | 2,097,166 | 256,425 | 0.12 | c2hls |

**cosim** (both sides measured)

*no paired benches*

### vs `scrape+noskills`

**csynth worst**

- n=11; CH wins=8; C2 wins=3; geomean(c2/ch)=2.565

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemm_ncubed | machsuite_64 | 4,627 | 306,388 | 66.22 | chathls |
| 3mm | hlsfactory | 7,018 | 307,576 | 43.83 | chathls |
| matmul | poly_mini | 660 | 3,230 | 4.89 | chathls |
| gemm | poly_mini | 902 | 3,251 | 3.60 | chathls |
| syrk | hlsfactory | 619,281 | 1,294,152 | 2.09 | chathls |
| bicg | poly_mini | 843 | 1,708 | 2.03 | chathls |
| atax | poly_mini | 915 | 1,288 | 1.41 | chathls |
| gemm_blocked | machsuite_64 | 2,097,166 | 2,719,783 | 1.30 | chathls |
| gesummv | poly_mini | 482 | 245 | 0.51 | c2hls |
| mvt | poly_mini | 1,234 | 499 | 0.40 | c2hls |
| covariance | poly_mini | 17,022 | 6,608 | 0.39 | c2hls |

**csynth avg**

- n=11; CH wins=8; C2 wins=3; geomean(c2/ch)=2.717

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemm_ncubed | machsuite_64 | 4,627 | 306,388 | 66.22 | chathls |
| 3mm | hlsfactory | 7,018 | 307,576 | 43.83 | chathls |
| matmul | poly_mini | 660 | 3,230 | 4.89 | chathls |
| gemm | poly_mini | 902 | 3,251 | 3.60 | chathls |
| syrk | hlsfactory | 388,881 | 1,099,272 | 2.83 | chathls |
| bicg | poly_mini | 843 | 1,708 | 2.03 | chathls |
| atax | poly_mini | 915 | 1,288 | 1.41 | chathls |
| gemm_blocked | machsuite_64 | 2,097,166 | 2,719,783 | 1.30 | chathls |
| covariance | poly_mini | 10,750 | 5,824 | 0.54 | c2hls |
| gesummv | poly_mini | 482 | 245 | 0.51 | c2hls |
| mvt | poly_mini | 1,234 | 499 | 0.40 | c2hls |

**csynth best**

- n=11; CH wins=9; C2 wins=2; geomean(c2/ch)=3.064

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemm_ncubed | machsuite_64 | 4,627 | 306,388 | 66.22 | chathls |
| 3mm | hlsfactory | 7,018 | 307,576 | 43.83 | chathls |
| syrk | hlsfactory | 164,241 | 912,072 | 5.55 | chathls |
| matmul | poly_mini | 660 | 3,230 | 4.89 | chathls |
| gemm | poly_mini | 902 | 3,251 | 3.60 | chathls |
| bicg | poly_mini | 843 | 1,708 | 2.03 | chathls |
| atax | poly_mini | 915 | 1,288 | 1.41 | chathls |
| gemm_blocked | machsuite_64 | 2,097,166 | 2,719,783 | 1.30 | chathls |
| covariance | poly_mini | 4,926 | 5,096 | 1.03 | chathls |
| gesummv | poly_mini | 482 | 245 | 0.51 | c2hls |
| mvt | poly_mini | 1,234 | 499 | 0.40 | c2hls |

**cosim** (both sides measured)

*no paired benches*

### vs `skills`

**csynth worst**

- n=14; CH wins=12; C2 wins=2; geomean(c2/ch)=7.797

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| symm | hlsfactory | 311,596 | 90,408,002 | 290.14 | chathls |
| gemm_ncubed | machsuite_64 | 4,627 | 1,319,077 | 285.08 | chathls |
| gemm | poly_mini | 902 | 31,419 | 34.83 | chathls |
| 3mm | hlsfactory | 7,018 | 92,640 | 13.20 | chathls |
| 2mm | hlsfactory | 7,582 | 53,914 | 7.11 | chathls |
| syr2k | hlsfactory | 394,721 | 2,709,768 | 6.87 | chathls |
| matmul | poly_mini | 660 | 3,640 | 5.52 | chathls |
| gesummv | poly_mini | 482 | 2,315 | 4.80 | chathls |
| syrk | hlsfactory | 619,281 | 2,710,074 | 4.38 | chathls |
| mvt | poly_mini | 1,234 | 5,007 | 4.06 | chathls |
| atax | poly_mini | 915 | 3,318 | 3.63 | chathls |
| bicg | poly_mini | 843 | 2,596 | 3.08 | chathls |
| gemm_blocked | machsuite_64 | 2,097,166 | 1,695,911 | 0.81 | c2hls |
| covariance | poly_mini | 17,022 | 6,626 | 0.39 | c2hls |

**csynth avg**

- n=14; CH wins=12; C2 wins=2; geomean(c2/ch)=7.014

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemm_ncubed | machsuite_64 | 4,627 | 1,319,077 | 285.08 | chathls |
| symm | hlsfactory | 311,596 | 34,190,402 | 109.73 | chathls |
| gemm | poly_mini | 902 | 31,419 | 34.83 | chathls |
| 3mm | hlsfactory | 7,018 | 92,640 | 13.20 | chathls |
| 2mm | hlsfactory | 7,582 | 53,914 | 7.11 | chathls |
| matmul | poly_mini | 660 | 3,640 | 5.52 | chathls |
| gesummv | poly_mini | 482 | 2,315 | 4.80 | chathls |
| mvt | poly_mini | 1,234 | 5,007 | 4.06 | chathls |
| atax | poly_mini | 915 | 3,318 | 3.63 | chathls |
| syrk | hlsfactory | 388,881 | 1,362,874 | 3.50 | chathls |
| syr2k | hlsfactory | 394,721 | 1,365,768 | 3.46 | chathls |
| bicg | poly_mini | 843 | 2,596 | 3.08 | chathls |
| gemm_blocked | machsuite_64 | 2,097,166 | 1,695,911 | 0.81 | c2hls |
| covariance | poly_mini | 10,750 | 6,234 | 0.58 | c2hls |

**csynth best**

- n=14; CH wins=11; C2 wins=3; geomean(c2/ch)=3.876

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemm_ncubed | machsuite_64 | 4,627 | 1,319,077 | 285.08 | chathls |
| gemm | poly_mini | 902 | 31,419 | 34.83 | chathls |
| 3mm | hlsfactory | 7,018 | 92,640 | 13.20 | chathls |
| 2mm | hlsfactory | 7,582 | 53,914 | 7.11 | chathls |
| matmul | poly_mini | 660 | 3,640 | 5.52 | chathls |
| gesummv | poly_mini | 482 | 2,315 | 4.80 | chathls |
| mvt | poly_mini | 1,234 | 5,007 | 4.06 | chathls |
| symm | hlsfactory | 311,596 | 1,200,002 | 3.85 | chathls |
| atax | poly_mini | 915 | 3,280 | 3.58 | chathls |
| bicg | poly_mini | 843 | 2,596 | 3.08 | chathls |
| covariance | poly_mini | 4,926 | 5,870 | 1.19 | chathls |
| gemm_blocked | machsuite_64 | 2,097,166 | 1,695,911 | 0.81 | c2hls |
| syrk | hlsfactory | 164,241 | 49,354 | 0.30 | c2hls |
| syr2k | hlsfactory | 394,721 | 55,368 | 0.14 | c2hls |

**cosim** (both sides measured)

*no paired benches*

### Native vs **best** DeepSeek poly flavor (per bench, per metric)

**csynth worst — best DS flavor**

- n=14; CH wins=7; C2 wins=7; geomean(c2/ch)=1.155

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemm_ncubed | machsuite_64 | 4,627 | 306,388 | 66.22 | chathls |
| 3mm | hlsfactory | 7,018 | 90,518 | 12.90 | chathls |
| 2mm | hlsfactory | 7,582 | 25,322 | 3.34 | chathls |
| gemm | poly_mini | 902 | 1,419 | 1.57 | chathls |
| syr2k | hlsfactory | 394,721 | 587,234 | 1.49 | chathls |
| atax | poly_mini | 915 | 1,288 | 1.41 | chathls |
| bicg | poly_mini | 843 | 982 | 1.16 | chathls |
| syrk | hlsfactory | 619,281 | 558,495 | 0.90 | c2hls |
| gesummv | poly_mini | 482 | 245 | 0.51 | c2hls |
| matmul | poly_mini | 660 | 333 | 0.50 | c2hls |
| mvt | poly_mini | 1,234 | 499 | 0.40 | c2hls |
| covariance | poly_mini | 17,022 | 6,290 | 0.37 | c2hls |
| symm | hlsfactory | 311,596 | 50,432 | 0.16 | c2hls |
| gemm_blocked | machsuite_64 | 2,097,166 | 256,425 | 0.12 | c2hls |

**csynth avg — best DS flavor**

- n=14; CH wins=6; C2 wins=8; geomean(c2/ch)=1.145

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemm_ncubed | machsuite_64 | 4,627 | 306,388 | 66.22 | chathls |
| 3mm | hlsfactory | 7,018 | 90,518 | 12.90 | chathls |
| 2mm | hlsfactory | 7,582 | 25,322 | 3.34 | chathls |
| gemm | poly_mini | 902 | 1,419 | 1.57 | chathls |
| atax | poly_mini | 915 | 1,288 | 1.41 | chathls |
| bicg | poly_mini | 843 | 982 | 1.16 | chathls |
| syr2k | hlsfactory | 394,721 | 392,034 | 0.99 | c2hls |
| syrk | hlsfactory | 388,881 | 363,295 | 0.93 | c2hls |
| gesummv | poly_mini | 482 | 245 | 0.51 | c2hls |
| matmul | poly_mini | 660 | 333 | 0.50 | c2hls |
| covariance | poly_mini | 10,750 | 5,114 | 0.48 | c2hls |
| mvt | poly_mini | 1,234 | 499 | 0.40 | c2hls |
| symm | hlsfactory | 311,596 | 50,432 | 0.16 | c2hls |
| gemm_blocked | machsuite_64 | 2,097,166 | 256,425 | 0.12 | c2hls |

**csynth best — best DS flavor**

- n=14; CH wins=6; C2 wins=8; geomean(c2/ch)=0.917

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemm_ncubed | machsuite_64 | 4,627 | 306,388 | 66.22 | chathls |
| 3mm | hlsfactory | 7,018 | 90,518 | 12.90 | chathls |
| 2mm | hlsfactory | 7,582 | 25,322 | 3.34 | chathls |
| gemm | poly_mini | 902 | 1,419 | 1.57 | chathls |
| atax | poly_mini | 915 | 1,288 | 1.41 | chathls |
| bicg | poly_mini | 843 | 982 | 1.16 | chathls |
| covariance | poly_mini | 4,926 | 2,937 | 0.60 | c2hls |
| gesummv | poly_mini | 482 | 245 | 0.51 | c2hls |
| matmul | poly_mini | 660 | 333 | 0.50 | c2hls |
| mvt | poly_mini | 1,234 | 499 | 0.40 | c2hls |
| syrk | hlsfactory | 164,241 | 38,471 | 0.23 | c2hls |
| symm | hlsfactory | 311,596 | 50,432 | 0.16 | c2hls |
| syr2k | hlsfactory | 394,721 | 55,368 | 0.14 | c2hls |
| gemm_blocked | machsuite_64 | 2,097,166 | 256,425 | 0.12 | c2hls |

## B. Csynth: ChatHLS **c2hlsport** vs DeepSeek **`hlsfactory_*`** (real c2hls poly sizes)

This is the fair poly compare. Both sides use HLSFactory dims (gemm **60×70×80**). Example: CH **342,901** vs c2hls **96,373**.

**csynth worst**

- n=8; CH wins=2; C2 wins=6; geomean(c2/ch)=0.479

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemver | hlsfactory | 30,916 | 158,742 | 5.13 | chathls |
| durbin | hlsfactory | 110,197 | 113,550 | 1.03 | chathls |
| trisolv | hlsfactory | 275,761 | 204,511 | 0.74 | c2hls |
| trmm | hlsfactory | 5,726,401 | 3,744,048 | 0.65 | c2hls |
| lu | hlsfactory | 42,785,041 | 14,683,113 | 0.34 | c2hls |
| gemm | hlsfactory | 342,901 | 96,373 | 0.28 | c2hls |
| covariance | hlsfactory | 5,418,752 | 1,139,839 | 0.21 | c2hls |
| syrk | hlsfactory | 7,809,761 | 414,420 | 0.05 | c2hls |

**csynth avg**

- n=8; CH wins=2; C2 wins=6; geomean(c2/ch)=0.548

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemver | hlsfactory | 30,916 | 158,742 | 5.13 | chathls |
| durbin | hlsfactory | 61,050 | 78,207 | 1.28 | chathls |
| trisolv | hlsfactory | 138,961 | 110,911 | 0.80 | c2hls |
| trmm | hlsfactory | 2,846,401 | 1,872,048 | 0.66 | c2hls |
| lu | hlsfactory | 10,735,441 | 3,925,833 | 0.37 | c2hls |
| gemm | hlsfactory | 342,901 | 96,373 | 0.28 | c2hls |
| covariance | hlsfactory | 2,717,952 | 599,039 | 0.22 | c2hls |
| syrk | hlsfactory | 3,931,361 | 408,020 | 0.10 | c2hls |

**csynth best**

- n=8; CH wins=6; C2 wins=2; geomean(c2/ch)=2.613

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| lu | hlsfactory | 721 | 30,873 | 42.82 | chathls |
| gemver | hlsfactory | 30,916 | 158,742 | 5.13 | chathls |
| trisolv | hlsfactory | 4,441 | 18,871 | 4.25 | chathls |
| durbin | hlsfactory | 11,903 | 43,340 | 3.64 | chathls |
| syrk | hlsfactory | 149,921 | 401,780 | 2.68 | chathls |
| trmm | hlsfactory | 62,401 | 62,448 | 1.00 | chathls |
| covariance | hlsfactory | 84,672 | 71,759 | 0.85 | c2hls |
| gemm | hlsfactory | 342,901 | 96,373 | 0.28 | c2hls |

## C. Csynth: machsuite **64×64** gemm (size-matched)

ChatHLS `gemm_ncubed` **4,627** is real top-XML (DSP≈1408). Compared only to same-N partners.

**native vs chathls poly — csynth worst**

- n=2; CH wins=1; C2 wins=1; geomean(c2/ch)=13.391

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemm_ncubed | machsuite_64 | 4,627 | 1,319,080 | 285.08 | chathls |
| gemm_blocked | machsuite_64 | 2,097,166 | 1,319,078 | 0.63 | c2hls |

**port vs machsuite campaign — csynth worst**

- n=2; CH wins=2; C2 wins=0; geomean(c2/ch)=63.455

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemm_ncubed | machsuite_64 | 4,627 | 10,571,928 | 2284.83 | chathls |
| stencil2d | machsuite_64 | 12,108 | 21,338 | 1.76 | chathls |

**native vs chathls poly — csynth avg**

- n=2; CH wins=1; C2 wins=1; geomean(c2/ch)=13.391

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemm_ncubed | machsuite_64 | 4,627 | 1,319,080 | 285.08 | chathls |
| gemm_blocked | machsuite_64 | 2,097,166 | 1,319,078 | 0.63 | c2hls |

**port vs machsuite campaign — csynth avg**

- n=2; CH wins=2; C2 wins=0; geomean(c2/ch)=63.455

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemm_ncubed | machsuite_64 | 4,627 | 10,571,928 | 2284.83 | chathls |
| stencil2d | machsuite_64 | 12,108 | 21,338 | 1.76 | chathls |

**native vs chathls poly — csynth best**

- n=2; CH wins=1; C2 wins=1; geomean(c2/ch)=13.391

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemm_ncubed | machsuite_64 | 4,627 | 1,319,080 | 285.08 | chathls |
| gemm_blocked | machsuite_64 | 2,097,166 | 1,319,078 | 0.63 | c2hls |

**port vs machsuite campaign — csynth best**

- n=2; CH wins=2; C2 wins=0; geomean(c2/ch)=63.455

| Bench | size | CH | C2 | c2/ch | winner |
|-------|------|---:|---:|------:|--------|
| gemm_ncubed | machsuite_64 | 4,627 | 10,571,928 | 2284.83 | chathls |
| stencil2d | machsuite_64 | 12,108 | 21,338 | 1.76 | chathls |

## D. Explicitly **rejected** (size mismatch)

| ChatHLS | size | c2hls | size | why rejected |
|---------|------|-------|------|--------------|
| c2hlsport `hlsfactory_gemm` = 342,901 | hlsfactory 60×70×80 | poly `chathls_gemm` ~3k–25k | poly_mini 20×25×30 | different NI/NJ/NK |
| c2hlsport `hlsfactory_atax` = 8,372 | hlsfactory 116×124 | poly `chathls_atax` ~1–3k | poly_mini 38×42 | different M/N |
| native Mini gemm = 902 | poly_mini | hlsfactory camp gemm = 96,373 | hlsfactory | different NI/NJ/NK |

## E. Coverage

| System | DeepSeek | Devstral | GLM-4.7 |
|--------|----------|----------|---------|
| ChatHLS U280 | native / c2hlsport / tierA | — | — |
| c2hls | poly Mini + hlsfactory + machsuite ports | poly + Tier-A | poly |

