# ii_target_miss — Phase B code snippets (all hlsfactory_*)

From `flash_skills_20260620_004507` baseline synthesis reports.
Each block is the loop HLS flagged with **II Violation** (feeds `ii_target_miss` bottleneck kind).

**Dominant root cause:** pipelined loops reading/writing **global DDR arrays** (`A[i][j]`, `B[i][j]`, …) through `m_axi` without local staging — scheduler reports **Memory Dependency** on `gmem` request/response.

## 2mm

### `VITIS_LOOP_34_1_VITIS_LOOP_35_2_VITIS_LOOP_37_3` — achieved II=168, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_2mm_phase_b_synth_003/kernel.cpp:35`

```cpp
      29|     const int nk = NK;
      30|     const int nl = NL;
      31| 
      32|     int i, j, k;
      33| 
      34|     for (i = 0; i < ni; i++)
>>>   35|         for (j = 0; j < nj; j++) {
      36|             tmp[i][j] = 0.0;
      37|             for (k = 0; k < nk; ++k)
      38|                 tmp[i][j] += alpha * A[i][k] * B[k][j];
      39|         }
      40|     for (i = 0; i < ni; i++)
      41|         for (j = 0; j < nl; j++) {
```

## 3mm

### `VITIS_LOOP_37_1_VITIS_LOOP_38_2` — achieved II=75, Resource Limitation

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_3mm_phase_b_synth_003/kernel.cpp:37`

```cpp
      31|     const int nk = NK;
      32|     const int nl = NL;
      33|     const int nm = NM;
      34| 
      35|     int i, j, k;
      36| 
>>>   37|     for (i = 0; i < ni; i++)
      38|         for (j = 0; j < nj; j++)
      39|         {
      40|             E[i][j] = 0.0;
      41|             for (k = 0; k < nk; ++k)
      42|                 E[i][j] += A[i][k] * B[k][j];
      43|         }
```

### `VITIS_LOOP_45_4_VITIS_LOOP_46_5_VITIS_LOOP_49_6` — achieved II=161, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_3mm_phase_b_synth_003/kernel.cpp:46`

```cpp
      40|             E[i][j] = 0.0;
      41|             for (k = 0; k < nk; ++k)
      42|                 E[i][j] += A[i][k] * B[k][j];
      43|         }
      44| 
      45|     for (i = 0; i < nj; i++)
>>>   46|         for (j = 0; j < nl; j++)
      47|         {
      48|             F[i][j] = 0.0;
      49|             for (k = 0; k < nm; ++k)
      50|                 F[i][j] += C[i][k] * D[k][j];
      51|         }
      52| 
```

### `VITIS_LOOP_53_7_VITIS_LOOP_54_8` — achieved II=75, Resource Limitation

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_3mm_phase_b_synth_003/kernel.cpp:53`

```cpp
      47|         {
      48|             F[i][j] = 0.0;
      49|             for (k = 0; k < nm; ++k)
      50|                 F[i][j] += C[i][k] * D[k][j];
      51|         }
      52| 
>>>   53|     for (i = 0; i < ni; i++)
      54|         for (j = 0; j < nl; j++)
      55|         {
      56|             G[i][j] = 0.0;
      57|             for (k = 0; k < nj; ++k)
      58|                 G[i][j] += E[i][k] * F[k][j];
      59|         }
```

## atax

### `VITIS_LOOP_30_3` — achieved II=7, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_atax_phase_b_synth_001/kernel.cpp:30`

```cpp
      24| 
      25|     for (i = 0; i < n; i++)
      26|         y[i] = 0;
      27|     for (i = 0; i < m; i++)
      28|     {
      29|         tmp[i] = 0.0;
>>>   30|         for (j = 0; j < n; j++)
      31|             tmp[i] = tmp[i] + A[i][j] * x[j];
      32|         for (j = 0; j < n; j++)
      33|             y[j] = y[j] + A[i][j] * tmp[i];
      34|     }
      35| }
      36| }
```

### `VITIS_LOOP_32_4` — achieved II=160, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_atax_phase_b_synth_001/kernel.cpp:32`

```cpp
      26|         y[i] = 0;
      27|     for (i = 0; i < m; i++)
      28|     {
      29|         tmp[i] = 0.0;
      30|         for (j = 0; j < n; j++)
      31|             tmp[i] = tmp[i] + A[i][j] * x[j];
>>>   32|         for (j = 0; j < n; j++)
      33|             y[j] = y[j] + A[i][j] * tmp[i];
      34|     }
      35| }
      36| }
```

## bicg

### `VITIS_LOOP_33_3` — achieved II=160, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_bicg_phase_b_synth_001/kernel.cpp:33`

```cpp
      27| 
      28|     for (i = 0; i < m; i++)
      29|         s[i] = 0;
      30|     for (i = 0; i < n; i++)
      31|     {
      32|         q[i] = 0.0;
>>>   33|         for (j = 0; j < m; j++)
      34|         {
      35|             s[j] = s[j] + r[i] * A[i][j];
      36|             q[i] = q[i] + A[i][j] * p[j];
      37|         }
      38|     }
      39| }
```

## cholesky

### `VITIS_LOOP_15_3` — achieved II=158, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_cholesky_phase_b_synth_001/kernel.cpp:15`

```cpp
       9| 
      10|   int i, j, k;
      11| 
      12|   for (i = 0; i < n; i++) {
      13| 
      14|      for (j = 0; j < i; j++) {
>>>   15|         for (k = 0; k < j; k++) {
      16|            A[i][j] -= A[i][k] * A[j][k];
      17|         }
      18|         A[i][j] /= A[j][j];
      19|      }
      20| 
      21|      for (k = 0; k < i; k++) {
```

### `VITIS_LOOP_21_4` — achieved II=149, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_cholesky_phase_b_synth_001/kernel.cpp:21`

```cpp
      15|         for (k = 0; k < j; k++) {
      16|            A[i][j] -= A[i][k] * A[j][k];
      17|         }
      18|         A[i][j] /= A[j][j];
      19|      }
      20| 
>>>   21|      for (k = 0; k < i; k++) {
      22|         A[i][i] -= A[i][k] * A[i][k];
      23|      }
      24|      A[i][i] = sqrt(A[i][i]);
      25|   }
      26| 
      27| }
```

## correlation

### `VITIS_LOOP_29_1_VITIS_LOOP_32_2` — achieved II=183, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_correlation_phase_b_synth_001/kernel.cpp:29`

```cpp
      23|     const int m = M;
      24| 
      25|     int i, j, k;
      26| 
      27|     double eps = 0.1;
      28| 
>>>   29|     for (j = 0; j < m; j++)
      30|     {
      31|         mean[j] = 0.0;
      32|         for (i = 0; i < n; i++)
      33|             mean[j] += data[i][j];
      34|         mean[j] /= float_n;
      35|     }
```

### `VITIS_LOOP_40_4` — achieved II=7, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_correlation_phase_b_synth_001/kernel.cpp:40`

```cpp
      34|         mean[j] /= float_n;
      35|     }
      36| 
      37|     for (j = 0; j < m; j++)
      38|     {
      39|         stddev[j] = 0.0;
>>>   40|         for (i = 0; i < n; i++)
      41|             stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
      42|         stddev[j] /= float_n;
      43|         stddev[j] = sqrt(stddev[j]);
      44| 
      45|         stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
      46|     }
```

### `VITIS_LOOP_48_5_VITIS_LOOP_49_6` — achieved II=184, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_correlation_phase_b_synth_001/kernel.cpp:48`

```cpp
      42|         stddev[j] /= float_n;
      43|         stddev[j] = sqrt(stddev[j]);
      44| 
      45|         stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
      46|     }
      47| 
>>>   48|     for (i = 0; i < n; i++)
      49|         for (j = 0; j < m; j++)
      50|         {
      51|             data[i][j] -= mean[j];
      52|             data[i][j] /= sqrt(float_n) * stddev[j];
      53|         }
      54| 
```

### `VITIS_LOOP_58_8_VITIS_LOOP_61_9` — achieved II=161, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_correlation_phase_b_synth_001/kernel.cpp:58`

```cpp
      52|             data[i][j] /= sqrt(float_n) * stddev[j];
      53|         }
      54| 
      55|     for (i = 0; i < m-1; i++)
      56|     {
      57|         corr[i][i] = 1.0;
>>>   58|         for (j = i+1; j < m; j++)
      59|         {
      60|             corr[i][j] = 0.0;
      61|             for (k = 0; k < n; k++)
      62|                 corr[i][j] += (data[k][i] * data[k][j]);
      63|             corr[j][i] = corr[i][j];
      64|         }
```

## covariance

### `VITIS_LOOP_24_1_VITIS_LOOP_27_2` — achieved II=183, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_covariance_phase_b_synth_001/kernel.cpp:24`

```cpp
      18| 
      19|     const int n = N;
      20|     const int m = M;
      21| 
      22|     int i, j, k;
      23| 
>>>   24|     for (j = 0; j < m; j++)
      25|     {
      26|         mean[j] = 0.0;
      27|         for (i = 0; i < n; i++)
      28|             mean[j] += data[i][j];
      29|         mean[j] /= float_n;
      30|     }
```

### `VITIS_LOOP_32_3_VITIS_LOOP_33_4` — achieved II=151, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_covariance_phase_b_synth_001/kernel.cpp:32`

```cpp
      26|         mean[j] = 0.0;
      27|         for (i = 0; i < n; i++)
      28|             mean[j] += data[i][j];
      29|         mean[j] /= float_n;
      30|     }
      31| 
>>>   32|     for (i = 0; i < n; i++)
      33|         for (j = 0; j < m; j++)
      34|             data[i][j] -= mean[j];
      35| 
      36|     for (i = 0; i < m; i++)
      37|         for (j = i; j < m; j++)
      38|         {
```

### `VITIS_LOOP_37_6_VITIS_LOOP_40_7` — achieved II=192, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_covariance_phase_b_synth_001/kernel.cpp:37`

```cpp
      31| 
      32|     for (i = 0; i < n; i++)
      33|         for (j = 0; j < m; j++)
      34|             data[i][j] -= mean[j];
      35| 
      36|     for (i = 0; i < m; i++)
>>>   37|         for (j = i; j < m; j++)
      38|         {
      39|             cov[i][j] = 0.0;
      40|             for (k = 0; k < n; k++)
      41|                 cov[i][j] += data[k][i] * data[k][j];
      42|             cov[i][j] /= (float_n - 1.0);
      43|             cov[j][i] = cov[i][j];
```

## doitgen

*No ii_target_miss (failed before Phase B)*

## durbin

### `VITIS_LOOP_30_2` — achieved II=7, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_durbin_phase_b_synth_001/kernel.cpp:30`

```cpp
      24|  beta = 1.0;
      25|  alpha = -r[0];
      26| 
      27|  for (k = 1; k < n; k++) {
      28|    beta = (1-alpha*alpha)*beta;
      29|    sum = 0.0;
>>>   30|    for (i=0; i<k; i++) {
      31|       sum += r[k-i-1]*y[i];
      32|    }
      33|    alpha = - (r[k] + sum)/beta;
      34| 
      35|    for (i=0; i<k; i++) {
      36|       z[i] = y[i] + alpha*y[k-i-1];
```

### `VITIS_LOOP_35_3` — achieved II=2, Resource Limitation

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_durbin_phase_b_synth_001/kernel.cpp:35`

```cpp
      29|    sum = 0.0;
      30|    for (i=0; i<k; i++) {
      31|       sum += r[k-i-1]*y[i];
      32|    }
      33|    alpha = - (r[k] + sum)/beta;
      34| 
>>>   35|    for (i=0; i<k; i++) {
      36|       z[i] = y[i] + alpha*y[k-i-1];
      37|    }
      38|    for (i=0; i<k; i++) {
      39|      y[i] = z[i];
      40|    }
      41|    y[k] = alpha;
```

## fdtd-2d

### `VITIS_LOOP_30_3_VITIS_LOOP_31_4` — achieved II=168, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_fdtd-2d_phase_b_synth_001/kernel.cpp:30`

```cpp
      24|     int t, i, j;
      25| 
      26|     for(t = 0; t < tmax; t++)
      27|     {
      28|         for (j = 0; j < ny; j++)
      29|             ey[0][j] = _fict_[t];
>>>   30|         for (i = 1; i < nx; i++)
      31|             for (j = 0; j < ny; j++)
      32|                 ey[i][j] = ey[i][j] - 0.5*(hz[i][j]-hz[i-1][j]);
      33|         for (i = 0; i < nx; i++)
      34|             for (j = 1; j < ny; j++)
      35|                 ex[i][j] = ex[i][j] - 0.5*(hz[i][j]-hz[i][j-1]);
      36|         for (i = 0; i < nx - 1; i++)
```

### `VITIS_LOOP_33_5_VITIS_LOOP_34_6` — achieved II=169, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_fdtd-2d_phase_b_synth_001/kernel.cpp:33`

```cpp
      27|     {
      28|         for (j = 0; j < ny; j++)
      29|             ey[0][j] = _fict_[t];
      30|         for (i = 1; i < nx; i++)
      31|             for (j = 0; j < ny; j++)
      32|                 ey[i][j] = ey[i][j] - 0.5*(hz[i][j]-hz[i-1][j]);
>>>   33|         for (i = 0; i < nx; i++)
      34|             for (j = 1; j < ny; j++)
      35|                 ex[i][j] = ex[i][j] - 0.5*(hz[i][j]-hz[i][j-1]);
      36|         for (i = 0; i < nx - 1; i++)
      37|             for (j = 0; j < ny - 1; j++)
      38|                 hz[i][j] = hz[i][j] - 0.7*(ex[i][j+1] - ex[i][j] +
      39|                                            ey[i+1][j] - ey[i][j]);
```

### `VITIS_LOOP_36_7_VITIS_LOOP_37_8` — achieved II=185, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_fdtd-2d_phase_b_synth_001/kernel.cpp:36`

```cpp
      30|         for (i = 1; i < nx; i++)
      31|             for (j = 0; j < ny; j++)
      32|                 ey[i][j] = ey[i][j] - 0.5*(hz[i][j]-hz[i-1][j]);
      33|         for (i = 0; i < nx; i++)
      34|             for (j = 1; j < ny; j++)
      35|                 ex[i][j] = ex[i][j] - 0.5*(hz[i][j]-hz[i][j-1]);
>>>   36|         for (i = 0; i < nx - 1; i++)
      37|             for (j = 0; j < ny - 1; j++)
      38|                 hz[i][j] = hz[i][j] - 0.7*(ex[i][j+1] - ex[i][j] +
      39|                                            ey[i+1][j] - ey[i][j]);
      40|     }
      41| }
      42| }
```

## floyd-warshall

### `VITIS_LOOP_12_1_VITIS_LOOP_13_2_VITIS_LOOP_14_3` — achieved II=143, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_floyd-warshall_phase_b_synth_001/kernel.cpp:12`

```cpp
       6| #pragma HLS INTERFACE s_axilite port=path bundle=control
       7| #pragma HLS INTERFACE s_axilite port=return bundle=control
       8| 
       9|     const int n = N;
      10|     int i, j, k;
      11| 
>>>   12|     for (k = 0; k < n; k++) {
      13|         for (i = 0; i < n; i++) {
      14|             for (j = 0; j < n; j++) {
      15|                 path[i][j] = path[i][j] < path[i][k] + path[k][j] ?
      16|                     path[i][j] : path[i][k] + path[k][j];
      17|             }
      18|         }
```

## gemm

### `VITIS_LOOP_27_2` — achieved II=151, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_gemm_phase_b_synth_003/kernel.cpp:27`

```cpp
      21|     const int ni = NI;
      22|     const int nj = NJ;
      23|     const int nk = NK;
      24| 
      25|     int i, j, k;
      26|     for (i = 0; i < ni; i++) {
>>>   27|         for (j = 0; j < nj; j++)
      28|             C[i][j] *= beta;
      29|         for (k = 0; k < nk; k++) {
      30|             for (j = 0; j < nj; j++)
      31|                 C[i][j] += alpha * A[i][k] * B[k][j];
      32|         }
      33|     }
```

### `VITIS_LOOP_30_4` — achieved II=159, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_gemm_phase_b_synth_003/kernel.cpp:30`

```cpp
      24| 
      25|     int i, j, k;
      26|     for (i = 0; i < ni; i++) {
      27|         for (j = 0; j < nj; j++)
      28|             C[i][j] *= beta;
      29|         for (k = 0; k < nk; k++) {
>>>   30|             for (j = 0; j < nj; j++)
      31|                 C[i][j] += alpha * A[i][k] * B[k][j];
      32|         }
      33|     }
      34| }
      35| }
```

## gemver

### `VITIS_LOOP_44_2` — achieved II=169, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_gemver_phase_b_synth_001/kernel.cpp:44`

```cpp
      38| 
      39|     const int n = N;
      40| 
      41|   int i, j;
      42| 
      43|   for (i = 0; i < n; i++)
>>>   44|     for (j = 0; j < n; j++)
      45|       A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
      46| 
      47|   for (i = 0; i < n; i++)
      48|     for (j = 0; j < n; j++)
      49|       x[i] = x[i] + beta * A[j][i] * y[j];
      50| 
```

### `VITIS_LOOP_48_4` — achieved II=7, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_gemver_phase_b_synth_001/kernel.cpp:48`

```cpp
      42| 
      43|   for (i = 0; i < n; i++)
      44|     for (j = 0; j < n; j++)
      45|       A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
      46| 
      47|   for (i = 0; i < n; i++)
>>>   48|     for (j = 0; j < n; j++)
      49|       x[i] = x[i] + beta * A[j][i] * y[j];
      50| 
      51|   for (i = 0; i < n; i++)
      52|     x[i] = x[i] + z[i];
      53| 
      54|   for (i = 0; i < n; i++)
```

### `VITIS_LOOP_51_5` — achieved II=151, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_gemver_phase_b_synth_001/kernel.cpp:51`

```cpp
      45|       A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
      46| 
      47|   for (i = 0; i < n; i++)
      48|     for (j = 0; j < n; j++)
      49|       x[i] = x[i] + beta * A[j][i] * y[j];
      50| 
>>>   51|   for (i = 0; i < n; i++)
      52|     x[i] = x[i] + z[i];
      53| 
      54|   for (i = 0; i < n; i++)
      55|     for (j = 0; j < n; j++)
      56|       w[i] = w[i] +  alpha * A[i][j] * x[j];
      57| }
```

### `VITIS_LOOP_55_7` — achieved II=7, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_gemver_phase_b_synth_001/kernel.cpp:55`

```cpp
      49|       x[i] = x[i] + beta * A[j][i] * y[j];
      50| 
      51|   for (i = 0; i < n; i++)
      52|     x[i] = x[i] + z[i];
      53| 
      54|   for (i = 0; i < n; i++)
>>>   55|     for (j = 0; j < n; j++)
      56|       w[i] = w[i] +  alpha * A[i][j] * x[j];
      57| }
      58| }
```

## gesummv

### `VITIS_LOOP_18_1_VITIS_LOOP_22_2` — achieved II=178, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_gesummv_phase_b_synth_001/kernel.cpp:18`

```cpp
      12| {
      13| 
      14|     const int n = N;
      15| 
      16|   int i, j;
      17| 
>>>   18|   for (i = 0; i < n; i++)
      19|     {
      20|       tmp[i] = 0.0;
      21|       y[i] = 0.0;
      22|       for (j = 0; j < n; j++)
      23| 	{
      24| 	  tmp[i] = A[i][j] * x[j] + tmp[i];
```

## gramschmidt

### `VITIS_LOOP_32_2` — achieved II=7, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_gramschmidt_phase_b_synth_001/kernel.cpp:32`

```cpp
      26| 
      27|   double nrm;
      28| 
      29|   for (k = 0; k < n; k++)
      30|     {
      31|       nrm = 0.0;
>>>   32|       for (i = 0; i < m; i++)
      33|         nrm += A[i][k] * A[i][k];
      34|       R[k][k] = sqrt(nrm);
      35|       for (i = 0; i < m; i++)
      36|         Q[i][k] = A[i][k] / R[k][k];
      37|       for (j = k + 1; j < n; j++)
      38|         {
```

## heat-3d

### `VITIS_LOOP_23_2_VITIS_LOOP_24_3_VITIS_LOOP_25_4` — achieved II=7, Resource Limitation

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_heat-3d_phase_b_synth_001/kernel.cpp:28`

```cpp
      22|     for (t = 1; t <= 40; t++) {
      23|         for (i = 1; i < n-1; i++) {
      24|             for (j = 1; j < n-1; j++) {
      25|                 for (k = 1; k < n-1; k++) {
      26|                     B[i][j][k] =   0.125 * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
      27|                                  + 0.125 * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k])
>>>   28|                                  + 0.125 * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1])
      29|                                  + A[i][j][k];
      30|                 }
      31|             }
      32|         }
      33|         for (i = 1; i < n-1; i++) {
      34|            for (j = 1; j < n-1; j++) {
```

### `VITIS_LOOP_33_5_VITIS_LOOP_34_6_VITIS_LOOP_35_7` — achieved II=7, Resource Limitation

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_heat-3d_phase_b_synth_001/kernel.cpp:38`

```cpp
      32|         }
      33|         for (i = 1; i < n-1; i++) {
      34|            for (j = 1; j < n-1; j++) {
      35|                for (k = 1; k < n-1; k++) {
      36|                    A[i][j][k] =   0.125 * (B[i+1][j][k] - 2.0 * B[i][j][k] + B[i-1][j][k])
      37|                                 + 0.125 * (B[i][j+1][k] - 2.0 * B[i][j][k] + B[i][j-1][k])
>>>   38|                                 + 0.125 * (B[i][j][k+1] - 2.0 * B[i][j][k] + B[i][j][k-1])
      39|                                 + B[i][j][k];
      40|                }
      41|            }
      42|        }
      43|     }
      44| }
```

## jacobi-1d

### `VITIS_LOOP_14_2` — achieved II=3, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_jacobi-1d_phase_b_synth_001/kernel.cpp:15`

```cpp
       9|     const int tsteps = TSTEPS;
      10| 
      11|     int t, i;
      12| 
      13|     for (t = 0; t < tsteps; t++) {
      14|         for (i = 1; i < n - 1; i++)
>>>   15|             B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
      16|         for (i = 1; i < n - 1; i++)
      17|             A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
      18|     }
      19| }
      20| }
      21| 
```

### `VITIS_LOOP_16_3` — achieved II=3, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_jacobi-1d_phase_b_synth_001/kernel.cpp:17`

```cpp
      11|     int t, i;
      12| 
      13|     for (t = 0; t < tsteps; t++) {
      14|         for (i = 1; i < n - 1; i++)
      15|             B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
      16|         for (i = 1; i < n - 1; i++)
>>>   17|             A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
      18|     }
      19| }
      20| }
      21| 
      22| extern "C" {
      23| void workload(double A[N + 0], double B[N + 0]) {
```

## jacobi-2d

### `VITIS_LOOP_20_2_VITIS_LOOP_21_3` — achieved II=5, Resource Limitation

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_jacobi-2d_phase_b_synth_003/kernel.cpp:22`

```cpp
      16| 
      17|     int t, i, j;
      18| 
      19|     for (t = 0; t < tsteps; t++) {
      20|         for (i = 1; i < n - 1; i++) {
      21|             for (j = 1; j < n - 1; j++) {
>>>   22|                 B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
      23|             }
      24|         }
      25|         for (i = 1; i < n - 1; i++) {
      26|             for (j = 1; j < n - 1; j++) {
      27|                 A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i][1+j] + B[1+i][j] + B[i-1][j]);
      28|             }
```

### `VITIS_LOOP_25_4_VITIS_LOOP_26_5` — achieved II=5, Resource Limitation

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_jacobi-2d_phase_b_synth_003/kernel.cpp:27`

```cpp
      21|             for (j = 1; j < n - 1; j++) {
      22|                 B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
      23|             }
      24|         }
      25|         for (i = 1; i < n - 1; i++) {
      26|             for (j = 1; j < n - 1; j++) {
>>>   27|                 A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i][1+j] + B[1+i][j] + B[i-1][j]);
      28|             }
      29|         }
      30|     }
      31| }
      32| }
```

## lu

### `VITIS_LOOP_10_3` — achieved II=158, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_lu_phase_b_synth_001/kernel.cpp:10`

```cpp
       4| void kernel_lu(double A[N + 0][N + 0]) {
       5|     const int n = N;
       6|     int i, j, k;
       7| 
       8|     for (i = 0; i < n; i++) {
       9|         for (j = 0; j < i; j++) {
>>>   10|             for (k = 0; k < j; k++) {
      11|                 A[i][j] -= A[i][k] * A[k][j];
      12|             }
      13|             A[i][j] /= A[j][j];
      14|         }
      15|         for (j = i; j < n; j++) {
      16|             for (k = 0; k < i; k++) {
```

### `VITIS_LOOP_15_4_VITIS_LOOP_16_5` — achieved II=158, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_lu_phase_b_synth_001/kernel.cpp:15`

```cpp
       9|         for (j = 0; j < i; j++) {
      10|             for (k = 0; k < j; k++) {
      11|                 A[i][j] -= A[i][k] * A[k][j];
      12|             }
      13|             A[i][j] /= A[j][j];
      14|         }
>>>   15|         for (j = i; j < n; j++) {
      16|             for (k = 0; k < i; k++) {
      17|                 A[i][j] -= A[i][k] * A[k][j];
      18|             }
      19|         }
      20|     }
      21| }
```

## ludcmp

### `VITIS_LOOP_29_3` — achieved II=7, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_ludcmp_phase_b_synth_001/kernel.cpp:29`

```cpp
      23| 
      24|   double w;
      25| 
      26|   for (i = 0; i < n; i++) {
      27|     for (j = 0; j <i; j++) {
      28|        w = A[i][j];
>>>   29|        for (k = 0; k < j; k++) {
      30|           w -= A[i][k] * A[k][j];
      31|        }
      32|         A[i][j] = w / A[j][j];
      33|     }
      34|    for (j = i; j < n; j++) {
      35|        w = A[i][j];
```

### `VITIS_LOOP_36_5` — achieved II=7, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_ludcmp_phase_b_synth_001/kernel.cpp:36`

```cpp
      30|           w -= A[i][k] * A[k][j];
      31|        }
      32|         A[i][j] = w / A[j][j];
      33|     }
      34|    for (j = i; j < n; j++) {
      35|        w = A[i][j];
>>>   36|        for (k = 0; k < i; k++) {
      37|           w -= A[i][k] * A[k][j];
      38|        }
      39|        A[i][j] = w;
      40|     }
      41|   }
      42| 
```

### `VITIS_LOOP_45_7` — achieved II=7, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_ludcmp_phase_b_synth_001/kernel.cpp:45`

```cpp
      39|        A[i][j] = w;
      40|     }
      41|   }
      42| 
      43|   for (i = 0; i < n; i++) {
      44|      w = b[i];
>>>   45|      for (j = 0; j < i; j++)
      46|         w -= A[i][j] * y[j];
      47|      y[i] = w;
      48|   }
      49| 
      50|    for (i = n-1; i >=0; i--) {
      51|      w = y[i];
```

### `VITIS_LOOP_52_9` — achieved II=7, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_ludcmp_phase_b_synth_001/kernel.cpp:52`

```cpp
      46|         w -= A[i][j] * y[j];
      47|      y[i] = w;
      48|   }
      49| 
      50|    for (i = n-1; i >=0; i--) {
      51|      w = y[i];
>>>   52|      for (j = i+1; j < n; j++)
      53|         w -= A[i][j] * x[j];
      54|      x[i] = w / A[i][i];
      55|   }
      56| 
      57| }
      58| }
```

## mvt

### `VITIS_LOOP_28_2` — achieved II=7, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_mvt_phase_b_synth_001/kernel.cpp:28`

```cpp
      22| 
      23|     const int n = N;
      24| 
      25|   int i, j;
      26| 
      27|   for (i = 0; i < n; i++)
>>>   28|     for (j = 0; j < n; j++)
      29|       x1[i] = x1[i] + A[i][j] * y_1[j];
      30|   for (i = 0; i < n; i++)
      31|     for (j = 0; j < n; j++)
      32|       x2[i] = x2[i] + A[j][i] * y_2[j];
      33| 
      34| }
```

### `VITIS_LOOP_31_4` — achieved II=7, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_mvt_phase_b_synth_001/kernel.cpp:31`

```cpp
      25|   int i, j;
      26| 
      27|   for (i = 0; i < n; i++)
      28|     for (j = 0; j < n; j++)
      29|       x1[i] = x1[i] + A[i][j] * y_1[j];
      30|   for (i = 0; i < n; i++)
>>>   31|     for (j = 0; j < n; j++)
      32|       x2[i] = x2[i] + A[j][i] * y_2[j];
      33| 
      34| }
      35| }
```

## nussinov

### `VITIS_LOOP_22_3` — achieved II=144, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_nussinov_phase_b_synth_001/kernel.cpp:23`

```cpp
      17|                     table[i][j] = ((table[i][j] >= table[i + 1][j - 1] + (((seq[i]) + (seq[j])) == 3 ? 1 : 0)) ? table[i][j] : table[i + 1][j - 1] + (((seq[i]) + (seq[j])) == 3 ? 1 : 0));
      18|                 else
      19|                     table[i][j] = ((table[i][j] >= table[i + 1][j - 1]) ? table[i][j] : table[i + 1][j - 1]);
      20|             }
      21| 
      22|             for (k = i + 1; k < j; k++) {
>>>   23|                 table[i][j] = ((table[i][j] >= table[i][k] + table[k + 1][j]) ? table[i][j] : table[i][k] + table[k + 1][j]);
      24|             }
      25|         }
      26|     }
      27| }
      28| }
```

## seidel-2d

### `VITIS_LOOP_14_1_VITIS_LOOP_15_2_VITIS_LOOP_16_3` — achieved II=238, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_seidel-2d_phase_b_synth_001/kernel.cpp:15`

```cpp
       9|     const int n = N;
      10|     const int tsteps = TSTEPS;
      11| 
      12|     int t, i, j;
      13| 
      14|     for (t = 0; t <= tsteps - 1; t++) {
>>>   15|         for (i = 1; i <= n - 2; i++) {
      16|             for (j = 1; j <= n - 2; j++) {
      17|                 A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
      18|                          + A[i][j-1] + A[i][j] + A[i][j+1]
      19|                          + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1]) / 9.0;
      20|             }
      21|         }
```

## symm

### `VITIS_LOOP_20_3` — achieved II=151, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_symm_phase_b_synth_003/kernel.cpp:20`

```cpp
      14|     int i, j, k;
      15|     double temp2;
      16|     for (i = 0; i < m; i++)
      17|         for (j = 0; j < n; j++)
      18|         {
      19|             temp2 = 0;
>>>   20|             for (k = 0; k < i; k++) {
      21|                 C[k][j] += alpha * B[i][j] * A[i][k];
      22|                 temp2 += B[k][j] * A[i][k];
      23|             }
      24|             C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
      25|         }
      26| }
```

## syr2k

### `VITIS_LOOP_28_3_VITIS_LOOP_29_4` — achieved II=176, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_syr2k_phase_b_synth_001/kernel.cpp:28`

```cpp
      22|     const int m = M;
      23| 
      24|     int i, j, k;
      25|     for (i = 0; i < n; i++) {
      26|         for (j = 0; j <= i; j++)
      27|             C[i][j] *= beta;
>>>   28|         for (k = 0; k < m; k++)
      29|             for (j = 0; j <= i; j++)
      30|             {
      31|                 C[i][j] += A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k];
      32|             }
      33|     }
      34| }
```

## syrk

### `VITIS_LOOP_25_3_VITIS_LOOP_26_4` — achieved II=165, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_syrk_phase_b_synth_001/kernel.cpp:25`

```cpp
      19|     const int m = M;
      20| 
      21|     int i, j, k;
      22|     for (i = 0; i < n; i++) {
      23|         for (j = 0; j <= i; j++)
      24|             C[i][j] *= beta;
>>>   25|         for (k = 0; k < m; k++) {
      26|             for (j = 0; j <= i; j++)
      27|                 C[i][j] += alpha * A[i][k] * A[j][k];
      28|         }
      29|     }
      30| }
      31| }
```

## trisolv

### `VITIS_LOOP_17_2` — achieved II=159, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_trisolv_phase_b_synth_003/kernel.cpp:17`

```cpp
      11| 
      12|   int i, j;
      13| 
      14|   for (i = 0; i < n; i++)
      15|     {
      16|       x[i] = b[i];
>>>   17|       for (j = 0; j <i; j++)
      18|         x[i] -= L[i][j] * x[j];
      19|       x[i] = x[i] / L[i][i];
      20|     }
      21| 
      22| }
      23| }
```

## trmm

### `VITIS_LOOP_22_3` — achieved II=157, Memory Dependency

Source: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/hls_synth__hlsfactory_trmm_phase_b_synth_003/kernel.cpp:23`

```cpp
      17|     const int n = N;
      18| 
      19|     int i, j, k;
      20|     for (i = 0; i < m; i++)
      21|         for (j = 0; j < n; j++) {
      22|             for (k = i+1; k < m; k++)
>>>   23|                 B[i][j] += A[k][i] * B[k][j];
      24|             B[i][j] = alpha * B[i][j];
      25|         }
      26| }
      27| }
```

