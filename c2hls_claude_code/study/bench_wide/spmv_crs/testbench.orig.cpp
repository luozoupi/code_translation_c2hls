/*
 * Vitis HLS C-simulation testbench for spmv_crs benchmark.
 * Creates a small deterministic sparse matrix in CRS format,
 * computes SpMV golden reference, compares against HLS workload().
 * Returns 0 on success, 1 on mismatch.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "spmv.h"

#define EPSILON 1.0e-6

/* Golden reference: plain-C sparse matrix-vector multiply */
static void spmv_ref(TYPE val[NNZ], int32_t cols[NNZ],
                     int32_t rowDelimiters[N + 1],
                     TYPE vec[N], TYPE out[N]) {
    int i, j;
    for (i = 0; i < N; i++) {
        TYPE sum = 0;
        for (j = rowDelimiters[i]; j < rowDelimiters[i + 1]; j++) {
            sum += val[j] * vec[cols[j]];
        }
        out[i] = sum;
    }
}

extern "C" void workload(TYPE* val, int32_t* cols, int32_t* rowDelimiters,
              TYPE* vec, TYPE* out);

int main() {
    int errors = 0;

    TYPE val[NNZ];
    int32_t cols[NNZ];
    int32_t rowDelimiters[N + 1];
    TYPE vec[N];
    TYPE ref_out[N], dut_out[N];

    /* Build a deterministic sparse matrix:
     * ~3.4 non-zeros per row (NNZ/N ~ 3.37) */
    srand(42);
    int nnz_idx = 0;
    rowDelimiters[0] = 0;
    for (int i = 0; i < N; i++) {
        /* Each row gets 2-5 non-zeros */
        int nnz_per_row = 2 + (rand() % 4);
        if (nnz_idx + nnz_per_row > NNZ) nnz_per_row = NNZ - nnz_idx;
        for (int j = 0; j < nnz_per_row; j++) {
            if (nnz_idx >= NNZ) break;
            val[nnz_idx] = (TYPE)(rand() % 200 - 100) / 50.0;
            cols[nnz_idx] = rand() % N;
            nnz_idx++;
        }
        rowDelimiters[i + 1] = nnz_idx;
    }
    /* Fill remaining if any */
    for (; nnz_idx < NNZ; nnz_idx++) {
        val[nnz_idx] = 0.0;
        cols[nnz_idx] = 0;
    }
    rowDelimiters[N] = NNZ;

    /* Initialize vector */
    for (int i = 0; i < N; i++)
        vec[i] = (TYPE)(rand() % 100) / 50.0 - 1.0;

    /* Compute golden reference */
    memset(ref_out, 0, sizeof(ref_out));
    spmv_ref(val, cols, rowDelimiters, vec, ref_out);

    /* Call HLS design under test */
    memset(dut_out, 0, sizeof(dut_out));
    workload(val, cols, rowDelimiters, vec, dut_out);

    /* Compare outputs */
    for (int i = 0; i < N; i++) {
        TYPE diff = fabs(dut_out[i] - ref_out[i]);
        if (diff > EPSILON) {
            if (errors < 10)
                printf("FAIL: out[%d] = %.6f, expected %.6f (diff=%.2e)\n",
                       i, dut_out[i], ref_out[i], diff);
            errors++;
        }
    }

    if (errors == 0) {
        printf("PASS: spmv_crs testbench — all %d outputs match\n", N);
    } else {
        printf("FAIL: spmv_crs testbench — %d / %d mismatches\n", errors, N);
    }

    return (errors > 0) ? 1 : 0;
}
