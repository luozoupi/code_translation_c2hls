/*
 * Strict Vitis HLS C-simulation testbench for spmv_crs.
 * Golden: spmv_ref() (same arithmetic as plain.cpp's spmv) on identical deterministic CRS data (srand(42)).
 * workload(val, cols, rowDelimiters, vec, out): val, cols, rowDelimiters, vec are inputs; out[N] is a pure output
 * (assigned per row), poisoned with 12345.0 for ref and DUT. Every out element is checked, double tol
 * 1e-6*fmax(1,|ref|) (SPEC), NaN = error. Prints "Passed!" or "Failed with N errors!".
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "spmv.h"

static int errors = 0;
static void chk_d(const char *name, int i, double d, double r) {
    if (!isfinite(r) || !(fabs(d - r) <= 1e-6 * fmax(1.0, fabs(r)))) {   /* NaN/inf in DUT output (or a bad reference) is an error */
        if (errors < 10) printf("%s[%d]: got %.9g expected %.9g\n", name, i, d, r);
        errors++;
    }
}

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
    for (int i = 0; i < N; i++) ref_out[i] = 12345.0;   /* poison */
    spmv_ref(val, cols, rowDelimiters, vec, ref_out);

    /* Call HLS design under test */
    for (int i = 0; i < N; i++) dut_out[i] = 12345.0;   /* poison */
    workload(val, cols, rowDelimiters, vec, dut_out);

    for (int i = 0; i < N; i++) chk_d("out", i, dut_out[i], ref_out[i]);
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
