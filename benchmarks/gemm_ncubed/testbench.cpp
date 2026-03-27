/*
 * Vitis HLS C-simulation testbench for gemm_ncubed benchmark.
 * Computes golden reference using naive O(n^3) matrix multiply,
 * compares against HLS workload() output.
 * Returns 0 on success, 1 on mismatch.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gemm.h"

#define EPSILON 1.0e-6

/* Golden reference: naive matrix multiply */
static void gemm_ref(TYPE m1[N], TYPE m2[N], TYPE prod[N]) {
    int i, j, k;
    for (i = 0; i < row_size; i++) {
        for (j = 0; j < col_size; j++) {
            TYPE sum = 0;
            for (k = 0; k < row_size; k++) {
                sum += m1[i * col_size + k] * m2[k * col_size + j];
            }
            prod[i * col_size + j] = sum;
        }
    }
}

extern "C" void workload(TYPE* m1, TYPE* m2, TYPE* prod);

int main() {
    int errors = 0;

    TYPE m1[N], m2[N];
    TYPE ref_prod[N], dut_prod[N];

    /* Initialize with deterministic pseudo-random data */
    srand(42);
    for (int i = 0; i < N; i++) {
        m1[i] = (TYPE)(rand() % 100) / 50.0 - 1.0;
        m2[i] = (TYPE)(rand() % 100) / 50.0 - 1.0;
    }

    /* Compute golden reference */
    memset(ref_prod, 0, sizeof(ref_prod));
    gemm_ref(m1, m2, ref_prod);

    /* Call HLS design under test */
    memset(dut_prod, 0, sizeof(dut_prod));
    workload(m1, m2, dut_prod);

    /* Compare outputs */
    for (int i = 0; i < N; i++) {
        TYPE diff = fabs(dut_prod[i] - ref_prod[i]);
        if (diff > EPSILON) {
            if (errors < 10)
                printf("FAIL: prod[%d] = %.6f, expected %.6f (diff=%.2e)\n",
                       i, dut_prod[i], ref_prod[i], diff);
            errors++;
        }
    }

    if (errors == 0) {
        printf("PASS: gemm_ncubed testbench — all %d outputs match\n", N);
    } else {
        printf("FAIL: gemm_ncubed testbench — %d / %d mismatches\n", errors, N);
    }

    return (errors > 0) ? 1 : 0;
}
