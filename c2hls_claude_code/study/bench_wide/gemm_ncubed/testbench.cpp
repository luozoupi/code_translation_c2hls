/*
 * Strict Vitis HLS C-simulation testbench for gemm_ncubed.
 * Golden: naive O(n^3) matrix multiply gemm_ref() (same arithmetic as plain.cpp's gemm) on identical deterministic
 * inputs (srand(42)). workload(m1, m2, prod): m1, m2 are inputs; prod is a pure output (assigned, never read first)
 * and is poisoned with 12345.0 for both reference and DUT. Every prod element is checked, double tol
 * 1e-6*fmax(1,|ref|) (SPEC), NaN = error. Prints "Passed!" or "Failed with N errors!".
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gemm.h"

static int errors = 0;
static void chk_d(const char *name, int i, double d, double r) {
    if (!isfinite(r) || !(fabs(d - r) <= 1e-6 * fmax(1.0, fabs(r)))) {   /* NaN/inf in DUT output (or a bad reference) is an error */
        if (errors < 10) printf("%s[%d]: got %.9g expected %.9g\n", name, i, d, r);
        errors++;
    }
}

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
    TYPE m1[N], m2[N];
    TYPE ref_prod[N], dut_prod[N];

    /* Initialize with deterministic pseudo-random data */
    srand(42);
    for (int i = 0; i < N; i++) {
        m1[i] = (TYPE)(rand() % 100) / 50.0 - 1.0;
        m2[i] = (TYPE)(rand() % 100) / 50.0 - 1.0;
    }

    /* Compute golden reference */
    for (int i = 0; i < N; i++) ref_prod[i] = 12345.0;   /* poison */
    gemm_ref(m1, m2, ref_prod);

    /* Call HLS design under test */
    for (int i = 0; i < N; i++) dut_prod[i] = 12345.0;   /* poison */
    workload(m1, m2, dut_prod);

    /* Compare outputs */
    for (int i = 0; i < N; i++) chk_d("prod", i, dut_prod[i], ref_prod[i]);
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
