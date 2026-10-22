/*
 * Strict Vitis HLS C-simulation testbench for stencil2D.
 * Golden: stencil_ref() (same algorithm as plain.cpp's stencil) on identical deterministic inputs (srand(42)).
 * workload(orig, sol, filter): orig, filter are inputs; sol is a pure output, poisoned with 0x5A5A5A5A for ref and DUT.
 * plain.cpp only defines sol for r < row_size-2, c < col_size-2 (the other cells are never assigned, i.e. undefined),
 * so exactly that valid region is compared, every element, exactly. Prints "Passed!" or "Failed with N errors!".
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stencil.h"

/* Golden reference: plain-C 2D stencil */
static void stencil_ref(TYPE orig[row_size * col_size],
                        TYPE sol[row_size * col_size],
                        TYPE filter[f_size]) {
    int r, c, k1, k2;
    TYPE temp, mul;
    for (r = 0; r < row_size - 2; r++) {
        for (c = 0; c < col_size - 2; c++) {
            temp = (TYPE)0;
            for (k1 = 0; k1 < 3; k1++) {
                for (k2 = 0; k2 < 3; k2++) {
                    mul = filter[k1 * 3 + k2] * orig[(r + k1) * col_size + c + k2];
                    temp += mul;
                }
            }
            sol[(r * col_size) + c] = temp;
        }
    }
}

extern "C" void workload(TYPE* orig, TYPE* sol, TYPE* filter);

static int errors = 0;
static void chk_i(const char *name, long i, long long d, long long r) {
    if (d != r) {
        if (errors < 10) printf("%s[%ld]: got %lld expected %lld\n", name, i, d, r);
        errors++;
    }
}

int main() {

    TYPE orig[row_size * col_size];
    TYPE ref_sol[row_size * col_size];
    TYPE dut_sol[row_size * col_size];
    TYPE filter[f_size];

    /* Initialize with deterministic data */
    srand(42);
    for (int i = 0; i < row_size * col_size; i++)
        orig[i] = (TYPE)(rand() % 1000);
    /* Simple averaging filter */
    for (int i = 0; i < f_size; i++)
        filter[i] = 1;

    /* Compute golden reference */
    for (int i = 0; i < row_size * col_size; i++) ref_sol[i] = 0x5A5A5A5A;   /* poison */
    stencil_ref(orig, ref_sol, filter);

    /* Call HLS design under test */
    for (int i = 0; i < row_size * col_size; i++) dut_sol[i] = 0x5A5A5A5A;   /* poison */
    workload(orig, dut_sol, filter);

    /* Compare outputs (valid region) */
    for (int r = 0; r < row_size - 2; r++)
        for (int c = 0; c < col_size - 2; c++)
            chk_i("sol", r * col_size + c, dut_sol[r * col_size + c], ref_sol[r * col_size + c]);
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
