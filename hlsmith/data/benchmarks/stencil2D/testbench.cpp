/*
 * Vitis HLS C-simulation testbench for stencil2D benchmark.
 * Applies a 3x3 filter to a 128x64 grid, compares against golden reference.
 * Returns 0 on success, 1 on mismatch.
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

int main() {
    int errors = 0;

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
    memset(ref_sol, 0, sizeof(ref_sol));
    stencil_ref(orig, ref_sol, filter);

    /* Call HLS design under test */
    memset(dut_sol, 0, sizeof(dut_sol));
    workload(orig, dut_sol, filter);

    /* Compare outputs */
    for (int r = 0; r < row_size - 2; r++) {
        for (int c = 0; c < col_size - 2; c++) {
            int idx = r * col_size + c;
            if (dut_sol[idx] != ref_sol[idx]) {
                if (errors < 10)
                    printf("FAIL: sol[%d][%d] = %d, expected %d\n",
                           r, c, dut_sol[idx], ref_sol[idx]);
                errors++;
            }
        }
    }

    if (errors == 0) {
        printf("PASS: stencil2D testbench — all outputs match\n");
    } else {
        printf("FAIL: stencil2D testbench — %d mismatches\n", errors);
    }

    return (errors > 0) ? 1 : 0;
}
