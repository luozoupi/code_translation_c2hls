/*
 * Vitis HLS C-simulation testbench for Pathfinder (DP shortest path).
 * Computes golden reference using plain dynamic programming,
 * then compares against HLS workload() output.
 * Returns 0 on success, 1 on mismatch.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pathfinder.h"

extern "C" void workload(int32_t J[ROWS * COLS], int32_t Jout[COLS]);

/* Golden reference: row-by-row DP with min of left/center/right neighbor */
static void pathfinder_ref(int32_t J[ROWS * COLS], int32_t Jout[COLS])
{
    int32_t dst[COLS], src[COLS];

    memcpy(dst, J, sizeof(int32_t) * COLS);

    for (int t = 0; t < ROWS - 1; t++) {
        for (int n = 0; n < COLS; n++) {
            int32_t m = dst[n];
            if (n > 0)      m = MIN(m, dst[n - 1]);
            if (n < COLS - 1) m = MIN(m, dst[n + 1]);
            src[n] = J[(t + 1) * COLS + n] + m;
        }
        memcpy(dst, src, sizeof(int32_t) * COLS);
    }
    memcpy(Jout, dst, sizeof(int32_t) * COLS);
}

int main() {
    int errors = 0;
    srand(42);

    int32_t* J       = new int32_t[ROWS * COLS];
    int32_t* ref_out = new int32_t[COLS];
    int32_t* dut_out = new int32_t[COLS];

    /* Generate deterministic cost matrix */
    for (int i = 0; i < ROWS * COLS; i++)
        J[i] = rand() % 100;

    memset(ref_out, 0, COLS * sizeof(int32_t));
    memset(dut_out, 0, COLS * sizeof(int32_t));

    /* Compute golden reference */
    pathfinder_ref(J, ref_out);

    /* Call HLS DUT */
    workload(J, dut_out);

    /* Compare (exact integer match) */
    int mismatches = 0;
    for (int i = 0; i < COLS; i++) {
        if (ref_out[i] != dut_out[i]) {
            if (mismatches < 5) {
                printf("  mismatch[%d]: ref=%d dut=%d\n", i, ref_out[i], dut_out[i]);
            }
            mismatches++;
        }
    }
    if (mismatches > 0) {
        printf("FAIL: Jout — %d mismatches out of %d\n", mismatches, COLS);
        errors++;
    }

    if (errors == 0) {
        printf("PASS: pathfinder testbench — all outputs match golden reference\n");
    } else {
        printf("FAIL: pathfinder testbench — %d error(s)\n", errors);
    }

    delete[] J; delete[] ref_out; delete[] dut_out;
    return (errors > 0) ? 1 : 0;
}
