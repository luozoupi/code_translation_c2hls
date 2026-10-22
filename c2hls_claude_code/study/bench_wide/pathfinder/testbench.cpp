/*
 * Strict Vitis HLS C-simulation testbench for pathfinder.
 * Golden: pathfinder_ref() (row-by-row DP, same algorithm as plain.cpp's workload) on an identical deterministic cost
 * matrix (srand(42)). workload(J, Jout): J is input only; Jout[COLS] is a pure output (workload never reads it, only
 * memcpy's the DP result into it), poisoned with 0x5A5A5A5A for ref and DUT (not a reachable path cost). Every Jout
 * element is compared exactly. Prints "Passed!" or "Failed with N errors!".
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

static int errors = 0;
static void chk_i(const char *name, long i, long long d, long long r) {
    if (d != r) {
        if (errors < 10) printf("%s[%ld]: got %lld expected %lld\n", name, i, d, r);
        errors++;
    }
}

int main() {
    srand(42);

    int32_t* J       = new int32_t[ROWS * COLS];
    int32_t* ref_out = new int32_t[COLS];
    int32_t* dut_out = new int32_t[COLS];

    /* Generate deterministic cost matrix */
    for (int i = 0; i < ROWS * COLS; i++)
        J[i] = rand() % 100;

    for (int i = 0; i < COLS; i++) { ref_out[i] = 0x5A5A5A5A; dut_out[i] = 0x5A5A5A5A; }   /* poison */

    /* Compute golden reference */
    pathfinder_ref(J, ref_out);

    /* Call HLS DUT */
    workload(J, dut_out);

    for (int i = 0; i < COLS; i++) chk_i("Jout", i, dut_out[i], ref_out[i]);

    delete[] J; delete[] ref_out; delete[] dut_out;
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
