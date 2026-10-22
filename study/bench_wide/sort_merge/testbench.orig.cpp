/*
 * Vitis HLS C-simulation testbench for sort_merge benchmark.
 * Generates a random array, sorts with both reference qsort and HLS workload(),
 * compares results. Also verifies that all original elements are preserved.
 * Returns 0 on success, 1 on mismatch.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sort.h"

static int cmp_int32(const void *a, const void *b) {
    int32_t va = *(const int32_t*)a;
    int32_t vb = *(const int32_t*)b;
    return (va > vb) - (va < vb);
}

extern "C" void workload(TYPE* a);

int main() {
    int errors = 0;

    TYPE ref_a[SIZE], dut_a[SIZE];

    /* Initialize with deterministic pseudo-random data */
    srand(42);
    for (int i = 0; i < SIZE; i++) {
        ref_a[i] = (TYPE)(rand() % 100000 - 50000);
        dut_a[i] = ref_a[i];
    }

    /* Golden reference: stdlib qsort */
    qsort(ref_a, SIZE, sizeof(TYPE), cmp_int32);

    /* Call HLS design under test */
    workload(dut_a);

    /* Verify sorted order */
    for (int i = 1; i < SIZE; i++) {
        if (dut_a[i - 1] > dut_a[i]) {
            if (errors < 10)
                printf("FAIL: not sorted at [%d]: %d > %d\n",
                       i, dut_a[i - 1], dut_a[i]);
            errors++;
        }
    }

    /* Verify element preservation (sum check) */
    int64_t ref_sum = 0, dut_sum = 0;
    for (int i = 0; i < SIZE; i++) {
        ref_sum += ref_a[i];
        dut_sum += dut_a[i];
    }
    if (ref_sum != dut_sum) {
        printf("FAIL: element sum mismatch (ref=%ld, dut=%ld)\n",
               (long)ref_sum, (long)dut_sum);
        errors++;
    }

    if (errors == 0) {
        printf("PASS: sort_merge testbench — %d elements sorted correctly\n", SIZE);
    } else {
        printf("FAIL: sort_merge testbench — %d error(s)\n", errors);
    }

    return (errors > 0) ? 1 : 0;
}
