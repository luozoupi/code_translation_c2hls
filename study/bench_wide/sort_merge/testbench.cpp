/*
 * Strict Vitis HLS C-simulation testbench for sort_merge.
 * Independent golden: plain.cpp's workload() copied below as workload_ref() and run on a copy of the same
 * deterministic input (srand(42)); additionally cross-checked against stdlib qsort. Every element of a[] is
 * compared exactly. a[] is in-out (input data, sorted in place), so it is not poisoned.
 * Prints exactly "Passed!" or "Failed with N errors!".
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

/* ---- reference: copy of plain.cpp ---- */

static void merge_ref(TYPE a[SIZE], int start, int m, int stop) {
    TYPE temp[SIZE];
    int i, j, k;

    merge_label1: for (i = start; i <= m; i++) {
        temp[i] = a[i];
    }
    merge_label2: for (j = m + 1; j <= stop; j++) {
        temp[m + 1 + stop - j] = a[j];
    }

    i = start;
    j = stop;

    merge_label3: for (k = start; k <= stop; k++) {
        TYPE tmp_j = temp[j];
        TYPE tmp_i = temp[i];
        if (tmp_j < tmp_i) {
            a[k] = tmp_j;
            j--;
        } else {
            a[k] = tmp_i;
            i++;
        }
    }
}

static void ms_mergesort_ref(TYPE a[SIZE]) {
    int start, stop;
    int i, m, from, mid, to;

    start = 0;
    stop = SIZE;

    mergesort_label1: for (m = 1; m < stop - start; m += m) {
        mergesort_label2: for (i = start; i < stop; i += m + m) {
            from = i;
            mid = i + m - 1;
            to = i + m + m - 1;
            if (to < stop) {
                merge_ref(a, from, mid, to);
            } else {
                merge_ref(a, from, mid, stop);
            }
        }
    }
}


static void workload_ref(TYPE* a) {

    TYPE l_a[SIZE];
    int i;
    for (i = 0; i < SIZE; i++) l_a[i] = a[i];
    ms_mergesort_ref(l_a);
    for (i = 0; i < SIZE; i++) a[i] = l_a[i];
}
/* ---- end reference ---- */

extern "C" void workload(TYPE* a);

int main() {
    int errors = 0;

    TYPE ref_a[SIZE], dut_a[SIZE], qs_a[SIZE];

    /* Initialize with deterministic pseudo-random data */
    srand(42);
    for (int i = 0; i < SIZE; i++) {
        ref_a[i] = (TYPE)(rand() % 100000 - 50000);
        dut_a[i] = ref_a[i];
        qs_a[i] = ref_a[i];
    }

    /* Golden reference: plain.cpp workload, cross-checked against qsort */
    workload_ref(ref_a);
    qsort(qs_a, SIZE, sizeof(TYPE), cmp_int32);

    /* Call HLS design under test */
    workload(dut_a);

    for (int i = 0; i < SIZE; i++) {
        if (dut_a[i] != ref_a[i] || dut_a[i] != qs_a[i]) {
            if (errors < 10)
                printf("a[%d]: got %d expected %d (qsort %d)\n", i, dut_a[i], ref_a[i], qs_a[i]);
            errors++;
        }
    }

    if (errors) {
        printf("Failed with %d errors!\n", errors);
        return 1;
    }
    printf("Passed!\n");
    return 0;
}
