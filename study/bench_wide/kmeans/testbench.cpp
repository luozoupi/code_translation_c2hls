/*
 * Strict Vitis HLS C-simulation testbench for kmeans (membership assignment).
 * Golden: kmeans_ref() (same arithmetic as plain.cpp's workload, first-minimum tie-break) on identical deterministic
 * inputs (srand(42)). workload(feature, clusters, membership): feature, clusters are inputs; membership is a pure
 * output, poisoned with -1 (0xFFFFFFFF, not a legal cluster id) for both ref and DUT. Every membership entry is
 * compared exactly. Prints "Passed!" or "Failed with N errors!".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "kmeans.h"

extern "C" void workload(float* feature, float* clusters, int* membership);

/* Golden reference */
static void kmeans_ref(float* feature, float* clusters, int* membership)
{
    for (int i = 0; i < NPOINTS; i++) {
        float min_dist = FLT_MAX;
        int index = 0;
        for (int j = 0; j < NCLUSTERS; j++) {
            float dist = 0.0f;
            for (int k = 0; k < NFEATURES; k++) {
                float diff = feature[NFEATURES * i + k] - clusters[NFEATURES * j + k];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                index = j;
            }
        }
        membership[i] = index;
    }
}

int main() {
    srand(42);

    float* feature  = new float[NPOINTS * NFEATURES];
    float* clusters = new float[NCLUSTERS * NFEATURES];
    int*   ref_mem  = new int[NPOINTS];
    int*   dut_mem  = new int[NPOINTS];

    /* Generate test data */
    for (int i = 0; i < NPOINTS * NFEATURES; i++)
        feature[i] = (float)(rand() % 10000) / 100.0f;
    for (int i = 0; i < NCLUSTERS * NFEATURES; i++)
        clusters[i] = (float)(rand() % 10000) / 100.0f;

    memset(ref_mem, -1, NPOINTS * sizeof(int));
    memset(dut_mem, -1, NPOINTS * sizeof(int));

    /* Compute golden reference */
    kmeans_ref(feature, clusters, ref_mem);

    /* Call HLS DUT */
    workload(feature, clusters, dut_mem);

    /* Compare (exact integer match) */
    int errors = 0;
    for (int i = 0; i < NPOINTS; i++) {
        if (ref_mem[i] != dut_mem[i]) {
            if (errors < 10) printf("membership[%d]: got %d expected %d\n", i, dut_mem[i], ref_mem[i]);
            errors++;
        }
    }

    delete[] feature; delete[] clusters;
    delete[] ref_mem; delete[] dut_mem;
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
