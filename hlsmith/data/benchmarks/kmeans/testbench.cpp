/*
 * Vitis HLS C-simulation testbench for KMeans clustering assignment.
 * Computes nearest cluster assignment as golden reference,
 * then compares against HLS workload() output.
 * Returns 0 on success, 1 on mismatch.
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
    int errors = 0;
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
    int mismatches = 0;
    for (int i = 0; i < NPOINTS; i++) {
        if (ref_mem[i] != dut_mem[i]) {
            if (mismatches < 5) {
                printf("  mismatch[%d]: ref=%d dut=%d\n", i, ref_mem[i], dut_mem[i]);
            }
            mismatches++;
        }
    }
    if (mismatches > 0) {
        printf("FAIL: membership — %d mismatches out of %d\n", mismatches, NPOINTS);
        errors++;
    }

    if (errors == 0) {
        printf("PASS: kmeans testbench — all outputs match golden reference\n");
    } else {
        printf("FAIL: kmeans testbench — %d error(s)\n", errors);
    }

    delete[] feature; delete[] clusters;
    delete[] ref_mem; delete[] dut_mem;
    return (errors > 0) ? 1 : 0;
}
