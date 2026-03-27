/*
 * Vitis HLS C-simulation testbench for KNN distance computation.
 * Computes squared Euclidean distances as golden reference,
 * then compares against HLS workload() output.
 * Returns 0 on success, 1 on mismatch.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "knn.h"

extern "C" void workload(
    float inputQuery[NUM_FEATURE],
    float searchSpace[NUM_PT_IN_SEARCHSPACE * NUM_FEATURE],
    float distance[NUM_PT_IN_SEARCHSPACE]);

/* Golden reference */
static void knn_ref(float query[NUM_FEATURE],
                    float space[NUM_PT_IN_SEARCHSPACE * NUM_FEATURE],
                    float dist[NUM_PT_IN_SEARCHSPACE])
{
    for (int i = 0; i < NUM_PT_IN_SEARCHSPACE; i++) {
        float sum = 0.0f;
        for (int j = 0; j < NUM_FEATURE; j++) {
            float d = space[i * NUM_FEATURE + j] - query[j];
            sum += d * d;
        }
        dist[i] = sum;
    }
}

int main() {
    int errors = 0;
    srand(42);

    float query[NUM_FEATURE];
    float* space    = new float[NUM_PT_IN_SEARCHSPACE * NUM_FEATURE];
    float* ref_dist = new float[NUM_PT_IN_SEARCHSPACE];
    float* dut_dist = new float[NUM_PT_IN_SEARCHSPACE];

    /* Generate test data */
    for (int j = 0; j < NUM_FEATURE; j++)
        query[j] = (float)(rand() % 1000) / 100.0f;
    for (int i = 0; i < NUM_PT_IN_SEARCHSPACE * NUM_FEATURE; i++)
        space[i] = (float)(rand() % 1000) / 100.0f;

    memset(ref_dist, 0, NUM_PT_IN_SEARCHSPACE * sizeof(float));
    memset(dut_dist, 0, NUM_PT_IN_SEARCHSPACE * sizeof(float));

    /* Compute golden reference */
    knn_ref(query, space, ref_dist);

    /* Call HLS DUT */
    workload(query, space, dut_dist);

    /* Compare with tolerance */
    float tol = 1e-4f;
    int mismatches = 0;
    for (int i = 0; i < NUM_PT_IN_SEARCHSPACE; i++) {
        if (fabsf(ref_dist[i] - dut_dist[i]) > tol * (fabsf(ref_dist[i]) + 1.0f)) {
            if (mismatches < 5) {
                printf("  mismatch[%d]: ref=%.6f dut=%.6f\n", i, ref_dist[i], dut_dist[i]);
            }
            mismatches++;
        }
    }
    if (mismatches > 0) {
        printf("FAIL: distance — %d mismatches out of %d\n", mismatches, NUM_PT_IN_SEARCHSPACE);
        errors++;
    }

    if (errors == 0) {
        printf("PASS: knn testbench — all outputs match golden reference\n");
    } else {
        printf("FAIL: knn testbench — %d error(s)\n", errors);
    }

    delete[] space; delete[] ref_dist; delete[] dut_dist;
    return (errors > 0) ? 1 : 0;
}
