/*
 * Strict Vitis HLS C-simulation testbench for knn (distance computation).
 * Golden: knn_ref() (same arithmetic as plain.cpp's workload) on identical deterministic inputs (srand(42)).
 * workload(inputQuery, searchSpace, distance): inputQuery, searchSpace are inputs; distance is a pure output (assigned
 * per point), poisoned with 12345.0f for both ref and DUT. Every distance is checked, float tol 1e-3*fmax(1,|ref|)
 * (SPEC), NaN = error. Prints "Passed!" or "Failed with N errors!".
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

static int errors = 0;
static void chk_f(const char *name, int i, float d, float r) {
    if (!isfinite(r) || !(fabsf(d - r) <= 1e-3f * fmaxf(1.0f, fabsf(r)))) {   /* NaN/inf in DUT output (or a bad reference) is an error */
        if (errors < 10) printf("%s[%d]: got %g expected %g\n", name, i, (double)d, (double)r);
        errors++;
    }
}

int main() {
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

    for (int i = 0; i < NUM_PT_IN_SEARCHSPACE; i++) { ref_dist[i] = 12345.0f; dut_dist[i] = 12345.0f; }   /* poison */

    /* Compute golden reference */
    knn_ref(query, space, ref_dist);

    /* Call HLS DUT */
    workload(query, space, dut_dist);

    for (int i = 0; i < NUM_PT_IN_SEARCHSPACE; i++) chk_f("distance", i, dut_dist[i], ref_dist[i]);

    delete[] space; delete[] ref_dist; delete[] dut_dist;
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
