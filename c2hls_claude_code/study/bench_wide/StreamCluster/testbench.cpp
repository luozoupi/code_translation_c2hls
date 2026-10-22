/*
 * Strict Vitis HLS C-simulation testbench for StreamCluster.
 * Golden: streamcluster_ref() (same algorithm as plain.cpp's workload) on identical deterministic inputs (srand(42)).
 * Classification from plain.cpp: coord, weight, cost, target, assign, center_table, num, numcenter are inputs.
 * switch_membership (set to 1 only when current_cost < 0, never otherwise written), work_mem (-=) and
 * cost_of_opening_x[0] (+=) are IN-OUT accumulators: plain.cpp never zeroes or fully assigns them, so they are NOT
 * poisoned; ref and DUT get the same real initial data (switch_membership 0 flags, work_mem 0..100, cost_x[0] 0..100).
 * Every element of the three in-out arrays is checked: switch_membership exactly, floats with tol
 * 1e-3*fmax(1,|ref|) (SPEC), NaN = error. Prints "Passed!" or "Failed with N errors!".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "streamcluster.h"

extern "C" void workload(
    float* coord, float* weight, float* cost, float* target,
    int* assign, int* center_table, char* switch_membership,
    float* work_mem, int num, float* cost_of_opening_x, int numcenter);

/* Golden reference: plain C implementation */
static void streamcluster_ref(
    float* coord, float* weight, float* cost, float* target,
    int* assign, int* center_table, char* switch_membership,
    float* work_mem, int num, float* cost_of_opening_x, int numcenter)
{
    for (int i = 0; i < BATCH_SIZE; i++) {
        float sum = 0;
        for (int j = 0; j < DIM; j++) {
            float a = coord[i * DIM + j] - target[j];
            sum += a * a;
        }
        float current_cost = sum * weight[i] - cost[i];
        int local_center_index = center_table[assign[i]];
        if (current_cost < 0) {
            switch_membership[i] = 1;
            cost_of_opening_x[0] += current_cost;
        } else {
            work_mem[local_center_index] -= current_cost;
        }
    }
}

static int errors = 0;
static void chk_f(const char *name, int i, float d, float r) {
    if (!isfinite(r) || !(fabsf(d - r) <= 1e-3f * fmaxf(1.0f, fabsf(r)))) {   /* NaN/inf in DUT output (or a bad reference) is an error */
        if (errors < 10) printf("%s[%d]: got %g expected %g\n", name, i, (double)d, (double)r);
        errors++;
    }
}
static void chk_i(const char *name, long i, long long d, long long r) {
    if (d != r) {
        if (errors < 10) printf("%s[%ld]: got %lld expected %lld\n", name, i, d, r);
        errors++;
    }
}

int main() {
    srand(42);

    float* coord  = new float[BATCH_SIZE * DIM];
    float* weight = new float[BATCH_SIZE];
    float* cost   = new float[BATCH_SIZE];
    float* target = new float[DIM];
    int*   assign = new int[BATCH_SIZE];
    int*   center_table = new int[MAX_WORK_MEM_SIZE];

    /* Reference outputs */
    char*  ref_switch = new char[BATCH_SIZE];
    float* ref_work   = new float[MAX_WORK_MEM_SIZE];
    float  ref_cost_x[1];

    /* DUT outputs */
    char*  dut_switch = new char[BATCH_SIZE];
    float* dut_work   = new float[MAX_WORK_MEM_SIZE];
    float  dut_cost_x[1];

    /* Generate deterministic test data */
    for (int i = 0; i < BATCH_SIZE * DIM; i++)
        coord[i] = (float)(rand() % 1000) / 100.0f;
    for (int i = 0; i < BATCH_SIZE; i++) {
        weight[i] = (float)(rand() % 100) / 10.0f + 0.1f;
        cost[i]   = (float)(rand() % 80000) / 10.0f;   /* 0..8000 so current_cost takes both signs */
        assign[i] = rand() % MAX_WORK_MEM_SIZE;
    }
    for (int i = 0; i < DIM; i++)
        target[i] = (float)(rand() % 1000) / 100.0f;
    for (int i = 0; i < MAX_WORK_MEM_SIZE; i++)
        center_table[i] = i;

    /* In-out accumulators: identical real initial data for ref and DUT */
    memset(ref_switch, 0, BATCH_SIZE);
    for (int i = 0; i < MAX_WORK_MEM_SIZE; i++) ref_work[i] = (float)(rand() % 10000) / 100.0f;
    ref_cost_x[0] = (float)(rand() % 10000) / 100.0f;

    memcpy(dut_switch, ref_switch, BATCH_SIZE);
    memcpy(dut_work, ref_work, MAX_WORK_MEM_SIZE * sizeof(float));
    dut_cost_x[0] = ref_cost_x[0];

    /* Compute golden reference */
    streamcluster_ref(coord, weight, cost, target, assign, center_table,
                      ref_switch, ref_work, BATCH_SIZE, ref_cost_x, MAX_WORK_MEM_SIZE);

    /* Call HLS DUT */
    workload(coord, weight, cost, target, assign, center_table,
             dut_switch, dut_work, BATCH_SIZE, dut_cost_x, MAX_WORK_MEM_SIZE);

    for (int i = 0; i < BATCH_SIZE; i++) chk_i("switch_membership", i, dut_switch[i], ref_switch[i]);
    for (int i = 0; i < MAX_WORK_MEM_SIZE; i++) chk_f("work_mem", i, dut_work[i], ref_work[i]);
    chk_f("cost_of_opening_x", 0, dut_cost_x[0], ref_cost_x[0]);

    delete[] coord; delete[] weight; delete[] cost; delete[] target;
    delete[] assign; delete[] center_table;
    delete[] ref_switch; delete[] ref_work;
    delete[] dut_switch; delete[] dut_work;

    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
