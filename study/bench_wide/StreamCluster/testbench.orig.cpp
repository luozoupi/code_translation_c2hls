/*
 * Vitis HLS C-simulation testbench for StreamCluster benchmark.
 * Computes golden reference for streaming k-means cost evaluation,
 * then compares against HLS workload() output.
 * Returns 0 on success, 1 on mismatch.
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

int main() {
    int errors = 0;
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
        cost[i]   = (float)(rand() % 500) / 10.0f;
        assign[i] = rand() % MAX_WORK_MEM_SIZE;
    }
    for (int i = 0; i < DIM; i++)
        target[i] = (float)(rand() % 1000) / 100.0f;
    for (int i = 0; i < MAX_WORK_MEM_SIZE; i++)
        center_table[i] = i;

    /* Initialize outputs */
    memset(ref_switch, 0, BATCH_SIZE);
    memset(ref_work, 0, MAX_WORK_MEM_SIZE * sizeof(float));
    ref_cost_x[0] = 0.0f;

    memset(dut_switch, 0, BATCH_SIZE);
    memset(dut_work, 0, MAX_WORK_MEM_SIZE * sizeof(float));
    dut_cost_x[0] = 0.0f;

    /* Compute golden reference */
    streamcluster_ref(coord, weight, cost, target, assign, center_table,
                      ref_switch, ref_work, BATCH_SIZE, ref_cost_x, MAX_WORK_MEM_SIZE);

    /* Call HLS DUT */
    workload(coord, weight, cost, target, assign, center_table,
             dut_switch, dut_work, BATCH_SIZE, dut_cost_x, MAX_WORK_MEM_SIZE);

    /* Compare switch_membership */
    int switch_mismatches = 0;
    for (int i = 0; i < BATCH_SIZE; i++) {
        if (ref_switch[i] != dut_switch[i]) switch_mismatches++;
    }
    if (switch_mismatches > 0) {
        printf("FAIL: switch_membership — %d mismatches out of %d\n", switch_mismatches, BATCH_SIZE);
        errors++;
    }

    /* Compare work_mem with tolerance */
    float tol = 1e-3f;
    int work_mismatches = 0;
    for (int i = 0; i < MAX_WORK_MEM_SIZE; i++) {
        if (fabsf(ref_work[i] - dut_work[i]) > tol * (fabsf(ref_work[i]) + 1.0f)) {
            work_mismatches++;
        }
    }
    if (work_mismatches > 0) {
        printf("FAIL: work_mem — %d mismatches\n", work_mismatches);
        errors++;
    }

    /* Compare cost_of_opening_x */
    if (fabsf(ref_cost_x[0] - dut_cost_x[0]) > tol * (fabsf(ref_cost_x[0]) + 1.0f)) {
        printf("FAIL: cost_of_opening_x — ref=%.6f dut=%.6f\n", ref_cost_x[0], dut_cost_x[0]);
        errors++;
    }

    if (errors == 0) {
        printf("PASS: StreamCluster testbench — all outputs match golden reference\n");
    } else {
        printf("FAIL: StreamCluster testbench — %d error(s)\n", errors);
    }

    delete[] coord; delete[] weight; delete[] cost; delete[] target;
    delete[] assign; delete[] center_table;
    delete[] ref_switch; delete[] ref_work;
    delete[] dut_switch; delete[] dut_work;

    return (errors > 0) ? 1 : 0;
}
