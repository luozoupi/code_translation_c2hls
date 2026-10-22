/*
 * Vitis HLS C-simulation testbench for LUD (LU Decomposition).
 * Computes in-place LU decomposition as golden reference,
 * then compares against HLS workload() output.
 * Returns 0 on success, 1 on mismatch.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lud.h"

extern "C" void workload(float result[GRID_ROWS * GRID_COLS]);

/* Golden reference: Doolittle LU decomposition */
static void lud_ref(float result[GRID_ROWS * GRID_COLS])
{
    float sum;
    for (int i = 0; i < SIZE; i++) {
        for (int j = i; j < SIZE; j++) {
            sum = result[i * SIZE + j];
            for (int k = 0; k < i; k++)
                sum -= result[i * SIZE + k] * result[k * SIZE + j];
            result[i * SIZE + j] = sum;
        }
        for (int j = i + 1; j < SIZE; j++) {
            sum = result[j * SIZE + i];
            for (int k = 0; k < i; k++)
                sum -= result[j * SIZE + k] * result[k * SIZE + i];
            result[j * SIZE + i] = sum / result[i * SIZE + i];
        }
    }
}

int main() {
    int errors = 0;
    srand(42);

    float* ref_mat = new float[GRID_ROWS * GRID_COLS];
    float* dut_mat = new float[GRID_ROWS * GRID_COLS];

    /* Generate a diagonally dominant matrix for numerical stability */
    for (int i = 0; i < GRID_ROWS; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < GRID_COLS; j++) {
            float val = (float)(rand() % 200 - 100) / 100.0f;
            ref_mat[i * GRID_COLS + j] = val;
            row_sum += fabsf(val);
        }
        /* Make diagonal dominant to avoid division by near-zero */
        ref_mat[i * GRID_COLS + i] = row_sum + 1.0f;
    }

    memcpy(dut_mat, ref_mat, GRID_ROWS * GRID_COLS * sizeof(float));

    /* Compute golden reference */
    lud_ref(ref_mat);

    /* Call HLS DUT */
    workload(dut_mat);

    /* Compare with tolerance (LU decomposition accumulates FP error) */
    float tol = 1e-3f;
    int mismatches = 0;
    for (int i = 0; i < GRID_ROWS * GRID_COLS; i++) {
        float diff = fabsf(ref_mat[i] - dut_mat[i]);
        float scale = fabsf(ref_mat[i]) + 1.0f;
        if (diff > tol * scale) {
            if (mismatches < 5) {
                printf("  mismatch[%d,%d]: ref=%.6f dut=%.6f diff=%.6e\n",
                       i / GRID_COLS, i % GRID_COLS, ref_mat[i], dut_mat[i], diff);
            }
            mismatches++;
        }
    }
    if (mismatches > 0) {
        printf("FAIL: LUD result — %d mismatches out of %d\n", mismatches, GRID_ROWS * GRID_COLS);
        errors++;
    }

    if (errors == 0) {
        printf("PASS: lud testbench — all outputs match golden reference\n");
    } else {
        printf("FAIL: lud testbench — %d error(s)\n", errors);
    }

    delete[] ref_mat; delete[] dut_mat;
    return (errors > 0) ? 1 : 0;
}
