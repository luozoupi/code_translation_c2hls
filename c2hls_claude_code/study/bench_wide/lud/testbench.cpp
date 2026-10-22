/*
 * Strict Vitis HLS C-simulation testbench for lud.
 * Golden: lud_ref() (same Doolittle algorithm as plain.cpp's lud) on an identical deterministic diagonally dominant
 * matrix (srand(42)). workload(result): result is IN-OUT (the matrix is decomposed in place), so it has real initial
 * data for both ref and DUT and there is no pure output to poison. Every element is checked, float tol
 * 1e-3*fmax(1,|ref|) (SPEC), NaN = error. Prints "Passed!" or "Failed with N errors!".
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

static int errors = 0;
static void chk_f(const char *name, int i, float d, float r) {
    if (!isfinite(r) || !(fabsf(d - r) <= 1e-3f * fmaxf(1.0f, fabsf(r)))) {   /* NaN/inf in DUT output (or a bad reference) is an error */
        if (errors < 10) printf("%s[%d]: got %g expected %g\n", name, i, (double)d, (double)r);
        errors++;
    }
}

int main() {
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

    for (int i = 0; i < GRID_ROWS * GRID_COLS; i++) chk_f("result", i, dut_mat[i], ref_mat[i]);

    delete[] ref_mat; delete[] dut_mat;
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
