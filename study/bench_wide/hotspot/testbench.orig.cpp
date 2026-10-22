/*
 * Vitis HLS C-simulation testbench for Hotspot thermal simulation.
 * Computes golden reference using plain C stencil,
 * then compares against HLS workload() output.
 * Returns 0 on success, 1 on mismatch.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hotspot.h"

extern "C" void workload(float result[GRID_ROWS * GRID_COLS],
                         float temp[GRID_ROWS * GRID_COLS],
                         float power[GRID_ROWS * GRID_COLS]);

/* Golden reference: single hotspot timestep */
static void hotspot_ref(float* result, float* temp, float* power,
                        float Cap_1, float Rx_1, float Ry_1, float Rz_1)
{
    float amb_temp = 80.0f;

    for (int r = 0; r < GRID_ROWS; r++) {
        for (int c = 0; c < GRID_COLS; c++) {
            int idx = r * GRID_COLS + c;
            float delta;

            if (r == 0 && c == 0) {
                delta = Cap_1 * (power[idx] +
                    (temp[idx+1] - temp[idx]) * Rx_1 +
                    (temp[idx+GRID_COLS] - temp[idx]) * Ry_1 +
                    (amb_temp - temp[idx]) * Rz_1);
            } else if (r == 0 && c == GRID_COLS-1) {
                delta = Cap_1 * (power[idx] +
                    (temp[idx-1] - temp[idx]) * Rx_1 +
                    (temp[idx+GRID_COLS] - temp[idx]) * Ry_1 +
                    (amb_temp - temp[idx]) * Rz_1);
            } else if (r == GRID_ROWS-1 && c == GRID_COLS-1) {
                delta = Cap_1 * (power[idx] +
                    (temp[idx-1] - temp[idx]) * Rx_1 +
                    (temp[idx-GRID_COLS] - temp[idx]) * Ry_1 +
                    (amb_temp - temp[idx]) * Rz_1);
            } else if (r == GRID_ROWS-1 && c == 0) {
                delta = Cap_1 * (power[idx] +
                    (temp[idx+1] - temp[idx]) * Rx_1 +
                    (temp[idx-GRID_COLS] - temp[idx]) * Ry_1 +
                    (amb_temp - temp[idx]) * Rz_1);
            } else if (r == 0) {
                delta = Cap_1 * (power[idx] +
                    (temp[idx+1] + temp[idx-1] - 2.0f*temp[idx]) * Rx_1 +
                    (temp[idx+GRID_COLS] - temp[idx]) * Ry_1 +
                    (amb_temp - temp[idx]) * Rz_1);
            } else if (r == GRID_ROWS-1) {
                delta = Cap_1 * (power[idx] +
                    (temp[idx+1] + temp[idx-1] - 2.0f*temp[idx]) * Rx_1 +
                    (temp[idx-GRID_COLS] - temp[idx]) * Ry_1 +
                    (amb_temp - temp[idx]) * Rz_1);
            } else if (c == 0) {
                delta = Cap_1 * (power[idx] +
                    (temp[idx+GRID_COLS] + temp[idx-GRID_COLS] - 2.0f*temp[idx]) * Ry_1 +
                    (temp[idx+1] - temp[idx]) * Rx_1 +
                    (amb_temp - temp[idx]) * Rz_1);
            } else if (c == GRID_COLS-1) {
                delta = Cap_1 * (power[idx] +
                    (temp[idx+GRID_COLS] + temp[idx-GRID_COLS] - 2.0f*temp[idx]) * Ry_1 +
                    (temp[idx-1] - temp[idx]) * Rx_1 +
                    (amb_temp - temp[idx]) * Rz_1);
            } else {
                delta = Cap_1 * (power[idx] +
                    (temp[idx+GRID_COLS] + temp[idx-GRID_COLS] - 2.0f*temp[idx]) * Ry_1 +
                    (temp[idx+1] + temp[idx-1] - 2.0f*temp[idx]) * Rx_1 +
                    (amb_temp - temp[idx]) * Rz_1);
            }
            result[idx] = temp[idx] + delta;
        }
    }
}

/* Golden reference: full workload (SIM_TIME/2 ping-pong iterations) */
static void workload_ref(float* result, float* temp, float* power)
{
    float grid_height = CHIP_HEIGHT / GRID_ROWS;
    float grid_width  = CHIP_WIDTH / GRID_COLS;
    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * T_CHIP * grid_width * grid_height;
    float Rx  = grid_width / (2.0f * K_SI * T_CHIP * grid_height);
    float Ry  = grid_height / (2.0f * K_SI * T_CHIP * grid_width);
    float Rz  = T_CHIP / (K_SI * grid_height * grid_width);
    float max_slope = MAX_PD / (FACTOR_CHIP * T_CHIP * SPEC_HEAT_SI);
    float step = PRECISION / max_slope / 1000.0f;

    float Rx_1 = 1.0f / Rx;
    float Ry_1 = 1.0f / Ry;
    float Rz_1 = 1.0f / Rz;
    float Cap_1 = step / Cap;

    for (int i = 0; i < SIM_TIME / 2; i++) {
        hotspot_ref(result, temp, power, Cap_1, Rx_1, Ry_1, Rz_1);
        hotspot_ref(temp, result, power, Cap_1, Rx_1, Ry_1, Rz_1);
    }
}

int main() {
    int errors = 0;
    int N = GRID_ROWS * GRID_COLS;
    srand(42);

    float* temp_ref   = new float[N];
    float* temp_dut   = new float[N];
    float* power_arr  = new float[N];
    float* result_ref = new float[N];
    float* result_dut = new float[N];

    /* Generate deterministic test data: temperatures ~80C, small power variations */
    for (int i = 0; i < N; i++) {
        temp_ref[i] = 80.0f + (float)(rand() % 200) / 100.0f;
        power_arr[i] = (float)(rand() % 100) / 10.0f;
    }
    memcpy(temp_dut, temp_ref, N * sizeof(float));
    memset(result_ref, 0, N * sizeof(float));
    memset(result_dut, 0, N * sizeof(float));

    /* Compute golden reference */
    workload_ref(result_ref, temp_ref, power_arr);

    /* Reset temp for DUT (workload modifies temp in-place via ping-pong) */
    memcpy(temp_dut, temp_ref, N * sizeof(float));

    /* Actually temp_ref was modified by workload_ref. Regenerate for DUT. */
    srand(42);
    for (int i = 0; i < N; i++) {
        temp_dut[i] = 80.0f + (float)(rand() % 200) / 100.0f;
    }

    /* Call HLS DUT */
    workload(result_dut, temp_dut, power_arr);

    /* Compare result with tolerance (accumulated FP over 64 timesteps) */
    float tol = 1e-2f;
    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(result_ref[i] - result_dut[i]);
        float scale = fabsf(result_ref[i]) + 1.0f;
        if (diff > tol * scale) {
            if (mismatches < 5) {
                printf("  mismatch[%d,%d]: ref=%.6f dut=%.6f\n",
                       i / GRID_COLS, i % GRID_COLS, result_ref[i], result_dut[i]);
            }
            mismatches++;
        }
    }
    if (mismatches > 0) {
        printf("FAIL: result — %d mismatches out of %d\n", mismatches, N);
        errors++;
    }

    if (errors == 0) {
        printf("PASS: hotspot testbench — all outputs match golden reference\n");
    } else {
        printf("FAIL: hotspot testbench — %d error(s)\n", errors);
    }

    delete[] temp_ref; delete[] temp_dut; delete[] power_arr;
    delete[] result_ref; delete[] result_dut;
    return (errors > 0) ? 1 : 0;
}
