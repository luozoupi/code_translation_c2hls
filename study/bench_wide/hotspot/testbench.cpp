/*
 * Strict Vitis HLS C-simulation testbench for hotspot.
 * Independent golden: plain.cpp (hotspot + workload) copied below as hotspot_ref/workload_ref and run on identical
 * copies of the deterministic inputs (srand(42)). workload(result, temp, power):
 *   - result : pure output (written by the first hotspot call before it is ever read) -> poisoned with 12345.0f
 *   - temp   : in-out (initial data, ping-ponged and overwritten) -> same real initial data for ref and DUT; also checked
 *   - power  : input only
 * Every element of result[] and temp[] is checked, float tolerance 1e-3*fmax(1,|ref|) (SPEC); NaN is an error.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hotspot.h"

extern "C" void workload(float result[GRID_ROWS * GRID_COLS],
                         float temp[GRID_ROWS * GRID_COLS],
                         float power[GRID_ROWS * GRID_COLS]);

/* ---- reference: copy of plain.cpp ---- */



static void hotspot_ref(float result[GRID_ROWS * GRID_COLS], float temp[GRID_ROWS * GRID_COLS], float power[GRID_ROWS * GRID_COLS], float Cap_1, float Rx_1, float Ry_1, float Rz_1) {
    float amb_temp = 80.0;
    float delta;

    for (int r = 0; r < GRID_ROWS; r++)
        for (int c = 0; c < GRID_COLS; c++) {
            if (r == 0 || c == 0 || r == GRID_ROWS - 1 || c == GRID_COLS - 1) {

                /* Corner 1 */
                if ((r == 0) && (c == 0)) {
                    delta = (Cap_1) * (power[0] +
                        (temp[1] - temp[0]) * Rx_1 +
                        (temp[GRID_COLS] - temp[0]) * Ry_1 +
                        (amb_temp - temp[0]) * Rz_1);
                }   
    
                /* Corner 2 */
                else if ((r == 0) && (c == GRID_COLS - 1)) {
                    delta = (Cap_1) * (power[c] +
                        (temp[c - 1] - temp[c]) * Rx_1 +
                        (temp[c + GRID_COLS] - temp[c]) * Ry_1 +
                        (amb_temp - temp[c]) * Rz_1);
                }   
    
                /* Corner 3 */
                else if ((r == GRID_ROWS - 1) && (c == GRID_COLS - 1)) {
                    delta = (Cap_1) * (power[r*GRID_COLS + c] +
                        (temp[r*GRID_COLS + c - 1] - temp[r*GRID_COLS + c]) * Rx_1 +
                        (temp[(r - 1)*GRID_COLS + c] - temp[r*GRID_COLS + c]) * Ry_1 +
                        (amb_temp - temp[r*GRID_COLS + c]) * Rz_1);
                }   
    
                /* Corner 4 */
                else if ((r == GRID_ROWS - 1) && (c == 0)) {
                    delta = (Cap_1) * (power[r*GRID_COLS] +
                        (temp[r*GRID_COLS + 1] - temp[r*GRID_COLS]) * Rx_1 +
                        (temp[(r - 1)*GRID_COLS] - temp[r*GRID_COLS]) * Ry_1 +
                        (amb_temp - temp[r*GRID_COLS]) * Rz_1);
                }   
    
                /* Edge 1 */
                else if (r == 0) {
                    delta = (Cap_1) * (power[c] +
                        (temp[c + 1] + temp[c - 1] - 2.0*temp[c]) * Rx_1 +
                        (temp[GRID_COLS + c] - temp[c]) * Ry_1 +
                        (amb_temp - temp[c]) * Rz_1);
                }   
    
                /* Edge 2 */
                else if (c == GRID_COLS - 1) {
                    delta = (Cap_1) * (power[r*GRID_COLS + c] +
                        (temp[(r + 1)*GRID_COLS + c] + temp[(r - 1)*GRID_COLS + c] - 2.0*temp[r*GRID_COLS + c]) * Ry_1 +
                        (temp[r*GRID_COLS + c - 1] - temp[r*GRID_COLS + c]) * Rx_1 +
                        (amb_temp - temp[r*GRID_COLS + c]) * Rz_1);
                }   
    
                /* Edge 3 */
                else if (r == GRID_ROWS - 1) {
                    delta = (Cap_1) * (power[r*GRID_COLS + c] +
                        (temp[r*GRID_COLS + c + 1] + temp[r*GRID_COLS + c - 1] - 2.0*temp[r*GRID_COLS + c]) * Rx_1 +
                        (temp[(r - 1)*GRID_COLS + c] - temp[r*GRID_COLS + c]) * Ry_1 +
                        (amb_temp - temp[r*GRID_COLS + c]) * Rz_1);
                }   
    
                /* Edge 4 */
                else if (c == 0) {
                    delta = (Cap_1) * (power[r*GRID_COLS] +
                        (temp[(r + 1)*GRID_COLS] + temp[(r - 1)*GRID_COLS] - 2.0*temp[r*GRID_COLS]) * Ry_1 +
                        (temp[r*GRID_COLS + 1] - temp[r*GRID_COLS]) * Rx_1 +
                        (amb_temp - temp[r*GRID_COLS]) * Rz_1);
                }

            }

            else {
                    delta = (Cap_1 * (power[r*GRID_COLS + c] +
                        (temp[(r + 1)*GRID_COLS + c] + temp[(r - 1)*GRID_COLS + c] - 2.f*temp[r*GRID_COLS + c]) * Ry_1 +
                        (temp[r*GRID_COLS + c + 1] + temp[r*GRID_COLS + c - 1] - 2.f*temp[r*GRID_COLS + c]) * Rx_1 +
                        (amb_temp - temp[r*GRID_COLS + c]) * Rz_1));
            }

            result[r*GRID_COLS + c] = temp[r*GRID_COLS + c] + delta;

        }

    return;
}




static void workload_ref(float result[GRID_ROWS * GRID_COLS], float temp[GRID_ROWS * GRID_COLS], float power[GRID_ROWS * GRID_COLS])
{

    
    
    
    float grid_height = CHIP_HEIGHT / GRID_ROWS;
    float grid_width = CHIP_WIDTH / GRID_COLS;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * T_CHIP * grid_width * grid_height;
    float Rx = grid_width / (2.0 * K_SI * T_CHIP * grid_height);
    float Ry = grid_height / (2.0 * K_SI * T_CHIP * grid_width);
    float Rz = T_CHIP / (K_SI * grid_height * grid_width);

    float max_slope = MAX_PD / (FACTOR_CHIP * T_CHIP * SPEC_HEAT_SI);
    float step = PRECISION / max_slope / 1000.0;

    float Rx_1=1.f / Rx;
    float Ry_1=1.f / Ry;
    float Rz_1=1.f / Rz;
    float Cap_1 = step / Cap;

    int i;
    for (i = 0; i < SIM_TIME/2; i++) {
       hotspot_ref(result, temp, power, Cap_1, Rx_1, Ry_1, Rz_1);
       
       hotspot_ref(temp, result, power, Cap_1, Rx_1, Ry_1, Rz_1);

    }

    return;
}

/* ---- end reference ---- */

static int errors = 0;
static void chk_f(const char *name, int i, float d, float r) {
    if (!(fabsf(d - r) <= 1e-3f * fmaxf(1.0f, fabsf(r)))) {   /* also catches NaN */
        if (errors < 10) printf("%s[%d]: got %g expected %g\n", name, i, (double)d, (double)r);
        errors++;
    }
}

int main() {
    int N = GRID_ROWS * GRID_COLS;
    srand(42);

    float* temp_ref   = new float[N];
    float* temp_dut   = new float[N];
    float* power_arr  = new float[N];
    float* result_ref = new float[N];
    float* result_dut = new float[N];

    /* deterministic data: temperatures ~80C, small power variations */
    for (int i = 0; i < N; i++) {
        temp_ref[i] = 80.0f + (float)(rand() % 200) / 100.0f;
        power_arr[i] = (float)(rand() % 100) / 10.0f;
    }
    memcpy(temp_dut, temp_ref, N * sizeof(float));
    for (int i = 0; i < N; i++) { result_ref[i] = 12345.0f; result_dut[i] = 12345.0f; }

    workload_ref(result_ref, temp_ref, power_arr);
    workload(result_dut, temp_dut, power_arr);

    for (int i = 0; i < N; i++) chk_f("result", i, result_dut[i], result_ref[i]);
    for (int i = 0; i < N; i++) chk_f("temp", i, temp_dut[i], temp_ref[i]);

    delete[] temp_ref; delete[] temp_dut; delete[] power_arr;
    delete[] result_ref; delete[] result_dut;
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
