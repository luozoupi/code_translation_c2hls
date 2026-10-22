/*
 * Strict Vitis HLS C-simulation testbench for lc_mgvf (no testbench shipped originally).
 * Golden: plain.cpp (heaviside, lc_mgvf, workload) copied below with _ref suffix and run on identical copies of the
 * deterministic inputs (srand(42), values 0..1). workload(result, imgvf, I):
 *   - result : pure output (written by the first lc_mgvf call before it is read) -> poisoned with 12345.0f
 *   - imgvf  : in-out (real data in, ping-ponged and overwritten) -> same data for ref and DUT; also checked
 *   - I      : input only
 * Every element of result[] and imgvf[] is checked, float tol 1e-3*fmax(1,|ref|), NaN = error.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lc_mgvf.h"

extern "C" void workload(float result[GRID_ROWS * GRID_COLS], float imgvf[GRID_ROWS * GRID_COLS], float I[GRID_ROWS * GRID_COLS]);

/* ---- reference: copy of plain.cpp ---- */



static float heaviside_ref(float x) {
    // A simpler, faster approximation of the Heaviside function
    float out = 0.0;
    if (x > -0.0001) out = 0.5;
    if (x >  0.0001) out = 1.0;
    return out; 
}

static float lc_mgvf_ref(float result[GRID_ROWS * GRID_COLS], float imgvf[GRID_ROWS * GRID_COLS], float I[GRID_ROWS * GRID_COLS])
{
    float total_diff = 0.0;

    for (int i = 0; i < GRID_ROWS; i++) {
        for (int j = 0; j < GRID_COLS; j++) {
            float old_val = imgvf[i * GRID_COLS + j];

            float UL    = ((i == 0              ||  j == 0              ) ? 0 : (imgvf[(i - 1   ) * GRID_COLS + (j - 1  )] - old_val));
            float U     = ((i == 0                                      ) ? 0 : (imgvf[(i - 1   ) * GRID_COLS + (j      )] - old_val));
            float UR    = ((i == 0              ||  j == GRID_COLS - 1  ) ? 0 : (imgvf[(i - 1   ) * GRID_COLS + (j + 1  )] - old_val));

            float L     = ((                        j == 0              ) ? 0 : (imgvf[(i       ) * GRID_COLS + (j - 1  )] - old_val));
            float R     = ((                        j == GRID_COLS - 1  ) ? 0 : (imgvf[(i       ) * GRID_COLS + (j + 1  )] - old_val));

            float DL    = ((i == GRID_ROWS - 1  ||  j == 0              ) ? 0 : (imgvf[(i + 1   ) * GRID_COLS + (j - 1  )] - old_val));
            float D     = ((i == GRID_ROWS - 1                          ) ? 0 : (imgvf[(i + 1   ) * GRID_COLS + (j      )] - old_val));
            float DR    = ((i == GRID_ROWS - 1  ||  j == GRID_COLS - 1  ) ? 0 : (imgvf[(i + 1   ) * GRID_COLS + (j + 1  )] - old_val));

            float vHe = old_val + MU_O_LAMBDA * (heaviside_ref(UL) * UL + heaviside_ref(U) * U + heaviside_ref(UR) * UR + heaviside_ref(L) * L + heaviside_ref(R) * R + heaviside_ref(DL) * DL + heaviside_ref(D) * D + heaviside_ref(DR) * DR);

            float vI = I[i * GRID_COLS + j];
            float new_val = vHe - (ONE_O_LAMBDA * vI * (vHe - vI));
            result[i * GRID_COLS + j] = new_val;

            total_diff += fabs(new_val - old_val);
        }
    }

    return (total_diff / (float)(GRID_ROWS * GRID_COLS));
}

static void workload_ref(float result[GRID_ROWS * GRID_COLS], float imgvf[GRID_ROWS * GRID_COLS], float I[GRID_ROWS * GRID_COLS])
{
    
    

    int i;
    float diff = 1.0;
    for (i = 0; i < ITERATION / 2; i++) {
        diff = lc_mgvf_ref(result, imgvf, I);
        diff = lc_mgvf_ref(imgvf, result, I);
    }
    return;

}








/* ---- end reference ---- */

static int errors = 0;
static void chk_f(const char *name, int i, float d, float r) {
    if (!isfinite(r) || !(fabsf(d - r) <= 1e-3f * fmaxf(1.0f, fabsf(r)))) {   /* NaN/inf in DUT output (or a bad reference) is an error */
        if (errors < 10) printf("%s[%d]: got %g expected %g\n", name, i, (double)d, (double)r);
        errors++;
    }
}
static float frand(float lo, float hi) { return lo + (hi - lo) * ((float)rand() / (float)RAND_MAX); }

#define N (GRID_ROWS * GRID_COLS)
static float I_arr[N], imgvf_ref[N], imgvf_dut[N], res_ref[N], res_dut[N];

int main() {
    srand(42);
    for (int i = 0; i < N; i++) {
        imgvf_ref[i] = frand(0.0f, 1.0f);
        imgvf_dut[i] = imgvf_ref[i];
        I_arr[i] = frand(0.0f, 1.0f);
        res_ref[i] = 12345.0f;
        res_dut[i] = 12345.0f;
    }
    workload_ref(res_ref, imgvf_ref, I_arr);
    workload(res_dut, imgvf_dut, I_arr);
    for (int i = 0; i < N; i++) chk_f("result", i, res_dut[i], res_ref[i]);
    for (int i = 0; i < N; i++) chk_f("imgvf", i, imgvf_dut[i], imgvf_ref[i]);
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
