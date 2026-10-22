/*
 * Strict Vitis HLS C-simulation testbench for lavaMD.
 * Golden: lavaMD_ref() (same algorithm as plain.cpp's lavaMD_baseline_padded) on identical deterministic inputs
 * (srand(42)). workload(pos_i, q_i, pos_o): pos_i, q_i are inputs. pos_o is IN-OUT: plain.cpp accumulates into it with
 * += and never zeroes it, so it is NOT poisoned; ref and DUT get the same non-zero initial values (0..1) and the
 * result must be initial + forces. Every pos_o element is checked, float tol 1e-3*fmax(1,|ref|) (SPEC), NaN = error.
 * Prints "Passed!" or "Failed with N errors!".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lavaMD.h"

extern "C" void workload(TYPE pos_i[N_PADDED * POS_DIM],
                         TYPE q_i[N_PADDED],
                         TYPE pos_o[N * POS_DIM]);

/* Golden reference: N-body force computation with padded layout */
static void lavaMD_ref(TYPE* pos_i, TYPE* q_i, TYPE* pos_o)
{
    int neighborOffset[FULL_NEIGHBOR_COUNT][3] = {
        {-1,-1,-1}, { 0,-1,-1}, { 1,-1,-1},
        {-1, 0,-1}, { 0, 0,-1}, { 1, 0,-1},
        {-1, 1,-1}, { 0, 1,-1}, { 1, 1,-1},
        {-1,-1, 0}, { 0,-1, 0}, { 1,-1, 0},
        {-1, 0, 0}, { 0, 0, 0}, { 1, 0, 0},
        {-1, 1, 0}, { 0, 1, 0}, { 1, 1, 0},
        {-1,-1, 1}, { 0,-1, 1}, { 1,-1, 1},
        {-1, 0, 1}, { 0, 0, 1}, { 1, 0, 1},
        {-1, 1, 1}, { 0, 1, 1}, { 1, 1, 1}
    };

    for (int i = 0; i < DIMENSION_3D; i++) {
        int C_idx = i * NUMBER_PAR_PER_BOX;
        int z = i / DIMENSION_2D;
        int remainder = i % DIMENSION_2D;
        int y = remainder / DIMENSION_1D;
        int x = remainder % DIMENSION_1D;

        int x_n = x + 1, y_n = y + 1, z_n = z + 1;
        int A_idx = (z_n * DIMENSION_2D_PADDED + y_n * DIMENSION_1D_PADDED + x_n)
                    * NUMBER_PAR_PER_BOX;

        for (int l = 0; l < FULL_NEIGHBOR_COUNT; l++) {
            int bx = x_n + neighborOffset[l][0];
            int by = y_n + neighborOffset[l][1];
            int bz = z_n + neighborOffset[l][2];
            int B_idx = (bz * DIMENSION_2D_PADDED + by * DIMENSION_1D_PADDED + bx)
                        * NUMBER_PAR_PER_BOX;

            for (int j = 0; j < NUMBER_PAR_PER_BOX; j++) {
                int Aj = A_idx + j;
                int Cj = C_idx + j;
                for (int k = 0; k < NUMBER_PAR_PER_BOX; k++) {
                    int Bk = B_idx + k;
                    TYPE r2 = pos_i[Aj*POS_DIM+V] + pos_i[Bk*POS_DIM+V] -
                        (pos_i[Aj*POS_DIM+X] * pos_i[Bk*POS_DIM+X] +
                         pos_i[Aj*POS_DIM+Y] * pos_i[Bk*POS_DIM+Y] +
                         pos_i[Aj*POS_DIM+Z] * pos_i[Bk*POS_DIM+Z]);
                    TYPE u2 = A2 * r2;
                    u2 = u2 * -1.0f;
                    TYPE vij = 1.0f + u2 + 0.5f*u2*u2 +
                               0.16666f*u2*u2*u2 +
                               0.041666f*u2*u2*u2*u2;
                    TYPE fs = 2.0f * vij;
                    TYPE dx = pos_i[Aj*POS_DIM+X] - pos_i[Bk*POS_DIM+X];
                    TYPE dy = pos_i[Aj*POS_DIM+Y] - pos_i[Bk*POS_DIM+Y];
                    TYPE dz = pos_i[Aj*POS_DIM+Z] - pos_i[Bk*POS_DIM+Z];

                    pos_o[Cj*POS_DIM+V] += q_i[Bk] * vij;
                    pos_o[Cj*POS_DIM+X] += q_i[Bk] * fs * dx;
                    pos_o[Cj*POS_DIM+Y] += q_i[Bk] * fs * dy;
                    pos_o[Cj*POS_DIM+Z] += q_i[Bk] * fs * dz;
                }
            }
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

    TYPE* pos_i     = new TYPE[N_PADDED * POS_DIM];
    TYPE* q_i       = new TYPE[N_PADDED];
    TYPE* ref_pos_o = new TYPE[N * POS_DIM];
    TYPE* dut_pos_o = new TYPE[N * POS_DIM];

    /* Generate test data: small position values, unit charges */
    memset(pos_i, 0, N_PADDED * POS_DIM * sizeof(TYPE));
    memset(q_i, 0, N_PADDED * sizeof(TYPE));

    /* Fill only the valid (non-padding) entries via the padded indexing */
    for (int i = 0; i < DIMENSION_3D; i++) {
        int z = i / DIMENSION_2D;
        int remainder = i % DIMENSION_2D;
        int y = remainder / DIMENSION_1D;
        int x = remainder % DIMENSION_1D;
        int base = ((z+1)*DIMENSION_2D_PADDED + (y+1)*DIMENSION_1D_PADDED + (x+1))
                   * NUMBER_PAR_PER_BOX;
        for (int j = 0; j < NUMBER_PAR_PER_BOX; j++) {
            int idx = base + j;
            pos_i[idx*POS_DIM+V] = (TYPE)(rand() % 100) / 50.0f;
            pos_i[idx*POS_DIM+X] = (TYPE)(rand() % 100) / 50.0f;
            pos_i[idx*POS_DIM+Y] = (TYPE)(rand() % 100) / 50.0f;
            pos_i[idx*POS_DIM+Z] = (TYPE)(rand() % 100) / 50.0f;
            q_i[idx] = (TYPE)(rand() % 10 + 1) / 10.0f;
        }
    }

    /* in-out accumulator: same real (non-zero) initial data for ref and DUT */
    for (int i = 0; i < N * POS_DIM; i++) {
        ref_pos_o[i] = (TYPE)(rand() % 100) / 100.0f;
        dut_pos_o[i] = ref_pos_o[i];
    }

    /* Compute golden reference */
    lavaMD_ref(pos_i, q_i, ref_pos_o);

    /* Call HLS DUT */
    workload(pos_i, q_i, dut_pos_o);

    for (int i = 0; i < N * POS_DIM; i++) chk_f("pos_o", i, dut_pos_o[i], ref_pos_o[i]);

    delete[] pos_i; delete[] q_i;
    delete[] ref_pos_o; delete[] dut_pos_o;
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
