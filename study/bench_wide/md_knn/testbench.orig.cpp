/*
 * Vitis HLS C-simulation testbench for md_knn (molecular dynamics) benchmark.
 * Generates random atomic positions and neighbor lists,
 * computes LJ forces with golden reference and HLS workload(),
 * compares force outputs.
 * Returns 0 on success, 1 on mismatch.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "md.h"

#define EPSILON 1.0e-6

/* Golden reference: plain-C MD kernel */
static void md_ref(TYPE force_x[nAtoms], TYPE force_y[nAtoms], TYPE force_z[nAtoms],
                   TYPE position_x[nAtoms], TYPE position_y[nAtoms], TYPE position_z[nAtoms],
                   int32_t NL[nAtoms * maxNeighbors]) {
    int i, j;
    for (i = 0; i < nAtoms; i++) {
        TYPE fx = 0, fy = 0, fz = 0;
        TYPE ix = position_x[i], iy = position_y[i], iz = position_z[i];
        for (j = 0; j < maxNeighbors; j++) {
            int jidx = NL[i * maxNeighbors + j];
            TYPE dx = ix - position_x[jidx];
            TYPE dy = iy - position_y[jidx];
            TYPE dz = iz - position_z[jidx];
            TYPE r2inv = 1.0 / (dx*dx + dy*dy + dz*dz);
            TYPE r6inv = r2inv * r2inv * r2inv;
            TYPE pot = r6inv * (lj1 * r6inv - lj2);
            TYPE f = r2inv * pot;
            fx += dx * f;
            fy += dy * f;
            fz += dz * f;
        }
        force_x[i] = fx;
        force_y[i] = fy;
        force_z[i] = fz;
    }
}

extern "C" void workload(TYPE* force_x, TYPE* force_y, TYPE* force_z,
              TYPE* position_x, TYPE* position_y, TYPE* position_z,
              int32_t* NL);

int main() {
    int errors = 0;

    TYPE pos_x[nAtoms], pos_y[nAtoms], pos_z[nAtoms];
    int32_t NL[nAtoms * maxNeighbors];
    TYPE ref_fx[nAtoms], ref_fy[nAtoms], ref_fz[nAtoms];
    TYPE dut_fx[nAtoms], dut_fy[nAtoms], dut_fz[nAtoms];

    /* Generate deterministic random positions (avoid zero distance) */
    srand(42);
    for (int i = 0; i < nAtoms; i++) {
        pos_x[i] = (TYPE)(rand() % 1000) / 100.0 + 0.1;
        pos_y[i] = (TYPE)(rand() % 1000) / 100.0 + 0.1;
        pos_z[i] = (TYPE)(rand() % 1000) / 100.0 + 0.1;
    }
    /* Generate neighbor lists (ensure no self-neighbor) */
    for (int i = 0; i < nAtoms; i++) {
        for (int j = 0; j < maxNeighbors; j++) {
            int nb;
            do { nb = rand() % nAtoms; } while (nb == i);
            NL[i * maxNeighbors + j] = nb;
        }
    }

    /* Compute golden reference */
    md_ref(ref_fx, ref_fy, ref_fz, pos_x, pos_y, pos_z, NL);

    /* Call HLS design under test */
    memset(dut_fx, 0, sizeof(dut_fx));
    memset(dut_fy, 0, sizeof(dut_fy));
    memset(dut_fz, 0, sizeof(dut_fz));
    workload(dut_fx, dut_fy, dut_fz, pos_x, pos_y, pos_z, NL);

    /* Compare outputs */
    for (int i = 0; i < nAtoms; i++) {
        TYPE dx = fabs(dut_fx[i] - ref_fx[i]);
        TYPE dy = fabs(dut_fy[i] - ref_fy[i]);
        TYPE dz = fabs(dut_fz[i] - ref_fz[i]);
        if (dx > EPSILON || dy > EPSILON || dz > EPSILON) {
            if (errors < 5)
                printf("FAIL: atom[%d] force=(%.6f,%.6f,%.6f) expected=(%.6f,%.6f,%.6f)\n",
                       i, dut_fx[i], dut_fy[i], dut_fz[i],
                       ref_fx[i], ref_fy[i], ref_fz[i]);
            errors++;
        }
    }

    if (errors == 0) {
        printf("PASS: md_knn testbench — all %d atom forces match\n", nAtoms);
    } else {
        printf("FAIL: md_knn testbench — %d / %d atom mismatches\n", errors, nAtoms);
    }

    return (errors > 0) ? 1 : 0;
}
