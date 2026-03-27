#include "md.h"

void md_kernel(TYPE force_x[nAtoms], TYPE force_y[nAtoms], TYPE force_z[nAtoms],
               TYPE position_x[nAtoms], TYPE position_y[nAtoms], TYPE position_z[nAtoms],
               int32_t NL[nAtoms * maxNeighbors]) {
    TYPE delx, dely, delz, r2inv;
    TYPE r6inv, potential, force, j_x, j_y, j_z;
    TYPE i_x, i_y, i_z, fx, fy, fz;
    int32_t i, j, jidx;

    loop_i: for (i = 0; i < nAtoms; i++) {
        i_x = position_x[i];
        i_y = position_y[i];
        i_z = position_z[i];
        fx = 0;
        fy = 0;
        fz = 0;
        loop_j: for (j = 0; j < maxNeighbors; j++) {
            jidx = NL[i * maxNeighbors + j];
            j_x = position_x[jidx];
            j_y = position_y[jidx];
            j_z = position_z[jidx];
            delx = i_x - j_x;
            dely = i_y - j_y;
            delz = i_z - j_z;
            r2inv = 1.0 / (delx * delx + dely * dely + delz * delz);
            r6inv = r2inv * r2inv * r2inv;
            potential = r6inv * (lj1 * r6inv - lj2);
            force = r2inv * potential;
            fx += delx * force;
            fy += dely * force;
            fz += delz * force;
        }
        force_x[i] = fx;
        force_y[i] = fy;
        force_z[i] = fz;
    }
}


void workload(TYPE* force_x, TYPE* force_y, TYPE* force_z,
              TYPE* position_x, TYPE* position_y, TYPE* position_z,
              int32_t* NL) {

    TYPE l_fx[nAtoms], l_fy[nAtoms], l_fz[nAtoms];
    TYPE l_px[nAtoms], l_py[nAtoms], l_pz[nAtoms];
    int32_t l_NL[nAtoms * maxNeighbors];
    int i;

    for (i = 0; i < nAtoms; i++) {
        l_px[i] = position_x[i];
        l_py[i] = position_y[i];
        l_pz[i] = position_z[i];
    }
    for (i = 0; i < nAtoms * maxNeighbors; i++) l_NL[i] = NL[i];

    md_kernel(l_fx, l_fy, l_fz, l_px, l_py, l_pz, l_NL);

    for (i = 0; i < nAtoms; i++) {
        force_x[i] = l_fx[i];
        force_y[i] = l_fy[i];
        force_z[i] = l_fz[i];
    }
}