/* Reduced-shape ABI adapter; upstream double precision retained. */
#ifndef HLSMITH_PILOT_JACOBI_1D_H
#define HLSMITH_PILOT_JACOBI_1D_H
#define N 10
#define TSTEPS 4

#ifdef __cplusplus
extern "C" {
#endif
void kernel_jacobi_1d(double A[N], double B[N]);
#ifdef __cplusplus
}
#endif
#endif
