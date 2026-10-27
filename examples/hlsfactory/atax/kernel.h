/* Reduced-shape ABI adapter; upstream double precision retained. */
#ifndef HLSMITH_PILOT_ATAX_H
#define HLSMITH_PILOT_ATAX_H
#define M 6
#define N 8

#ifdef __cplusplus
extern "C" {
#endif
void kernel_atax(double A[M][N], double x[N], double y[N], double tmp[M]);
#ifdef __cplusplus
}
#endif
#endif
