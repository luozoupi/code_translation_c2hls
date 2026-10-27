/* Reduced-shape ABI adapter; upstream double precision retained. */
#ifndef HLSMITH_PILOT_MVT_H
#define HLSMITH_PILOT_MVT_H
#define N 8

#ifdef __cplusplus
extern "C" {
#endif
void kernel_mvt(double x1[N], double x2[N], double y_1[N], double y_2[N], double A[N][N]);
#ifdef __cplusplus
}
#endif
#endif
