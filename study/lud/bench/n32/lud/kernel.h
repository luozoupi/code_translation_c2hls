#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef float data_t;
#define N 32

#ifdef __cplusplus
extern "C" void lu_kernel(data_t A[N][N], data_t L[N][N], data_t U[N][N]);
#endif
