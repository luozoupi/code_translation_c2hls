/* New deterministic pilot harness; preserved historical harness is in vendor/. */
#include <stdio.h>
#include "kernel.h"

int main(void) {
  double A[M][N], x[N], y[N], tmp[M];
  for (int c = 0; c < 3; ++c) {
    for (int j = 0; j < N; ++j) {
      x[j] = (double)((j * 3 + c) % 11 - 5) / 8.0;
      y[j] = 7.0 + (double)j;
    }
    for (int i = 0; i < M; ++i) {
      tmp[i] = -9.0 - (double)i;
      for (int j = 0; j < N; ++j)
        A[i][j] = c == 2 ? 0.0 : (double)((i * 5 + j * 3 + c) % 13 - 6) / 8.0;
    }
    kernel_atax(A, x, y, tmp);
    printf("case%d y", c);
    for (int j = 0; j < N; ++j) printf(" %.17g", y[j]);
    printf("\ntmp");
    for (int i = 0; i < M; ++i) printf(" %.17g", tmp[i]);
    printf("\nx");
    for (int j = 0; j < N; ++j) printf(" %.17g", x[j]);
    printf("\nA");
    for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) printf(" %.17g", A[i][j]);
    printf("\n");
  }
  return 0;
}
