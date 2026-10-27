/* New deterministic pilot harness; preserved historical harness is in vendor/. */
#include <stdio.h>
#include "kernel.h"

int main(void) {
  double A[N], B[N];
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < N; ++i) {
      A[i] = c == 2 ? 0.0 : (double)((i * 3 + c) % 11 - 5) / 8.0;
      B[i] = (double)((i * 5 + 2 * c) % 13 - 6) / 16.0;
    }
    A[0] = 0.25 + (double)c / 8.0;
    A[N - 1] = -0.5 - (double)c / 8.0;
    B[0] = -0.125 - (double)c / 16.0;
    B[N - 1] = 0.375 + (double)c / 16.0;
    kernel_jacobi_1d(A, B);
    printf("case%d A", c);
    for (int i = 0; i < N; ++i) printf(" %.17g", A[i]);
    printf("\nB");
    for (int i = 0; i < N; ++i) printf(" %.17g", B[i]);
    printf("\n");
  }
  return 0;
}
