/* New deterministic pilot harness; preserved historical harness is in vendor/. */
#include <stdio.h>
#include "kernel.h"

int main(void) {
  double A[N][N], x1[N], x2[N], y1[N], y2[N];
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < N; ++i) {
      x1[i] = (double)(i - 3 + c) / 8.0;
      x2[i] = (double)(2 * i + 1 - c) / 16.0;
      y1[i] = (double)((3 * i + c) % 7 - 2) / 8.0;
      y2[i] = (double)((5 * i + 2 * c) % 9 - 4) / 16.0;
      for (int j = 0; j < N; ++j)
        A[i][j] = c == 2 ? 0.0 : (double)((i * 3 + j * 5 + c) % 13 - 6) / 8.0;
    }
    kernel_mvt(x1, x2, y1, y2, A);
    printf("case%d x1", c);
    for (int i = 0; i < N; ++i) printf(" %.17g", x1[i]);
    printf("\nx2");
    for (int i = 0; i < N; ++i) printf(" %.17g", x2[i]);
    printf("\ny1");
    for (int i = 0; i < N; ++i) printf(" %.17g", y1[i]);
    printf("\ny2");
    for (int i = 0; i < N; ++i) printf(" %.17g", y2[i]);
    printf("\nA");
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) printf(" %.17g", A[i][j]);
    printf("\n");
  }
  return 0;
}
