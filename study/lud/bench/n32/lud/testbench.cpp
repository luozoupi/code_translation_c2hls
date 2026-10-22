#include "kernel.h"
extern "C" void lu_kernel(data_t A[N][N], data_t L[N][N], data_t U[N][N]);

// Golden: AutoSA's lu_cpu (same recurrences, 3-D temporaries). Checks every output element:
// U[k][i] for i >= k and L[k][j] for j > k. (AutoSA's own testbench compares L only where nothing is written.)
static void init_array(data_t A[N][N]) {
  static data_t T[N][N];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j <= i; j++) T[i][j] = (data_t)(-j % N) / N + 1;
    for (int j = i + 1; j < N; j++) T[i][j] = 0;
    T[i][i] = 1;
  }
  for (int r = 0; r < N; r++)
    for (int s = 0; s < N; s++) {
      data_t acc = 0;
      for (int t = 0; t < N; t++) acc += T[r][t] * T[s][t];
      A[r][s] = acc;
    }
}

static void lu_golden(data_t A[N][N], data_t L[N][N], data_t U[N][N]) {
  static data_t V[N][N], P[N][N];
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) V[i][j] = A[i][j];
  for (int k = 0; k < N; k++) {
    for (int i = k; i < N; i++) U[k][i] = V[i][k];                 // U[j][i] at j == k
    for (int j = k + 1; j < N; j++) L[k][j] = V[k][j] / V[k][k];   // L[i][j] at i == k
    for (int j = k + 1; j < N; j++)
      for (int i = k; i < N; i++) P[i][j] = V[i][j] - L[k][j] * V[i][k];
    for (int j = k + 1; j < N; j++)
      for (int i = k; i < N; i++) V[i][j] = P[i][j];
  }
}

int main() {
  static data_t A[N][N], L[N][N], U[N][N], Lg[N][N], Ug[N][N];
  init_array(A);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) { L[i][j] = 12345.0f; U[i][j] = 12345.0f; Lg[i][j] = 0; Ug[i][j] = 0; }
  lu_kernel(A, L, U);
  lu_golden(A, Lg, Ug);
  int err = 0;
  for (int k = 0; k < N; k++)
    for (int x = 0; x < N; x++) {
      if (x >= k && fabs((float)Ug[k][x] - (float)U[k][x]) > 1e-3f * (1.0f + fabs((float)Ug[k][x]))) err++;
      if (x > k && fabs((float)Lg[k][x] - (float)L[k][x]) > 1e-3f * (1.0f + fabs((float)Lg[k][x]))) err++;
    }
  if (err) printf("Failed with %d errors!\n", err);
  else printf("Passed!\n");
  return 0;
}
