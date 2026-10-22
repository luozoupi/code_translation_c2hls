#include "kernel.h"

// LU decomposition without pivoting (AutoSA autosa_tests/lu, uniform-dependence form, not in place).
// Outputs: U[k][i] for i >= k (upper triangle incl. diagonal) and L[k][j] for j > k (multipliers, stored transposed:
// L[k][j] is the factor of row j at step k). Other elements of L and U are not outputs.
extern "C" void lu_kernel(data_t A[N][N], data_t L[N][N], data_t U[N][N]) {
  data_t prev_V[N][N];
  data_t V[N][N];
  data_t U_tmp[N][N];
  data_t L_tmp[N][N];
  for (int k = 0; k < N; k++) {
    for (int j = k; j < N; j++)
      for (int i = k; i < N; i++) {
        if (k == 0)
          prev_V[i][j] = A[i][j];
        else
          prev_V[i][j] = V[i][j];
        if (j == k) {
          U_tmp[i][j] = prev_V[i][j];
          U[j][i] = U_tmp[i][j];
        } else {
          U_tmp[i][j] = U_tmp[i][j - 1];
          if (i == k) {
            L_tmp[i][j] = prev_V[i][j] / U_tmp[i][j];
            L[i][j] = L_tmp[i][j];
          } else {
            L_tmp[i][j] = L_tmp[i - 1][j];
          }
          V[i][j] = prev_V[i][j] - L_tmp[i][j] * U_tmp[i][j];
        }
      }
  }
}
