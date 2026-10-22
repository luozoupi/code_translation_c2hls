#include <assert.h>
#include <stdio.h>
#include "kernel_kernel.h"

#include "kernel.h"

void init_array(data_t A[N][N])
{
  int i, j;

  for (i = 0; i < N; i++)
  {
    for (j = 0; j <= i; j++)
      A[i][j] = (data_t)(-j % N) / N + 1;
    for (j = i + 1; j < N; j++) {
      A[i][j] = 0;
    }
    A[i][i] = 1;
  }

  /* Make the matrix positive semi-definite. */
  /* not necessary for LU, but using same code as cholesky */
  int r, s, t;
  data_t B[N][N];
  for (r = 0; r < N; r++)
    for (s = 0; s < N; s++) 
      B[r][s] = 0;
  for (t = 0; t < N; t++)
    for (r = 0; r < N; r++)
      for (s = 0; s < N; s++)
        B[r][s] += A[r][t] * A[s][t];
  for (r = 0; r < N; r++)        
    for (s = 0; s < N; s++)
      A[r][s] = B[r][s];
}

void lu_cpu(data_t A[N][N], data_t L[N][N], data_t U[N][N]) {
  data_t prev_V[N][N][N];
  data_t V_tmp[N][N][N];
  data_t U_tmp[N][N][N];
  data_t L_tmp[N][N][N];

  for (int k = 0; k < N; k++)
    for (int j = k; j < N; j++)
      for (int i = k; i < N; i++) {
        if (k == 0)
          prev_V[i][j][k] = A[i][j];
        else
          prev_V[i][j][k] = V_tmp[i][j][k - 1];
        
        if (j == k) {
          U_tmp[i][j][k] = prev_V[i][j][k];
          U[j][i] = U_tmp[i][j][k];
        } else {
          U_tmp[i][j][k] = U_tmp[i][j - 1][k];

          if (i == k) {            
            L_tmp[i][j][k] = prev_V[i][j][k] / U_tmp[i][j - 1][k]; // final
            L[i][j] = L_tmp[i][j][k];
          } else {
            L_tmp[i][j][k] = L_tmp[i - 1][j][k];
          }
          V_tmp[i][j][k] = prev_V[i][j][k] - L_tmp[i][j][k] * U_tmp[i][j - 1][k];
        }
      }  
}

void lu_device(data_t A[N][N], data_t L[N][N], data_t U[N][N])
{
