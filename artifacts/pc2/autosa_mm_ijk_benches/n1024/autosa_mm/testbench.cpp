#include "kernel.h"
extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]);

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  data_t (*A)[K] = (data_t (*)[K])malloc(sizeof(data_t) * (size_t)I * (size_t)K);
  data_t (*B)[K] = (data_t (*)[K])malloc(sizeof(data_t) * (size_t)J * (size_t)K);
  data_t (*C)[J] = (data_t (*)[J])malloc(sizeof(data_t) * (size_t)I * (size_t)J);
  data_t (*C_golden)[J] = (data_t (*)[J])malloc(sizeof(data_t) * (size_t)I * (size_t)J);
  if (!A || !B || !C || !C_golden) {
    printf("Failed to allocate\n");
    return 1;
  }

  for (int i = 0; i < I; i++)
    for (int k = 0; k < K; k++) {
      A[i][k] = (data_t)rand() / RAND_MAX;
    }

  for (int j = 0; j < J; j++)
    for (int k = 0; k < K; k++) {
      B[j][k] = (data_t)rand() / RAND_MAX;
    }

  autosa_mm(A, B, C);

  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {
      C_golden[i][j] = 0;
      for (int k = 0; k < K; k++) {
        C_golden[i][j] = C_golden[i][j] + A[i][k] * B[j][k];
      }
    }

  int err = 0;
  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {
      if (fabs((float)C_golden[i][j] - (float)C[i][j]) > 0.001)
        err++;
    }

  free(A);
  free(B);
  free(C);
  free(C_golden);

  if (err)
    printf("Failed with %d errors!\n", err);
  else
    printf("Passed!\n");

  return 0;
}
