#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#define N 32
#define M 32
#define P 32

extern "C" {
void matmul(int A[N][M], int B[M][P], int AB[N][P]);
}

static int v0_A[N][M];
static int v1_B[M][P];
static int v2_AB[N][P];

int main() {
  memset(v0_A, 0, sizeof(v0_A));
  memset(v1_B, 0, sizeof(v1_B));
  memset(v2_AB, 0, sizeof(v2_AB));
  matmul(v0_A, v1_B, v2_AB);
  printf("PASS smoke\n");
  return 0;
}
