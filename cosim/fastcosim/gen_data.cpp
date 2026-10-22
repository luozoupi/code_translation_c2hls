/* Data + golden for fast cosim: the bench testbench's data and golden logic, unchanged.
 * Build: g++ -O2 -I<dir with the bench kernel.h> -DEXPECT_N=1024 gen_data.cpp -o gen_data
 * Run:   gen_data OUTDIR   -> A.bin B.bin C.bin(zeros) C_golden.bin C_golden.memh (raw little-endian float32,
 *        row-major, i.e. m_axi byte order; .memh = one 512-bit beat per line, float 15 first). */
#include <stdint.h>
#include <string.h>
#include "kernel.h"  /* the bench header: I = J = K = 1024 */
static_assert(I == EXPECT_N && J == EXPECT_N && K == EXPECT_N, "fast cosim must use the bench problem size");

static void dump(const char *dir, const char *name, const void *p, size_t bytes) {
  char path[4096];
  snprintf(path, sizeof path, "%s/%s", dir, name);
  FILE *f = fopen(path, "wb");
  if (!f || fwrite(p, 1, bytes, f) != bytes) { perror(path); exit(2); }
  fclose(f);
}

int main(int argc, char **argv) {
  if (argc != 2) { fprintf(stderr, "usage: gen_data OUTDIR\n"); return 2; }
  data_t (*A)[K] = (data_t (*)[K])malloc(sizeof(data_t) * (size_t)I * K);
  data_t (*B)[K] = (data_t (*)[K])malloc(sizeof(data_t) * (size_t)J * K);
  data_t (*C)[J] = (data_t (*)[J])calloc((size_t)I * J, sizeof(data_t));
  data_t (*G)[J] = (data_t (*)[J])malloc(sizeof(data_t) * (size_t)I * J);
  if (!A || !B || !C || !G) return 2;
  printf("TB_SIZE I=%d J=%d K=%d\n", I, J, K);
  /* Same rand() sequence as the bench testbench: all of A, then all of B. */
  for (int i = 0; i < I; i++)
    for (int k = 0; k < K; k++) A[i][k] = (data_t)rand() / RAND_MAX;
  for (int j = 0; j < J; j++)
    for (int k = 0; k < K; k++) B[j][k] = (data_t)rand() / RAND_MAX;
  /* Same golden loop as the bench (float, k in order). */
  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {
      G[i][j] = 0;
      for (int k = 0; k < K; k++) G[i][j] = G[i][j] + A[i][k] * B[j][k];
    }
  dump(argv[1], "A.bin", A, sizeof(data_t) * (size_t)I * K);
  dump(argv[1], "B.bin", B, sizeof(data_t) * (size_t)J * K);
  dump(argv[1], "C.bin", C, sizeof(data_t) * (size_t)I * J);
  dump(argv[1], "C_golden.bin", G, sizeof(data_t) * (size_t)I * J);
  char path[4096];
  snprintf(path, sizeof path, "%s/C_golden.memh", argv[1]);
  FILE *f = fopen(path, "w");
  const uint32_t *w = (const uint32_t *)G;
  size_t n = (size_t)I * J;
  for (size_t b = 0; b < n / 16; b++) {
    for (int l = 15; l >= 0; l--) fprintf(f, "%08x", w[b * 16 + l]);
    fputc('\n', f);
  }
  fclose(f);
  return 0;
}
