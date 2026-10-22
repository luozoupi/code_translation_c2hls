// Hardened evaluation bench for C = A * B^T (overwrite), I = J = K = 1024, fp32 (study/bench_hard, 2026-10-21).
// Used only for final verification; agents never see it. Differences from bench_fixed:
//  * two input sets per run, kernel called once per set (catches state carried between calls):
//      set 1: signed uniform in [-4, 4);  set 2: wide range, sign * 2^e * (1 + f), e in [-10, 10], ~3% exact zeros;
//  * a seed the agents did not see; golden in double precision;
//  * tolerance |C - ref| <= 1e-4 * sum_k |A[i][k] * B[j][k]| + 1e-6, which admits any fp32 summation order
//    (worst-case fp32 error for K = 1024 is about 6e-5 of that sum) but not reduced precision or fixed point;
//  * NaN or Inf in C is an error; C is poisoned with 12345 before each call.
#include "kernel.h"
#include <cstdint>
extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]);

static uint64_t s = 0x9E3779B97F4A7C15ull ^ 20261021ull;
static inline double u01() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (double)(s >> 11) / 9007199254740992.0; }
static float wide() {
  if (u01() < 0.03) return 0.0f;
  int e = (int)(u01() * 21.0) - 10;
  float v = ldexpf(1.0f + (float)u01(), e);
  return u01() < 0.5 ? -v : v;
}

int main() {
  data_t (*A)[K] = (data_t (*)[K])malloc(sizeof(data_t) * (size_t)I * K);
  data_t (*B)[K] = (data_t (*)[K])malloc(sizeof(data_t) * (size_t)J * K);
  data_t (*C)[J] = (data_t (*)[J])malloc(sizeof(data_t) * (size_t)I * J);
  if (!A || !B || !C) { printf("Failed to allocate\n"); return 1; }
  long err = 0;
  for (int set = 1; set <= 2; set++) {
    for (int i = 0; i < I; i++) for (int k = 0; k < K; k++) A[i][k] = set == 1 ? (float)(u01() * 8.0 - 4.0) : wide();
    for (int j = 0; j < J; j++) for (int k = 0; k < K; k++) B[j][k] = set == 1 ? (float)(u01() * 8.0 - 4.0) : wide();
    for (int i = 0; i < I; i++) for (int j = 0; j < J; j++) C[i][j] = (data_t)12345.0f;
    autosa_mm(A, B, C);
    long e1 = 0;
    for (int i = 0; i < I; i++)
      for (int j = 0; j < J; j++) {
        double ref = 0, mag = 0;
        for (int k = 0; k < K; k++) { double p = (double)A[i][k] * (double)B[j][k]; ref += p; mag += fabs(p); }
        double c = (double)C[i][j];
        if (!(fabs(c - ref) <= 1e-4 * mag + 1e-6)) {   // also true for NaN / Inf
          if (e1 < 5) printf("set %d: C[%d][%d] = %.9g, ref %.9g\n", set, i, j, c, ref);
          e1++;
        }
      }
    printf("set %d: %ld errors\n", set, e1);
    err += e1;
  }
  free(A); free(B); free(C);
  if (err) printf("Failed with %ld errors!\n", err); else printf("Passed!\n");
  return err ? 1 : 0;
}
