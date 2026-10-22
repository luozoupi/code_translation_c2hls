#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "3mm.h"


void init_array(int ni, int nj, int nk, int nl, int nm,
		double A[ NI + 0][NK + 0],
		double B[ NK + 0][NJ + 0],
		double C[ NJ + 0][NM + 0],
		double D[ NM + 0][NL + 0])
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (double) ((i*j+1) % ni) / (5*ni);
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (double) ((i*(j+1)+2) % nj) / (5*nj);
  for (i = 0; i < nj; i++)
    for (j = 0; j < nm; j++)
      C[i][j] = (double) (i*(j+3) % nl) / (5*nl);
  for (i = 0; i < nm; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (double) ((i*(j+2)+2) % nk) / (5*nk);
}


void print_array(int ni, int nl,
		 double G[ NI + 0][NL + 0])
{
  int i, j;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "G");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
	fprintf (stderr, "%0.6lf ", G[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "G");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

/* ---- strict checker (bench_wide SPEC) ---- */
/* golden: static copy of plain.cpp top function, run on identical input copies */



static void kernel_3mm_ref(    
		double E[ NI + 0][NJ + 0],
		double A[ NI + 0][NK + 0],
		double B[ NK + 0][NJ + 0],
		double F[ NJ + 0][NL + 0],
		double C[ NJ + 0][NM + 0],
		double D[ NM + 0][NL + 0],
		double G[ NI + 0][NL + 0])
{


    const int ni = NI;
    const int nj = NJ;
    const int nk = NK;
    const int nl = NL;
    const int nm = NM;

  int i, j, k;


  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      {
	E[i][j] = 0.0;
	for (k = 0; k < nk; ++k)
	  E[i][j] += A[i][k] * B[k][j];
      }

  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      {
	F[i][j] = 0.0;
	for (k = 0; k < nm; ++k)
	  F[i][j] += C[i][k] * D[k][j];
      }

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      {
	G[i][j] = 0.0;
	for (k = 0; k < nj; ++k)
	  G[i][j] += E[i][k] * F[k][j];
      }

}
static int g_errs = 0;
static void chk_d(const char *nm, const double *d, const double *r, size_t n)
{
  for (size_t q = 0; q < n; q++) {
    int bad = 0;
    if (isnan(d[q]) || isnan(r[q])) bad = 1; /* NaN in kernel (or golden) output is an error */
    else bad = !(fabs(d[q] - r[q]) <= 1e-6 * fmax(1.0, fabs(r[q])));
    if (bad) { if (g_errs < 10) printf("MISMATCH %s[%zu]: got %.9g expected %.9g\n", nm, q, d[q], r[q]); g_errs++; }
  }
}
static void chk_i(const char *nm, const int *d, const int *r, size_t n)
{
  for (size_t q = 0; q < n; q++)
    if (d[q] != r[q]) { if (g_errs < 10) printf("MISMATCH %s[%zu]: got %d expected %d\n", nm, q, d[q], r[q]); g_errs++; }
}
static void chk_c(const char *nm, const char *d, const char *r, size_t n)
{
  for (size_t q = 0; q < n; q++)
    if (d[q] != r[q]) { if (g_errs < 10) printf("MISMATCH %s[%zu]: got %d expected %d\n", nm, q, (int)d[q], (int)r[q]); g_errs++; }
}

int main(int argc, char** argv)
{

  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;
  int nm = NM;


   double E[ NI + 0][NJ + 0];
   double A[ NI + 0][NK + 0];
   double B[ NK + 0][NJ + 0];
   double F[ NJ + 0][NL + 0];
   double C[ NJ + 0][NM + 0];
   double D[ NM + 0][NL + 0];
   double G[ NI + 0][NL + 0];


  init_array (ni, nj, nk, nl, nm,
	      A,
	      B,
	      C,
	      D);


    /* reference copies */
  static double E_ref[ NI + 0][NJ + 0];
  static double A_ref[ NI + 0][NK + 0];
  static double B_ref[ NK + 0][NJ + 0];
  static double F_ref[ NJ + 0][NL + 0];
  static double C_ref[ NJ + 0][NM + 0];
  static double D_ref[ NM + 0][NL + 0];
  static double G_ref[ NI + 0][NL + 0];
  /* poison pure outputs */
  for (size_t q = 0; q < sizeof(E)/sizeof(double); q++) ((double*)&E)[q] = 12345.0;
  for (size_t q = 0; q < sizeof(F)/sizeof(double); q++) ((double*)&F)[q] = 12345.0;
  for (size_t q = 0; q < sizeof(G)/sizeof(double); q++) ((double*)&G)[q] = 12345.0;
  memcpy(&E_ref, &E, sizeof(E));
  memcpy(&A_ref, &A, sizeof(A));
  memcpy(&B_ref, &B, sizeof(B));
  memcpy(&F_ref, &F, sizeof(F));
  memcpy(&C_ref, &C, sizeof(C));
  memcpy(&D_ref, &D, sizeof(D));
  memcpy(&G_ref, &G, sizeof(G));
  /* golden */
  kernel_3mm_ref (
	      E_ref,
	      A_ref,
	      B_ref,
	      F_ref,
	      C_ref,
	      D_ref,
	      G_ref);
  /* kernel under test */
  kernel_3mm (
	      E,
	      A,
	      B,
	      F,
	      C,
	      D,
	      G);
  chk_d("G", (const double*)&G, (const double*)&G_ref, sizeof(G)/sizeof(double));
  if (g_errs) { printf("Failed with %d errors!\n", g_errs); return 1; }
  printf("Passed!\n");
  return 0;
}
