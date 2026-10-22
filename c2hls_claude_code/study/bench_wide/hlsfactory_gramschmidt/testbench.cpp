#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "gramschmidt.h"


void init_array(int m, int n,
		double A[ M + 0][N + 0],
		double R[ N + 0][N + 0],
		double Q[ M + 0][N + 0])
{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      A[i][j] = (((double) ((i*j) % m) / m )*100) + 10;
      Q[i][j] = 0.0;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      R[i][j] = 0.0;
}


void print_array(int m, int n,
		 double A[ M + 0][N + 0],
		 double R[ N + 0][N + 0],
		 double Q[ M + 0][N + 0])
{
  int i, j;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "R");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	if ((i*n+j) % 20 == 0) fprintf (stderr, "\n");
	fprintf (stderr, "%0.6lf ", R[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "R");

  fprintf(stderr, "begin dump: %s", "Q");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i*n+j) % 20 == 0) fprintf (stderr, "\n");
	fprintf (stderr, "%0.6lf ", Q[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "Q");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

/* ---- strict checker (bench_wide SPEC) ---- */
/* golden: static copy of plain.cpp top function, run on identical input copies */



static void kernel_gramschmidt_ref( 
			double A[ M + 0][N + 0],
			double R[ N + 0][N + 0],
			double Q[ M + 0][N + 0])
{


    const int m = M;
    const int n = N;

  int i, j, k;

  double nrm;

  for (k = 0; k < n; k++)
    {
      nrm = 0.0;
      for (i = 0; i < m; i++)
        nrm += A[i][k] * A[i][k];
      R[k][k] = sqrt(nrm);
      for (i = 0; i < m; i++)
        Q[i][k] = A[i][k] / R[k][k];
      for (j = k + 1; j < n; j++)
	{
	  R[k][j] = 0.0;
	  for (i = 0; i < m; i++)
	    R[k][j] += Q[i][k] * A[i][j];
	  for (i = 0; i < m; i++)
	    A[i][j] = A[i][j] - Q[i][k] * R[k][j];
	}
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

  int m = M;
  int n = N;


  double A[ M + 0][N + 0];
  double R[ N + 0][N + 0];
  double Q[ M + 0][N + 0];


  init_array (m, n,
	      A,
	      R,
	      Q);


    /* reference copies */
  static double A_ref[ M + 0][N + 0];
  static double R_ref[ N + 0][N + 0];
  static double Q_ref[ M + 0][N + 0];
  /* poison pure outputs */
  for (size_t q = 0; q < sizeof(Q)/sizeof(double); q++) ((double*)&Q)[q] = 12345.0;
  memcpy(&A_ref, &A, sizeof(A));
  memcpy(&R_ref, &R, sizeof(R));
  memcpy(&Q_ref, &Q, sizeof(Q));
  /* golden */
  kernel_gramschmidt_ref (
		      A_ref,
		      R_ref,
		      Q_ref);
  /* kernel under test */
  kernel_gramschmidt (
		      A,
		      R,
		      Q);
  chk_d("A", (const double*)&A, (const double*)&A_ref, sizeof(A)/sizeof(double));
  chk_d("R", (const double*)&R, (const double*)&R_ref, sizeof(R)/sizeof(double));
  chk_d("Q", (const double*)&Q, (const double*)&Q_ref, sizeof(Q)/sizeof(double));
  if (g_errs) { printf("Failed with %d errors!\n", g_errs); return 1; }
  printf("Passed!\n");
  return 0;
}
