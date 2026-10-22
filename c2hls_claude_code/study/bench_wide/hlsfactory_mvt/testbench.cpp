#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "mvt.h"


void init_array(int n,
		double x1[ N + 0],
		double x2[ N + 0],
		double y_1[ N + 0],
		double y_2[ N + 0],
		double A[ N + 0][N + 0])
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      x1[i] = (double) (i % n) / n;
      x2[i] = (double) ((i + 1) % n) / n;
      y_1[i] = (double) ((i + 3) % n) / n;
      y_2[i] = (double) ((i + 4) % n) / n;
      for (j = 0; j < n; j++)
	A[i][j] = (double) (i*j % n) / n;
    }
}


void print_array(int n,
		 double x1[ N + 0],
		 double x2[ N + 0])

{
  int i;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "x1");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (stderr, "\n");
    fprintf (stderr, "%0.6lf ", x1[i]);
  }
  fprintf(stderr, "\nend   dump: %s\n", "x1");

  fprintf(stderr, "begin dump: %s", "x2");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (stderr, "\n");
    fprintf (stderr, "%0.6lf ", x2[i]);
  }
  fprintf(stderr, "\nend   dump: %s\n", "x2");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

/* ---- strict checker (bench_wide SPEC) ---- */
/* golden: static copy of plain.cpp top function, run on identical input copies */



static void kernel_mvt_ref(
		double x1[ N + 0],
		double x2[ N + 0],
		double y_1[ N + 0],
		double y_2[ N + 0],
		double A[ N + 0][N + 0])
{


    const int n = N;

  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      x1[i] = x1[i] + A[i][j] * y_1[j];
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      x2[i] = x2[i] + A[j][i] * y_2[j];

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

  int n = N;


   double A[ N + 0][N + 0];
   double x1[ N + 0];
   double x2[ N + 0];
   double y_1[ N + 0];
   double y_2[ N + 0];


  init_array (n,
	      x1,
	      x2,
	      y_1,
	      y_2,
	      A);


    /* reference copies */
  static double A_ref[ N + 0][N + 0];
  static double x1_ref[ N + 0];
  static double x2_ref[ N + 0];
  static double y_1_ref[ N + 0];
  static double y_2_ref[ N + 0];
  /* poison pure outputs */
  memcpy(&A_ref, &A, sizeof(A));
  memcpy(&x1_ref, &x1, sizeof(x1));
  memcpy(&x2_ref, &x2, sizeof(x2));
  memcpy(&y_1_ref, &y_1, sizeof(y_1));
  memcpy(&y_2_ref, &y_2, sizeof(y_2));
  /* golden */
  kernel_mvt_ref (
	      x1_ref,
	      x2_ref,
	      y_1_ref,
	      y_2_ref,
	      A_ref);
  /* kernel under test */
  kernel_mvt (
	      x1,
	      x2,
	      y_1,
	      y_2,
	      A);
  chk_d("x1", (const double*)&x1, (const double*)&x1_ref, sizeof(x1)/sizeof(double));
  chk_d("x2", (const double*)&x2, (const double*)&x2_ref, sizeof(x2)/sizeof(double));
  if (g_errs) { printf("Failed with %d errors!\n", g_errs); return 1; }
  printf("Passed!\n");
  return 0;
}
