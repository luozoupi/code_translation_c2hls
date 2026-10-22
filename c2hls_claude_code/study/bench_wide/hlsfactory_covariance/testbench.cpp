#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "covariance.h"


void init_array (int m, int n,
		 double *float_n,
		 double data[ N + 0][M + 0])
{
  int i, j;

  *float_n = (double)n;

  for (i = 0; i < 100; i++)
    for (j = 0; j < 80; j++)
      data[i][j] = ((double) i*j) / 80;
}


void print_array(int m,
		 double cov[ M + 0][M + 0])

{
  int i, j;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "cov");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf (stderr, "\n");
      fprintf (stderr, "%0.6lf ", cov[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "cov");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

/* ---- strict checker (bench_wide SPEC) ---- */
/* golden: static copy of plain.cpp top function, run on identical input copies */



static void kernel_covariance_ref( 
		       double float_n,
		       double data[ N + 0][M + 0],
		       double cov[ M + 0][M + 0],
		       double mean[ M + 0])
{


    const int n = N;
    const int m = M;

  int i, j, k;

  for (j = 0; j < m; j++)
    {
      mean[j] = 0.0;
      for (i = 0; i < n; i++)
        mean[j] += data[i][j];
      mean[j] /= float_n;
    }

  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      data[i][j] -= mean[j];

  for (i = 0; i < m; i++)
    for (j = i; j < m; j++)
      {
        cov[i][j] = 0.0;
        for (k = 0; k < n; k++)
	  cov[i][j] += data[k][i] * data[k][j];
        cov[i][j] /= (float_n - 1.0);
        cov[j][i] = cov[i][j];
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

  int n = N;
  int m = M;


  double float_n;
  double data[ N + 0][M + 0];
  double cov[ M + 0][M + 0];
  double mean[ M + 0];


  init_array (m, n, &float_n, data);


    /* reference copies */
  static double data_ref[ N + 0][M + 0];
  static double cov_ref[ M + 0][M + 0];
  static double mean_ref[ M + 0];
  /* poison pure outputs */
  for (size_t q = 0; q < sizeof(cov)/sizeof(double); q++) ((double*)&cov)[q] = 12345.0;
  for (size_t q = 0; q < sizeof(mean)/sizeof(double); q++) ((double*)&mean)[q] = 12345.0;
  memcpy(&data_ref, &data, sizeof(data));
  memcpy(&cov_ref, &cov, sizeof(cov));
  memcpy(&mean_ref, &mean, sizeof(mean));
  /* golden */
  kernel_covariance_ref ( float_n,
		     data_ref,
		     cov_ref,
		     mean_ref);
  /* kernel under test */
  kernel_covariance ( float_n,
		     data,
		     cov,
		     mean);
  chk_d("cov", (const double*)&cov, (const double*)&cov_ref, sizeof(cov)/sizeof(double));
  if (g_errs) { printf("Failed with %d errors!\n", g_errs); return 1; }
  printf("Passed!\n");
  return 0;
}
