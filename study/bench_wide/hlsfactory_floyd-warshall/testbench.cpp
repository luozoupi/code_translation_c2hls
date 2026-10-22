#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "floyd-warshall.h"


void init_array (int n,
		 int path[ N + 0][N + 0])
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      path[i][j] = i*j%7+1;
      if ((i+j)%13 == 0 || (i+j)%7==0 || (i+j)%11 == 0)
         path[i][j] = 999;
    }
}


void print_array(int n,
		 int path[ N + 0][N + 0])

{
  int i, j;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "path");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf (stderr, "\n");
      fprintf (stderr, "%d ", path[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "path");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

/* ---- strict checker (bench_wide SPEC) ---- */
/* golden: static copy of plain.cpp top function, run on identical input copies */



static void kernel_floyd_warshall_ref(
			   int path[ N + 0][N + 0])
{


    const int n = N;

  int i, j, k;

  for (k = 0; k < n; k++)
    {
      for(i = 0; i < n; i++)
	for (j = 0; j < n; j++)
	  path[i][j] = path[i][j] < path[i][k] + path[k][j] ?
	    path[i][j] : path[i][k] + path[k][j];
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


   int path[ N + 0][N + 0];


  init_array (n, path);


    /* reference copies */
  static int path_ref[ N + 0][N + 0];
  /* poison pure outputs */
  memcpy(&path_ref, &path, sizeof(path));
  /* golden */
  kernel_floyd_warshall_ref (path_ref);
  /* kernel under test */
  kernel_floyd_warshall (path);
  chk_i("path", (const int*)&path, (const int*)&path_ref, sizeof(path)/sizeof(int));
  /* second input: sparse random graph (3% edges, weights 1..20, INF=100000 elsewhere, 0 diagonal) so that shortest paths need many intermediate k */
  {
    unsigned long long st = 12345ULL;
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
      st = st * 6364136223846793005ULL + 1442695040888963407ULL;
      unsigned v = (unsigned)(st >> 33);
      path[i][j] = (i == j) ? 0 : ((v % 100) < 3 ? (int)(1 + (v / 100) % 20) : 100000);
    }
    memcpy(&path_ref, &path, sizeof(path));
    kernel_floyd_warshall_ref(path_ref);
    kernel_floyd_warshall(path);
    chk_i("path(input2)", (const int*)&path, (const int*)&path_ref, sizeof(path)/sizeof(int));
  }
  if (g_errs) { printf("Failed with %d errors!\n", g_errs); return 1; }
  printf("Passed!\n");
  return 0;
}
