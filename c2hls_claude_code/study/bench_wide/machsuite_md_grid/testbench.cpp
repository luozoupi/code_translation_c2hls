/*
 * Strict Vitis HLS C-simulation testbench for machsuite_md_grid.
 * MachSuite harness (input.data -> run_benchmark -> check), with:
 *  - golden = check.data (parsed with the harness's output_to_data; it covers every output / in-out array);
 *    every element is compared (no early exit), prints at most 10 mismatches, then exactly
 *    "Passed!" (return 0) or "Failed with N errors!" (return 1) instead of the harness's check_data()/Success. text.
 *  - force is IN-OUT: plain.cpp reads force[..][p_idx] into its running sums before writing it (and never touches
 *    slots p_idx >= n_points), so it keeps its real initial data (zeros from input_to_data) and is NOT poisoned;
 *    n_points and position are inputs. Every force component of every slot is compared.
 *  - double tolerance 1e-6*fmax(1,|ref|) (SPEC), NaN = error.
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <assert.h>
#include "md.h"

/* MachSuite support is C; force C linkage from this C++ TB. */
extern "C" {
#include "support.h"
}

/* ---- checking helpers (golden = check.data) ---- */
static int errors = 0;
#define REPORT(...) do { if (errors < 10) printf(__VA_ARGS__); errors++; } while (0)
static int chk_d(const char *name, long i, double d, double r) {
    if (!isfinite(r) || !(fabs(d - r) <= 1e-6 * fmax(1.0, fabs(r)))) {   /* NaN/inf in DUT output is an error */
        REPORT("%s[%ld]: got %.9g expected %.9g\n", name, i, d, r);
        return 1;
    }
    return 0;
}

/* ---- end ---- */


int main(int argc, char **argv)
{
  const char *in_file = "input.data";
  const char *check_file = "check.data";
  assert(argc < 4 && "Usage: ./benchmark <input_file> <check_file>");
  if (argc > 1) in_file = argv[1];
  if (argc > 2) check_file = argv[2];

  char *data = (char *)malloc(INPUT_SIZE);
  char *gold = (char *)malloc(INPUT_SIZE);
  char *chk = (char *)malloc(INPUT_SIZE);
  assert(data && gold && chk && "Out of memory");
  int in_fd = open(in_file, O_RDONLY);
  assert(in_fd > 0 && "Couldn't open input data file");
  input_to_data(in_fd, data);
  close(in_fd);

  struct bench_args_t *d = (struct bench_args_t *)data;
  struct bench_args_t *g = (struct bench_args_t *)gold;
  struct bench_args_t *c = (struct bench_args_t *)chk;

  /* identical inputs / in-out arrays for reference and kernel */
  memcpy(gold, data, INPUT_SIZE);
  /* golden is check.data (covers every output / in-out array) */

  /* poison pure outputs of the kernel under test (after inputs are loaded) */
/* no pure outputs: force is an in-out accumulator */

  run_benchmark(data);

  int check_fd = open(check_file, O_RDONLY);
  assert(check_fd > 0 && "Couldn't open check data file");
  output_to_data(check_fd, chk);
  close(check_fd);

  for (int i = 0; i < blockSide; i++)
    for (int j = 0; j < blockSide; j++)
      for (int k = 0; k < blockSide; k++)
        for (int p = 0; p < densityFactor; p++) {
          long idx = (((i*blockSide + j)*blockSide + k)*densityFactor + p);
          chk_d("force.x", idx, d->force[i][j][k][p].x, c->force[i][j][k][p].x);
          chk_d("force.y", idx, d->force[i][j][k][p].y, c->force[i][j][k][p].y);
          chk_d("force.z", idx, d->force[i][j][k][p].z, c->force[i][j][k][p].z);
        }
  free(data); free(gold); free(chk);
  if (errors) {
    printf("Failed with %d errors!\n", errors);
    return 1;
  }
  printf("Passed!\n");
  return 0;
}
