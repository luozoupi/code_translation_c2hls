#ifndef JACOBI3D_H
#define JACOBI3D_H

#include <stdint.h>

/* SODA benchmark 'jacobi3d' (tests/src/jacobi3d.soda). Arrays are plain C arrays, row-major; SODA's FIRST index is the
   fastest-varying one, so SODA name(dx,dy[,dz]) is name[y+dy][x+dx] ([z+dz][y+dy][x+dx] in 3D). */
#define WIDTH 128
#define HEIGHT 128
#define DEPTH 128
#define ITERATE 6

typedef float t1_t;
typedef float t0_t;

#ifdef __cplusplus
extern "C" {
#endif
void workload(const float t1[DEPTH][HEIGHT][WIDTH],
              float t0[DEPTH][HEIGHT][WIDTH]);
#ifdef __cplusplus
}
#endif

#endif
