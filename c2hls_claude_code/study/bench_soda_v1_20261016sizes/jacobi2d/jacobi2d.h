#ifndef JACOBI2D_H
#define JACOBI2D_H

#include <stdint.h>

/* SODA benchmark 'jacobi2d' (tests/src/jacobi2d.soda). Arrays are plain C arrays, row-major; SODA's FIRST index is the
   fastest-varying one, so SODA name(dx,dy[,dz]) is name[y+dy][x+dx] ([z+dz][y+dy][x+dx] in 3D). */
#define WIDTH 1024
#define HEIGHT 1024
#define ITERATE 10

typedef float t1_t;
typedef float t0_t;

#ifdef __cplusplus
extern "C" {
#endif
void workload(const float t1[HEIGHT][WIDTH],
              float t0[HEIGHT][WIDTH]);
#ifdef __cplusplus
}
#endif

#endif
