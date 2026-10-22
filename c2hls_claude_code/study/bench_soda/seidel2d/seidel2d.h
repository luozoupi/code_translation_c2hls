#ifndef SEIDEL2D_H
#define SEIDEL2D_H

#include <stdint.h>

/* SODA benchmark 'seidel2d' (tests/src/seidel2d.soda). Arrays are plain C arrays, row-major; SODA's FIRST index is the
   fastest-varying one, so SODA name(dx,dy[,dz]) is name[y+dy][x+dx] ([z+dz][y+dy][x+dx] in 3D). */
#define WIDTH 1024
#define HEIGHT 1024
#define ITERATE 9

typedef float input_t;
typedef float output_t;

#ifdef __cplusplus
extern "C" {
#endif
void workload(const float input[HEIGHT][WIDTH],
              float output[HEIGHT][WIDTH]);
#ifdef __cplusplus
}
#endif

#endif
