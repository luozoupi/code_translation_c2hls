#ifndef DENOISE2D_H
#define DENOISE2D_H

#include <stdint.h>

/* SODA benchmark 'denoise2d' (tests/src/denoise2d.soda). Arrays are plain C arrays, row-major; SODA's FIRST index is the
   fastest-varying one, so SODA name(dx,dy[,dz]) is name[y+dy][x+dx] ([z+dz][y+dy][x+dx] in 3D). */
#define WIDTH 1024
#define HEIGHT 1024
#define ITERATE 1

typedef float f_t;
typedef float u_t;
typedef float output_t;

#ifdef __cplusplus
extern "C" {
#endif
void workload(const float f[HEIGHT][WIDTH],
              const float u[HEIGHT][WIDTH],
              float output[HEIGHT][WIDTH]);
#ifdef __cplusplus
}
#endif

#endif
