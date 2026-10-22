#ifndef DENOISE3D_H
#define DENOISE3D_H

#include <stdint.h>

/* SODA benchmark 'denoise3d' (tests/src/denoise3d.soda). Arrays are plain C arrays, row-major; SODA's FIRST index is the
   fastest-varying one, so SODA name(dx,dy[,dz]) is name[y+dy][x+dx] ([z+dz][y+dy][x+dx] in 3D). */
#define WIDTH 256
#define HEIGHT 256
#define DEPTH 256
#define ITERATE 1

typedef float f_t;
typedef float u_t;
typedef float output_t;

#ifdef __cplusplus
extern "C" {
#endif
void workload(const float f[DEPTH][HEIGHT][WIDTH],
              const float u[DEPTH][HEIGHT][WIDTH],
              float output[DEPTH][HEIGHT][WIDTH]);
#ifdef __cplusplus
}
#endif

#endif
