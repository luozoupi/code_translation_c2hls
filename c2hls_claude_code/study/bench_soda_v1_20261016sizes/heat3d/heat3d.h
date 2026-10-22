#ifndef HEAT3D_H
#define HEAT3D_H

#include <stdint.h>

/* SODA benchmark 'heat3d' (tests/src/heat3d.soda). Arrays are plain C arrays, row-major; SODA's FIRST index is the
   fastest-varying one, so SODA name(dx,dy[,dz]) is name[y+dy][x+dx] ([z+dz][y+dy][x+dx] in 3D). */
#define WIDTH 128
#define HEIGHT 128
#define DEPTH 128
#define ITERATE 5

typedef float in_t;
typedef float out_t;

#ifdef __cplusplus
extern "C" {
#endif
void workload(const float in[DEPTH][HEIGHT][WIDTH],
              float out[DEPTH][HEIGHT][WIDTH]);
#ifdef __cplusplus
}
#endif

#endif
