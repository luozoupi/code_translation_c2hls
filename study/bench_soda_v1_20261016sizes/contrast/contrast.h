#ifndef CONTRAST_H
#define CONTRAST_H

#include <stdint.h>

/* SODA benchmark 'contrast' (tests/src/contrast.soda). Arrays are plain C arrays, row-major; SODA's FIRST index is the
   fastest-varying one, so SODA name(dx,dy[,dz]) is name[y+dy][x+dx] ([z+dz][y+dy][x+dx] in 3D). */
#define WIDTH 512
#define HEIGHT 512
#define ITERATE 1

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
