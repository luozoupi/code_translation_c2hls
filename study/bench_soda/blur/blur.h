#ifndef BLUR_H
#define BLUR_H

#include <stdint.h>

/* SODA benchmark 'blur' (tests/src/blur.soda). Arrays are plain C arrays, row-major; SODA's FIRST index is the
   fastest-varying one, so SODA name(dx,dy[,dz]) is name[y+dy][x+dx] ([z+dz][y+dy][x+dx] in 3D). */
#define WIDTH 1024
#define HEIGHT 1024
#define ITERATE 1

typedef uint16_t input_t;
typedef uint16_t blur_y_t;

#ifdef __cplusplus
extern "C" {
#endif
void workload(const uint16_t input[HEIGHT][WIDTH],
              uint16_t blur_y[HEIGHT][WIDTH]);
#ifdef __cplusplus
}
#endif

#endif
