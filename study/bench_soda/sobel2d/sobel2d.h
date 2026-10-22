#ifndef SOBEL2D_H
#define SOBEL2D_H

#include <stdint.h>

/* SODA benchmark 'sobel2d' (tests/src/sobel2d.soda). Arrays are plain C arrays, row-major; SODA's FIRST index is the
   fastest-varying one, so SODA name(dx,dy[,dz]) is name[y+dy][x+dx] ([z+dz][y+dy][x+dx] in 3D). */
#define WIDTH 1024
#define HEIGHT 1024
#define ITERATE 1

typedef int16_t img_t;
typedef uint16_t mag_t;

#ifdef __cplusplus
extern "C" {
#endif
void workload(const int16_t img[HEIGHT][WIDTH],
              uint16_t mag[HEIGHT][WIDTH]);
#ifdef __cplusplus
}
#endif

#endif
