#ifndef XCORR_H
#define XCORR_H

#include <stdint.h>

/* SODA benchmark 'xcorr' (tests/src/xcorr.soda). Arrays are plain C arrays, row-major; SODA's FIRST index is the
   fastest-varying one, so SODA name(dx,dy[,dz]) is name[y+dy][x+dx] ([z+dz][y+dy][x+dx] in 3D). */
#define WIDTH 1024
#define HEIGHT 1024
#define ITERATE 1

typedef int16_t input_t;
typedef int16_t tmp3_t;

#ifdef __cplusplus
extern "C" {
#endif
void workload(const int16_t input[HEIGHT][WIDTH],
              int16_t tmp3[HEIGHT][WIDTH]);
#ifdef __cplusplus
}
#endif

#endif
