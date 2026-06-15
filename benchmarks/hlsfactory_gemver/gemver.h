#pragma once
// >>> c2hls auto-macro guards (do not edit between markers)
#ifndef N
#define N 120
#endif
// <<< c2hls auto-macro guards
#include <cmath>


extern "C" {
void kernel_gemver(
		   double alpha,
		   double beta,
		   double A[ N + 0][N + 0],
		   double u1[ N + 0],
		   double v1[ N + 0],
		   double u2[ N + 0],
		   double v2[ N + 0],
		   double w[ N + 0],
		   double x[ N + 0],
		   double y[ N + 0],
		   double z[ N + 0]);
}