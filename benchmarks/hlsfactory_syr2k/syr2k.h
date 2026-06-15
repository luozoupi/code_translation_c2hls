#pragma once
// >>> c2hls auto-macro guards (do not edit between markers)
#ifndef N
#define N 80
#endif
#ifndef M
#define M 60
#endif
// <<< c2hls auto-macro guards
#include <cmath>


extern "C" {
void kernel_syr2k( 
		  double alpha,
		  double beta,
		  double C[ N + 0][N + 0],
		  double A[ N + 0][M + 0],
		  double B[ N + 0][M + 0]);
}