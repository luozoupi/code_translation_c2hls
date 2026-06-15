#pragma once
// >>> c2hls auto-macro guards (do not edit between markers)
#ifndef M
#define M 60
#endif
#ifndef N
#define N 80
#endif
// <<< c2hls auto-macro guards
#include <cmath>


extern "C" {
void kernel_trmm( 
		 double alpha,
		 double A[ M + 0][M + 0],
		 double B[ M + 0][N + 0]);
}