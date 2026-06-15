#pragma once
// >>> c2hls auto-macro guards (do not edit between markers)
#ifndef N
#define N 90
#endif
// <<< c2hls auto-macro guards
#include <cmath>


extern "C" {
void kernel_gesummv(
		    double alpha,
		    double beta,
		    double A[ N + 0][N + 0],
		    double B[ N + 0][N + 0],
		    double tmp[ N + 0],
		    double x[ N + 0],
		    double y[ N + 0]);
}