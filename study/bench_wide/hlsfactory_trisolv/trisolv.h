#pragma once
// >>> c2hls auto-macro guards (do not edit between markers)
#ifndef N
#define N 120
#endif
// <<< c2hls auto-macro guards
#include <cmath>


extern "C" {
void kernel_trisolv(
		    double L[ N + 0][N + 0],
		    double x[ N + 0],
		    double b[ N + 0]);
}