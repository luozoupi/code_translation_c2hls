#pragma once
// >>> c2hls auto-macro guards (do not edit between markers)
#ifndef N
#define N 120
#endif
#ifndef TSTEPS
#define TSTEPS 40
#endif
// <<< c2hls auto-macro guards
#include <cmath>


extern "C" {
void kernel_jacobi_1d(
			    
			    double A[ N + 0],
			    double B[ N + 0]);
}