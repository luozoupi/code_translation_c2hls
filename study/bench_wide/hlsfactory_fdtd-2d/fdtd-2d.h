#pragma once
// >>> c2hls auto-macro guards (do not edit between markers)
#ifndef TMAX
#define TMAX 40
#endif
#ifndef NX
#define NX 60
#endif
#ifndef NY
#define NY 80
#endif
// <<< c2hls auto-macro guards
#include <cmath>


extern "C" {
void kernel_fdtd_2d(
		    
		    
		    double ex[ NX + 0][NY + 0],
		    double ey[ NX + 0][NY + 0],
		    double hz[ NX + 0][NY + 0],
		    double _fict_[ TMAX + 0]);
}