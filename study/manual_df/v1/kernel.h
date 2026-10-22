#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef float data_t;
//#define I 256 
//#define J 264 
//#define K 256

//#define I 128 
//#define J 128 
//#define K 128

//#define I 64
//#define J 64
//#define K 64

#ifdef __cplusplus
#define I 1024
#define J 1024
#define K 1024
extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]);
#endif
