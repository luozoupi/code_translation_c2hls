#ifndef SOBEL_H_
#define SOBEL_H_
#include "ap_int.h"
#include "params.h"

void sobel_x(ap_uint<8> input_image[H][W], ap_uint<1> output_image[H][W]);

#endif
