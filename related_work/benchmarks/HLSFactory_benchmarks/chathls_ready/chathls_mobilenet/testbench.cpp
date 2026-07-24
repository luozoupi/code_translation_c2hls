#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>


typedef int8_t fm_t;
typedef int8_t wt_t;
typedef int32_t bias_t;
typedef int32_t acc_t;
typedef int32_t final_wide_t;
#ifndef INPUT_IMG_SIZE
#define IMG_DIM 128
#define IMG_CH 3
#define INPUT_IMG_SIZE (IMG_DIM * IMG_DIM * IMG_CH)
#define NUM_CLASSES 5
#endif

extern "C" {
void mobilenet(int image_in_stream[INPUT_IMG_SIZE], int prediction[NUM_CLASSES]);
}

static int v0_image_in_stream[INPUT_IMG_SIZE];
static int v1_prediction[NUM_CLASSES];

int main() {
  memset(v0_image_in_stream, 0, sizeof(v0_image_in_stream));
  memset(v1_prediction, 0, sizeof(v1_prediction));
  mobilenet(v0_image_in_stream, v1_prediction);
  printf("PASS smoke\n");
  return 0;
}
