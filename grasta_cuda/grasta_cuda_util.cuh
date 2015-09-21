#ifndef UTILS_H
#define UTILS_H
#include <math.h>
#include "grasta_cuda_kernels.cuh"

/* Defined in grasta_random_masl_gen.h
const int kSCREEN_WIDTH = 320 * 2;
const int kSCREEN_HEIGHT = 240 * 2;
*/

cublasStatus_t cublasInit(cublasHandle_t &handle);


#endif // UTILS_H
