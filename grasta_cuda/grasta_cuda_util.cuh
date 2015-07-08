#include <math.h>
#include "grasta_cuda_kernels.cuh"

const int kSCREEN_WIDTH = 320;
const int kSCREEN_HEIGHT = 240;

cublasStatus_t cublasInit(cublasHandle_t &handle);

