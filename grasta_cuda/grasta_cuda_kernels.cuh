#ifndef KERNELS_H
#define KERNELS_H
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h" 

const int kBLOCKSIZE = 512;   // number of threads in a block on GPU

__global__ void shrinkKernel(const float *x, float *s, float gamma);

__global__ void larbOrthAltKernel(float *B,
                                  float *x,
                                  float *w,
                                  float *s,
                                  float *y,
                                  float rho,
                                  float maxiter);

#endif // KERNELS_H