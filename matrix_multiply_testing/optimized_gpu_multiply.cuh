#ifndef OPTIMIZED_GPU_H
#define OPTIMIZED_GPU_H

#include "matrix_mult_util.cuh"

#define OPT_BLOCK_X 16
#define OPT_BLOCK_Y 16
#define OPT_BLOCK_Z 1
#define OPT_THREADS_PER_BLOCK (OPT_BLOCK_X * OPT_BLOCK_Y * OPT_BLOCK_Z)

void DoOptimizedGpuMatrixMult();

cudaError_t OptimizedGpuMult(float* matA, const size_t A_width, const size_t A_height,
                             float* matB, const size_t B_width, const size_t B_height,
                             float* resMat);

__global__ void OptimizedMatMultKern(float* dev_A_in, float* dev_B_in, float* dev_res_in, size_t width);

#endif // OPTIMIZED_GPU_H