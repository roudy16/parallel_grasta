#ifndef NAIVE_GPU_H
#define NAIVE_GPU_H

#include "matrix_mult_util.cuh"

#define NAIVE_BLOCK_X 16
#define NAIVE_BLOCK_Y 16
#define NAIVE_BLOCK_Z 1
#define NAIVE_THREADS_PER_BLOCK (NAIVE_BLOCK_X * NAIVE_BLOCK_Y * NAIVE_BLOCK_Z)

void DoNaiveGpuMatrixMult();

cudaError_t NaiveGpuMult(float* matA, const size_t A_width, const size_t A_height,
                         float* matB, const size_t B_width, const size_t B_height,
                         float* resMat);

__global__ void NaiveMatMultKern(float* dev_A_in, float* dev_B_in, float* dev_res_in, size_t width);

#endif // NAIVE_GPU_H