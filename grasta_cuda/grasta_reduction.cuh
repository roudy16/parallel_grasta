#ifndef REDUCTION_H
#define REDUCTION_H
/*
#include "grasta_cuda_util.cuh"

// Keep this a multiple of 32
#define REDUCTION_BLOCK_SIZE 512
#define REDUCTION_NUM_BLOCKS 256

cudaError_t cudaSimpleReduction(float* data_to_sum, unsigned int num, float &accu);

// Simple reduction algorithm. The implementation is here due to some nuances
// with templates.
template <unsigned int blockSize>
__global__ void simple_reduce(float *g_idata, float *g_odata, unsigned int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0.0;

    // Note: Need to ensure that the number of elements to be summed is a multiple
    // of blockSize * 2
    while(i < n)
    {
        sdata[tid] += g_idata[i] + g_idata[i + blockSize];
        i += gridSize;
    }

    __syncthreads();

    if(blockSize >= 512)
    {
        if(tid < 256) sdata[tid] += sdata[tid + 256];

        __syncthreads();
    }
    if(blockSize >= 256)
    {
        if(tid < 128) sdata[tid] += sdata[tid + 128];

        __syncthreads();
    }
    if(blockSize >= 128)
    {
        if(tid < 64) sdata[tid] += sdata[tid + 64];

        __syncthreads();
    }

    if(tid < 32)
    {
        if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if(blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if(blockSize >= 16) sdata[tid] += sdata[tid +  8];
        if(blockSize >=  8) sdata[tid] += sdata[tid +  4];
        if(blockSize >=  4) sdata[tid] += sdata[tid +  2];
        if(blockSize >=  2) sdata[tid] += sdata[tid +  1];
    }

    if(tid == 0) g_odata[blockIdx.x] = sdata[0];
}
#endif //REDUCTION_H
*/