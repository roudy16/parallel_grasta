#ifndef KERNELS_H
#define KERNELS_H
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h" 

const int kBLOCKSIZE = 512;   // number of threads in a block on GPU
const int kNUMBLOCKS = 256;

__global__ void shrinkKernel(const float *x, float *s, float gamma);

__global__ void larbOrthAltKernel(float *B,
                                  float *x,
                                  float *w,
                                  float *s,
                                  float *y,
                                  float rho,
                                  float maxiter);

template<int blockSize>
__global__ void setupSparseKernel_1(int use_number,
                                  int   *use_index,
                                  float *x,
                                  float *smallx,
                                  float *smallB,
                                  float *tB,
                                  float *outData);

__global__ void setupSparseKernel_2(int use_number,
                                    int m,
                                    int n,
                                    float scale,
                                    float *smallx,
                                    float *smallB,
                                    float *B,
                                    int   *use_index);

template<int blockSize>
__global__ void setupSparseKernel_1(int use_number,
                                  int   *use_index,
                                  float *x,
                                  float *smallx,
                                  float *smallB,
                                  float *tB,
                                  float *outData)
{
    __shared__ float share_data[blockSize];

    int tx = threadIdx.x;
    int i = blockIdx.x * blockSize + tx;
    //int gridSize = blockSize * 2 * gridDim.x;
    share_data[tx] = 0.0f;

    /*
    int index_end = use_number - blockSize;
    while( i < index_end ) {
        smallx[i] = x[use_index[i]];
        smallx[i + blockSize] = x[use_index[i + blockSize]];
        share_data[tx] = share_data[tx] + fabs(smallx[i]) + fabs(smallx[i + blockSize]);
        i += gridSize;
    }
    */

    smallx[i] = x[use_index[i]];
    share_data[tx] = share_data[tx] + fabsf(smallx[i]);

    __syncthreads();

    if(blockSize >= 512)
    {
        if(tx < 256) share_data[tx] += share_data[tx + 256];

        __syncthreads();
    }
    if(blockSize >= 256)
    {
        if(tx < 128) share_data[tx] += share_data[tx + 128];

        __syncthreads();
    }
    if(blockSize >= 128)
    {
        if(tx < 64) share_data[tx] += share_data[tx + 64];

        __syncthreads();
    }

    if(tx < 32)
    {
        if(blockSize >= 64) share_data[tx] += share_data[tx + 32];
        if(blockSize >= 32) share_data[tx] += share_data[tx + 16];
        if(blockSize >= 16) share_data[tx] += share_data[tx +  8];
        if(blockSize >=  8) share_data[tx] += share_data[tx +  4];
        if(blockSize >=  4) share_data[tx] += share_data[tx +  2];
        if(blockSize >=  2) share_data[tx] += share_data[tx +  1];
    }

    if(tx == 0) outData[blockIdx.x] = share_data[0];

}
#endif // KERNELS_H