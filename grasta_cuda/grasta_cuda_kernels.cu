#include "grasta_cuda_kernels.cuh"

__global__ void shrinkKernel(const float *x, float *s, float gamma){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    const float t = x[i];
    s[i] = (t - gamma * ((t > 0) - (t < 0))) * (fabsf(t) > gamma);
}

__global__ void larbOrthAltKernel(float *B,
                                  float *x,
                                  float *w,
                                  float *s,
                                  float *y,
                                  float rho,
                                  float maxiter)
{
    for(int i = 0; i < maxiter; ++i){
        
    }
}


__global__ void setupSparseKernel_2(int use_number,
                                    int m,
                                    int n,
                                    float scale,
                                    float *smallx,
                                    float *smallB,
                                    float *B,
                                    int   *use_index)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    smallx[i] = smallx[i] / scale;

    __syncthreads();

    for(int j = 0; j < n; ++j){
        smallB[ j * use_number + i ] = B[ j * m + use_index[i]];
    }
}