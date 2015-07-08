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