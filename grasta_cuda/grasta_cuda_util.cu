#include <cstdlib>
#include <cstdio>
#include "grasta_cuda_util.cuh"

cublasStatus_t cublasInit(cublasHandle_t &handle){
    return cublasCreate(&handle);
}