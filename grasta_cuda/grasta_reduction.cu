#include <cassert>
#include <cstdio>
#include <cstdlib>
#include "grasta_cuda_util.cuh"
#include "grasta_reduction.cuh"

using namespace std;

cudaError_t cudaSimpleReduction(float* data_to_sum, unsigned int num, float &accu){
    float *dev_data_to_sum = 0; // array of elements to sum that reside on the device
    float *dev_temp_sums = 0;   // holds accumulations of elements between kernel calls
    float *temp_sums = 0;       // storage for temps sums from device
    cudaError_t cudaStatus;

    temp_sums = (float*) malloc(REDUCTION_BLOCK_SIZE * sizeof(float));

    // This invariant makes the reduction algorithm easier to implement
    assert(num % (REDUCTION_BLOCK_SIZE * 2) == 0);

    // Allocate GPU buffers for element
    cudaStatus = cudaMalloc((void**)&dev_data_to_sum, num * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_temp_sums, REDUCTION_BLOCK_SIZE * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_data_to_sum, data_to_sum, num * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU
    const unsigned int kSMEM_BYTES = REDUCTION_BLOCK_SIZE * sizeof(float);
    simple_reduce<REDUCTION_BLOCK_SIZE><<< REDUCTION_NUM_BLOCKS, REDUCTION_BLOCK_SIZE, kSMEM_BYTES >>>
        (dev_data_to_sum, dev_temp_sums, num / (REDUCTION_BLOCK_SIZE * 2)); // kernel args

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(temp_sums, dev_temp_sums, REDUCTION_BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_temp_sums);
    cudaFree(dev_data_to_sum);
    free(temp_sums);
    
    return cudaStatus;
}
