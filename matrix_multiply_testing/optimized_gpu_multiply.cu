#include <cmath>

#include "optimized_gpu_multiply.cuh"

void DoOptimizedGpuMatrixMult()
{
    const size_t kMatWidth = MAT_WIDTH;
    const size_t kMatHeight = MAT_HEIGHT;

    cudaError_t cuErr = OptimizedGpuMult( MatrixUtil::GetMatrix1(), kMatWidth, kMatHeight,
                                          MatrixUtil::GetMatrix2(), kMatWidth, kMatHeight,
                                          MatrixUtil::GetResultMat() );

    if( cuErr != cudaSuccess )
    {
        fprintf( stderr, "OptimizedGpuMult Failed!\n" );
        return;
    }
}

cudaError_t OptimizedGpuMult(float* matA, const size_t A_width, const size_t A_height,
                             float* matB, const size_t B_width, const size_t B_height,
                             float* resMat)
{
    cudaError_t cuErr;

    // Device pointers to matrix data
    float* dev_A;
    float* dev_B;
    float* dev_res;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cuErr = cudaSetDevice(0);
    if (cuErr != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cuErr;
    }

    /////////////////////////////////////////////
    // Allocate memory for matrices on the device

    cuErr = cudaMalloc((void**)&dev_A, A_width * A_height * sizeof(float));
    if (cuErr != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cuErr = cudaMalloc((void**)&dev_B, B_width * B_height * sizeof(float));
    if (cuErr != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cuErr = cudaMalloc((void**)&dev_res, A_width * B_height * sizeof(float));
    if (cuErr != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    ////////////////////////////////////////////
    // Copy matrices A and B to device
    cuErr = cudaMemcpy(dev_A, matA, A_width * A_height * sizeof(float), cudaMemcpyHostToDevice);
    if (cuErr != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cuErr = cudaMemcpy(dev_B, matB, B_width * B_height * sizeof(float), cudaMemcpyHostToDevice);
    if (cuErr != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    dim3 BlockDimensions( OPT_BLOCK_X, OPT_BLOCK_Y, OPT_BLOCK_Z );
    dim3 NumBlocks( ceil(MAT_WIDTH / (float)OPT_BLOCK_X), ceil(MAT_HEIGHT / (float)OPT_BLOCK_Y), 1);

    /////////////////////////////////////////
    // Launch the Naive kernel
    OptimizedMatMultKern<<< NumBlocks, BlockDimensions>>>(dev_A,
                                                      dev_B,
                                                      dev_res,
                                                      MAT_WIDTH);

    // Check for any errors launching the kernel
    cuErr = cudaGetLastError();
    if (cuErr != cudaSuccess) {
        fprintf(stderr, "Optimized kernel launch failed: %s\n", cudaGetErrorString(cuErr));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cuErr = cudaDeviceSynchronize();
    if (cuErr != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Optimized Kernel!\n", cuErr);
        goto Error;
    }

    // Copy result from GPU buffer to host memory.
    cuErr = cudaMemcpy(resMat, dev_res, A_width * B_height * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuErr != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cuErr = cudaDeviceSynchronize();

    MatrixUtil::PrintMatrix(resMat, MAT_WIDTH, MAT_HEIGHT);

// Labels are stupid
Error:
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_res);

    return cuErr;
}

// The Kernel code that runs on the device
__global__ void OptimizedMatMultKern(float* dev_A_in, float* dev_B_in, float* dev_res_in, size_t width)
{
    // Shared memory for tiles. Offset of 1 added to reduce bank conflicts
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH * 2];
    __shared__ float Nds[TILE_WIDTH * 2][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    for (int m = 0; m < width/(TILE_WIDTH * 2); ++m)
    {
        // Collaborative loading into shared memory
        // Nds is stored in column major format to reduce bank conflicts
        Mds[ty][tx]              = dev_A_in[row * width + m * 2 * TILE_WIDTH  + tx];
        Mds[ty][TILE_WIDTH + tx] = dev_A_in[row * width + m * 2 * TILE_WIDTH  + tx + TILE_WIDTH];
        Nds[ty][tx]              = dev_B_in[ (m * 2 * TILE_WIDTH + ty) * width + col ];
        Nds[ty + TILE_WIDTH][tx] = dev_B_in[ (m * 2 * TILE_WIDTH + TILE_WIDTH + ty) * width  + col ];
        __syncthreads();

        for (int k = 0; k < 2 * TILE_WIDTH; ++k)
        {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    dev_res_in[row * width + col] = Pvalue;
}