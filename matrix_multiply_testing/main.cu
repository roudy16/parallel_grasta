#include "matrix_mult_util.cuh"
#include "cpu_matrix_multiply.cuh"
#include "naive_gpu_multiply.cuh"
#include "optimized_gpu_multiply.cuh"

using namespace std;

void Test()
{
    for(int i = 0; i < 10; ++i){ cout << i; }
}

int main()
{
    MatrixUtil matUtil;
    matUtil.InitTheMatrices();

    TimeFunction(&CpuMatrixMult, "CPU Mult");
    TimeFunction(&CpuMatrixMult, "CPU Mult");
    TimeFunction(&DoNaiveGpuMatrixMult, "Naive GPU");
    TimeFunction(&DoOptimizedGpuMatrixMult, "Opt GPU");
    TimeFunction(&DoNaiveGpuMatrixMult, "Naive GPU");
    TimeFunction(&DoOptimizedGpuMatrixMult, "Opt GPU");
    TimeFunction(&DoNaiveGpuMatrixMult, "Naive GPU");
    TimeFunction(&DoOptimizedGpuMatrixMult, "Opt GPU");
    TimeFunction(&DoNaiveGpuMatrixMult, "Naive GPU");
    TimeFunction(&DoOptimizedGpuMatrixMult, "Opt GPU");

    matUtil.FreeMatrices();

    return 0;
}