#ifndef MAT_MULT_UTIL_H
#define MAT_MULT_UTIL_H

#include <iostream>
#include <iomanip>
#include <Windows.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "device_functions.h"
#include "matrix_mult_constants.cuh"

class MatrixUtil {
private:
    static float* theMatrix1;
    static float* theMatrix2;
    static float* resultMat;

public:
    MatrixUtil(){}

    // Assign arbitrary values to the matrices to be multiplied
    void InitTheMatrices();

    // Free memory allocated in InitTheMatrices
    void FreeMatrices();

    // Return pointer to Matrix1
    static float* GetMatrix1();

    // Return pointer to Matrix2
    static float* GetMatrix2();

    // Return pointer to resultMat
    static float* GetResultMat();

    // Print the matrix if it is a reasonable size
    static void PrintMatrix(float* mat, size_t width, size_t height);

};

// Time a function of form void func()
void TimeFunction( void(*func)(void), const char* func_name);


#endif // MAT_MULT_UTIL_H