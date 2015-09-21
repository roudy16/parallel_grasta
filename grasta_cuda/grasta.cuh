#ifndef GRASTA_MAIN_H
#define GRASTA_MAIN_H

#include "larb.cuh"
#include "cublas_v2.h"
#include "grasta_cuda_kernels.cuh"


#define N_VAL 9

struct DataPtrs {
    float *B,*x,*w,*bb,*ff, *tau; // CPU
    int *use_index;

    // Used in grasta_substep
	float* s;
	float* y;
	float* tB;
	float* smallx;
	float* smallB;
    float* pismallB;
	float* g1;
	float* uw;
	float* Uw;
	float* g2t;
	float* g2;
	float* g;
    float* outFromGpu;

    // Device pointers (GPU)
    float *dev_B, *dev_x, *dev_w, *dev_bb,
          *dev_ff, *dev_tau;
    int   *dev_use_index;
    float* dev_tB;
    float* dev_smallx;
    float* dev_smallB;
    float* dev_outFromGpu;
};


void grasta_step (float* B, float* x, float* w, int m, int n, float dt,float rho, int maxiter,
                  float* dev_B);

void grasta_step_subsample(int m, int n, float dt,float rho,
                           int maxiter, int *use_index, const int use_number, DataPtrs& data_ptrs);

#endif // GRASTA_MAIN_H
