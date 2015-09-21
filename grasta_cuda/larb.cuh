#ifndef LARB_H
#define LARB_H

#include "grasta_cuda_util.cuh"

cudaError_t cudaShrink(float* x, float* s, float gamma, int N);

void shrink(float* x,float* s,float gamma,int N);

void pinv_qr_m_big(float* B, float* pB,int m,int n);// assumes m>n. destroys B; make a copy.

void larb_orthogonal(float* B,int m,int n,float* x,float* c, float gamma,float maxiter, cublasHandle_t &handle);

void larb_orthogonal_alt(float* B,int m,int n,float* x,float* w, float* s, float* y,float rho,float maxiter);

void larb_no_orthogonal_alt(float* sB,float* B,int m,int n,float* x,float* w, float* s, float* y,float rho,float maxiter);

#endif // LARB_H