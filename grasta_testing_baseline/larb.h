#ifndef LARB_H
#define LARB_H

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include "mkl.h"
#include "mkl_blas.h"

namespace Larb{

void print_matrix_colmajor(int numrows,int numcols,float* M);
void pinv_qr_m_big(float* B, float* pB,int m,int n);// assumes m>n. destroys B; make a copy.
void shrink(float* x,float* s,float gamma,int N);
void larb_orthogonal(float* B,int m,int n,float* x,float* c, float gamma,float maxiter);//assumes B is an orthogonal matrix. 
void larb_no_orthogonal(float* pB, float* B,int m,int n,float* x,float* c, float gamma,float maxiter);//does not assume B is an orthogonal matrix; instead, the pseudoinverse pB is passed in.
void larb_orthogonal_alt(float* B,int m,int n,float* x,float* w, float* s, float* y,float rho,float maxiter);//assumes B is an orthogonal matrix. 
void larb_no_orthogonal_alt(float* sB,float* B,int m,int n,float* x,float* w, float* s, float* y,float rho,float maxiter); 

} // Larb namespace
#endif // LARB_H