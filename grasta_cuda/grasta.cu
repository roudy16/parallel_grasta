#include "grasta.cuh"

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include "mkl.h"
#include "mkl_blas.h"

#define N_VAL 9

void grasta_step (float* B, float* x, float* w, int m, int n, float dt,float rho, int maxiter,
                  float* dev_B)
{
	float one=1.0f;
	int oneinc=1;
	float zero=0.0f;
	int ii,jj;
	//float* w=(float*)malloc(n*sizeof(float));
	float* s=(float*)calloc(m,sizeof(float));
	float* y=(float*)calloc(m,sizeof(float));
	float* g1=(float*)malloc(m*sizeof(float));
	float* uw=(float*)malloc(m*sizeof(float));
	float* g2t=(float*)malloc(n*sizeof(float));
	float* g2=(float*)malloc(m*sizeof(float));
	float* g=(float*)malloc(m*sizeof(float));
	float sigma;
	float normg;
	float normw;
	float cs;
	float ss;

	larb_orthogonal_alt(B,m,n,x,w,s,y,rho,maxiter * 1.0f);
	//printf("%f\n",w[5]);

	sgemv("N",&m,&n,&one,B,&m,w,&oneinc,&zero,uw,&oneinc);
	for(jj=0;jj<m;jj++){
		g1[jj]=y[jj]+rho*(uw[jj]+s[jj]-x[jj]);
	}

	sgemv("T",&m,&n,&one,B,&m,g1,&oneinc,&zero,g2t,&oneinc);
	sgemv("N",&m,&n,&one,B,&m,g2t,&oneinc,&zero,g2,&oneinc);


	for(jj=0;jj<m;jj++){
		g[jj]=g1[jj]-g2[jj];
	}

	normg=sdot(&m,g,&oneinc,g,&oneinc);
	normg=sqrt(normg);

	normw=sdot(&n,w,&oneinc,w,&oneinc);
	normw=sqrt(normw);

	sigma=normg*normw;

	cs=(cos(dt*sigma)-1)/(normw*normw);
	ss=sin(dt*sigma)/sigma;

	//reuse g1 as temp:
	for(jj=0;jj<m;jj++){
		g1[jj]=cs*uw[jj]-ss*g[jj];
	}

	for(ii=0;ii<m;ii++){//row
		for(jj=0;jj<n;jj++){//column
			B[jj*m+ii]=B[jj*m+ii]+g1[ii]*w[jj];
		}
	}
    ///////////////////////////////////////

	free (s);
	free(y);
	free(g1);
	free(uw);
	free(g2t);
	free(g2);
	free(g);
}

void grasta_step_subsample(int m, int n, float dt,float rho,
                           int maxiter, int *use_index, const int use_number, DataPtrs &data_ptrs)
{
	float one=1.0f;
	int oneinc=1;
	float zero=0.0f;
	int ii,jj;

    memset(data_ptrs.s, 0, use_number * sizeof(float));
    memset(data_ptrs.y, 0, use_number * sizeof(float));

	float sigma;
	float normg;
	float normw;
	float cs;
	float ss;
	float scale = 0.0f;
	float fuse_number = use_number * 1.0f;

    // TODO Error Handling and Memory deallocation

    if(use_number % ( kBLOCKSIZE) != 0){
        std::cout << "Invalid use_number dimension\n" << std::flush;
        return;
    }
    setupSparseKernel_1<kBLOCKSIZE><<<use_number / ( kBLOCKSIZE ), kBLOCKSIZE>>>(use_number,
                                                              data_ptrs.dev_use_index,
                                                              data_ptrs.dev_x,
                                                              data_ptrs.dev_smallx,
                                                              data_ptrs.dev_smallB,
                                                              data_ptrs.dev_tB,
                                                              data_ptrs.dev_outFromGpu); 

    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if(cudaStatus != cudaSuccess){
        std::cout << "Error launching setupSparseKernel_1: " << cudaGetErrorString(cudaStatus) << "\n" << std::flush;
        return;
    }

    cudaMemcpy(data_ptrs.outFromGpu, data_ptrs.dev_outFromGpu, (use_number / kBLOCKSIZE) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaStatus = cudaDeviceSynchronize();
    if(cudaStatus != cudaSuccess){
        std::cout << "Memcopying from setupSparseKernel_1: " << cudaGetErrorString(cudaStatus) << "\n" << std::flush;
        return;
    }

    for(int i = 0; i < use_number / ( kBLOCKSIZE ); ++i){
        scale += data_ptrs.outFromGpu[i];
        std::cout << data_ptrs.outFromGpu[i] << '\n';
    }
    std::cout << scale << '\n';
    scale = scale * 8.0f / fuse_number; // This works but where is the multiple of 8 lost?
    std::cout << scale << std::endl;

    setupSparseKernel_2<<<use_number / kBLOCKSIZE, kBLOCKSIZE >>>(use_number,
                                                                  m,
                                                                  n,
                                                                  scale,
                                                                  data_ptrs.dev_smallx,
                                                                  data_ptrs.dev_smallB,
                                                                  data_ptrs.dev_B,
                                                                  data_ptrs.dev_use_index);

    cudaStatus = cudaDeviceSynchronize();
    if(cudaStatus != cudaSuccess){
        std::cout << "Error launching setupSparseKernel_2\n" << std::flush;
        return;
    }

    cudaMemcpy(data_ptrs.dev_tB, data_ptrs.dev_smallB, use_number * n * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(data_ptrs.smallx, data_ptrs.dev_smallx, use_number * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(data_ptrs.smallB, data_ptrs.dev_smallB, use_number * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(data_ptrs.tB, data_ptrs.dev_tB, use_number * n * sizeof(float), cudaMemcpyDeviceToHost);

    // ### Convert to GPU kernel
    /*
	for (ii = 0; ii < use_number; ++ii){
		data_ptrs.smallx[ii] = data_ptrs.x[use_index[ii]];
		scale = scale + fabs(data_ptrs.smallx[ii]);
        std::cout << fabs(data_ptrs.smallx[ii]) << '\n';
	}
    std::cout << scale << '\n';
	scale=scale/fuse_number;
    std::cout << scale << std::endl;
	for (ii = 0; ii < use_number; ++ii){
		data_ptrs.smallx[ii] = data_ptrs.smallx[ii]/scale;
	}
	for (jj = 0; jj < n; ++jj){
		for (ii = 0; ii < use_number; ++ii){
			data_ptrs.smallB[ jj * use_number + ii] = data_ptrs.B[ jj * m + use_index[ii] ];
		}
	}
	for (ii=0; ii < use_number * n; ++ii){
		data_ptrs.tB[ii] = data_ptrs.smallB[ii];
	}
    */
    // ### End Kernel

	pinv_qr_m_big(data_ptrs.tB, data_ptrs.pismallB, use_number, n);

larb_no_orthogonal_alt(data_ptrs.pismallB, data_ptrs.smallB, use_number, n, data_ptrs.smallx, data_ptrs.w, data_ptrs.s, data_ptrs.y, rho, maxiter * 1.0f);

/*
    ticks t2 = getticks();
	fprintf(stderr,"t2=%g\n",elapsed(t1,t2));

    [s_t, w, ldual, ~] = sparse_residual_pursuit(U_Omega, y_Omega, OPTS)

	fprintf(stderr,"B[5]=%f\n",B[5]);
	fprintf(stderr,"w[5]=%f\n",w[5]);
*/

sgemv("N",&use_number,&n,&one,data_ptrs.smallB,&use_number,data_ptrs.w,&oneinc,&zero,data_ptrs.uw,&oneinc);//uw=B_idx w

	for(jj=0;jj<use_number;jj++){
		data_ptrs.g1[jj]=data_ptrs.y[jj]+rho*(data_ptrs.uw[jj]+data_ptrs.s[jj]-data_ptrs.smallx[jj]);//-s?   check me!  todo!!!
	}
//gamma_1 = ldual + OPTS.RHO*(U_Omega*w + s_t - y_Omega);



sgemv("T",&use_number,&n,&one,data_ptrs.smallB,&use_number,data_ptrs.g1,&oneinc,&zero,data_ptrs.g2t,&oneinc);//n x use_number g2t=smallB'*g1

//UtDual_omega = U_Omega' * gamma_1;
	sgemv("N",&m,&n,&one,data_ptrs.B,&m,data_ptrs.g2t,&oneinc,&zero,data_ptrs.g2,&oneinc);//m x n g2=B*g2t
								//gamma_2 = U0 * UtDual_omega;
//	ticks t3 = getticks();
//	fprintf(stderr,"t3=%g\n",elapsed(t2,t3));

	for(jj = 0; jj < m; ++jj){
		data_ptrs.g[jj] = -data_ptrs.g2[jj];
	}
	for(jj = 0; jj < use_number; ++jj){
		data_ptrs.g[use_index[jj]] += data_ptrs.g1[jj];
	}

/*
	gamma = zeros(DIM_M,1);
	gamma(idx) = gamma_1;
	gamma = gamma - gamma_2;
*/

	normg = sdot(&m,data_ptrs.g,&oneinc,data_ptrs.g,&oneinc);
	normg = sqrt(normg);
	normw = sdot(&n,data_ptrs.w,&oneinc,data_ptrs.w,&oneinc);
	normw = sqrt(normw);
	sigma = normg*normw;

/*
	gamma_norm = norm(gamma);
	w_norm     = norm(w);
	sG = gamma_norm * w_norm;
*/

/*
	cs=(cos(dt*sigma)-1)/(normw*normw);
	ss=sin(dt*sigma)/sigma;

	if ((normw>.01)&&(normw>.01)){
		//reuse g2 as temp:
		for(jj=0;jj<use_number;jj++){
			g2[use_index[jj]]=cs*Uw[jj];
		}
		for(jj=0;jj<m;jj++){
			g2[jj]-=ss*g[jj];//
		}

		for(ii=0;ii<m;ii++){//row
			for(jj=0;jj<n;jj++){//column
				B[jj*m+ii]=B[jj*m+ii]+g2[ii]*w[jj];
			}
		}
	}
*/

//	ticks t4 = getticks();
//	fprintf(stderr,"t4=%g\n",elapsed(t3,t4));

cs=0;
ss=0;

float* alpha;
alpha = (float*)calloc(m,sizeof(float));

if (normw>0){
	cs = (cos(dt*sigma)-1);
	for (ii = 0; ii < n; ++ii){
		alpha[ii] = data_ptrs.w[ii]/normw;
		data_ptrs.w[ii] = scale*data_ptrs.w[ii];
	}
}

float* beta;
beta = (float*)calloc(m,sizeof(float));
if (normg > 0){
	for (ii = 0; ii < m; ++ii){
		beta[ii] = data_ptrs.g[ii]/normg;
	}
	ss = sin(dt*sigma);
}

sgemv("N",&m,&n,&one,data_ptrs.B,&m,alpha,&oneinc,&zero,data_ptrs.Uw,&oneinc);//Uw=Bw

for(ii = 0; ii < m; ++ii){//row
	for(jj = 0; jj < n; ++jj){//column
        data_ptrs.B[jj*m+ii] = data_ptrs.B[jj*m+ii]+cs*data_ptrs.Uw[ii]*alpha[jj]-ss*beta[ii]*alpha[jj];
	}
}

// Take the gradient step along Grassmannian geodesic.
/*
alpha = w/w_norm;
beta  = gamma/gamma_norm;
step = (cos(t)-1)*U0*(alpha*alpha')  - sin(t)*beta*alpha';
*/
	free(alpha);
	free(beta);
	//free (s);
	//free(y);
	//free(g1);
	//free(uw);
	//free(Uw);
	//free(g2t);
	//free(g2);
	//free(g);

	//cudaFreeHost(tB);
	//cudaFreeHost(smallx);
	//cudaFreeHost(smallB);
    //cudaFreeHost(outFromGpu);
    //cudaFree(dev_tB);
    //cudaFree(dev_smallx);
    //cudaFree(dev_smallB);
    //cudaFree(dev_use_index); // Array of selected indices (mask)
    //cudaFree(dev_outFromGpu);
}
