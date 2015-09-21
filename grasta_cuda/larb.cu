#include "larb.cuh"
#include <cstdlib>
#include <cstdio>

#include <math.h>
#include "grasta_cuda_util.cuh"
#include "mkl.h"
#include "mkl_blas.h"

cudaError_t cudaShrink(float* x, float* s, float gamma, int N){
    float *dev_x = 0; // device vector
    float *dev_s = 0; // device vector
    cudaError_t cudaStatus;

    // Allocate GPU buffers for vectors
    cudaStatus = cudaMalloc((void**)&dev_x, N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_s, N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_s, s, N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

#pragma warning(push)
#pragma warning(disable: 4244)
    // Launch a kernel on the GPU 
    shrinkKernel<<<ceil(N / (float)kBLOCKSIZE), kBLOCKSIZE>>>(dev_x, dev_s, gamma);
#pragma warning(pop)

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
    cudaStatus = cudaMemcpy(s, dev_s, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    Error:
    cudaFree(dev_x);
    cudaFree(dev_s);
    
    return cudaStatus;
}

void shrink(float* x,float* s,float gamma,int N){
	int ii;
	float t=0;
	for(ii=0;ii<N;ii++){
		t=x[ii];
		s[ii]=(t-gamma*((t > 0) - (t < 0)))*(fabsf(t)>gamma);
	}
}

void pinv_qr_m_big(float* B, float* pB,int m,int n){// assumes m>n. destroys B; make a copy.
	float one=1.0f;
	//int oneinc=1;
	float zero=0.0f;
	float* r;
	r=(float*)calloc(n*n,sizeof(float));
	float* tri;
	tri=(float*)calloc(n*n,sizeof(float));

	float *tau;
	tau=(float*)malloc(m*sizeof(float));
	int ii,jj;


	float  twork=0;
	int lwork=-1;
	int info;

	sgeqrf( &m, &n, B, &m, tau, &twork, &lwork, &info);	
	lwork=(int) twork;	
//	printf("\n lwork=%d\n", lwork );		
	float *work;
	work=(float*)malloc(lwork*sizeof(float));

	sgeqrf(&m, &n, B, &m, tau, work, &lwork, &info );


/*	print_matrix_colmajor(m,n,B);
	printf("\n\n");*/

	for (ii=0;ii<n;ii++){//column index
		for (jj=0;jj<(ii+1);jj++){
			r[ii*n+jj]=B[ii*m+jj];
		}
	}


/*
print_matrix_colmajor(n,n,r);
	printf("\n\n");*/

	sorgqr(&m, &n, &n, B, &m, tau, work, &lwork, &info );


	strtri("U","N",&n,r,&n,&info);

	for (ii=0;ii<n;ii++){//column index
		for (jj=0;jj<(ii+1);jj++){
			tri[jj*n+ii]=r[ii*n+jj];
		}
	}

	
	sgemm("N","N",&m,&n,&n,&one,B,&m,tri,&n,&zero,pB,&m);
	free(work);
	free(tri);
	free(r);
	free(tau);
}

void larb_orthogonal(float* B,int m,int n,float* x,float* c, float gamma,float maxiter, cublasHandle_t &handle){//assumes B is an orthogonal matrix. 
/*
c=B'*x;
u=B*c-x;
y=0;
for k=1:maxiter
    a=shrink(B*c-x+y,gamma);
    c=B'*(x+a-y);
    y=y-a+(B*c-x);
end

*/
	int ii=0,jj=0;
	float* y=(float*)calloc(m,sizeof(float));
	float* u=(float*)calloc(m,sizeof(float));
	float* a=(float*)calloc(m,sizeof(float));
	float* junk=(float*)calloc(m,sizeof(float));
	float one=1.0f;
	int oneinc=1;
	float zero=0.0f;

	sgemv("T",&m,&n,&one,B,&m,x,&oneinc,&zero,c,&oneinc);//calculate c
	sgemv("N",&m,&n,&one,B,&m,c,&oneinc,&zero,u,&oneinc);//calculate u=Bc;


	for(jj=0;jj<m;jj++){	
		u[jj]=u[jj]-x[jj];//u=Bc-x;
	}
	//main loop:
	for (ii=0;ii<maxiter;ii++){
		for(jj=0;jj<m;jj++){	
			u[jj]=u[jj]+y[jj];//u=Bc-x+y;
		}

		shrink(u,a,gamma,m);//a=shrink(Bc-x+y,gamma);
        //cudaShrink(u, a, gamma, m);

		for(jj=0;jj<m;jj++){	
			junk[jj]=x[jj]+a[jj]-y[jj];//junk=x+a-y
		}

		sgemv("T",&m,&n,&one,B,&m,junk,&oneinc,&zero,c,&oneinc);//calculate c=B'junk
		sgemv("N",&m,&n,&one,B,&m,c,&oneinc,&zero,u,&oneinc);//calculate u=Bc;
		for(jj=0;jj<m;jj++){	
			u[jj]=u[jj]-x[jj];//u=Bc-x;
		}
		for(jj=0;jj<m;jj++){	
			y[jj]=y[jj]-a[jj]+u[jj];//y <-- y-a+u
		}
	}
//	sgemv(chn,&dp,&d,&one,P+offp,&dp,X+offx,&oneinc,&zero,Z+offz,&oneinc);
}

void larb_orthogonal_alt(float* B,int m,int n,float* x,float* w, float* s, float* y,float rho,float maxiter){//assumes B is an orthogonal matrix. 
	int ii=0,jj=0;
	float* u=(float*)calloc(m,sizeof(float));
	float* junk=(float*)calloc(m,sizeof(float));
	float one=1.0f;
	int oneinc=1;
	float zero=0.0f;
	float irho=1/rho;

    //cudaMalloc(


	//main loop:
	for (ii=0;ii<maxiter;ii++){



		for(jj=0;jj<m;jj++){	
			junk[jj]=rho*(x[jj]-s[jj])-y[jj];
		}
		
		sgemv("T",&m,&n,&irho,B,&m,junk,&oneinc,&zero,w,&oneinc);//calculate w=(B'(rho(x-s-y)))/rho
		sgemv("N",&m,&n,&one,B,&m,w,&oneinc,&zero,u,&oneinc);//calculate junk=Bw;


		for(jj=0;jj<m;jj++){	
			junk[jj]=x[jj]-u[jj]-y[jj];
		}
	
		shrink(junk,s,1/(1+rho),m);
        //cudaShrink(junk, s, (1 / (1 + rho)), m);


		for(jj=0;jj<m;jj++){	
			y[jj]=y[jj]+rho*(s[jj]+u[jj]-x[jj]);//
		}
/*
		print_matrix_colmajor(n,1,w);
		printf("\n");*/

	}

free(u);
free(junk);

}

// called in grasta_step_subsample()
void larb_no_orthogonal_alt(float* sB,float* B,int m,int n,float* x,float* w, float* s, float* y,float rho,float maxiter){ 
	int ii=0,jj=0;
//	float* y=(float*)calloc(m,sizeof(float));
	float* u=(float*)calloc(m,sizeof(float));
//	float* a=(float*)calloc(m,sizeof(float));
	float* junk=(float*)calloc(m,sizeof(float));
	float one=1.0f;
	int oneinc=1;
	float zero=0.0f;
	float irho=1/rho;
	//main loop:
	for (ii=0;ii<maxiter;ii++){



		for(jj=0;jj<m;jj++){	
			junk[jj]=rho*(x[jj]-s[jj])-y[jj];
		}
		
		sgemv("T",&m,&n,&irho,sB,&m,junk,&oneinc,&zero,w,&oneinc);//calculate w=(B'(rho(x-s_-y)))/rho

		sgemv("N",&m,&n,&one,B,&m,w,&oneinc,&zero,u,&oneinc);//calculate junk=Bw;


		for(jj=0;jj<m;jj++){	
			junk[jj]=x[jj]-u[jj]-y[jj];
		}
	
		shrink(junk,s,1/(1+rho),m);
        //cudaShrink(junk, s, (1 / (1 + rho)), m);

		for(jj=0;jj<m;jj++){	
			y[jj]=y[jj]+rho*(s[jj]+u[jj]-x[jj]);//
		}
	}
free(u);
free(junk);
}
