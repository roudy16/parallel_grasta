#pragma warning(push, 0)
#include <opencv/highgui.h>
#include <opencv/cv.h>
#pragma warning(pop)
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <time.h>
#include "math.h"
#include "mkl.h"
#include "mkl_blas.h"
#include <stdlib.h>
#include <stdio.h>
#include "grasta.cuh"
#include "../rand_index_set_generator/random_mask_reader.h"
#include "../rand_index_set_generator/grasta_random_mask_gen.h"

//g++ -I /usr/local/include/opencv/ -L /usr/local/lib/ -lhighgui -lcvaux -lcxcore -L/opt/intel/composerxe-2011.4.191/mkl/lib/intel64  -Wl,--start-group -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -Wl,--end-group -liomp5 -lpthread -lm grasta_cam_test_subsample.cpp -o gcts

#define PROFILE_MODE // comment this to remove profiling code

///////////////////////////////////////////////////////////
///main
////////////////////////////////////////////////////////////

int main( int argc, char* argv[] ) {
    cudaError_t stat;
    cublasHandle_t handle;

    // Choose which GPU to run on, change this on a multi-GPU system.
    stat = cudaSetDevice(0);
    if (stat != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return -1;
    }

    unsigned int device_flags = cudaDeviceScheduleBlockingSync;
    stat = cudaGetDeviceFlags(&device_flags);
    if (stat != cudaSuccess) {
        fprintf(stderr, "setting cuda flags failed!\n");
        return -1;
    }

    // Initialize cuBLAS
    if(cublasInit(handle) != CUBLAS_STATUS_SUCCESS){ return -1; }

    RandomMaskReader maskReader; // It is probably silly for this to be class
                                 // instead just a function.
    RandMaskInfo maskInfo = maskReader.ReadMasksFromFile(); // Just call this function

    DataPtrs data_ptrs;
    memset(&data_ptrs, 0, sizeof(DataPtrs));

    srand(time(0) & 0xFFFFFFFF);
    float one=1.0f;
    int oneinc=1;
    float zero=0.0f;

    float dt=.000001f;
    float rho=1;

    int m, n, ii; //jj;
    int hh = kSCREEN_HEIGHT;
    int ww = kSCREEN_WIDTH;
    m=hh*ww;
    n= N_VAL;

    // TODO Error Checking
    // Allocate page-locked memory in RAM
    cudaHostAlloc((void**)&data_ptrs.B, m*n*sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&data_ptrs.w, n*sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&data_ptrs.x, m*sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&data_ptrs.bb, m*sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&data_ptrs.ff, m*sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&data_ptrs.tau, m*sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&data_ptrs.use_index, maskInfo.maskSize*sizeof(int), cudaHostAllocDefault);

    if (cudaMalloc((void**)&data_ptrs.dev_B, m * n * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**)&data_ptrs.dev_w, n * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**)&data_ptrs.dev_x, m * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**)&data_ptrs.dev_bb, m * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**)&data_ptrs.dev_ff, m * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**)&data_ptrs.dev_tau, m * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**)&data_ptrs.dev_use_index, maskInfo.maskSize * sizeof(int)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(data_ptrs.dev_B); // TODO Complete error handling here
        return -1;
    }

    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = (cudaError_t)(cudaStatus + cudaHostAlloc((void**)&data_ptrs.smallx, maskInfo.maskSize * sizeof(float), cudaHostAllocDefault));
    cudaStatus = (cudaError_t)(cudaStatus + cudaHostAlloc((void**)&data_ptrs.smallB, maskInfo.maskSize * n * sizeof(float), cudaHostAllocDefault));
    cudaStatus = (cudaError_t)(cudaStatus + cudaHostAlloc((void**)&data_ptrs.tB, maskInfo.maskSize * n * sizeof(float), cudaHostAllocDefault));
    cudaStatus = (cudaError_t)(cudaStatus + cudaHostAlloc((void**)&data_ptrs.outFromGpu, (maskInfo.maskSize / kBLOCKSIZE) * sizeof(float), cudaHostAllocDefault));
    cudaStatus = (cudaError_t)(cudaStatus + cudaHostAlloc((void**)&data_ptrs.pismallB, maskInfo.maskSize * n * sizeof(float), cudaHostAllocDefault));
    if(cudaStatus != cudaSuccess){
        std::cout << "Error allocating host memory in Grasta substep\n" << std::flush;
        return -2;
    }
    cudaStatus = (cudaError_t)(cudaStatus + cudaMalloc((void**)&data_ptrs.dev_smallx, maskInfo.maskSize * sizeof(float)));
    cudaStatus = (cudaError_t)(cudaStatus + cudaMalloc((void**)&data_ptrs.dev_smallB, maskInfo.maskSize * n * sizeof(float)));
    cudaStatus = (cudaError_t)(cudaStatus + cudaMalloc((void**)&data_ptrs.dev_tB, maskInfo.maskSize * n * sizeof(float)));
    cudaStatus = (cudaError_t)(cudaStatus + cudaMalloc((void**)&data_ptrs.dev_outFromGpu, (maskInfo.maskSize / kBLOCKSIZE) * sizeof(float)));
    if(cudaStatus != cudaSuccess){
        std::cout << "Error allocating device memory in Grasta substep\n" << std::flush;
        return -3;
    }
    cudaStatus = (cudaError_t)(cudaStatus + cudaHostAlloc((void**)&data_ptrs.g1, maskInfo.maskSize*sizeof(float), cudaHostAllocDefault));
    cudaStatus = (cudaError_t)(cudaStatus + cudaHostAlloc((void**)&data_ptrs.uw, maskInfo.maskSize*sizeof(float), cudaHostAllocDefault));
    cudaStatus = (cudaError_t)(cudaStatus + cudaHostAlloc((void**)&data_ptrs.Uw, m*sizeof(float), cudaHostAllocDefault));
    cudaStatus = (cudaError_t)(cudaStatus + cudaHostAlloc((void**)&data_ptrs.g2t, n*sizeof(float), cudaHostAllocDefault));
    cudaStatus = (cudaError_t)(cudaStatus + cudaHostAlloc((void**)&data_ptrs.g2, m*sizeof(float), cudaHostAllocDefault));
    cudaStatus = (cudaError_t)(cudaStatus + cudaHostAlloc((void**)&data_ptrs.g, m*sizeof(float), cudaHostAllocDefault));
    cudaStatus = (cudaError_t)(cudaStatus + cudaHostAlloc((void**)&data_ptrs.s, maskInfo.maskSize*sizeof(float), cudaHostAllocDefault));
    cudaStatus = (cudaError_t)(cudaStatus + cudaHostAlloc((void**)&data_ptrs.y, maskInfo.maskSize*sizeof(float), cudaHostAllocDefault));
    if(cudaStatus != cudaSuccess){
        std::cout << "Error allocating host memory part 2\n" << std::flush;
        return -4;
    }

    // seed matrix B with random values
    for (ii=0;ii<m*n;ii++){
        data_ptrs.B[ii]=rand() * 1.0f;
    }

    float  twork=0; // while in a split?
    int lwork=-1;
    int info;

    sgeqrf( &m, &n, data_ptrs.B, &m, data_ptrs.tau, &twork, &lwork, &info);	
    lwork=(int) twork;	
    //	printf("\n lwork=%d\n", lwork );		
    float *work;
    work=(float*)malloc(lwork*sizeof(float));

    sgeqrf(&m, &n, data_ptrs.B, &m, data_ptrs.tau, work, &lwork, &info );
    sorgqr(&m, &n, &n, data_ptrs.B, &m, data_ptrs.tau, work, &lwork, &info );

    //cvNamedWindow( "selected location", 1 );
    cvNamedWindow( "capture", 1 );
    cvNamedWindow( "background?", 1 );
    cvNamedWindow( "foreground?", 1 );

    CvCapture* capture = cvCreateCameraCapture(1);
    if(!capture){
	    printf("failed to capture video from usb camera, trying built in camera\n");
	    capture = cvCreateCameraCapture(CV_CAP_ANY);
	    if(!capture){
		    printf("failed to capture video\n");
		    return(1);
	    }
    }   

    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, .5, .5, 0, 1, CV_AA);

    IplImage* frame;
    frame = cvQueryFrame(capture);

    IplImage* outbw = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
    IplImage* outg = cvCreateImage(cvGetSize(frame),IPL_DEPTH_32F,1);
    IplImage* outgs = cvCreateImage(cvSize(ww,hh),IPL_DEPTH_32F,1);
    IplImage* outgsb = cvCreateImage(cvSize(ww,hh),IPL_DEPTH_32F,1);
    IplImage* outgsf = cvCreateImage(cvSize(ww,hh),IPL_DEPTH_32F,1);

    //cvCopy(out,sframe,0);

    #ifdef PROFILE_MODE
    clock_t start_time(clock());
    #endif

    char dtstring[40];
    int c;

    double sample_percent=.1;
    //double  rm=double(RAND_MAX);

    //int tcount=0;
    int turbo=0;
    float ff_l1_norm=0;
    while( 1 ) {

        //if (tcount++>4) break;	
        frame = cvQueryFrame(capture);
        if( !frame ) break;
        cvCvtColor(frame,outbw,CV_BGR2GRAY);
        cvCvtScale(outbw,outg,.0039,0);//scale to 1/255
        cvResize(outg,outgs);

        data_ptrs.x=(float*)outgs->imageData;

        #ifdef PROFILE_MODE
        start_time = clock() - start_time;
        sprintf(dtstring,"FPS = %.4f", (CLOCKS_PER_SEC/(float)start_time));
        cvPutText(outgs,dtstring , cvPoint(10, 60), &font, cvScalar(0, 0, 0, 0));
        start_time = clock();
        #endif

        sprintf(dtstring,"dt = %.8f",dt);
        cvPutText(outgs,dtstring , cvPoint(10, 40), &font, cvScalar(0, 0, 0, 0));
        cvShowImage("capture", outgs);

        /*
        rm=sample_percent*((double)RAND_MAX);
        use_number=0;	
        for (ii=0;ii<m;ii++){
            if (rand()<rm){
                use_index[use_number]=ii;
                use_number++;
            }
        }
        */
        //fprintf(stderr,"use_number=%d\n",use_number);

        // TODO Error Handling
        // Transfer data from CPU RAM to VRAM
        data_ptrs.use_index = maskInfo.GetRandomMask();
        cudaMemcpy(data_ptrs.dev_use_index, data_ptrs.use_index, maskInfo.maskSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(data_ptrs.dev_B, data_ptrs.B, m * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(data_ptrs.dev_x, data_ptrs.x, m * sizeof(float), cudaMemcpyHostToDevice); // Image data

        if (turbo<5) {
            grasta_step (data_ptrs.B,data_ptrs.x,data_ptrs.w,m,n,dt,rho,20,data_ptrs.dev_B);
        }else{
            grasta_step_subsample (m,n,dt,rho,40, data_ptrs.use_index, maskInfo.maskSize, data_ptrs);
        }

        sgemv("N",&m,&n,&one,data_ptrs.B,&m,data_ptrs.w,&oneinc,&zero,data_ptrs.bb,&oneinc);

        // TODO examine what this loop is actually checking for -Steve
        // Update: looks like checking for changes in the L1 norm
        ff_l1_norm=0;	
        for (ii=0;ii<m;ii++){
            data_ptrs.ff[ii]=data_ptrs.x[ii]-data_ptrs.bb[ii];
            if (fabs(data_ptrs.ff[ii])>.05){
                ff_l1_norm ++;
                //ff_l1_norm += fabs(ff[ii]);
            }
        }
        //fprintf(stderr,"%f\n",ff_l1_norm);
        // If more than 60% of the L1 norms have changed
        if (ff_l1_norm>m*.6){
            turbo=0;
        }
        else{
            turbo++;
        }

    /*  for(jj=0;jj<m;jj++){	
            g[jj]=g1[jj]-g2[jj];
        }*/


        outgsb->imageData = (char*)data_ptrs.bb;
        outgsb->imageDataOrigin = outgsb->imageData;
        cvShowImage( "background?", outgsb);

        outgsf->imageData = (char*)data_ptrs.ff;
        outgsf->imageDataOrigin = outgsf->imageData;
        cvNormalize(outgsf, outgsf,1,0,CV_MINMAX);
        cvShowImage( "foreground?", outgsf);

        c = cvWaitKey(5);
        if( (char)c == 27 )
            break;
        switch( (char) c )
        {
            case 'm':
                sample_percent=sample_percent+.05;
                printf("sample percent up %.8f \n",sample_percent);
                break;
            case 'l':
                sample_percent=sample_percent-.05;
                printf("sample percent down %.8f \n",sample_percent);
                break;
            case 'u':
                dt=3*dt/2;
                printf("dt up %.8f \n",dt);
                break;
            case 'd':
                dt=2*dt/3;
                printf("dt down %.8f\n",dt );
                break;
            default:
                ;
        }
    } // End Frame Loop

    // TODO Deallocate Memory
}



