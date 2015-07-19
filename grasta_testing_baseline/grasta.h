#ifndef GRASTA_H
#define GRASTA_H

#include <opencv/highgui.h>
#include <string.h>
#include <opencv/cv.h>
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
#include "larb.h"

#define kSCREEN_HEIGHT 480
#define kSCREEN_WIDTH 640

namespace Grasta{

void grasta_step (float* B, float* x, float* w, int m, int n, float dt,float rho, int maxiter);
void grasta_step_subsample(float* B, float* x, float* w, int m, int n, float dt,float rho, int maxiter, int* use_index, int use_number);

} // Grasta namespace
#endif // GRASTA_H