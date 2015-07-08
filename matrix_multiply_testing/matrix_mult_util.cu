#include "matrix_mult_util.cuh"
#include "matrix_mult_constants.cuh"

using namespace std;

float* MatrixUtil::theMatrix1 = NULL;
float* MatrixUtil::theMatrix2 = NULL;
float* MatrixUtil::resultMat = NULL;

float* MatrixUtil::GetMatrix1(){ return theMatrix1; }
float* MatrixUtil::GetMatrix2(){ return theMatrix2; }
float* MatrixUtil::GetResultMat(){ return resultMat; }

void MatrixUtil::InitTheMatrices()
{
    theMatrix1 = new float[MAT_HEIGHT * MAT_WIDTH];
    theMatrix2 = new float[MAT_HEIGHT * MAT_WIDTH];
    resultMat = new float[MAT_HEIGHT * MAT_WIDTH];

    for (size_t i = 0; i < MAT_HEIGHT; ++i)
    {
        for (size_t j = 0; j < MAT_WIDTH; ++j)
        {
            theMatrix1[i * MAT_WIDTH + j] = i + j;//(float)(j % 7 + i);
            theMatrix2[i * MAT_WIDTH + j] = i;//(float)(j % 11 + i);
        }
    }

    //PrintMatrix(theMatrix1, MAT_WIDTH, MAT_HEIGHT);
    //PrintMatrix(theMatrix2, MAT_WIDTH, MAT_HEIGHT);
}

void MatrixUtil::FreeMatrices()
{
    delete [] theMatrix1;
    delete [] theMatrix2;
    delete [] resultMat;
}

void MatrixUtil::PrintMatrix(float* mat, size_t width, size_t height)
{
    if( (width * height) > MAX_PRINT_SIZE ) { cout << "Matrix too large to print\n"; return;}

    for (size_t row = 0; row < height; ++row)
    {
        for ( size_t col = 0; col < width; ++col)
        {
            cout << setw(5) << mat[row * width + col] << " ";
        }

        cout << '\n';
    }
}

void TimeFunction( void(*func)(), const char* func_name )
{
    LARGE_INTEGER start, end;

    // Windows API calls for timing
    QueryPerformanceCounter(&start);
    (*func)();
    QueryPerformanceCounter(&end);

    cout << setw(10) << func_name << " executed in " << setw(10) 
         << end.LowPart - start.LowPart << " perf counts\n";
}
