#include "cpu_matrix_multiply.cuh"
#include "matrix_mult_util.cuh"

void CpuMatrixMult()
{
    float *Mat1, *Mat2;
    float *result;
    
    //result = new float[MAT_WIDTH * MAT_HEIGHT];

    Mat1 = MatrixUtil::GetMatrix1();
    Mat2 = MatrixUtil::GetMatrix2();
    result = MatrixUtil::GetResultMat();

    for(size_t row1 = 0; row1 < MAT_HEIGHT; ++row1)
    {
        for(size_t col1 = 0; col1 < MAT_WIDTH; ++col1)
        {
            float acc = 0.0;
            for(size_t idx = 0; idx < MAT_HEIGHT; ++idx)
            {
                acc += (Mat1[row1 * MAT_WIDTH + idx] * Mat2[idx * MAT_WIDTH + col1 ]);
            }
            result[row1 * MAT_WIDTH + col1] = acc;
        }
    }

    MatrixUtil::PrintMatrix(result, MAT_WIDTH, MAT_HEIGHT);
}