#include "grasta_random_mask_gen.h"
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <fstream>
#include <cassert>


// HELPER FUNCTION DECLARATIONS
///////////////////////////////
bool MaskAlreadyContainsIndex(int *const maskBase, 
                              const int maskSize, 
                              const int value);
int intComp(const void* p1, const void* p2);

/* Default constructor
    m_maskSize is a multiple of REDUCTION_BLOCKSIZE_SIZE rounded up. This allows
    for performance improvements in some CUDA kernels.
*/
RandomMaskGenerator::RandomMaskGenerator()
    : p_data(nullptr),
    m_maskSize( (((int)((float)(kSCREEN_HEIGHT * kSCREEN_WIDTH) * kRANDOMSAMPLEPERCENTAGE)) + 512)
              - ((int)((float)(kSCREEN_HEIGHT * kSCREEN_WIDTH) * kRANDOMSAMPLEPERCENTAGE)) % 512),
      m_gen(std::clock()),
      m_distribution(0, kSCREEN_HEIGHT * kSCREEN_WIDTH)
{
    
    // Allocate memory for all the sample masks and initialize all bits to 1
    p_data = new int[m_maskSize * kNUMRANDOMMASKS];
    for(unsigned long i = 0; i < m_maskSize * kNUMRANDOMMASKS; ++i)
    {
        p_data[i] = INT_MAX;
    }
    //std::fill_n(p_data, m_maskSize * kNUMRANDOMMASKS, ULONG_MAX);

    std::cout << "Mask Size: " << m_maskSize << std::endl;

    for(unsigned int i = 0; i < kNUMRANDOMMASKS; ++i)
    {
        std::cout << "Making Mask " << i << std::endl;

        for(int j = 0; j < m_maskSize; ++j)
        {
            int newIndex = m_distribution(m_gen);
            while( MaskAlreadyContainsIndex(p_data + (i * m_maskSize), m_maskSize, newIndex))
            {
                newIndex = m_distribution(m_gen);
            }
            p_data[i * m_maskSize + j] = newIndex;
        }
    }

    for(unsigned int i = 0; i < kNUMRANDOMMASKS; ++i)
    {
        std::cout << "Sorting Mask " << i << std::endl;

        qsort(reinterpret_cast<void*>(p_data + (i * m_maskSize)), m_maskSize, sizeof(int), &intComp);
    }
}

RandomMaskGenerator::~RandomMaskGenerator()
{
    delete [] p_data;
    p_data = nullptr;
    p_Instance = nullptr;
}

int* RandomMaskGenerator::GetRandomMask()
{
    return p_data + (m_gen() % kNUMRANDOMMASKS) * m_maskSize;
}

const int RandomMaskGenerator::GetMaskSize()
{
    return m_maskSize;
}

RandomMaskGenerator* RandomMaskGenerator::p_Instance = nullptr;

RandomMaskGenerator* RandomMaskGenerator::Instance()
{
    if(p_Instance == nullptr)
    {
        p_Instance = new RandomMaskGenerator();
    }
    return p_Instance;
}

bool MaskAlreadyContainsIndex(int * const maskBase, 
                              const int maskSize, 
                              const int value)
{
    for(int i = 0; i < maskSize; ++i)
    {
        if(maskBase[i] == value)
        {
            return true;
        }
    }
    return false;
}

void RandomMaskGenerator::PrintDataToFile(const char* filename)
{
    assert(p_data != nullptr);

    std::ofstream filestream(filename, std::ios_base::out | std::ios_base::binary);

    filestream.write(reinterpret_cast<const char*>(&kSCREEN_WIDTH), sizeof(&kSCREEN_WIDTH));
    filestream.write(reinterpret_cast<const char*>(&kSCREEN_HEIGHT), sizeof(&kSCREEN_HEIGHT));
    filestream.write(reinterpret_cast<const char*>(&m_maskSize), sizeof(&m_maskSize));
    filestream.write(reinterpret_cast<const char*>(&kNUMRANDOMMASKS), sizeof(kNUMRANDOMMASKS));

    for(unsigned int i = 0; i < kNUMRANDOMMASKS; ++i)
    {
        filestream.write(reinterpret_cast<const char*>(&p_data[i * m_maskSize]),
                         m_maskSize * sizeof(int));
    }

    filestream.close();
}

int intComp(const void* p1, const void* p2)
{
    return *(int*)p1 - *(int*)p2;
}