#ifndef RANDOM_MASK_H
#define RANDOM_MASK_H

#include <random>

const float kRANDOMSAMPLEPERCENTAGE = 0.1f; // The percentage of total number if scalar values that
                                            // are selected in each random mask

const unsigned int kNUMRANDOMMASKS = 1024; // The number of random masks to produce when program starts

/*  This class will generate a number of random masks that are used to select the indices for
    the scalar values that are chosen for each grasta subsample. The intention of these masks
    is to reduce the runtime cost of generating random numbers during the running of grasta.
*/
class RandomMaskGenerator
{
public:
    static RandomMaskGenerator* Instance();
    int* GetRandomMask();
    const int GetMaskSize();

private:
    RandomMaskGenerator();
    ~RandomMaskGenerator();
    

    static RandomMaskGenerator* p_Instance;

    int * p_data;         // Pointer to base of collection of sample masks
    const int  m_maskSize;     // Size of each sample mask, ie the number of indices in each.

    std::mt19937 m_gen;
    std::uniform_int_distribution<int> m_distribution;
};


#endif // RANDOM_MASK_H
