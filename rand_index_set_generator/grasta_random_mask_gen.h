#ifndef RANDOM_MASK_H
#define RANDOM_MASK_H

#include <random>

#define RANDOMSAMPLEPERCENT 0.1f
#define SCREEN_WIDTH 640
#define SCREEN_HEIGHT 480
#define NUM_MASKS 1024

const float kRANDOMSAMPLEPERCENTAGE = RANDOMSAMPLEPERCENT; // The percentage of total number if scalar values that
                                            // are selected in each random mask

const unsigned int kSCREEN_HEIGHT  = SCREEN_HEIGHT;
const unsigned int kSCREEN_WIDTH   = SCREEN_WIDTH;
const unsigned int kNUMRANDOMMASKS = NUM_MASKS; // The number of random masks to produce when program starts
const int kSUBSAMPLE_SIZE = (((int)((float)(SCREEN_HEIGHT * SCREEN_WIDTH) * RANDOMSAMPLEPERCENT)) + 512)
    - ((int)((float)(SCREEN_HEIGHT * SCREEN_WIDTH) * RANDOMSAMPLEPERCENT)) % 512;

/*  This class will generate a number of random masks that are used to select the indices for
    the scalar values that are chosen for each grasta subsample. The intention of these masks
    is to reduce the runtime cost of generating random numbers during the running of grasta.
*/
class RandomMaskGenerator
{
public:
    static RandomMaskGenerator* Instance();
    int* GetRandomMask();
    int GetMaskSize();
    void PrintDataToFile(const char* filename);

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
