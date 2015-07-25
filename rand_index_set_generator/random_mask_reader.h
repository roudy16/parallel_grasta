#ifndef MASK_READER_H
#define MASK_READER_H

#include <fstream>

struct RandMaskInfo
{
    unsigned int width;
    unsigned int height;
    int          maskSize;
    unsigned int numMasks;
    int*         data;

private:
    RandMaskInfo& operator=(const RandMaskInfo &rhs);
};



class RandomMaskReader
{
public:
    RandomMaskReader();
    RandMaskInfo ReadMasksFromFile();

private:
    const char* filename;
};

#endif // MASK_READER_H