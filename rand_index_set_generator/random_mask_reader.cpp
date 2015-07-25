#include "random_mask_reader.h"

RandMaskInfo& RandMaskInfo::operator=(const RandMaskInfo &rhs)
{
    if(this == &rhs)
    {
        return *this;
    }

    width = rhs.width;
    height = rhs.height;
    maskSize = rhs.maskSize;
    numMasks = rhs.numMasks;
    data = rhs.data;

    return *this;
}

RandomMaskReader::RandomMaskReader() : filename("randomMasks.data") {}

RandMaskInfo RandomMaskReader::ReadMasksFromFile()
{
    RandMaskInfo info;
    std::ifstream ifs(filename);

    ifs.read(reinterpret_cast<char*>(&info.width), sizeof(info.width));
    ifs.read(reinterpret_cast<char*>(&info.height), sizeof(info.height));
    ifs.read(reinterpret_cast<char*>(&info.maskSize), sizeof(info.maskSize));
    ifs.read(reinterpret_cast<char*>(&info.numMasks), sizeof(info.numMasks));

    info.data = new int[info.maskSize * info.numMasks];

    for(unsigned long long i = 0; i < info.numMasks; ++i)
    {
        ifs.read( reinterpret_cast<char*>(&info.data[i * info.maskSize]),
                  sizeof(int) * info.maskSize);
    }

    return info;
}