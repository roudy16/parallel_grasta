#include "random_mask_reader.h"

RandMaskInfo& RandMaskInfo::operator=(const RandMaskInfo &rhs)
{
    if(this == &rhs)
    {
        return *this;
    }

    width = rhs.width;
    height = rhs.height;
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
    ifs.read(reinterpret_cast<char*>(&info.numMasks), sizeof(info.numMasks));

    info.data = new int[info.width * info.height * info.numMasks];

    for(unsigned long long i = 0; i < info.numMasks; ++i)
    {
        ifs.read( reinterpret_cast<char*>(&info.data[i * info.width * info.height]),
                  sizeof(int) * info.width * info.height );
    }

    return info;
}