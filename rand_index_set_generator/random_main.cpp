#include "grasta_random_mask_gen.h"
#include "random_mask_reader.h"
#include <stdio.h>
#include <fstream>
#include <iostream>

int main()
{
    /*const char* filename = "randomMasks.data";
    RandomMaskGenerator* pMasks = RandomMaskGenerator::Instance();

    pMasks->PrintDataToFile(filename);
    */

    RandomMaskReader maskReader;
    RandMaskInfo maskInfo = maskReader.ReadMasksFromFile();

    std::ofstream ofs("output.txt");


    std::cout << "Width: " << maskInfo.width << '\n'
              << "Height: " << maskInfo.height << '\n'
              << "Mask Size: " << maskInfo.maskSize << '\n'
              << "Num Masks: " << maskInfo.numMasks << std::endl;

    for(unsigned long i = 0; i < maskInfo.maskSize; ++i)
    {
        ofs << maskInfo.data[i] << '\n';
    }

    std::cout << std::flush;

    return 0;
}
