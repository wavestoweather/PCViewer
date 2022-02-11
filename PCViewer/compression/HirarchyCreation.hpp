#pragma once
#include "DataLoader.hpp"
#include "../Data.hpp"
#include <string_view>

// functions for creating the compressed hierarchy
// the compressed hierarchy files have the following format:
// [number]dimensions [number]compressedByteLength [number]symbolsSize [number]fullDataSize [number]quantizationStep [number]epsilon \n //Note: newline character has to be taken into account for reading
// [vector<float>]centerOfLevel             //note: the centerOfLevel point contaings ALL columns (including counts), compressedBytes also include the center of this livel to avoid special treatments
// [vector<byte>]compressedBytes
namespace compression{
    // creates ready to use compressed hierarchy in the 'outputfolder' 
    // with the data given in 'loader'
    // Note: This method uses 'createTempHirarchy' and 'compressTempHirarchy' to create temporary hierarchy files to cope with large datasets
    // TODO: add Hierarchy create node pointer to the function to be able ot exchange the hierarchy creation to any method
    void createHirarchy(const std::string_view& outputFolder, DataLoader* loader, float lvl0eps, int levels, int lvlMultiplier, int maxMemoryMB, int amtOfThreads, float quantizationStep);
    void createTempHirarchy(const std::string_view& outputFolder, DataLoader* loader, float lvl0eps, int levels, int lvlMultiplier, int maxMemoryMB, int amtOfThreads);
    void compressTempHirarchy(const std::string_view& outputFolder, int amtOfThreads, float quantizationStep);
    void loadAndDecompress(const std::string_view& file, Data& data);
    void loadHirarchy(const std::string_view& outputFolder, Data& data);
    // combines all data objects into a single one, with the first in the vector being the resulting Data object
    void combineData(std::vector<Data>& data, Data& dst);
};