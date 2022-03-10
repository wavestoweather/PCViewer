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
    enum class CompressionMethod{
        Leaders,
        VectorLeaders,
        Hash,
        HashVectorLeaders,
        HashLeader
    };
    enum class CachingMethod{
        Native,
        Bundled,
        Single,
        MethodCount
    };
    const std::string_view CachingMethodNames[3]{"NativeCaching", "BundledCaching", "SingleCaching"};

    // creates ready to use compressed hierarchy in the 'outputfolder' 
    // with the data given in 'loader'
    // Note: This method uses 'createTempHirarchy' and 'compressTempHirarchy' to create temporary hierarchy files to cope with large datasets
    // TODO: add Hierarchy create node pointer to the function to be able ot exchange the hierarchy creation to any method
    void createHirarchy(const std::string_view& outputFolder, DataLoader* loader, float lvl0eps, int levels, int lvlMultiplier, int maxMemoryMB, int amtOfThreads, float quantizationStep);
    void createTempHirarchy(const std::string_view& outputFolder, DataLoader* loader, float lvl0eps, int levels, int lvlMultiplier, int maxMemoryMB, int amtOfThreads);
    void compressTempHirarchy(const std::string_view& outputFolder, int amtOfThreads, float quantizationStep);
    //does the same as compressTempHierarchy but without compression. Is tried to evaluate if compression is beneficial
    void convertTempHierarchy(const std::string_view& outputFolder, int amtOfThreads);
    void compressBundledTempHierarchy(const std::string_view& outputFolder, int amtOfThreads, float quantizationStep);
    void loadAndDecompress(const std::string_view& file, Data& data);
    void loadAndDecompressBundled(const std::string_view& levelFile, size_t offset, Data& data);    //levelFile is the levelX.info file because the extra information is needed
    void loadHirarchy(const std::string_view& outputFolder, Data& data);
    // combines all data objects into a single one, with the first in the vector being the resulting Data object
    void combineData(std::vector<Data>& data, Data& dst);
    // checks for a folder

    //--------------------------------------------------------------------------------
    // section for n-dimensional hierarchy creation
    //--------------------------------------------------------------------------------
    // These hierarchies are created as follows:
    //      For each attribute axis the hierarchy points are created (independent of all other axes). This enables extremely efficient storage of the cluster at each axis combined with a very efficient check for point size for single attribute brushing
    //      The n-dimensional clusters are then only counts for each cluster between n axes for all possible cluster pairs.
    // Stored is this structure as followes:
    //      A .axis file contains all hierarchy points for all axes (This information is assumed to be holdable in memory, as the memory footprint ~ N * C * 3 * sizeof(float)), with C being the max number of clusters. 
    
    // method to create a NDHierarchy
    // outputFolder     : Folder name where the resulting hierarchy will be put
    // loader           : DataLoader which contains the data to be converted
    // cachingMethod    : Caching method to be used for inbetween writeouts, should the memory be not large enough
    // startCluster     : Amt of cluster per axis at hierarchy level 0
    // clusterMultiplic.: Multiplicator for the cluster amt per level
    // dimensionality   : Dimensionality of the resulting hierarchy clusters (2d cluster for example can not be used for spline rendering)
    // levels           : Amount of levels
    // maxMb            : Maximum available RAM for hierarchy creation (is used to trigger caching events)
    // amtOfThreads     : Amount of threads to be used for hierarchy creation
    void createNDHierarchy(const std::string_view& outputFolder, DataLoader* loader, CachingMethod cachingMethod, int startCluster, int clusterMultiplikator, int dimensionality, int levels, int maxMb, int amtOfThreads);
    void convertNDHierarchy(const std::string_view& outputFolder, int amtOfThreads);
};