#include "PackInc.hpp"

namespace vkCompress
{
    // the following constants have to be consistent with the shaders!!!!
    const int SCAN_CTA_SIZE = 128;                   /**< Number of threads in a CTA */
    const int LOG2_SCAN_CTA_SIZE = 7;                /**< log_2(CTA_SIZE) */

    const int SCAN_ELTS_PER_THREAD = 8;              /**< Number of elements per scan thread */

    bool unpackInc16(GpuInstance* pInstance, uint* dpValues, const ushort* dpValueIncrements, uint valueCount)
    {
        if(valueCount == 0)
            return true;
        
        // doing the scan operations
        // corresponds to void scanArray in https://github.com/m0bl0/cudaCompress/blob/master/src/cudaCompress/scan/scan_app.cui
        // with template types <ushort, uint, false, identity)
        const uint blockSize = SCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE;
        uint numBlocks = (valueCount + blockSize - 1) / blockSize;

        uint sharedEltsPerBlock = SCAN_CTA_SIZE * 2;

        uint sharedMemSize = sizeof(uint) * sharedEltsPerBlock;

        // divide pitch by four since scan's load/store addresses are for vec4 elements
        uint blockSumRowPitch = 1;
        bool fullBlock = (valueCount == numBlocks * SCAN_CTA_SIZE * SCAN_ELTS_PER_THREAD);

        // setting up execution parameters
        uint dispatchX = numBlocks;
        

        return true;
    }

    void packInc16CPU(const uint* pValues, ushort* pValueIncrements, uint valueCount) 
    {
        // written to work correctly even when pValues and pValueIncrements alias
        uint prev = pValues[0];
        pValueIncrements[0] = prev;
        for(uint i = 1; i < valueCount; i++) {
            uint value = pValues[i];
            pValueIncrements[i] = pValues[i] - prev;
            prev = value;
        }
    }

    void unpackInc16CPU(uint* pValues, const ushort* pValueIncrements, uint valueCount) 
    {
        uint total = 0;
        for(uint i = 0; i < valueCount; i++) {
            total += pValueIncrements[i];
            pValues[i] = total;
        }
    }
}