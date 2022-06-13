#pragma once

#include "../cpuCompression/global.h"
#include "../../VkUtil.h"


namespace cudaCompress {

class ReducePlan
{
public:
    ReducePlan(VkUtil::Context& context , size_t elemSizeBytes, size_t numElements);
    ~ReducePlan();

    size_t m_numElements;     // Maximum number of input elements
    size_t m_elemSizeBytes;   // Size of each element in bytes, i.e. sizeof(T)
    uint   m_threadsPerBlock; // number of threads to launch per block
    uint   m_maxBlocks;       // maximum number of blocks to launch
    //void*  m_blockSums;       // Intermediate block sums array
    VkBuffer m_blockSums;     // corresponding vulkan resources for the cuda resources
    VkDeviceMemory m_blockSumsMem;

    VkUtil::Context m_context;
};

}