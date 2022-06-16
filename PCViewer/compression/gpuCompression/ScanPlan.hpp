#pragma once
#include "../cpuCompression/global.h"
#include <cstring>
#include "../../VkUtil.h"

namespace vkCompress {

class ScanPlan
{
public:
    ScanPlan(VkUtil::Context& context, size_t elemSizeBytes, size_t numElements);
    ScanPlan(VkUtil::Context& context, size_t elemSizeBytes, size_t numElements, size_t numRows, size_t rowPitch);
    ~ScanPlan();

    size_t  m_numElements;   // Maximum number of input elements
    size_t  m_elemSizeBytes; // Size of each element in bytes, i.e. sizeof(T)
    std::vector<VkBuffer> m_blockSums;  // the same as the old m_blockSums for vulkan
    std::vector<uint32_t> m_blockSumsOffsets;   // holds the memory offest for each blockSums buffer
    VkDeviceMemory m_blockSumsMemory;
    //void**  m_blockSums;     // Intermediate block sums array
    size_t  m_numLevels;     // Number of levels (in m_blockSums)
    size_t  m_numRows;       // Number of rows
    size_t* m_rowPitches;    // Pitch of each row in elements

    VkUtil::Context m_context;

private:
    void allocate(size_t elemSizeBytes, size_t numElements, size_t numRows, size_t rowPitch);
};

}