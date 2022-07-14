#include "ScanPlan.hpp"
#include <cstdlib>
#include "../cpuCompression/util.h"
#include "../../range.hpp"

namespace vkCompress {
const int SCAN_ELTS_PER_THREAD = 8;
const int SCAN_CTA_SIZE = 128;

ScanPlan::ScanPlan(VkUtil::Context& context, size_t elemSizeBytes, size_t numElements)
: m_numElements(numElements),
  m_elemSizeBytes(elemSizeBytes),
  m_blockSums(0),
  m_numLevels(0),
  m_numRows(0),
  m_rowPitches(nullptr),
  m_context(context)
{
    allocate(elemSizeBytes, numElements, 1, 0);
}

ScanPlan::ScanPlan(VkUtil::Context& context, size_t elemSizeBytes, size_t numElements, size_t numRows, size_t rowPitch)
: m_numElements(numElements),
  m_elemSizeBytes(elemSizeBytes),
  m_blockSums(0),
  m_numLevels(0),
  m_numRows(0),
  m_rowPitches(nullptr),
  m_context(context)
{
    allocate(elemSizeBytes, numElements, numRows, rowPitch);
}
  
void ScanPlan::allocate(size_t elemSizeBytes, size_t numElements, size_t numRows, size_t rowPitch)
{
    const size_t blockSize = SCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE;

    m_numElements = numElements;
    m_numRows = numRows;
    m_elemSizeBytes = elemSizeBytes;

    // find required number of levels
    size_t level = 0;
    size_t numElts = m_numElements;
    do
    {
        size_t numBlocks = (numElts + blockSize - 1) / blockSize;
        if (numBlocks > 1)
        {
            level++;
        }
        numElts = numBlocks;
    } while (numElts > 1);

    m_numLevels = level;

    m_blockSums.resize(m_numLevels);//  = (void**) malloc(m_numLevels * sizeof(void*));
    m_blockSumsOffsets.resize(m_numLevels);

    if (m_numRows > 1)
    {
        m_rowPitches = (size_t*) malloc((m_numLevels + 1) * sizeof(size_t));
        m_rowPitches[0] = rowPitch;
    }

    // allocate storage for block sums
    numElts = m_numElements;
    level = 0;
    size_t sizeSum{};
    do
    {
        size_t numBlocks = (numElts + blockSize - 1) / blockSize;
        if (numBlocks > 1) 
        {
            // Use cudaMallocPitch for multi-row block sums to ensure alignment
            if (m_numRows > 1)
            {
                size_t dpitch;
                // doing row aligning to 16 bytes to ensure every alignment there is...
                size_t rowLength = cudaCompress::getAlignedSize(numBlocks * m_elemSizeBytes, 16);
                //cudaSafeCall(cudaMallocPitch((void**)&(m_blockSums[level]), &dpitch, numBlocks * m_elemSizeBytes, numRows));
                //VkUtil::createBuffer(m_context.device, rowLength * numRows, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &m_blockSums[level]);
                m_blockSumsOffsets[level] = sizeSum;
                sizeSum += rowLength * numRows; 

                m_rowPitches[level+1] = rowLength / m_elemSizeBytes;
            }
            else
            {
                //cudaSafeCall(cudaMalloc((void**)&(m_blockSums[level]), numBlocks * m_elemSizeBytes));
                //VkUtil::createBuffer(m_context.device, numBlocks * m_elemSizeBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &m_blockSums[level]);
                m_blockSumsOffsets[level] = sizeSum;
                //vkGetBufferMemoryRequirements(m_context.device, m_blockSums[level], &memReq);
                sizeSum += numBlocks * m_elemSizeBytes;
            }
            level++;
        }
        numElts = numBlocks;
    } while (numElts > 1);

    //allocInfo.memoryTypeIndex = VkUtil::findMemoryType(m_context.physicalDevice, allocInfo.memoryTypeIndex, 0);
    auto [buffer, offset, memory] = VkUtil::createMultiBufferBound(m_context, {sizeSum}, {VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT}, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_memorySize = sizeSum;
    m_buffer = buffer[0];
    m_blockSumsMemory = memory;

    VkDeviceAddress baseAddress = VkUtil::getBufferAddress(m_context.device, m_buffer);
    
    for(int i: irange(m_blockSums)){
        //vkBindBufferMemory(m_context.device, m_blockSums[i], m_blockSumsMemory, m_blockSumsOffsets[i]);
        m_blockSums[i] = baseAddress + m_blockSumsOffsets[i];
    }
}

ScanPlan::~ScanPlan()
{
    vkDestroyBuffer(m_context.device, m_buffer, nullptr);
    vkFreeMemory(m_context.device,  m_blockSumsMemory, nullptr);

    if(m_numRows > 1)
    {
        free((void*)m_rowPitches);
        m_rowPitches = nullptr;
    }
    m_numElements = 0;
    m_numLevels = 0;
}

}