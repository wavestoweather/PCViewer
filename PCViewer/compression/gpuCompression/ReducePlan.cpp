#include "ReducePlan.hpp"
#include <algorithm>

namespace vkCompress {
const int REDUCE_CTA_SIZE = 256;  
ReducePlan::ReducePlan(VkUtil::Context& context, size_t  elemSizeBytes, size_t numElements)
: m_numElements(numElements),
  m_elemSizeBytes(elemSizeBytes),
  m_threadsPerBlock(REDUCE_CTA_SIZE),
  m_maxBlocks(64),
  m_blockSums(0),
  m_context(context)
{
    uint blocks = std::min(m_maxBlocks, (uint(m_numElements) + m_threadsPerBlock - 1) / m_threadsPerBlock);
    //cudaMalloc(&m_blockSums, blocks * m_elemSizeBytes);
    VkUtil::createBuffer(context.device, blocks * m_elemSizeBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &m_blockSums);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    VkMemoryRequirements memReq{};
    vkGetBufferMemoryRequirements(m_context.device, m_blockSums, &memReq);
    allocInfo.allocationSize += memReq.size;
    allocInfo.memoryTypeIndex |= memReq.memoryTypeBits;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(m_context.physicalDevice, allocInfo.memoryTypeIndex, 0);
    vkAllocateMemory(m_context.device, &allocInfo, nullptr, &m_blockSumsMem);
    vkBindBufferMemory(context.device, m_blockSums, m_blockSumsMem, 0);
}

ReducePlan::~ReducePlan()
{
    //cudaFree(m_blockSums);
    if(m_blockSums)
        vkDestroyBuffer(m_context.device, m_blockSums, nullptr);
    if(m_blockSumsMem)
        vkFreeMemory(m_context.device, m_blockSumsMem, nullptr);
    m_blockSums = 0;
}

}