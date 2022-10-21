#include "UploadManager.hpp"
#include "../range.hpp"
#include <algorithm>
#include <execution>

UploadManager::UploadManager(const VkUtil::Context& context, uint32_t transferQueueIndex, uint32_t amtStagingBuffer, uint32_t stagingBufferSize):
    stagingBufferSize(stagingBufferSize),
    _vkContext(context), 
    _transfers(amtStagingBuffer),
    _transferCommands(amtStagingBuffer)
{
    size_t alignedStagingSize = PCUtil::alignedSize(stagingBufferSize, 256);
    std::vector<size_t> sizes(amtStagingBuffer, alignedStagingSize);
    std::vector<VkBufferUsageFlags> usages(amtStagingBuffer, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    std::tie(_transferBuffers, _transferOffsets, _transferMemory) = VkUtil::createMultiBufferBound(context, sizes, usages, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkMapMemory(context.device, _transferMemory, 0, amtStagingBuffer * alignedStagingSize, 0, &_mappedMemory);  // stayes mapped upon destruction

    for(int i: irange(amtStagingBuffer))
        _transferFences.push_back(VkUtil::createFence(context.device, 0));

    if(transferQueueIndex != (uint32_t)-1){
        vkGetDeviceQueue(context.device, transferQueueIndex, 0, &_transferQueue);
        VkCommandPoolCreateInfo info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        info.queueFamilyIndex = transferQueueIndex;
        vkCreateCommandPool(context.device, &info, nullptr, &_transferPool);
    }

    // starting the transfer thread
    _transferThread = std::thread(threadExec, this);
}

UploadManager::~UploadManager(){
    _running = false;
    _transferThread.join();
    VkDevice& d = _vkContext.device;
    vkUnmapMemory(d, _transferMemory);
    for(auto fence: _transferFences)
        vkDestroyFence(d, fence, nullptr);
    if(_transferPool)
        vkDestroyCommandPool(d, _transferPool, nullptr);
    else
        vkFreeCommandBuffers(d, _vkContext.commandPool, _transferCommands.size(), _transferCommands.data());

    for(auto b: _transferBuffers)
        vkDestroyBuffer(d, b, nullptr);
    if(_transferMemory)
        vkFreeMemory(d, _transferMemory, nullptr);
}

VkFence UploadManager::uploadTask(const void* data, size_t byteSize, VkBuffer dstBuffer, size_t dstBufferOffset){
    {
        std::scoped_lock cacheLock(_cacheMutex);
        if(useCachedData){
            if(cachedData[dstBuffer] == data){
                //std::cout << "Skipping upload task" << std::endl;
                return VK_NULL_HANDLE;
            }
        }
        cachedData[dstBuffer] = data;
    }
    
    _taskSemaphore.acquire();   // trying to get free space, waiting if no space available
    uint32_t uplaodIndex = _nextFreeTransfer % _transferBuffers.size();
    _transfers[uplaodIndex] = {data, byteSize, dstBuffer, dstBufferOffset};

    ++_nextFreeTransfer;
    return _transferFences[uplaodIndex];
}

void UploadManager::threadExec(UploadManager* m){
    // make space for upload tasks to be recorded
    m->_taskSemaphore.releaseN(m->_transferBuffers.size());

    auto& curTransferIndex = m->_curTransferIndex;
    auto& semaphore = m->_taskSemaphore;
    uint32_t& doneTransferIndex = m->_doneTransferIndex;
    VkCommandPool pool = m->_transferPool ? m->_transferPool: m->_vkContext.commandPool;
    VkQueue queue = m->_transferQueue ? m->_transferQueue: m->_vkContext.queue;

    while(m->_running){
        assert(m->_nextFreeTransfer - doneTransferIndex <= m->_transferBuffers.size());
        uint32_t transfersInFlight = curTransferIndex - doneTransferIndex;
        // check if the next transfer in flight is done, reset fence if so and signal the semaphor that a new task can be recorded
        if(transfersInFlight > 0){
            uint32_t index = doneTransferIndex % m->_transferBuffers.size();
            if(vkWaitForFences(m->_vkContext.device, 1, &m->_transferFences[index], VK_TRUE, 0) == VK_SUCCESS){
                //std::cout << "Reset for " << doneTransferIndex << std::endl; std::cout.flush();
                vkResetFences(m->_vkContext.device, 1, &m->_transferFences[index]);
                ++doneTransferIndex;
                semaphore.release();        // signaling 1 task is able to be recorded
            }
        }

        // check for new transfer task
        if(curTransferIndex == m->_nextFreeTransfer)
            continue;

        if(m->_doneTransferIndex == m->_nextFreeTransfer && m->_idleSemaphore.peekCount() > 0)
            m->_idleSemaphore.releaseN(m->_idleSemaphore.peekCount());
        
        // working on copy task
        auto transferIndex = curTransferIndex++ % m->_transferBuffers.size();
        uint8_t* finalPointer = reinterpret_cast<uint8_t*>(m->_mappedMemory) + m->_transferOffsets[transferIndex];
        //std::cout << "[upload] " << m->_transfers[transferIndex].data  << "  to  " << (void*)finalPointer << std::endl; std::cout.flush();
        constexpr size_t threadAmt = 8;
        std::array<int, threadAmt> iter;
        std::iota(iter.begin(), iter.end(), 0);
        const uint32_t threadSize = (m->_transfers[transferIndex].byteSize + threadAmt - 1) / threadAmt;
        std::for_each(std::execution::par ,iter.begin(), iter.end(), [&](int i){
            std::memcpy(finalPointer + i * threadSize, m->_transfers[transferIndex].data + i * threadSize, threadSize);
        });
        
        VkCommandBuffer& commands = m->_transferCommands[transferIndex];
        if(commands)
            vkFreeCommandBuffers(m->_vkContext.device, pool, 1, &commands);
        VkUtil::createCommandBuffer(m->_vkContext.device, pool, &commands);

        VkBufferCopy copy{};
        copy.dstOffset = m->_transfers[transferIndex].dstBufferOffset;
        copy.size = m->_transfers[transferIndex].byteSize;
        vkCmdCopyBuffer(commands, m->_transferBuffers[transferIndex], m->_transfers[transferIndex].dstBuffer, 1, &copy);
        //std::cout << "Using " << curTransferIndex - 1 << std::endl; std::cout.flush();
        assert(vkWaitForFences(m->_vkContext.device, 1, &m->_transferFences[transferIndex], VK_TRUE, 0) == VK_TIMEOUT);
        // flush before command buffer submission to guarantee data upload
        assert(m->_transfers[transferIndex].byteSize <= m->stagingBufferSize);
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = m->_transferMemory;
        range.offset = m->_transferOffsets[transferIndex];
        range.size = PCUtil::alignedSize(m->_transfers[transferIndex].byteSize, 0x40);
        vkFlushMappedMemoryRanges(m->_vkContext.device, 1, &range);
        VkUtil::commitCommandBuffer(queue, commands, m->_transferFences[transferIndex]);
    }
    vkDeviceWaitIdle(m->_vkContext.device);
}
