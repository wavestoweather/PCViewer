#pragma once
#include "../VkUtil.h"
#include "../PCUtil.h"
#include "../largeVis/DrawlistUpdateCoupler.hpp"
#include <vector>
#include <thread>
#include <atomic>

// class that makes it possible to asynchronously upload dat to the gpu via staging buffers, which are mapped all the time
// the amount of staging buffers can be specified, uploading tasks can be recorded until all staging buffers are in use
// when no staging buffer is free, thread that calls the uploadTask() method will be halted until the task was recorded
class UploadManager: public DrawlistUpdateCoupler{
public:
    // Attribute section ------------------------------------------------------------
    const uint32_t stagingBufferSize;
    bool useCachedData{true};                       // determines if the lastBufferData should be used to check if data is cached
    std::map<VkBuffer, const void*> cachedData;     // map storing the last buffer data that was uploaded. When uploadTask() is called, it is checked if the buffer holds already the data and returns if it does
    
    // Method section ---------------------------------------------------------------
    UploadManager(const VkUtil::Context& context, uint32_t transferQueueIndex, uint32_t amtStagingBuffer, uint32_t stagingBufferSize);
    ~UploadManager();
    UploadManager(const UploadManager&) = delete;
    UploadManager& operator=(const UploadManager&) = delete;

    VkFence uploadTask(const void* data, size_t byteSize, VkBuffer dstBuffer, size_t dstBufferOffset = 0);
    void uploadTaskMulti(const std::vector<const void*>& data, size_t byteSize, const std::vector<VkBuffer>& dstBuffer, bool last = false, const std::vector<size_t>& dstBufferOffset = {});

    bool idle() const {return _doneTransferIndex == _nextFreeTransfer;};
    void queueWaitIdle() const{if(_transferQueue) check_vk_result(vkQueueWaitIdle(_transferQueue));};
    void waitForIdle() {_idleSemaphore.acquire();};

private:
    // Attribute section ------------------------------------------------------------
    struct Transfer{
        const void* data;
        size_t byteSize;
        VkBuffer dstBuffer;
        size_t dstBufferOffset;
    };
    VkUtil::Context _vkContext{};
    VkQueue _transferQueue{};               // if not null, special transfer queue is used for copy
    VkCommandPool _transferPool{};          // if not null, special command poos is used for copy commands
    std::vector<VkFence> _transferFences{}; // used for Gpu snyc after copying is done
    std::vector<VkCommandBuffer> _transferCommands{};
    std::vector<VkBuffer> _transferBuffers{};
    std::vector<size_t> _transferOffsets{};
    std::vector<Transfer> _transfers{};
    std::vector<Transfer> _singleCallTransfers{};
    VkDeviceMemory _transferMemory{};

    void* _mappedMemory;

    PCUtil::Semaphore _taskSemaphore{};
    PCUtil::Semaphore _idleSemaphore{};
    std::mutex _cacheMutex{};
    std::thread _transferThread{};
    std::atomic_uint32_t _nextFreeTransfer{};
    std::atomic_bool _lastMultiUpdate{};
    uint32_t _curTransferIndex{};
    uint32_t _doneTransferIndex{};

    bool _running{true};

    // Method section ---------------------------------------------------------------
    static void threadExec(UploadManager* m);
};