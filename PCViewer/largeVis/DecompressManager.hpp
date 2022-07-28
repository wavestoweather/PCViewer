#pragma once
#include "../VkUtil.h"
#include "../compression/gpuCompression/RLHuffData.hpp"
#include "../compression/gpuCompression/GpuInstance.hpp"
#include "../compression/gpuCompression/Encode.hpp"
#include "../compression/gpuCompression/Quantize.hpp"
#include "../compression/gpuCompression/DWT.hpp"
#include "../range.hpp"
#include "TimingInfo.hpp"
#include <thread>

// currently uses the gpu instance from an external source for decompression.
// Could b better to inherit the manager into this class
class DecompressManager{
public:
    // cpuColumns is a vector of pointers to the cpu data which should be decompressed. If one is a nullptr the data should not be decompressed
    using CpuColumns = std::vector<const RLHuffDecodeDataCpu*>;
    // gpuColumns is the same as cpuColumns for the gpu data
    using GpuColumns = std::vector<const RLHuffDecodeDataGpu*>;
    enum class DecompressedType{
        halfF
    };

    std::vector<VkBuffer> buffers{};
    std::vector<size_t> bufferOffsets{};
    VkDeviceMemory bufferMemory{};
    DecompressedType decompressedType{DecompressedType::halfF};
    size_t decompressedElementsPerBuffer{0};

    DecompressManager(const DecompressManager&) = delete;           //no copy constructor
    DecompressManager& operator=(const DecompressManager&) = delete;//no copy assignment
    DecompressManager(DecompressManager&& o){
        std::memcpy(this, &o, sizeof(o));
        std::memset(&o, 0, sizeof(o));
    }
    DecompressManager& operator=(DecompressManager&& o){
        std::memcpy(this, &o, sizeof(o));
        std::memset(&o, 0, sizeof(o));
        return *this;
    }
    // quick init via external column blocks (avoids initialization on first call to recordBlockDecompression(...))
    DecompressManager(uint32_t symbolCountPerBlock, vkCompress::GpuInstance& gpu ,const CpuColumns& cpuData, const GpuColumns& gpuData):
        _vkContext(gpu.vkContext)
        {resizeOrCreateBuffers(symbolCountPerBlock, gpu, cpuData, gpuData);};
    
    ~DecompressManager(){
        deleteVkResources(true);
    }

    void recordBlockDecompression(VkCommandBuffer commands, uint32_t symbolCountPerBlock, vkCompress::GpuInstance& gpu ,const CpuColumns& cpuData, const GpuColumns& gpuData, float quantizationStep){
        if(!_vkContext.device)
            _vkContext = gpu.vkContext;
        // ensure buffer size is large enough
        resizeOrCreateBuffers(symbolCountPerBlock, gpu, cpuData, gpuData);

        // check if decompression is necessary
        bool isSubset = _lastDecompressed == gpuData;
        if(!isSubset && _lastDecompressed.size() == gpuData.size()){
            isSubset = true;
            for(int i: irange(_lastDecompressed)){
                if(_lastDecompressed[i] && _lastDecompressed[i] != gpuData[i]){
                    isSubset = false;
                    break;
                }
            }
        }
        if(isSubset){    // return if current decoding is a subset
            std::cout << "Quitting blockDecompression" << std::endl;
            return;
        }

        // recording decode commands
        VkDeviceAddress cacheA = VkUtil::getBufferAddress(_vkContext.device, _cacheBuffers[0]);
        VkDeviceAddress cacheB = VkUtil::getBufferAddress(_vkContext.device, _cacheBuffers[1]);
        for(int i: irange(gpuData)){
            if(!gpuData[i])
                continue;   // skipping attributes which should not be decompressed

            // cacheA has to be zeroed, as otherwise rl decoding might leave old values in the buffer
            vkCmdFillBuffer(commands, _cacheBuffers[0], 0, symbolCountPerBlock * sizeof(uint16_t), 0);
            vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, {}, 0, {}, 0, {});

            vkCompress::decodeRLHuffHalf(&gpu, *cpuData[i], *gpuData[i], cacheA, commands);
            vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, {}, 0, {}, 0, {});
    
            vkCompress::unquantizeFromSymbols(&gpu, commands, cacheB, cacheA, symbolCountPerBlock, quantizationStep);
            vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, {}, 0, {}, 0, {});


            vkCompress::dwtFloatInverse(&gpu, commands, cacheA, cacheB, symbolCountPerBlock / 2, symbolCountPerBlock / 2, symbolCountPerBlock / 2);
            vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, {}, 0, {}, 0, {});
            VkBufferCopy cpy{};
            cpy.dstOffset = 0;
            cpy.srcOffset = 0;
            cpy.size = symbolCountPerBlock / 2 * sizeof(float);
            vkCmdCopyBuffer(commands, _cacheBuffers[1], _cacheBuffers[0], 1, &cpy);
            vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, {}, 0, {}, 0, {});
            
            VkDeviceAddress finalBlock = VkUtil::getBufferAddress(_vkContext.device, buffers[i]);
            vkCompress::dwtFloatToHalfInverse(&gpu, commands, finalBlock, cacheA, symbolCountPerBlock);
            vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 0, {}, 0, {}, 0, {});
        }

        // setting _lastDecompressed to remember the decoded columns
        _lastDecompressed = gpuData;
    }

    // creates a command buffer itself and commits it, return the vkevent that will be signaled upon finishing
    VkSemaphore executeBlockDecompression(uint32_t symbolCountPerBlock, vkCompress::GpuInstance& gpu ,const CpuColumns& cpuData, const GpuColumns& gpuData, float quantizationStep, VkSemaphore prevPipeSemaphore = {}, TimingInfo timingInfo = {}){
        assert((symbolCountPerBlock & 0b11) == 0);

        auto err = vkWaitForFences(_vkContext.device, 1, &_decompFence, VK_TRUE, 1e9); check_vk_result(err);
        assert(err == VK_SUCCESS);
        vkResetFences(_vkContext.device, 1, &_decompFence);

        std::scoped_lock<std::mutex> lock(*_vkContext.queueMutex);
        if(_commands)
            vkFreeCommandBuffers(_vkContext.device, _vkContext.commandPool, 1, &_commands);
        VkUtil::createCommandBuffer(_vkContext.device, _vkContext.commandPool, &_commands);

        if(timingInfo.queryPool){
            vkCmdResetQueryPool(_commands, timingInfo.queryPool, timingInfo.startIndex, 2);
            vkCmdWriteTimestamp(_commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timingInfo.queryPool, timingInfo.startIndex); 
        }

        recordBlockDecompression(_commands, symbolCountPerBlock, gpu, cpuData, gpuData, quantizationStep);

        if(timingInfo.queryPool)
            vkCmdWriteTimestamp(_commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timingInfo.queryPool, timingInfo.endIndex); 

        std::vector<VkSemaphore> waitSem;
        if(prevPipeSemaphore) waitSem.push_back(prevPipeSemaphore);
        VkUtil::commitCommandBuffer(_vkContext.queue, _commands, _decompFence, waitSem, {_syncSemaphore});
        return _syncSemaphore;
    }
private:
    std::vector<VkBuffer> _cacheBuffers{};   // needed for intermediate
    std::vector<size_t> _cacheBufferOffsets{};
    VkSemaphore _syncSemaphore{};
    VkFence _decompFence{};
    VkUtil::Context _vkContext{};

    GpuColumns _lastDecompressed{};  // stores the last recorded decompressed gpu data to avoid re-recording of the same decompression

    VkCommandBuffer _commands{};

    inline void deleteVkResources(bool final = false){
        auto device = _vkContext.device;
        for(auto& b: buffers)
            vkDestroyBuffer(device, b, nullptr);
        for(auto& b: _cacheBuffers)
            vkDestroyBuffer(device, b, nullptr);
        if(bufferMemory)
            vkFreeMemory(device, bufferMemory, nullptr);
        if(_syncSemaphore && final)
            vkDestroySemaphore(device, _syncSemaphore, nullptr);
        if(_decompFence && final)
            vkDestroyFence(device, _decompFence, nullptr);
    }

    void resizeOrCreateBuffers(uint32_t symbolCountPerBlock, vkCompress::GpuInstance& gpu ,const CpuColumns& cpuData, const GpuColumns& gpuData){
        if(!_syncSemaphore){
            VkSemaphoreCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            vkCreateSemaphore(_vkContext.device, &info, nullptr, &_syncSemaphore);
        }
        if(!_decompFence)
            _decompFence = VkUtil::createFence(_vkContext.device, VK_FENCE_CREATE_SIGNALED_BIT);

        if(cpuData.size() == buffers.size() && decompressedElementsPerBuffer >= symbolCountPerBlock){
            // nothing to do, return
            return;
        }

        // deleting old resources if available
        deleteVkResources();

        // creating needed resources
        // decompressed data resources
        uint32_t elementByteSize = decompressedType == DecompressedType::halfF ? 2: 0;
        std::vector<VkBufferUsageFlags> usages(cpuData.size(), VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        std::vector<size_t> sizes(cpuData.size(), symbolCountPerBlock * elementByteSize);
        // caching resources
        usages.push_back(VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        usages.push_back(VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        sizes.push_back(symbolCountPerBlock * sizeof(float));
        sizes.push_back(symbolCountPerBlock * sizeof(float));
        auto [bs, os, m] = VkUtil::createMultiBufferBound(_vkContext, sizes, usages, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        // copying the buffers to the corresponding variables
        buffers = std::vector<VkBuffer>(bs.begin(), bs.end() - 2); // all buffers except the last 2 are decompressed data
        bufferOffsets = std::vector<size_t>(os.begin(), os.end() - 2);
        _cacheBuffers = std::vector<VkBuffer>(bs.end() - 2, bs.end());  // last to buffers are cache buffers
        _cacheBufferOffsets = std::vector<size_t>(os.end() - 2, os.end());
        bufferMemory = m;
        decompressedElementsPerBuffer = symbolCountPerBlock;
    };
};