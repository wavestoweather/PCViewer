#pragma once
#include "../VkUtil.h"
#include "../compression/gpuCompression/RLHuffData.hpp"
#include "../compression/gpuCompression/GpuInstance.hpp"
#include "../compression/gpuCompression/Encode.hpp"
#include "../compression/gpuCompression/Quantize.hpp"
#include "../compression/gpuCompression/DWT.hpp"
#include "../range.hpp"

// currently uses the gpu instance from an external source for decompression.
// Could b better to inherit the manager into this class
class DecompressManager{
public:
    // cpuColumns is a vector of pointers to the cpu data which should be decompressed. If one is a nullptr the data should not be decompressed
    using cpuColumns = std::vector<const RLHuffDecodeDataCpu*>;
    // gpuColumns is the same as cpuColumns for the gpu data
    using gpuColumns = std::vector<const RLHuffDecodeDataGpu*>;
    enum class DecompressedType{
        halfF
    };

    std::vector<VkBuffer> buffers{};
    std::vector<size_t> bufferOffsets{};
    VkDeviceMemory bufferMemory{};
    DecompressedType decompressedType{DecompressedType::halfF};
    size_t decompressedElementsPerBuffer{0};

    // quick init via external column blocks
    DecompressManager(uint32_t symbolCountPerBlock, vkCompress::GpuInstance& gpu ,const cpuColumns& cpuData, const gpuColumns& gpuData):
        _vkContext(gpu.vkContext)
        {resizeOrCreateBuffers(symbolCountPerBlock, gpu, cpuData, gpuData);};
    
    ~DecompressManager(){
        deleteVkResources();
    }

    void recordBlockDecompression(VkCommandBuffer commands, uint32_t symbolCountPerBlock, vkCompress::GpuInstance& gpu ,const cpuColumns& cpuData, const gpuColumns& gpuData, float quantizationStep){
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
        if(isSubset)    // return if current decoding is a subset
            return;

        // recording decode commands
        VkDeviceAddress cacheA = VkUtil::getBufferAddress(_vkContext.device, _cacheBuffers[0]);
        VkDeviceAddress cacheB = VkUtil::getBufferAddress(_vkContext.device, _cacheBuffers[1]);
        for(int i: irange(gpuData)){
            if(!gpuData[i])
                continue;   // skipping attributes which should not be decompressed
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
private:
    std::vector<VkBuffer> _cacheBuffers{};   // needed for intermediate
    std::vector<size_t> _cacheBufferOffsets{};
    VkUtil::Context _vkContext;

    gpuColumns _lastDecompressed{};  // stores the last recorded decompressed gpu data to avoid re-recording of the same decompression

    inline void deleteVkResources(){
        auto device = _vkContext.device;
        for(auto& b: buffers)
            vkDestroyBuffer(device, b, nullptr);
        for(auto& b: _cacheBuffers)
            vkDestroyBuffer(device, b, nullptr);
        if(bufferMemory)
            vkFreeMemory(device, bufferMemory, nullptr);
    }

    void resizeOrCreateBuffers(uint32_t symbolCountPerBlock, vkCompress::GpuInstance& gpu ,const cpuColumns& cpuData, const gpuColumns& gpuData){
        if(cpuData.size() == buffers.size() && decompressedElementsPerBuffer >= symbolCountPerBlock){
            // nothing to do, return
            return;
        }

        // deleting old resources if available
        deleteVkResources();

        // creating needed resources
        // decompressed data resources
        uint32_t elementByteSize = decompressedType == DecompressedType::halfF ? 2: 0;
        std::vector<VkBufferUsageFlags> usages(cpuData.size(), VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        std::vector<size_t> sizes(cpuData.size(), symbolCountPerBlock * elementByteSize);
        // caching resources
        usages.push_back(VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        usages.push_back(VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        sizes.push_back(symbolCountPerBlock * sizeof(float));
        sizes.push_back(symbolCountPerBlock * sizeof(float));
        auto [bs, os, m] = VkUtil::createMultiBufferBound(_vkContext, sizes, usages, 0);
        // copying the buffers to the corresponding variables
        buffers = std::vector<VkBuffer>(bs.begin(), bs.end() - 2); // all buffers except the last 2 are decompressed data
        bufferOffsets = std::vector<size_t>(os.begin(), os.end() - 2);
        _cacheBuffers = std::vector<VkBuffer>(bs.end() - 2, bs.end());  // last to buffers are cache buffers
        _cacheBufferOffsets = std::vector<size_t>(os.end() - 2, os.end());
        bufferMemory = m;
    };
};