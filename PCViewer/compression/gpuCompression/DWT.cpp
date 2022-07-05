#include "DWT.hpp"
namespace vkCompress
{
    struct PC{
        uint srcOffset;
        uint dstOffset;
        uint size;
        uint padding;
        VkDeviceAddress dstAddress;
        VkDeviceAddress srcAddress;
    }pc{};

    auto powerOfTwo = [](uint32_t n){
        int count = 0;
        while(n > 0){
            count += n & 1;
            n >>= 1;
        }
        return count == 1;
    };

    void dwtFloatInverse(GpuInstance* pInstance, VkCommandBuffer commands, VkDeviceAddress dstAddress, VkDeviceAddress srcAddress, uint size, uint dstRowPitch, uint srcRowPitch) 
    {
        assert(powerOfTwo(size));
        const int xBlockSizeX = 128;
        const int xResultBlockCount = 8;

        pc.srcOffset = srcRowPitch;
        pc.dstOffset = dstRowPitch;
        pc.size = size;
        pc.dstAddress = dstAddress;
        pc.srcAddress = srcAddress;

        int dispatchX = size / (xResultBlockCount * xBlockSizeX);

        // no special case for image sizes of 64 and 128
        vkCmdPushConstants(commands, pInstance->DWT.floatInverseInfo.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pInstance->DWT.floatInverseInfo.pipeline);
        vkCmdDispatch(commands, dispatchX, 1, 1);
    }
    
    void dwtFloatToHalfInverse(GpuInstance* pInstance, VkCommandBuffer commands, VkDeviceAddress dstAddress, VkDeviceAddress srcAddress, uint size, uint dstRowPitch, uint srcRowPitch) 
    {
        assert(powerOfTwo(size));
        const int xBlockSizeX = 128;
        const int xResultBlockCount = 8;

        pc.srcOffset = srcRowPitch;
        pc.dstOffset = dstRowPitch;
        pc.size = size;
        pc.dstAddress = dstAddress;
        pc.srcAddress = srcAddress;

        int dispatchX = size / (xResultBlockCount * xBlockSizeX);

        // no special case for image sizes of 64 and 128
        vkCmdPushConstants(commands, pInstance->DWT.floatToHalfInverseInfo.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pInstance->DWT.floatToHalfInverseInfo.pipeline);
        vkCmdDispatch(commands, dispatchX, 1, 1);
    }
}