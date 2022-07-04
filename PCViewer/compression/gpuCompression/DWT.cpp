#include "DWT.hpp"
namespace vkCompress
{
    auto powerOfTwo = [](uint32_t n){
        int count = 0;
        while(n > 0){
            count += n & 1;
            n >>= 1;
        }
        return count;
    };

    void dwtFloatInverse(GpuInstance* pInstance, VkCommandBuffer commands, VkDeviceAddress dstAddress, VkDeviceAddress srcAddress, uint size, uint dstRowPitch, uint srcRowPitch) 
    {
        assert(powerOfTwo(size));
        const int xBlockSizeX = 32;
        const int xResultBlockCount = 8;

        int dispatchX = size / (xResultBlockCount * xBlockSizeX);

        // no special case for image sizes of 64 and 128
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pInstance->DWT.floatInverseInfo.pipeline);
        vkCmdDispatch(commands, dispatchX, 1, 1);
    }
    
    void dwtFloatToHalfInverse(GpuInstance* pInstance, VkCommandBuffer commands, VkDeviceAddress dstAddress, VkDeviceAddress srcAddress, uint size, uint dstRowPitch, uint srcRowPitch) 
    {
        assert(powerOfTwo(size));
        const int xBlockSizeX = 32;
        const int xResultBlockCount = 8;

        int dispatchX = size / (xResultBlockCount * xBlockSizeX);

        // no special case for image sizes of 64 and 128
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pInstance->DWT.floatToHalfInverseInfo.pipeline);
        vkCmdDispatch(commands, dispatchX, 1, 1);
    }
}