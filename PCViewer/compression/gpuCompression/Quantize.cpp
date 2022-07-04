#include "Quantize.hpp"


namespace vkCompress
{
void unquantizeFromSymbols(GpuInstance* pInstance, VkCommandBuffer commands, VkDeviceAddress dstFloatBufferAddress, VkDeviceAddress srcShortBufferAddress, uint size, float quantizationStep, EQuantizeType quantType) 
{
    const uint workGroupSize = 256;

    struct PC{
        uint size;
        float quantizationStep;
        uint quantizatonType;
        uint padding;
        VkDeviceAddress dst;
        VkDeviceAddress src;
    }pc{};

    pc.size = size;
    pc.quantizationStep = quantizationStep;
    pc.quantizatonType = quantType;
    pc.dst = dstFloatBufferAddress;
    pc.src = srcShortBufferAddress;

    vkCmdPushConstants(commands, pInstance->Quantization.unquantizeUShortFloatInfo.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pInstance->Quantization.unquantizeUShortFloatInfo.pipeline);
    uint dispatchX = (size + workGroupSize - 1) / workGroupSize;
    vkCmdDispatch(commands, dispatchX, 1, 1);
}
}