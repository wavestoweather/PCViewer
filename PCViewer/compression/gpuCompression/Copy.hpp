#pragma once
#include "GpuInstance.hpp"

namespace vkCompress{

    // note that the buffer addresses are the discrete places where the readout and write is done
    void copy(GpuInstance* pInstance, VkCommandBuffer commands, VkDeviceAddress dstBufferAddress, VkDeviceAddress srcBufferAddress, uint32_t byteSize){
        constexpr uint workGroupSize = 256;
        constexpr uint bytePerWorkGropu = workGroupSize * 4;

        struct PC{
            uint byteSize;
            uint pa,dd,ing;
            VkDeviceAddress dst;
            VkDeviceAddress src;
        }pc;

        pc.byteSize = byteSize;
        pc.dst = dstBufferAddress;
        pc.src = srcBufferAddress;

        vkCmdPushConstants(commands, pInstance->Copy.pipelineInfo.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pInstance->Copy.pipelineInfo.pipeline);
        uint dispatchX = (byteSize + bytePerWorkGropu - 1) / bytePerWorkGropu;
        vkCmdDispatch(commands, dispatchX, 1, 1);
    };
}