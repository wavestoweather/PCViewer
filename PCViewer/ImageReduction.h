#pragma once
#include "VkUtil.h"
#include "PCUtil.h"

namespace Internals {
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkCommandPool commandPool;
    VkQueue queue;
    VkDescriptorPool descriptorPool;
    namespace MaxY {
        struct UBO {
            int width;
            int height;
            int padding[2];
        };
        std::string shaderFile = "shader/imagereducey.comp.spv";
        VkPipeline pipeline;
        VkPipelineLayout pipelineLayout;
        VkDescriptorSetLayout descriptorSetLayout;
        VkDescriptorSet descriptorSet;
        VkBuffer uboBuffer;
        VkDeviceMemory uboMemroy;
    }
}

namespace ImageReduction{
    void init(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool);
    void cleanup();
    void reduceImageMaxY();
}