#ifndef VkUtil_H
#define VkUtil_H

#include <vulkan/vulkan.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cstring>
#include <mutex>
#include <optional>

static void check_vk_result(VkResult err)
{
    if (err == 0) return;
    printf("VkResult %d\n", err);
    if (err < 0)
        abort();
}

namespace VkUtil{
    struct Context{
        uint32_t screenSize[2];
        VkPhysicalDevice physicalDevice;
        VkDevice device;
        VkDescriptorPool descriptorPool;
        VkCommandPool commandPool;
        VkQueue queue;
        std::mutex* queueMutex;    // needed for queue synchroniztion
    };

    struct PipelineInfo{
        VkPipeline pipeline;
        VkPipelineLayout pipelineLayout;
        VkDescriptorSetLayout descriptorSetLayout;

        void vkDestroy(const Context& c){
            if(pipeline) vkDestroyPipeline(c.device, pipeline, nullptr);
            if(pipelineLayout) vkDestroyPipelineLayout(c.device, pipelineLayout, nullptr);
            if(descriptorSetLayout) vkDestroyDescriptorSetLayout(c.device, descriptorSetLayout, nullptr);
        }
    };

    struct BlendInfo {
        VkPipelineColorBlendAttachmentState blendAttachment;
        VkPipelineColorBlendStateCreateInfo createInfo;
    }; 

    enum PassType {
        PASS_TYPE_COLOR_OFFLINE,
        PASS_TYPE_COLOR_EXPORT,
        PASS_TYPE_COLOR_OFFLINE_NO_CLEAR,
        PASS_TYPE_COLOR16_OFFLINE,
        PASS_TYPE_COLOR16_OFFLINE_NO_CLEAR,
        PASS_TYPE_COLOR16UNORM_OFFLINE_NO_CLEAR,
        PASS_TYPE_COLOR32_OFFLINE_NO_CLEAR,
        PASS_TYPE_DEPTH_OFFLINE,
        PASS_TYPE_DEPTH_STENCIL_OFFLINE,
        PASS_TYPE_UINT32,
        PASS_TYPE_FLOAT,
        PASS_TYPE_NONE
    };

    uint32_t findMemoryType(VkPhysicalDevice physicalDevice ,uint32_t typeFilter, VkMemoryPropertyFlags properties);

    void createMipMaps(VkCommandBuffer commandBuffer, VkImage image, uint32_t mipLevels, uint32_t imageWidth, uint32_t imageHeight, VkImageLayout oldLayout, VkAccessFlags oldAccess, VkPipelineStageFlags oldPipelineStage);
    void createMipMaps(VkCommandBuffer commandBuffer, VkImage image, uint32_t mipLevels, uint32_t imageWidth, uint32_t imageHeight, uint32_t imageDepth, VkImageLayout oldLayout, VkAccessFlags oldAccess, VkPipelineStageFlags oldPipelineStage);
    void createCommandBuffer(VkDevice device, VkCommandPool commandPool, VkCommandBuffer *commandBuffer);
    void createBuffer(VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer* buffer);
    // creates readily bound bound buffers. All buffers are bound to a single memory
    // @return [buffers, offsets, memory]
    std::tuple<std::vector<VkBuffer>, std::vector<VkDeviceSize>, VkDeviceMemory> createMultiBufferBound(const Context& context,const std::vector<VkDeviceSize>& sizes, const std::vector<VkBufferUsageFlags>& usages, VkMemoryPropertyFlags memoryProperty);
    void createBufferView(VkDevice device, VkBuffer buffer, VkFormat format, uint32_t offset, VkDeviceSize range, VkBufferView* bufferView);
    VkDeviceAddress getBufferAddress(VkDevice device, VkBuffer buffer);
    void commitCommandBuffer( VkQueue queue, VkCommandBuffer commandBuffer, VkFence fence = {},const std::vector<VkSemaphore>& waitSemaphores = {}, const std::vector<VkSemaphore>& siganlSemaphore = {});
    void beginRenderPass(VkCommandBuffer commandBuffer, const std::vector<VkClearValue>& clearValues, VkRenderPass renderPass, VkFramebuffer framebuffer, VkExtent2D extend);
    void createPipeline(VkDevice device, VkPipelineVertexInputStateCreateInfo* vertexInfo, float frameWidth, float frameHight, const std::vector<VkDynamicState>& dynamicStates, VkShaderModule* shaderModules, VkPrimitiveTopology topology, VkPipelineRasterizationStateCreateInfo* rasterizerInfo, VkPipelineMultisampleStateCreateInfo* multisamplingInfo, VkPipelineDepthStencilStateCreateInfo* depthStencilInfo, BlendInfo* blendInfo, const std::vector<VkDescriptorSetLayout>& descriptorSetLayouts, VkRenderPass* renderPass, VkPipelineLayout* pipelineLayout, VkPipeline* pipeline, const std::vector<VkPushConstantRange>& pushConstantRanges = {});
    void createComputePipeline(VkDevice device, VkShaderModule& shaderModule, std::vector<VkDescriptorSetLayout> descriptorLayouts, VkPipelineLayout* pipelineLayout, VkPipeline* pipeline, VkSpecializationInfo* specializationInfo = VK_NULL_HANDLE, const std::vector<VkPushConstantRange>& pushConstants = {});
    void destroyPipeline(VkDevice device, VkPipeline pipeline);
    void createRenderPass(VkDevice device, VkUtil::PassType passType, VkRenderPass* renderPass, VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT);
    void createFrameBuffer(VkDevice device, VkRenderPass renderPass, const std::vector<VkImageView>& attachments, uint32_t width, uint32_t height, VkFramebuffer* frambuffer);
    void fillDescriptorSetLayoutBinding(uint32_t bindingNumber, VkDescriptorType descriptorType, uint32_t amtOfDescriptors, VkShaderStageFlags shaderStages, VkDescriptorSetLayoutBinding* uboLayoutBinding);
    void createDescriptorSetLayout(VkDevice device, const std::vector<VkDescriptorSetLayoutBinding>& bindings, VkDescriptorSetLayout* descriptorSetLayout);
    void createDescriptorSetLayoutPartiallyBound(VkDevice device, const std::vector<VkDescriptorSetLayoutBinding>& bindings, const std::vector<bool>& enableValidation, VkDescriptorSetLayout* descriptorSetLayout);
    void destroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout);
    void createDescriptorPool(VkDevice device, const std::vector<VkDescriptorPoolSize>& poolSizes, VkDescriptorPool* descriptorPool);
    void destroyDescriptorPool(VkDevice device, VkDescriptorPool pool);
    void createDescriptorSets(VkDevice device, const std::vector<VkDescriptorSetLayout>& layouts, VkDescriptorPool pool, VkDescriptorSet* descriptorSetArray);
    void updateDescriptorSet(VkDevice device, VkBuffer buffer, VkDeviceSize size, uint32_t binding, VkDescriptorSet descriptorSet);
    void updateDescriptorSet(VkDevice device, VkBuffer buffer, VkDeviceSize size, uint32_t binding, uint32_t offset, VkDescriptorSet descriptorSet);
    void updateDescriptorSet(VkDevice device, VkBuffer buffer, VkDeviceSize size, uint32_t binding, VkDescriptorType descriptorType, VkDescriptorSet descriptorSet);
    void updateDescriptorSet(VkDevice device, VkBuffer buffer, VkDeviceSize size, uint32_t binding, uint32_t offset, VkDescriptorType descriptorType, VkDescriptorSet descriptorSet);
    void updateArrayDescriptorSet(VkDevice device, VkBuffer buffer, VkDeviceSize size, uint32_t binding, uint32_t arrayIndex, VkDescriptorType descriptorType, VkDescriptorSet descriptorSet);
    void updateTexelBufferDescriptorSet(VkDevice device, VkBufferView bufferView, uint32_t binding, VkDescriptorSet descriptorSet);
    void updateImageDescriptorSet(VkDevice device, VkSampler sampler, VkImageView imageView, VkImageLayout imageLayout, uint32_t binding, VkDescriptorSet descriptorSet);
    void updateImageArrayDescriptorSet(VkDevice device, std::vector<VkSampler>& sampler, std::vector<VkImageView>& imageViews, std::vector<VkImageLayout>& imageLayouts, uint32_t binding, VkDescriptorSet descriptorSet);
    void updateStorageImageDescriptorSet(VkDevice device, VkImageView imageView, VkImageLayout imageLayout, uint32_t binding, VkDescriptorSet descriptorSet);
    void updateStorageImageArrayDescriptorSet(VkDevice device, std::vector<VkSampler>& sampler, std::vector<VkImageView>& imageViews, std::vector<VkImageLayout>& imageLayouts, uint32_t binding, VkDescriptorSet descriptorSet);
    void copyImage(VkCommandBuffer commandBuffer, VkImage srcImage, int32_t width, int32_t height, VkImageLayout srcImageLayout, VkImage dstImage, VkImageLayout dstImageLayout);
    void copyBuffer(VkCommandBuffer commandBuffer, VkBuffer src, VkBuffer dst, uint32_t byteSize, uint32_t srcOffset, uint32_t dstOffset);
    void copyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    void copyBufferTo3dImage(VkCommandBuffer commandBuffer, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, uint32_t depth);
    void copy3dImageToBuffer(VkCommandBuffer commandBuffer, VkBuffer buffer, VkImage image, VkImageLayout imageLayout, uint32_t width, uint32_t height, uint32_t depth);
    void transitionImageLayout(VkCommandBuffer commandBuffer, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
    void transitionImageLayoutDirect(VkDevice device, VkCommandPool pool, VkQueue queue, const std::vector<VkImage>& images, const std::vector<VkFormat>& formats, const std::vector<VkImageLayout>& oldLayouts, const std::vector<VkImageLayout>& newLayouts);
    void createImage(VkDevice device, uint32_t width, uint32_t height, VkFormat imageFormat, VkImageUsageFlags usageFlags, VkImage* image);
    void create1dImage(VkDevice device, uint32_t width, VkFormat imageFormat, VkImageUsageFlags usageFlags, VkImage* image);
    void create3dImage(VkDevice device, uint32_t width, uint32_t height, uint32_t depth, VkFormat imageFormat, VkImageUsageFlags usageFlags, VkImage* image);
    void create3dImage(VkDevice device, uint32_t width, uint32_t height, uint32_t depth, VkFormat imageFormat, VkImageUsageFlags usageFlags, uint32_t mipLevel, VkImage* image);
    void createImageView(VkDevice device, VkImage image, VkFormat imageFormat, uint32_t mipLevelCount, VkImageAspectFlags aspectMask, VkImageView* imageView);
    void create1dImageView(VkDevice device, VkImage image, VkFormat imageFormat, uint32_t mipLevelCount, VkImageView* imageView);
    void create3dImageView(VkDevice device, VkImage image, VkFormat imageFormat, uint32_t mipLevelCount, VkImageView* imageView);
    void createImageSampler(VkDevice device, VkSamplerAddressMode adressMode, VkFilter filter, uint16_t maxAnisotropy, uint16_t mipLevels, VkSampler* sampler);
    VkImage createImage(VkDevice device, uint32_t width, uint32_t height, uint32_t depth, VkFormat imageFormat, VkImageUsageFlags flags, uint32_t mipLevel = 1, VkSampleCountFlagBits sampleCounts = VK_SAMPLE_COUNT_1_BIT);
    //VkImageView createImageView(VkDevice device, VkImage image, VkFormat imageFormat, uint32_t mipLevel = 1, VkSampleCountFlagBits sampleCounts = VK_SAMPLE_COUNT_1_BIT);
    void uploadData(VkDevice device, VkDeviceMemory memory, uint32_t offset, uint32_t byteSize,const void* data);
    void downloadData(VkDevice device, VkDeviceMemory memory, uint32_t offset, uint32_t byteSize,void* data);
    void uploadDataIndirect(const Context& context, VkBuffer dstBuffer, uint32_t byteSize,const void* data);
    void downloadDataIndirect(const Context& context, VkBuffer dstBuffer, uint32_t byteSize,void* data);
    void uploadImageData(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkImage image, VkImageLayout imageLayout, VkFormat imageFormat, uint32_t x, uint32_t y, uint32_t z, void* data, uint32_t byteSize);
    void downloadImageData(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkImage image, VkFormat format, VkImageLayout imageLayout, uint32_t x, uint32_t y, uint32_t z, void* data, uint32_t byteSize);
    void addImageToAllocInfo(VkDevice device, VkImage image, VkMemoryAllocateInfo& allocInfo);
    VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& byteArr);
    VkEvent createEvent(VkDevice device, VkEventCreateFlags createFlags);
    VkFence createFence(VkDevice device, VkFenceCreateFlags createFlags);
    VkSemaphore createSemaphore(VkDevice device, VkSemaphoreCreateFlags createFlags);
};

#endif