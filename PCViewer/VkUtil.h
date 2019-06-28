#ifndef VkUtil_H
#define VkUtil_H

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include "PCUtil.h"

static void check_vk_result(VkResult err)
{
	if (err == 0) return;
	printf("VkResult %d\n", err);
	if (err < 0)
		abort();
}

class VkUtil{
public:
	struct BlendInfo {
		VkPipelineColorBlendAttachmentState blendAttachment;
		VkPipelineColorBlendStateCreateInfo createInfo;
	}; 

	enum PassType {
		PASS_TYPE_COLOR_OFFLINE,
		PASS_TYPE_DEPTH_OFFLINE
	};

	static void createMipMaps(VkCommandBuffer commandBuffer, VkImage image, uint32_t mipLevels, uint32_t imageWidth, uint32_t imageHeight, VkImageLayout oldLayout, VkAccessFlags oldAccess, VkPipelineStageFlags oldPipelineStage);
	static void createCommandBuffer(VkDevice device, VkCommandPool commandPool, VkCommandBuffer *commandBuffer);
	static void commitCommandBuffer( VkQueue queue, VkCommandBuffer commandBuffer);
	static void beginRenderPass(VkCommandBuffer commandBuffer, std::vector<VkClearValue>& clearValues, VkRenderPass renderPass, VkFramebuffer framebuffer, VkExtent2D extend);
	static void createPipeline(VkDevice device, VkPipelineVertexInputStateCreateInfo* vertexInfo, float frameWidth, float frameHight, const std::vector<VkDynamicState>& dynamicStates, VkShaderModule* shaderModules, VkPrimitiveTopology topology, VkPipelineRasterizationStateCreateInfo* rasterizerInfo, VkPipelineMultisampleStateCreateInfo* multisamplingInfo, VkPipelineDepthStencilStateCreateInfo* depthStencilInfo, BlendInfo* blendInfo, const std::vector<VkDescriptorSetLayout>& descriptorSetLayouts, VkRenderPass* renderPass, VkPipelineLayout* pipelineLayout, VkPipeline* pipeline);
	static void destroyPipeline(VkDevice device, VkPipeline pipeline);
	static void createPcPlotRenderPass(VkDevice device, VkUtil::PassType passType, VkRenderPass* renderPass);
	static void fillDescriptorSetLayoutBinding(uint32_t bindingNumber, VkDescriptorType descriptorType, uint32_t amtOfDescriptors, VkShaderStageFlags shaderStages, VkDescriptorSetLayoutBinding* uboLayoutBinding);
	static void createDescriptorSetLayout(VkDevice device, const std::vector<VkDescriptorSetLayoutBinding>& bindings, VkDescriptorSetLayout* descriptorSetLayout);
	static void destroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout);
	static void createDescriptorPool(VkDevice device, const std::vector<VkDescriptorPoolSize>& poolSizes, VkDescriptorPool* descriptorPool);
	static void destroyDescriptorPool(VkDevice device, VkDescriptorPool pool);
	static void createDescriptorSets(VkDevice device, const std::vector<VkDescriptorSetLayout>& layouts, VkDescriptorPool pool, VkDescriptorSet* descriptorSetArray);
	static void updateDescriptorSet(VkDevice device, VkBuffer buffer, uint32_t size, VkDescriptorSet descriptorSet);
	static void updateImageDescriptorSet(VkDevice device, VkSampler sampler, VkImageView imageView, VkImageLayout imageLayout, VkDescriptorSet descriptorSet);
	static void copyImage(VkCommandBuffer commandBuffer, VkImage srcImage, int32_t width, int32_t height, VkImageLayout srcImageLayout, VkImage dstImage, VkImageLayout dstImageLayout);
	static void transitionImageLayout(VkCommandBuffer commandBuffer, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
	static void createImage(VkDevice device, uint32_t width, uint32_t height, VkFormat imageFormat, VkImageUsageFlags usageFlags, VkImage* image);
	static void createImageView(VkDevice device, VkImage image, VkFormat imageFormat, uint32_t mipLevelCount, VkImageView* imageView);
	static void createImageSampler(VkDevice device, VkSamplerAddressMode adressMode, uint16_t maxAnisotropy, uint16_t mipLevels, VkSampler* sampler);
	static VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& byteArr);
};

#endif