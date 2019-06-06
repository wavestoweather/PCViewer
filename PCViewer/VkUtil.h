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
		Color,
		Depth
	};

	static void createMipMaps(VkCommandBuffer commandBuffer, VkImage image, uint32_t mipLevels, uint32_t imageWidth, uint32_t imageHeight, VkImageLayout oldLayout, VkAccessFlags oldAccess, VkPipelineStageFlags oldPipelineStage);
	static void createCommandBuffer(VkDevice device, VkCommandPool commandPool, VkCommandBuffer *commandBuffer);
	static void commitCommandBuffer(VkDevice device, VkQueue queue, VkCommandBuffer commandBuffer);
	static void createPipeline(VkDevice device, const std::string& vertexShaderPath, const std::string& fragmentShaderPath, VkPipelineVertexInputStateCreateInfo* vertexInfo, float frameWidth, float frameHight, VkDynamicState* dynamicStates, VkPipeline* pipeline);

private:
	static VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& byteArr);
};

#endif