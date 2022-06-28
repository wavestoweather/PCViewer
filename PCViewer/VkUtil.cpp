#include "VkUtil.h"
#include "range.hpp"

uint32_t VkUtil::findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
	VkPhysicalDeviceMemoryProperties memProps;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

	for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
	{
		if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
		{
			return i;
		}
	}
	// safety call to see whther a valid type Index was found
#ifdef _DEBUG
	std::cerr << "The memory type which is needed is not available!" << std::endl;
	exit(-1);
#endif
	return 0xffffffff;
}

void VkUtil::createMipMaps(VkCommandBuffer commandBuffer, VkImage image, uint32_t mipLevels, uint32_t imageWidth, uint32_t imageHeight, VkImageLayout oldLayout, VkAccessFlags oldAccess, VkPipelineStageFlags oldPipelineStage)
{
	VkImageMemoryBarrier use_barrier[1] = {};
	use_barrier[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	use_barrier[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	use_barrier[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	use_barrier[0].image = image;
	use_barrier[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	use_barrier[0].subresourceRange.baseArrayLayer = 0;
	use_barrier[0].subresourceRange.levelCount = 1;
	use_barrier[0].subresourceRange.layerCount = 1;

	int32_t mipWidth = (int32_t)imageWidth;
	int32_t mipHeight = (int32_t)imageHeight;

	for (uint32_t i = 1; i < mipLevels; i++)
	{
		use_barrier[0].srcAccessMask = oldAccess;
		use_barrier[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		use_barrier[0].oldLayout = oldLayout;
		use_barrier[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		use_barrier[0].subresourceRange.baseMipLevel = i - 1;

		vkCmdPipelineBarrier(commandBuffer, oldPipelineStage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, use_barrier);

		VkImageBlit blit = {};
		blit.srcOffsets[0] = {0, 0, 0};
		blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
		blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		blit.srcSubresource.mipLevel = i - 1;
		blit.srcSubresource.baseArrayLayer = 0;
		blit.srcSubresource.layerCount = 1;
		blit.dstOffsets[0] = {0, 0, 0};
		blit.dstOffsets[1] = {mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1};
		blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		blit.dstSubresource.mipLevel = i;
		blit.dstSubresource.baseArrayLayer = 0;
		blit.dstSubresource.layerCount = 1;

		vkCmdBlitImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

		use_barrier[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		use_barrier[0].dstAccessMask = oldAccess;
		use_barrier[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		use_barrier[0].newLayout = oldLayout;

		vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, oldPipelineStage, 0, 0, NULL, 0, NULL, 1, use_barrier);

		if (mipWidth > 1)
			mipWidth /= 2;
		if (mipHeight > 1)
			mipHeight /= 2;
	}
}

void VkUtil::createMipMaps(VkCommandBuffer commandBuffer, VkImage image, uint32_t mipLevels, uint32_t imageWidth, uint32_t imageHeight, uint32_t imageDepth, VkImageLayout oldLayout, VkAccessFlags oldAccess, VkPipelineStageFlags oldPipelineStage)
{
	VkImageMemoryBarrier use_barrier[1] = {};
	use_barrier[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	use_barrier[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	use_barrier[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	use_barrier[0].image = image;
	use_barrier[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	use_barrier[0].subresourceRange.baseArrayLayer = 0;
	use_barrier[0].subresourceRange.levelCount = 1;
	use_barrier[0].subresourceRange.layerCount = 1;

	int32_t mipWidth = (int32_t)imageWidth;
	int32_t mipHeight = (int32_t)imageHeight;
	int32_t mipDepth = (int32_t)imageDepth;

	for (uint32_t i = 1; i < mipLevels; i++)
	{
		use_barrier[0].srcAccessMask = oldAccess;
		use_barrier[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		use_barrier[0].oldLayout = oldLayout;
		use_barrier[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		use_barrier[0].subresourceRange.baseMipLevel = i - 1;

		vkCmdPipelineBarrier(commandBuffer, oldPipelineStage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, use_barrier);

		VkImageBlit blit = {};
		blit.srcOffsets[0] = {0, 0, 0};
		blit.srcOffsets[1] = {mipWidth, mipHeight, mipDepth};
		blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		blit.srcSubresource.mipLevel = i - 1;
		blit.srcSubresource.baseArrayLayer = 0;
		blit.srcSubresource.layerCount = 1;
		blit.dstOffsets[0] = {0, 0, 0};
		blit.dstOffsets[1] = {mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, mipDepth > 1 ? mipDepth / 2 : 1};
		blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		blit.dstSubresource.mipLevel = i;
		blit.dstSubresource.baseArrayLayer = 0;
		blit.dstSubresource.layerCount = 1;

		vkCmdBlitImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

		use_barrier[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		use_barrier[0].dstAccessMask = oldAccess;
		use_barrier[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		use_barrier[0].newLayout = oldLayout;

		vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, oldPipelineStage, 0, 0, NULL, 0, NULL, 1, use_barrier);

		if (mipWidth > 1)
			mipWidth /= 2;
		if (mipHeight > 1)
			mipHeight /= 2;
		if (mipDepth > 1)
			mipDepth /= 2;
	}
}

void VkUtil::createCommandBuffer(VkDevice device, VkCommandPool commandPool, VkCommandBuffer *commandBuffer)
{
	VkResult err;

	VkCommandBufferAllocateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	bufferInfo.commandPool = commandPool;
	bufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	bufferInfo.commandBufferCount = 1;

	err = vkAllocateCommandBuffers(device, &bufferInfo, commandBuffer);
	check_vk_result(err);

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
	beginInfo.pInheritanceInfo = nullptr;

	err = vkBeginCommandBuffer(*commandBuffer, &beginInfo);
	check_vk_result(err);
}

void VkUtil::createBuffer(VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer *buffer)
{
	VkResult err;

	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = size;
	bufferInfo.usage = usage;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	err = vkCreateBuffer(device, &bufferInfo, nullptr, buffer);
	check_vk_result(err);
}

std::tuple<std::vector<VkBuffer>, std::vector<VkDeviceSize>, VkDeviceMemory> VkUtil::createMultiBufferBound(const Context& context,const std::vector<VkDeviceSize>& sizes, const std::vector<VkBufferUsageFlags>& usages, VkMemoryPropertyFlags memoryProperty) 
{
	const int alignment = 128;
	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	std::vector<VkBuffer> buffers(sizes.size());
	std::vector<VkDeviceSize> offsets(sizes.size());
	bool useBufferDeviceAddress{false};
	for(int i: irange(buffers)){
		auto alignedSize = (sizes[i] + alignment - 1) / alignment * alignment;
		createBuffer(context.device, alignedSize, usages[i], &buffers[i]);
		VkMemoryRequirements memReq{};
		vkGetBufferMemoryRequirements(context.device, buffers[i], &memReq);
		offsets[i] = allocInfo.allocationSize;
		allocInfo.allocationSize += memReq.size;
		allocInfo.memoryTypeIndex |= memReq.memoryTypeBits;
		if(usages[i] & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
			useBufferDeviceAddress = true;
	}
	allocInfo.memoryTypeIndex = findMemoryType(context.physicalDevice, allocInfo.memoryTypeIndex, memoryProperty);
	VkMemoryAllocateFlagsInfo allocFlagInfo{};
	allocFlagInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
	if(useBufferDeviceAddress)
		allocFlagInfo.flags |= VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
	allocInfo.pNext = &allocFlagInfo;
	VkDeviceMemory memory;
	auto err = vkAllocateMemory(context.device, &allocInfo, nullptr, &memory); check_vk_result(err);
	for(int i: irange(buffers)){
		err = vkBindBufferMemory(context.device, buffers[i], memory, offsets[i]); check_vk_result(err);
	}
	return {buffers, offsets, memory};
};

VkDeviceAddress VkUtil::getBufferAddress(VkDevice device, VkBuffer buffer){
	VkBufferDeviceAddressInfo info{};
	info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	info.buffer = buffer;
	return vkGetBufferDeviceAddress(device, &info);
}

void VkUtil::createBufferView(VkDevice device, VkBuffer buffer, VkFormat format, uint32_t offset, VkDeviceSize range, VkBufferView *bufferView)
{
	VkResult err;

	VkBufferViewCreateInfo info = {};
	info.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
	info.buffer = buffer;
	info.format = format;
	info.offset = offset;
	info.range = range;
	err = vkCreateBufferView(device, &info, nullptr, bufferView);
	check_vk_result(err);
}

void VkUtil::commitCommandBuffer(VkQueue queue, VkCommandBuffer commandBuffer)
{
	VkResult err;

	err = vkEndCommandBuffer(commandBuffer);
	check_vk_result(err);

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.signalSemaphoreCount = 0;
	submitInfo.waitSemaphoreCount = 0;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	err = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	check_vk_result(err);
}

void VkUtil::beginRenderPass(VkCommandBuffer commandBuffer, const std::vector<VkClearValue> &clearValues, VkRenderPass renderPass, VkFramebuffer framebuffer, VkExtent2D extend)
{
	VkRenderPassBeginInfo renderPassInfo = {};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassInfo.renderPass = renderPass;
	renderPassInfo.framebuffer = framebuffer;
	renderPassInfo.renderArea.offset = {0, 0};
	renderPassInfo.renderArea.extent = extend;

	renderPassInfo.clearValueCount = clearValues.size();
	renderPassInfo.pClearValues = clearValues.data();

	vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
}

void VkUtil::createPipeline(VkDevice device, VkPipelineVertexInputStateCreateInfo *vertexInfo, float frameWidth, float frameHight, const std::vector<VkDynamicState> &dynamicStates, VkShaderModule *shaderModules, VkPrimitiveTopology topology, VkPipelineRasterizationStateCreateInfo *rasterizerInfo, VkPipelineMultisampleStateCreateInfo *multisamplingInfo, VkPipelineDepthStencilStateCreateInfo *depthStencilInfo, BlendInfo *blendInfo, const std::vector<VkDescriptorSetLayout> &descriptorSetLayouts, VkRenderPass *renderPass, VkPipelineLayout *pipelineLayout, VkPipeline *pipeline, const std::vector<VkPushConstantRange> &pushConstantRanges)
{
	VkResult err;

	std::vector<VkPipelineShaderStageCreateInfo> shaderStages;

	if (shaderModules[0])
	{
		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = shaderModules[0];
		vertShaderStageInfo.pName = "main";
		shaderStages.push_back(vertShaderStageInfo);
	}

	if (shaderModules[1])
	{
		VkPipelineShaderStageCreateInfo tessControlShaderStageInfo = {};
		tessControlShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		tessControlShaderStageInfo.stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
		tessControlShaderStageInfo.module = shaderModules[1];
		tessControlShaderStageInfo.pName = "main";
		shaderStages.push_back(tessControlShaderStageInfo);
	}

	if (shaderModules[2])
	{
		VkPipelineShaderStageCreateInfo tessEvaluationShaderStageInfo = {};
		tessEvaluationShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		tessEvaluationShaderStageInfo.stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
		tessEvaluationShaderStageInfo.module = shaderModules[2];
		tessEvaluationShaderStageInfo.pName = "main";
		shaderStages.push_back(tessEvaluationShaderStageInfo);
	}

	if (shaderModules[3])
	{
		VkPipelineShaderStageCreateInfo geoShaderStageInfo = {};
		geoShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		geoShaderStageInfo.stage = VK_SHADER_STAGE_GEOMETRY_BIT;
		geoShaderStageInfo.module = shaderModules[3];
		geoShaderStageInfo.pName = "main";
		shaderStages.push_back(geoShaderStageInfo);
	}

	if (shaderModules[4])
	{
		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = shaderModules[4];
		fragShaderStageInfo.pName = "main";
		shaderStages.push_back(fragShaderStageInfo);
	}

	VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = topology;
	inputAssembly.primitiveRestartEnable = (topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP || topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY || topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP || topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY) ? VK_TRUE : VK_FALSE;

	VkViewport viewport = {}; // description for our viewport for transformation operation after rasterization
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = frameWidth;
	viewport.height = frameHight;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	VkRect2D scissor = {}; // description for cutting the rendered result if wanted
	scissor.offset = {0, 0};
	scissor.extent = {(uint32_t)frameWidth, (uint32_t)frameHight};

	VkPipelineViewportStateCreateInfo viewportState = {};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;

	VkPipelineDynamicStateCreateInfo dynamicState = {};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = dynamicStates.size();
	dynamicState.pDynamicStates = dynamicStates.data();

	VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = descriptorSetLayouts.size();
	pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
	pipelineLayoutInfo.pushConstantRangeCount = pushConstantRanges.size();
	pipelineLayoutInfo.pPushConstantRanges = pushConstantRanges.data();

	err = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, pipelineLayout);
	check_vk_result(err);

	VkGraphicsPipelineCreateInfo pipelineInfo = {};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = shaderStages.size();
	pipelineInfo.pStages = shaderStages.data();
	pipelineInfo.pVertexInputState = vertexInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = rasterizerInfo;
	pipelineInfo.pMultisampleState = multisamplingInfo;
	pipelineInfo.pDepthStencilState = depthStencilInfo;
	pipelineInfo.pColorBlendState = &blendInfo->createInfo;
	pipelineInfo.pDynamicState = &dynamicState;
	pipelineInfo.layout = *pipelineLayout;
	pipelineInfo.renderPass = *renderPass;
	pipelineInfo.subpass = 0;

	err = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, pipeline);
	check_vk_result(err);

	if (shaderModules[0])
		vkDestroyShaderModule(device, shaderModules[0], nullptr);
	if (shaderModules[1])
		vkDestroyShaderModule(device, shaderModules[1], nullptr);
	if (shaderModules[2])
		vkDestroyShaderModule(device, shaderModules[2], nullptr);
	if (shaderModules[3])
		vkDestroyShaderModule(device, shaderModules[3], nullptr);
	if (shaderModules[4])
		vkDestroyShaderModule(device, shaderModules[4], nullptr);
}

void VkUtil::createComputePipeline(VkDevice device, VkShaderModule &shaderModule, std::vector<VkDescriptorSetLayout> descriptorLayouts, VkPipelineLayout *pipelineLayout, VkPipeline *pipeline, VkSpecializationInfo *specializationInfo)
{
	VkResult err;

	VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = descriptorLayouts.size();
	pipelineLayoutInfo.pSetLayouts = descriptorLayouts.data();
	pipelineLayoutInfo.pushConstantRangeCount = 0;
	pipelineLayoutInfo.pPushConstantRanges = nullptr;

	err = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, pipelineLayout);
	check_vk_result(err);

	VkPipelineShaderStageCreateInfo shaderInfo = {};
	shaderInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.module = shaderModule;
	shaderInfo.pName = "main";
	shaderInfo.pSpecializationInfo = specializationInfo;

	VkComputePipelineCreateInfo pipeInfo = {};
	pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipeInfo.layout = *pipelineLayout;
	pipeInfo.stage = shaderInfo;

	err = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, pipeline);
	check_vk_result(err);

	vkDestroyShaderModule(device, shaderModule, nullptr);
}

void VkUtil::destroyPipeline(VkDevice device, VkPipeline pipeline)
{
	vkDestroyPipeline(device, pipeline, nullptr);
}

VkShaderModule VkUtil::createShaderModule(VkDevice device, const std::vector<char> &byteArr)
{
	VkShaderModuleCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.pCode = reinterpret_cast<const uint32_t *>(byteArr.data());
	createInfo.codeSize = byteArr.size();

	VkShaderModule shaderModule;
	VkResult err = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
	check_vk_result(err);

	return shaderModule;
}

void VkUtil::createRenderPass(VkDevice device, VkUtil::PassType passType, VkRenderPass *renderPass)
{
	VkResult err;

	std::vector<VkAttachmentDescription> colorAttachments;
	VkAttachmentReference colorAttachmentRef = {};
	VkAttachmentReference depthAttachmentRef = {};
	VkSubpassDescription subpass = {};
	VkAttachmentDescription attachment = {};

	switch (passType)
	{
	case VkUtil::PASS_TYPE_COLOR_OFFLINE:
		attachment.format = VK_FORMAT_R8G8B8A8_UNORM;
		attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		colorAttachments.push_back(attachment);

		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		break;
	case VkUtil::PASS_TYPE_COLOR_EXPORT:
		attachment.format = VK_FORMAT_B8G8R8A8_UNORM;
		attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		colorAttachments.push_back(attachment);

		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		break;
	case VkUtil::PASS_TYPE_COLOR_OFFLINE_NO_CLEAR:
		attachment.format = VK_FORMAT_R8G8B8A8_UNORM;
		attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		colorAttachments.push_back(attachment);

		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		break;
	case VkUtil::PASS_TYPE_COLOR16_OFFLINE:
		attachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
		attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		colorAttachments.push_back(attachment);

		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		break;
	case VkUtil::PASS_TYPE_COLOR16_OFFLINE_NO_CLEAR:
		attachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
		attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		colorAttachments.push_back(attachment);

		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		break;
	case VkUtil::PASS_TYPE_DEPTH_OFFLINE:
		attachment.format = VK_FORMAT_R8G8B8A8_UNORM;
		attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		colorAttachments.push_back(attachment);

		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		attachment.format = VK_FORMAT_D16_UNORM;
		attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		colorAttachments.push_back(attachment);

		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;
		subpass.pDepthStencilAttachment = &depthAttachmentRef;

		break;
	case VkUtil::PASS_TYPE_DEPTH_STENCIL_OFFLINE:
		attachment.format = VK_FORMAT_R8G8B8A8_UNORM;
		attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		colorAttachments.push_back(attachment);

		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		attachment.format = VK_FORMAT_D24_UNORM_S8_UINT;
		attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		colorAttachments.push_back(attachment);

		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;
		subpass.pDepthStencilAttachment = &depthAttachmentRef;

		break;
	case VkUtil::PASS_TYPE_UINT32:
		attachment.format = VK_FORMAT_R32_UINT;
		attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		colorAttachments.push_back(attachment);

		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;
		break;
	case VkUtil::PASS_TYPE_FLOAT:
		attachment.format = VK_FORMAT_R32_SFLOAT;
		attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		colorAttachments.push_back(attachment);

		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;
		break;
	case VkUtil::PASS_TYPE_NONE:

		break;
	}
	VkRenderPassCreateInfo renderPassInfo = {};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = static_cast<uint32_t>(colorAttachments.size());
	renderPassInfo.pAttachments = colorAttachments.data();
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;

	err = vkCreateRenderPass(device, &renderPassInfo, nullptr, renderPass);
	check_vk_result(err);
}

void VkUtil::createFrameBuffer(VkDevice device, VkRenderPass renderPass, const std::vector<VkImageView> &attachments, uint32_t width, uint32_t height, VkFramebuffer *frambuffer)
{
	VkResult err;

	VkFramebufferCreateInfo framebufferInfo = {};
	framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	framebufferInfo.renderPass = renderPass;
	framebufferInfo.attachmentCount = attachments.size();
	framebufferInfo.pAttachments = attachments.data();
	framebufferInfo.width = width;
	framebufferInfo.height = height;
	framebufferInfo.layers = 1;

	err = vkCreateFramebuffer(device, &framebufferInfo, nullptr, frambuffer);
	check_vk_result(err);
}

void VkUtil::fillDescriptorSetLayoutBinding(uint32_t bindingNumber, VkDescriptorType descriptorType, uint32_t amtOfDescriptors, VkShaderStageFlags shaderStages, VkDescriptorSetLayoutBinding *uboLayoutBinding)
{
	uboLayoutBinding->binding = bindingNumber;
	uboLayoutBinding->descriptorType = descriptorType;
	uboLayoutBinding->descriptorCount = amtOfDescriptors;
	uboLayoutBinding->stageFlags = shaderStages;
}

void VkUtil::createDescriptorSetLayout(VkDevice device, const std::vector<VkDescriptorSetLayoutBinding> &bindings, VkDescriptorSetLayout *descriptorSetLayout)
{
	VkResult err;

	VkDescriptorSetLayoutCreateInfo layoutInfo = {};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
	layoutInfo.pBindings = bindings.data();

	err = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, descriptorSetLayout);
	check_vk_result(err);
}

void VkUtil::createDescriptorSetLayoutPartiallyBound(VkDevice device, const std::vector<VkDescriptorSetLayoutBinding> &bindings, const std::vector<bool> &enableValidation, VkDescriptorSetLayout *descriptorSetLayout)
{
	VkResult err;

	std::vector<VkDescriptorBindingFlagsEXT> bindFlag(bindings.size(), VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT);
	for (int i = 0; i < enableValidation.size(); ++i)
	{
		bindFlag[i] = (enableValidation[i]) ? 0 : VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT;
	}

	VkDescriptorSetLayoutBindingFlagsCreateInfoEXT extendedInfo = {};
	extendedInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT;
	extendedInfo.pNext = nullptr;
	extendedInfo.bindingCount = bindings.size();
	extendedInfo.pBindingFlags = bindFlag.data();

	VkDescriptorSetLayoutCreateInfo layoutInfo = {};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
	layoutInfo.pBindings = bindings.data();
	layoutInfo.pNext = &extendedInfo;

	err = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, descriptorSetLayout);
	check_vk_result(err);
}

void VkUtil::destroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout)
{
	vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

void VkUtil::createDescriptorPool(VkDevice device, const std::vector<VkDescriptorPoolSize> &poolSizes, VkDescriptorPool *descriptorPool)
{
	VkResult err;
	VkDescriptorPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
	poolInfo.pPoolSizes = poolSizes.data();

	uint32_t maxSets = 0;
	for (auto pool : poolSizes)
	{
		maxSets += pool.descriptorCount;
	}
	poolInfo.maxSets = maxSets;

	err = vkCreateDescriptorPool(device, &poolInfo, nullptr, descriptorPool);
	check_vk_result(err);
}

void VkUtil::destroyDescriptorPool(VkDevice device, VkDescriptorPool pool)
{
	vkDestroyDescriptorPool(device, pool, nullptr);
}

void VkUtil::createDescriptorSets(VkDevice device, const std::vector<VkDescriptorSetLayout> &layouts, VkDescriptorPool pool, VkDescriptorSet *descriptorSetArray)
{
	VkResult err;

	VkDescriptorSetAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = pool;
	allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
	allocInfo.pSetLayouts = layouts.data();

	err = vkAllocateDescriptorSets(device, &allocInfo, descriptorSetArray);
	check_vk_result(err);
}

void VkUtil::updateDescriptorSet(VkDevice device, VkBuffer buffer, VkDeviceSize size, uint32_t binding, VkDescriptorSet descriptorSet)
{
	VkDescriptorBufferInfo desBufferInfo = {};
	desBufferInfo.buffer = buffer;
	desBufferInfo.offset = 0;
	desBufferInfo.range = size;

	VkWriteDescriptorSet descriptorWrite = {};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = descriptorSet;
	descriptorWrite.dstBinding = binding;
	descriptorWrite.dstArrayElement = 0;
	descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.pBufferInfo = &desBufferInfo;

	vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
}

void VkUtil::updateDescriptorSet(VkDevice device, VkBuffer buffer, VkDeviceSize size, uint32_t binding, uint32_t offset, VkDescriptorSet descriptorSet)
{
	VkDescriptorBufferInfo desBufferInfo = {};
	desBufferInfo.buffer = buffer;
	desBufferInfo.offset = offset;
	desBufferInfo.range = size;

	VkWriteDescriptorSet descriptorWrite = {};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = descriptorSet;
	descriptorWrite.dstBinding = binding;
	descriptorWrite.dstArrayElement = 0;
	descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.pBufferInfo = &desBufferInfo;

	vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
}

void VkUtil::updateDescriptorSet(VkDevice device, VkBuffer buffer, VkDeviceSize size, uint32_t binding, VkDescriptorType descriptorType, VkDescriptorSet descriptorSet)
{
	VkDescriptorBufferInfo desBufferInfo = {};
	desBufferInfo.buffer = buffer;
	desBufferInfo.offset = 0;
	desBufferInfo.range = size;

	VkWriteDescriptorSet descriptorWrite = {};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = descriptorSet;
	descriptorWrite.dstBinding = binding;
	descriptorWrite.dstArrayElement = 0;
	descriptorWrite.descriptorType = descriptorType;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.pBufferInfo = &desBufferInfo;

	vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
}

void VkUtil::updateDescriptorSet(VkDevice device, VkBuffer buffer, VkDeviceSize size, uint32_t binding, uint32_t offset, VkDescriptorType descriptorType, VkDescriptorSet descriptorSet)
{
	VkDescriptorBufferInfo desBufferInfo = {};
	desBufferInfo.buffer = buffer;
	desBufferInfo.offset = offset;
	desBufferInfo.range = size;

	VkWriteDescriptorSet descriptorWrite = {};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = descriptorSet;
	descriptorWrite.dstBinding = binding;
	descriptorWrite.dstArrayElement = 0;
	descriptorWrite.descriptorType = descriptorType;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.pBufferInfo = &desBufferInfo;

	vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
}

void VkUtil::updateImageDescriptorSet(VkDevice device, VkSampler sampler, VkImageView imageView, VkImageLayout imageLayout, uint32_t binding, VkDescriptorSet descriptorSet)
{
	VkDescriptorImageInfo desc_image[1] = {};
	desc_image[0].sampler = sampler;
	desc_image[0].imageView = imageView;
	desc_image[0].imageLayout = imageLayout;
	VkWriteDescriptorSet write_desc[1] = {};
	write_desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write_desc[0].dstSet = descriptorSet;
	write_desc[0].descriptorCount = 1;
	write_desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	write_desc[0].pImageInfo = desc_image;
	write_desc[0].dstBinding = binding;
	vkUpdateDescriptorSets(device, 1, write_desc, 0, NULL);
}

void VkUtil::updateArrayDescriptorSet(VkDevice device, VkBuffer buffer, VkDeviceSize size, uint32_t binding, uint32_t arrayIndex, VkDescriptorType descriptorType, VkDescriptorSet descriptorSet)
{
	VkDescriptorBufferInfo desBufferInfo = {};
	desBufferInfo.buffer = buffer;
	desBufferInfo.offset = 0;
	desBufferInfo.range = size;

	VkWriteDescriptorSet descriptorWrite = {};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = descriptorSet;
	descriptorWrite.dstBinding = binding;
	descriptorWrite.dstArrayElement = arrayIndex;
	descriptorWrite.descriptorType = descriptorType;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.pBufferInfo = &desBufferInfo;

	vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
}

void VkUtil::updateImageArrayDescriptorSet(VkDevice device, std::vector<VkSampler> &sampler, std::vector<VkImageView> &imageViews, std::vector<VkImageLayout> &imageLayouts, uint32_t binding, VkDescriptorSet descriptorSet)
{
	VkDescriptorImageInfo *desc_images = new VkDescriptorImageInfo[imageViews.size()];
	for (int i = 0; i < imageViews.size(); ++i)
	{
		desc_images[i].sampler = sampler[i];
		desc_images[i].imageView = imageViews[i];
		desc_images[i].imageLayout = imageLayouts[i];
	}

	VkWriteDescriptorSet write_desc[1] = {};
	write_desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write_desc[0].dstSet = descriptorSet;
	write_desc[0].descriptorCount = imageViews.size();
	write_desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	write_desc[0].pImageInfo = desc_images;
	write_desc[0].dstBinding = binding;
	vkUpdateDescriptorSets(device, 1, write_desc, 0, NULL);
	delete[] desc_images;
}

void VkUtil::updateStorageImageDescriptorSet(VkDevice device, VkImageView imageView, VkImageLayout imageLayout, uint32_t binding, VkDescriptorSet descriptorSet)
{
	VkDescriptorImageInfo desc_image[1] = {};
	desc_image[0].sampler = nullptr;
	desc_image[0].imageView = imageView;
	desc_image[0].imageLayout = imageLayout;
	VkWriteDescriptorSet write_desc[1] = {};
	write_desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write_desc[0].dstSet = descriptorSet;
	write_desc[0].descriptorCount = 1;
	write_desc[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	write_desc[0].pImageInfo = desc_image;
	write_desc[0].dstBinding = binding;
	vkUpdateDescriptorSets(device, 1, write_desc, 0, NULL);
}

void VkUtil::updateStorageImageArrayDescriptorSet(VkDevice device, std::vector<VkSampler> &sampler, std::vector<VkImageView> &imageViews, std::vector<VkImageLayout> &imageLayouts, uint32_t binding, VkDescriptorSet descriptorSet)
{
	VkDescriptorImageInfo *desc_images = new VkDescriptorImageInfo[imageViews.size()];
	for (int i = 0; i < imageViews.size(); ++i)
	{
		desc_images[i].sampler = sampler[i];
		desc_images[i].imageView = imageViews[i];
		desc_images[i].imageLayout = imageLayouts[i];
	}

	VkWriteDescriptorSet write_desc[1] = {};
	write_desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write_desc[0].dstSet = descriptorSet;
	write_desc[0].descriptorCount = imageViews.size();
	write_desc[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	write_desc[0].pImageInfo = desc_images;
	write_desc[0].dstBinding = binding;
	vkUpdateDescriptorSets(device, 1, write_desc, 0, NULL);
	delete[] desc_images;
}

void VkUtil::updateTexelBufferDescriptorSet(VkDevice device, VkBufferView bufferView, uint32_t binding, VkDescriptorSet descriptorSet)
{
	VkWriteDescriptorSet write_desc[1] = {};
	write_desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write_desc[0].dstSet = descriptorSet;
	write_desc[0].descriptorCount = 1;
	write_desc[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
	write_desc[0].dstBinding = binding;
	write_desc[0].pTexelBufferView = &bufferView;
	vkUpdateDescriptorSets(device, 1, write_desc, 0, NULL);
}

void VkUtil::copyImage(VkCommandBuffer commandBuffer, VkImage srcImage, int32_t width, int32_t height, VkImageLayout srcImageLayout, VkImage dstImage, VkImageLayout dstImageLayout)
{
	VkImageBlit blit = {};
	blit.srcOffsets[0] = {0, 0, 0};
	blit.srcOffsets[1] = {width, height, 1};
	blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	blit.srcSubresource.mipLevel = 0;
	blit.srcSubresource.baseArrayLayer = 0;
	blit.srcSubresource.layerCount = 1;
	blit.dstOffsets[0] = {0, 0, 0};
	blit.dstOffsets[1] = {width, height, 1};
	blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	blit.dstSubresource.mipLevel = 0;
	blit.dstSubresource.baseArrayLayer = 0;
	blit.dstSubresource.layerCount = 1;

	vkCmdBlitImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, 1, &blit, VK_FILTER_LINEAR);
}

void VkUtil::copyBuffer(VkCommandBuffer commandBuffer, VkBuffer src, VkBuffer dst, uint32_t byteSize, uint32_t srcOffset, uint32_t dstOffset)
{
	VkBufferCopy bufCpy{};
	bufCpy.srcOffset = srcOffset;
	bufCpy.dstOffset = dstOffset;
	bufCpy.size = byteSize;
	vkCmdCopyBuffer(commandBuffer, src, dst, 1, &bufCpy);
}

void VkUtil::copyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
{
	VkBufferImageCopy region = {};
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;

	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount = 1;

	region.imageOffset = {0, 0, 0};
	region.imageExtent = {
		width,
		height,
		1};

	vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

void VkUtil::copyBufferTo3dImage(VkCommandBuffer commandBuffer, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, uint32_t depth)
{
	VkBufferImageCopy region = {};
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;

	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount = 1;

	region.imageOffset = {0, 0, 0};
	region.imageExtent = {
		width,
		height,
		depth};

	vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

void VkUtil::copy3dImageToBuffer(VkCommandBuffer commandBuffer, VkBuffer buffer, VkImage image, VkImageLayout imageLayout, uint32_t width, uint32_t height, uint32_t depth)
{
	VkBufferImageCopy region = {};
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;

	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount = 1;

	region.imageOffset = {0, 0, 0};
	region.imageExtent = {
		width,
		height,
		depth};

	vkCmdCopyImageToBuffer(commandBuffer, image, imageLayout, buffer, 1, &region);
}

void VkUtil::transitionImageLayout(VkCommandBuffer commandBuffer, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
{
	VkImageMemoryBarrier barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = newLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;

	VkPipelineStageFlags sourceStage = 0;
	VkPipelineStageFlags destinationStage = 0;

	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
	{
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
	{
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
	{
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;

		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_GENERAL)
	{
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;

		sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		destinationStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
	{
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL)
	{
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;

		sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		destinationStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
	{
		barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		sourceStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
		destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
	{
		barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		sourceStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
		destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
	{
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

		sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
	{
		barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		sourceStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
	{
		barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

		sourceStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL)
	{
		barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;

		sourceStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		destinationStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
	{
		barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

		sourceStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
	{
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
	{
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
	{
		barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		sourceStage = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
		destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
	{
		barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		sourceStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		destinationStage = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL)
	{
		barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;

		sourceStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		destinationStage = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	}
	else
	{
		std::cerr << "Unknown Layout transition from " << oldLayout << " to " << newLayout << "!" << std::endl;
		exit(-1);
	}

	vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void VkUtil::createImage(VkDevice device, uint32_t width, uint32_t height, VkFormat imageFormat, VkImageUsageFlags usageFlags, VkImage *image)
{
	VkResult err;

	// creating the VkImage for the PcPlot
	VkImageCreateInfo imageInfo = {};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.extent.width = width;
	imageInfo.extent.height = height;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.format = imageFormat;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage = usageFlags;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

	err = vkCreateImage(device, &imageInfo, nullptr, image);
	check_vk_result(err);
}

void VkUtil::create1dImage(VkDevice device, uint32_t width, VkFormat imageFormat, VkImageUsageFlags usageFlags, VkImage *image)
{
	VkResult err;

	// creating the VkImage for the PcPlot
	VkImageCreateInfo imageInfo = {};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_1D;
	imageInfo.extent.width = width;
	imageInfo.extent.height = 1;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.format = imageFormat;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage = usageFlags;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

	err = vkCreateImage(device, &imageInfo, nullptr, image);
	check_vk_result(err);
}

void VkUtil::create3dImage(VkDevice device, uint32_t width, uint32_t height, uint32_t depth, VkFormat imageFormat, VkImageUsageFlags usageFlags, VkImage *image)
{
	VkResult err;

	VkImageCreateInfo imageInfo = {};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_3D;
	imageInfo.extent.width = width;
	imageInfo.extent.height = height;
	imageInfo.extent.depth = depth;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.format = imageFormat;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage = usageFlags;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

	err = vkCreateImage(device, &imageInfo, nullptr, image);
	check_vk_result(err);
}

void VkUtil::create3dImage(VkDevice device, uint32_t width, uint32_t height, uint32_t depth, VkFormat imageFormat, VkImageUsageFlags usageFlags, uint32_t mipLevel, VkImage *image)
{
	VkResult err;

	VkImageCreateInfo imageInfo = {};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_3D;
	imageInfo.extent.width = width;
	imageInfo.extent.height = height;
	imageInfo.extent.depth = depth;
	imageInfo.mipLevels = mipLevel;
	imageInfo.arrayLayers = 1;
	imageInfo.format = imageFormat;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage = usageFlags;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

	err = vkCreateImage(device, &imageInfo, nullptr, image);
	check_vk_result(err);
}

void VkUtil::createImageView(VkDevice device, VkImage image, VkFormat imageFormat, uint32_t mipLevelCount, VkImageAspectFlags aspectMask, VkImageView *imageView)
{
	VkResult err;

	VkImageViewCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	createInfo.image = image;
	createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	createInfo.format = imageFormat;
	createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
	createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
	createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
	createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
	createInfo.subresourceRange.aspectMask = aspectMask;
	createInfo.subresourceRange.baseMipLevel = 0;
	createInfo.subresourceRange.levelCount = mipLevelCount;
	createInfo.subresourceRange.baseArrayLayer = 0;
	createInfo.subresourceRange.layerCount = 1;

	err = vkCreateImageView(device, &createInfo, nullptr, imageView);
	check_vk_result(err);
}

void VkUtil::create1dImageView(VkDevice device, VkImage image, VkFormat imageFormat, uint32_t mipLevelCount, VkImageView *imageView)
{
	VkResult err;

	VkImageViewCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	createInfo.image = image;
	createInfo.viewType = VK_IMAGE_VIEW_TYPE_1D;
	createInfo.format = imageFormat;
	createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
	createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
	createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
	createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
	createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	createInfo.subresourceRange.baseMipLevel = 0;
	createInfo.subresourceRange.levelCount = mipLevelCount;
	createInfo.subresourceRange.baseArrayLayer = 0;
	createInfo.subresourceRange.layerCount = 1;

	err = vkCreateImageView(device, &createInfo, nullptr, imageView);
	check_vk_result(err);
}

void VkUtil::create3dImageView(VkDevice device, VkImage image, VkFormat imageFormat, uint32_t mipLevelCount, VkImageView *imageView)
{
	VkResult err;

	VkImageViewCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	createInfo.image = image;
	createInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
	createInfo.format = imageFormat;
	createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
	createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
	createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
	createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
	createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	createInfo.subresourceRange.baseMipLevel = 0;
	createInfo.subresourceRange.levelCount = mipLevelCount;
	createInfo.subresourceRange.baseArrayLayer = 0;
	createInfo.subresourceRange.layerCount = 1;

	err = vkCreateImageView(device, &createInfo, nullptr, imageView);
	check_vk_result(err);
}

void VkUtil::createImageSampler(VkDevice device, VkSamplerAddressMode adressMode, VkFilter filter, uint16_t maxAnisotropy, uint16_t mipLevels, VkSampler *sampler)
{
	VkResult err;

	VkSamplerCreateInfo samplerInfo = {};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = filter;
	samplerInfo.minFilter = filter;
	samplerInfo.addressModeU = adressMode;
	samplerInfo.addressModeV = adressMode;
	samplerInfo.addressModeW = adressMode;
	samplerInfo.anisotropyEnable = maxAnisotropy > 0 ? VK_TRUE : VK_FALSE;
	samplerInfo.maxAnisotropy = maxAnisotropy;
	samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerInfo.mipLodBias = 0.0f;
	samplerInfo.minLod = 0.0f;
	samplerInfo.maxLod = (float)(mipLevels - 1);

	err = vkCreateSampler(device, &samplerInfo, nullptr, sampler);
	check_vk_result(err);
}

void VkUtil::uploadData(VkDevice device, VkDeviceMemory memory, uint32_t offset, uint32_t byteSize, const void *data)
{
	void *d;
	vkMapMemory(device, memory, offset, byteSize, 0, &d);
	memcpy(d, data, byteSize);
	vkUnmapMemory(device, memory);
}

void VkUtil::downloadData(VkDevice device, VkDeviceMemory memory, uint32_t offset, uint32_t byteSize, void *data)
{
	void *d;
	vkMapMemory(device, memory, offset, byteSize, 0, &d);
	memcpy(data, d, byteSize);
	vkUnmapMemory(device, memory);
}

void VkUtil::uploadImageData(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkImage image, VkImageLayout imageLayout, VkFormat imageFormat, uint32_t x, uint32_t y, uint32_t z, void *data, uint32_t byteSize)
{
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingMemory;
	createBuffer(device, byteSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &stagingBuffer);
	VkMemoryRequirements memReq;
	vkGetBufferMemoryRequirements(device, stagingBuffer, &memReq);
	VkMemoryAllocateInfo memAlloc{};
	memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memAlloc.allocationSize = memReq.size;
	memAlloc.memoryTypeIndex = findMemoryType(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	vkAllocateMemory(device, &memAlloc, nullptr, &stagingMemory);
	vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0);

	uploadData(device, stagingMemory, 0, byteSize, data);
	VkCommandBuffer commands;
	createCommandBuffer(device, commandPool, &commands);
	if (imageLayout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		transitionImageLayout(commands, image, imageFormat, imageLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	copyBufferTo3dImage(commands, stagingBuffer, image, x, y, z);
	if (imageLayout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		transitionImageLayout(commands, image, imageFormat, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, imageLayout);
	commitCommandBuffer(queue, commands);
	check_vk_result(vkQueueWaitIdle(queue));

	vkDestroyBuffer(device, stagingBuffer, nullptr);
	vkFreeMemory(device, stagingMemory, nullptr);
	vkFreeCommandBuffers(device, commandPool, 1, &commands);
}

void VkUtil::downloadImageData(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkImage image, VkFormat format, VkImageLayout imageLayout, uint32_t x, uint32_t y, uint32_t z, void *data, uint32_t byteSize)
{
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingMemory;
	createBuffer(device, byteSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &stagingBuffer);
	VkMemoryRequirements memReq;
	vkGetBufferMemoryRequirements(device, stagingBuffer, &memReq);
	VkMemoryAllocateInfo memAlloc{};
	memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memAlloc.allocationSize = memReq.size;
	memAlloc.memoryTypeIndex = findMemoryType(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	vkAllocateMemory(device, &memAlloc, nullptr, &stagingMemory);
	vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0);

	VkCommandBuffer commands;
	createCommandBuffer(device, commandPool, &commands);
	if (imageLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
		transitionImageLayout(commands, image, format, imageLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	copy3dImageToBuffer(commands, stagingBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, x, y, z);
	if (imageLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
		transitionImageLayout(commands, image, format, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, imageLayout);
	commitCommandBuffer(queue, commands);
	check_vk_result(vkQueueWaitIdle(queue));
	downloadData(device, stagingMemory, 0, byteSize, data);

	vkDestroyBuffer(device, stagingBuffer, nullptr);
	vkFreeMemory(device, stagingMemory, nullptr);
	vkFreeCommandBuffers(device, commandPool, 1, &commands);
}

namespace VkUtil
{
	void addImageToAllocInfo(VkDevice device, VkImage image, VkMemoryAllocateInfo& allocInfo) 
	{
		VkMemoryRequirements memReq{};
		vkGetImageMemoryRequirements(device, image, &memReq);
		allocInfo.allocationSize += memReq.size;
		allocInfo.memoryTypeIndex |= memReq.memoryTypeBits;
	}

	void transitionImageLayoutDirect(VkDevice device, VkCommandPool pool, VkQueue queue, const std::vector<VkImage>& images, const std::vector<VkFormat>& formats, const std::vector<VkImageLayout>& oldLayouts, const std::vector<VkImageLayout>& newLayouts) 
	{
		VkCommandBuffer commands;
		createCommandBuffer(device, pool, &commands);
		for(int i = 0; i < images.size(); ++i){
			transitionImageLayout(commands, images[i], formats[i], oldLayouts[i], newLayouts[i]);
		}
		commitCommandBuffer(queue, commands);
		VkResult res = vkQueueWaitIdle(queue); check_vk_result(res);
	}
}