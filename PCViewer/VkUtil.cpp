#include "VkUtil.h"

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

	for (uint32_t i = 1; i < mipLevels; i++) {
		use_barrier[0].srcAccessMask = oldAccess;
		use_barrier[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		use_barrier[0].oldLayout = oldLayout;
		use_barrier[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		use_barrier[0].subresourceRange.baseMipLevel = i - 1;

		vkCmdPipelineBarrier(commandBuffer, oldPipelineStage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, use_barrier);

		VkImageBlit blit = {};
		blit.srcOffsets[0] = { 0, 0, 0 };
		blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
		blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		blit.srcSubresource.mipLevel = i - 1;
		blit.srcSubresource.baseArrayLayer = 0;
		blit.srcSubresource.layerCount = 1;
		blit.dstOffsets[0] = { 0, 0, 0 };
		blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
		blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		blit.dstSubresource.mipLevel = i;
		blit.dstSubresource.baseArrayLayer = 0;
		blit.dstSubresource.layerCount = 1;

		vkCmdBlitImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1, &blit, VK_FILTER_LINEAR);

		use_barrier[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		use_barrier[0].dstAccessMask = oldAccess;
		use_barrier[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		use_barrier[0].newLayout = oldLayout;

		vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, oldPipelineStage, 0, 0, NULL, 0, NULL, 1, use_barrier);


		if (mipWidth > 1) mipWidth /= 2;
		if (mipHeight > 1) mipHeight /= 2;
	}
}

void VkUtil::createCommandBuffer(VkDevice device, VkCommandPool commandPool, VkCommandBuffer* commandBuffer)
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

void VkUtil::commitCommandBuffer(VkDevice device, VkQueue queue, VkCommandBuffer commandBuffer)
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

void VkUtil::createPipeline(VkDevice device, const std::string& vertexShaderPath, const std::string& fragmentShaderPath, VkPipeline* pipeline)
{
	VkResult err;

	auto vertShader = PCUtil::readByteFile(vertexShaderPath);
	auto fragShader = PCUtil::readByteFile(fragmentShaderPath);

	VkShaderModule vertShaderModule = createShaderModule(device,vertShader);
	VkShaderModule fragShaderModule = createShaderModule(device,fragShader);

	VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
	vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vertShaderStageInfo.module = vertShaderModule;
	vertShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
	fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragShaderStageInfo.module = fragShaderModule;
	fragShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo,fragShaderStageInfo };

	VkVertexInputBindingDescription bindingDescripiton = {};		//describes how big the vertex data is and how to read the data
	bindingDescripiton.binding = 0;
	bindingDescripiton.stride = sizeof(Vertex);
	bindingDescripiton.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputAttributeDescription attributeDescription = {};	//describes the attribute of the vertex. If more than 1 attribute is used this has to be an array
	attributeDescription.binding = 0;
	attributeDescription.location = 0;
	attributeDescription.format = VK_FORMAT_R32_SFLOAT;
	attributeDescription.offset = offsetof(Vertex, y);

	VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.pVertexBindingDescriptions = &bindingDescripiton;
	vertexInputInfo.vertexAttributeDescriptionCount = 1;
	vertexInputInfo.pVertexAttributeDescriptions = &attributeDescription;

	VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	VkViewport viewport = {};					//description for our viewport for transformation operation after rasterization
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)g_PcPlotWidth;
	viewport.height = (float)g_PcPlotHeight;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	VkRect2D scissor = {};						//description for cutting the rendered result if wanted
	scissor.offset = { 0, 0 };
	scissor.extent = { g_PcPlotWidth,g_PcPlotHeight };

	VkPipelineViewportStateCreateInfo viewportState = {};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;

	VkPipelineRasterizationStateCreateInfo rasterizer = {};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizer.lineWidth = 1.0f;
	rasterizer.cullMode = VK_CULL_MODE_NONE;
	rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
	rasterizer.depthBiasEnable = VK_FALSE;
	rasterizer.depthBiasClamp = 0.0f;
	rasterizer.depthBiasConstantFactor = 0.0f;
	rasterizer.depthBiasSlopeFactor = 0.0f;

	VkPipelineMultisampleStateCreateInfo multisampling = {};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	multisampling.minSampleShading = 1.0f;
	multisampling.pSampleMask = nullptr;
	multisampling.alphaToCoverageEnable = VK_FALSE;
	multisampling.alphaToOneEnable = VK_FALSE;

	VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_TRUE;
	colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
	colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

	VkPipelineColorBlendStateCreateInfo colorBlending = {};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;
	colorBlending.blendConstants[0] = 0.0f;
	colorBlending.blendConstants[1] = 0.0f;
	colorBlending.blendConstants[2] = 0.0f;
	colorBlending.blendConstants[3] = 0.0f;

	VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_LINE_WIDTH };

	VkPipelineDynamicStateCreateInfo dynamicState = {};			//enables change of the linewidth at runtime
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = 1;
	dynamicState.pDynamicStates = dynamicStates;

	VkDescriptorSetLayoutBinding uboLayoutBinding = {};
	uboLayoutBinding.binding = 0;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uboLayoutBinding.descriptorCount = 1;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_ALL;

	VkDescriptorSetLayoutCreateInfo layoutInfo = {};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 1;
	layoutInfo.pBindings = &uboLayoutBinding;

	err = vkCreateDescriptorSetLayout(g_Device, &layoutInfo, nullptr, &g_PcPlotDescriptorLayout);
	check_vk_result(err);

	VkDescriptorPoolSize poolSize = {};
	poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	poolSize.descriptorCount = 1;

	VkDescriptorPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = 1;
	poolInfo.pPoolSizes = &poolSize;
	poolInfo.maxSets = 1;

	err = vkCreateDescriptorPool(g_Device, &poolInfo, nullptr, &g_PcPlotDescriptorPool);
	check_vk_result(err);

	VkDescriptorSetAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = g_PcPlotDescriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &g_PcPlotDescriptorLayout;

	err = vkAllocateDescriptorSets(g_Device, &allocInfo, &g_PcPlotDescriptorSet);
	check_vk_result(err);

	VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &g_PcPlotDescriptorLayout;
	pipelineLayoutInfo.pushConstantRangeCount = 0;
	pipelineLayoutInfo.pPushConstantRanges = nullptr;

	err = vkCreatePipelineLayout(g_Device, &pipelineLayoutInfo, nullptr, &g_PcPlotPipelineLayout);
	check_vk_result(err);

	VkGraphicsPipelineCreateInfo pipelineInfo = {};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = shaderStages;
	pipelineInfo.pVertexInputState = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pDepthStencilState = nullptr;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDynamicState = &dynamicState;
	pipelineInfo.layout = g_PcPlotPipelineLayout;
	pipelineInfo.renderPass = g_PcPlotRenderPass;
	pipelineInfo.subpass = 0;

	err = vkCreateGraphicsPipelines(g_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &g_PcPlotPipeline);
	check_vk_result(err);

	vkDestroyShaderModule(g_Device, fragShaderModule, nullptr);
	vkDestroyShaderModule(g_Device, vertShaderModule, nullptr);
}

VkShaderModule VkUtil::createShaderModule(VkDevice device, const std::vector<char>& byteArr)
{
	VkShaderModuleCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.pCode = reinterpret_cast<const uint32_t*>(byteArr.data());
	createInfo.codeSize = byteArr.size();

	VkShaderModule shaderModule;
	VkResult err = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
	check_vk_result(err);

	return shaderModule;
}
