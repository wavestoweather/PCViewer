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

void VkUtil::commitCommandBuffer( VkQueue queue, VkCommandBuffer commandBuffer)
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

void VkUtil::createPipeline(VkDevice device, VkPipelineVertexInputStateCreateInfo* vertexInfo, float frameWidth, float frameHight, const std::vector<VkDynamicState>& dynamicStates, VkShaderModule* shaderModules, VkPrimitiveTopology topology, VkPipelineRasterizationStateCreateInfo* rasterizerInfo, VkPipelineMultisampleStateCreateInfo* multisamplingInfo, VkPipelineDepthStencilStateCreateInfo* depthStencilInfo, BlendInfo* blendInfo, const std::vector<VkDescriptorSetLayout>& descriptorSetLayouts, VkRenderPass* renderPass, VkPipelineLayout* pipelineLayout, VkPipeline* pipeline)
{
	VkResult err;

	std::vector<VkPipelineShaderStageCreateInfo> shaderStages;

	if (shaderModules[0]) {
		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = shaderModules[0];
		vertShaderStageInfo.pName = "main";
		shaderStages.push_back(vertShaderStageInfo);
	}

	if (shaderModules[1]) {
		VkPipelineShaderStageCreateInfo tessControlShaderStageInfo = {};
		tessControlShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		tessControlShaderStageInfo.stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
		tessControlShaderStageInfo.module = shaderModules[1];
		tessControlShaderStageInfo.pName = "main";
		shaderStages.push_back(tessControlShaderStageInfo);
	}

	if (shaderModules[2]) {
		VkPipelineShaderStageCreateInfo tessEvaluationShaderStageInfo = {};
		tessEvaluationShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		tessEvaluationShaderStageInfo.stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
		tessEvaluationShaderStageInfo.module = shaderModules[2];
		tessEvaluationShaderStageInfo.pName = "main";
		shaderStages.push_back(tessEvaluationShaderStageInfo);
	}

	if (shaderModules[3]) {
		VkPipelineShaderStageCreateInfo geoShaderStageInfo = {};
		geoShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		geoShaderStageInfo.stage = VK_SHADER_STAGE_GEOMETRY_BIT;
		geoShaderStageInfo.module = shaderModules[3];
		geoShaderStageInfo.pName = "main";
		shaderStages.push_back(geoShaderStageInfo);
	}

	if (shaderModules[4]) {
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
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	VkViewport viewport = {};					//description for our viewport for transformation operation after rasterization
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = frameWidth;
	viewport.height = frameHight;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	VkRect2D scissor = {};						//description for cutting the rendered result if wanted
	scissor.offset = { 0, 0 };
	scissor.extent = { (uint32_t)frameWidth,(uint32_t)frameHight };

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

	VkPipelineDynamicStateCreateInfo dynamicState = {};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = dynamicStates.size();
	dynamicState.pDynamicStates = dynamicStates.data();

	VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = descriptorSetLayouts.size();
	pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
	pipelineLayoutInfo.pushConstantRangeCount = 0;
	pipelineLayoutInfo.pPushConstantRanges = nullptr;

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

void VkUtil::destroyPipeline(VkDevice device,VkPipeline pipeline) {
	vkDestroyPipeline(device, pipeline, nullptr);
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

void VkUtil::createPcPlotRenderPass(VkDevice device, VkUtil::PassType passType,VkRenderPass* renderPass) {
	VkResult err;

	std::vector<VkAttachmentDescription> colorAttachments;
	VkAttachmentReference colorAttachmentRef = {};
	VkAttachmentReference depthAttachmentRef = {};
	VkSubpassDescription subpass = {};
	VkAttachmentDescription attachment = {};

	switch (passType) {
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

void VkUtil::fillDescriptorSetLayoutBinding(uint32_t bindingNumber, VkDescriptorType descriptorType, uint32_t amtOfDescriptors, VkShaderStageFlags shaderStages, VkDescriptorSetLayoutBinding* uboLayoutBinding)
{
	uboLayoutBinding->binding = bindingNumber;
	uboLayoutBinding->descriptorType = descriptorType;
	uboLayoutBinding->descriptorCount = amtOfDescriptors;
	uboLayoutBinding->stageFlags = shaderStages;
}

void VkUtil::createDescriptorSetLayout(VkDevice device,const std::vector<VkDescriptorSetLayoutBinding>& bindings, VkDescriptorSetLayout* descriptorSetLayout) {
	VkResult err;

	VkDescriptorSetLayoutCreateInfo layoutInfo = {};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
	layoutInfo.pBindings = bindings.data();

	err = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, descriptorSetLayout);
	check_vk_result(err);
}

void VkUtil::destroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout) {
	vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

void VkUtil::createDescriptorPool(VkDevice device, const std::vector<VkDescriptorPoolSize>& poolSizes, VkDescriptorPool* descriptorPool) {
	VkResult err;
	VkDescriptorPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
	poolInfo.pPoolSizes = poolSizes.data();

	uint32_t maxSets = 0;
	for (auto pool : poolSizes) {
		maxSets += pool.descriptorCount;
	}
	poolInfo.maxSets = maxSets;

	err = vkCreateDescriptorPool(device, &poolInfo, nullptr, descriptorPool);
	check_vk_result(err);
}

void VkUtil::destroyDescriptorPool(VkDevice device, VkDescriptorPool pool) {
	vkDestroyDescriptorPool(device, pool, nullptr);
}

void VkUtil::createDescriptorSets(VkDevice device,const std::vector<VkDescriptorSetLayout>& layouts, VkDescriptorPool pool, VkDescriptorSet* descriptorSetArray) {
	VkResult err;

	VkDescriptorSetAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = pool;
	allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
	allocInfo.pSetLayouts = layouts.data();

	err = vkAllocateDescriptorSets(device, &allocInfo, descriptorSetArray);
	check_vk_result(err);
}

void VkUtil::updateDescriptorSet(VkDevice device, VkBuffer buffer, uint32_t size, VkDescriptorSet descriptorSet)
{
	VkDescriptorBufferInfo desBufferInfo = {};
	desBufferInfo.buffer = buffer;
	desBufferInfo.offset = 0;
	desBufferInfo.range = size;

	VkWriteDescriptorSet descriptorWrite = {};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = descriptorSet;
	descriptorWrite.dstBinding = 0;
	descriptorWrite.dstArrayElement = 0;
	descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.pBufferInfo = &desBufferInfo;

	vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
}
