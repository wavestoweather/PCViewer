#include "View3d.h"

char View3d::vertPath[]= "shader/3dVert.spv";
char View3d::fragPath[]= "shader/3dFrag.spv";

View3d::View3d(uint32_t height, uint32_t width, VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool)
{
	imageHeight = 0;
	imageWidth = 0;
	image3dHeight = 0;
	image3dWidth = 0;
	image3dDepth = 0;
	this->device = device;
	this->physicalDevice = physicalDevice;
	this->commandPool = commandPool;
	this->queue = queue;
	this->descriptorPool = descriptorPool;
	imageMemory = VK_NULL_HANDLE;
	image =	VK_NULL_HANDLE;
	imageView =	VK_NULL_HANDLE;
	sampler = VK_NULL_HANDLE;
	image3dMemory = VK_NULL_HANDLE;
	image3d = VK_NULL_HANDLE;
	image3dView = VK_NULL_HANDLE;
	image3dSampler = VK_NULL_HANDLE;
	descriptorSetLayout = VK_NULL_HANDLE,
	descriptorSet = VK_NULL_HANDLE;
	pipeline = VK_NULL_HANDLE;
	renderPass = VK_NULL_HANDLE; 
	pipelineLayout = VK_NULL_HANDLE;
	constantMemory = VK_NULL_HANDLE;
	vertexBuffer = VK_NULL_HANDLE;
	indexBuffer = VK_NULL_HANDLE;
	uniformBuffer = VK_NULL_HANDLE;
	commandBuffer = VK_NULL_HANDLE;
	prepareImageCommand = VK_NULL_HANDLE;
	frameBuffer = VK_NULL_HANDLE;
	imageDescriptorSet = VK_NULL_HANDLE;

	camPos = glm::vec3(2, 2, 2);

	//setting up graphic resources
	createBuffer();
	createPipeline();
	createDescriptorSets();
	resize(width, height);
}

View3d::~View3d()
{
	if (imageMemory) {
		vkFreeMemory(device, imageMemory, nullptr);
	}
	if (image) {
		vkDestroyImage(device, image, nullptr);
	}
	if (imageView) {
		vkDestroyImageView(device, imageView, nullptr);
	}
	if (sampler) {
		vkDestroySampler(device, sampler, nullptr);
	}
	if (image3dMemory) {
		vkFreeMemory(device, image3dMemory, nullptr);
	}
	if (image3d) {
		vkDestroyImage(device, image, nullptr);
	}
	if (image3dView) {
		vkDestroyImageView(device, imageView, nullptr);
	}
	if (frameBuffer) {
		vkDestroyFramebuffer(device, frameBuffer, nullptr);
	}
	if (image3dSampler) {
		vkDestroySampler(device, sampler, nullptr);
	}
	if (descriptorSetLayout) {
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
	}
	if (pipeline) {
		vkDestroyPipeline(device, pipeline, nullptr);
	}
	if (pipelineLayout) {
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
	}
	if (renderPass) {
		vkDestroyRenderPass(device, renderPass, nullptr);
	}
	if (constantMemory) {
		vkFreeMemory(device, constantMemory, nullptr);
	}
	if (vertexBuffer) {
		vkDestroyBuffer(device, vertexBuffer, nullptr);
	}
	if (indexBuffer) {
		vkDestroyBuffer(device, indexBuffer, nullptr);
	}
	if (uniformBuffer) {
		vkDestroyBuffer(device, uniformBuffer, nullptr);
	}
}

void View3d::resize(uint32_t width, uint32_t height)
{
	if (imageWidth == width && imageHeight == height) {
		return;
	}
	imageWidth = width;
	imageHeight = height;
	
	if (image) {
		vkDestroyImage(device, image, nullptr);
	}
	if (imageView) {
		vkDestroyImageView(device, imageView, nullptr);
	}
	if (frameBuffer) {
		vkDestroyFramebuffer(device, frameBuffer, nullptr);
	}
	if (sampler) {
		vkDestroySampler(device, sampler, nullptr);
	}

	createImageResources();

	//transforming the image to the right format
	createPrepareImageCommandBuffer();
	VkResult err;

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.signalSemaphoreCount = 0;
	submitInfo.waitSemaphoreCount = 0;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &prepareImageCommand;

	err = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	check_vk_result(err);

	//udating the command buffer
	VkUtil::createCommandBuffer(device, commandPool, &commandBuffer);
	VkUtil::transitionImageLayout(commandBuffer, image, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	std::vector<VkClearValue> clearValues;
	clearValues.push_back({ 0,0,0,1 });
	VkUtil::beginRenderPass(commandBuffer, clearValues, renderPass, frameBuffer, { imageWidth,imageHeight });

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, offsets);
	vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

	VkViewport viewport = {};					//description for our viewport for transformation operation after rasterization
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = imageWidth;
	viewport.height = imageHeight;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

	VkRect2D scissor = {};						//description for cutting the rendered result if wanted
	scissor.offset = { 0, 0 };
	scissor.extent = { (uint32_t)imageWidth,(uint32_t)imageHeight };
	vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

	vkCmdDrawIndexed(commandBuffer, 3 * 6 * 2, 1, 0, 0, 0);

	vkCmdEndRenderPass(commandBuffer);

	VkUtil::transitionImageLayout(commandBuffer, image, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	err = vkEndCommandBuffer(commandBuffer);
	check_vk_result(err);

	err = vkDeviceWaitIdle(device);
	check_vk_result(err);

	render();
}

void View3d::update3dImage(uint32_t width, uint32_t height, uint32_t depth, float* data)
{
	if ((width != image3dWidth) || (height != image3dHeight) || (depth != image3dDepth)) {
		image3dWidth = width;
		image3dHeight = height;
		image3dDepth = depth;

		//destroying old resources
		if (image3dMemory) {
			vkFreeMemory(device, image3dMemory, nullptr);
		}
		if (image3d) {
			vkDestroyImage(device, image3d, nullptr);
		}
		if (image3dView) {
			vkDestroyImageView(device, image3dView, nullptr);
		}
		if (image3dSampler) {
			vkDestroySampler(device, image3dSampler, nullptr);
		}

		VkUtil::create3dImage(device, image3dWidth, image3dHeight, image3dDepth, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT, &image3d);
		VkUtil::create3dImageView(device, image3d, VK_FORMAT_R16G16B16A16_UNORM, 1, &image3dView);

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image3d, &memRequirements);

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memRequirements.memoryTypeBits, 0);
	}

	//uploading the data with a staging buffer

}

void View3d::updateCameraPos(float* mouseMovement)
{
	//rotation matrix for height adjustment
	glm::mat4 vertical = glm::rotate(glm::mat4(1.0f), mouseMovement[1] * VERTICALPANSPEED, glm::normalize(glm::cross(camPos, glm::vec3(0, 1, 0))));
	//rotation matrix for horizontal adjustment
	glm::mat4 horizontal = glm::rotate(glm::mat4(1.0f), mouseMovement[0] * HORIZONTALPANSPEED, glm::vec3(0, 1, 0));
	camPos = horizontal * vertical * glm::vec4(camPos,1);

	//adding zooming
	glm::vec3 zoomDir = -camPos;
	camPos += ZOOMSPEED * zoomDir * mouseMovement[2];
}

void View3d::render()
{
	VkResult err;

	//uploading the uniformBuffer
	UniformBuffer ubo;
	ubo.mvp = glm::perspective(glm::radians(45.0f), (float)imageWidth / (float)imageHeight, 0.1f, 100.0f);;
	ubo.mvp[1][1] *= -1;
	glm::mat4 look = glm::lookAt(camPos, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
	ubo.mvp = ubo.mvp * look;
	ubo.camPos = camPos;
	void* d;
	vkMapMemory(device, constantMemory, uniformBufferOffset, sizeof(UniformBuffer), 0, &d);
	memcpy(d, &ubo, sizeof(UniformBuffer));
	vkUnmapMemory(device, constantMemory);


	//submitting the command buffer
	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.signalSemaphoreCount = 0;
	submitInfo.waitSemaphoreCount = 0;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	err = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	check_vk_result(err);
}

void View3d::setImageDescriptorSet(VkDescriptorSet descriptor)
{
	imageDescriptorSet = descriptor;
}

VkDescriptorSet View3d::getImageDescriptorSet()
{
	return imageDescriptorSet;
}

VkSampler View3d::getImageSampler()
{
	return sampler;
}

VkImageView View3d::getImageView()
{
	return imageView;
}

void View3d::createPrepareImageCommandBuffer()
{
	VkUtil::createCommandBuffer(device, commandPool, &prepareImageCommand);
	VkUtil::transitionImageLayout(prepareImageCommand, image, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	vkEndCommandBuffer(prepareImageCommand);
}

void View3d::createImageResources()
{
	VkResult err;
	
	VkUtil::createImage(device, imageWidth, imageHeight, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT, &image);

	VkMemoryRequirements memReq = {};
	vkGetImageMemoryRequirements(device, image, &memReq);
	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memReq.size;
	allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, 0);
	err = vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory);
	check_vk_result(err);

	vkBindImageMemory(device, image, imageMemory, 0);

	VkUtil::createImageView(device, image, VK_FORMAT_R16G16B16A16_UNORM, 1, &imageView);
	std::vector<VkImageView> views;
	views.push_back(imageView);
	VkUtil::createFrameBuffer(device, renderPass, views, imageWidth, imageHeight, &frameBuffer);
	VkUtil::createImageSampler(device, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 16, 1, &sampler);
}

void View3d::createBuffer()
{
	VkUtil::createBuffer(device, 8 * sizeof(glm::vec3), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vertexBuffer);
	VkUtil::createBuffer(device, 12 * 3 * sizeof(uint16_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &indexBuffer);
	VkUtil::createBuffer(device, sizeof(UniformBuffer), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &uniformBuffer);

	VkResult err;

	VkMemoryRequirements memReq;
	vkGetBufferMemoryRequirements(device, vertexBuffer, &memReq);

	uint32_t memoryTypeBits = memReq.memoryTypeBits;
	VkMemoryAllocateInfo memAlloc = {};
	memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memAlloc.allocationSize = memReq.size;
	uint32_t indexBufferOffset = memReq.size;

	vkGetBufferMemoryRequirements(device, indexBuffer, &memReq);
	memAlloc.allocationSize += memReq.size;
	memoryTypeBits |= memReq.memoryTypeBits;
	uniformBufferOffset = memAlloc.allocationSize;

	vkGetBufferMemoryRequirements(device, uniformBuffer, &memReq);
	memAlloc.allocationSize += memReq.size;
	memoryTypeBits |= memReq.memoryTypeBits;

	memAlloc.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
	err = vkAllocateMemory(device, &memAlloc, nullptr, &constantMemory);
	check_vk_result(err);

	vkBindBufferMemory(device, vertexBuffer, constantMemory, 0);
	vkBindBufferMemory(device, indexBuffer, constantMemory, indexBufferOffset);
	vkBindBufferMemory(device, uniformBuffer, constantMemory, uniformBufferOffset);

	//creating the data for the buffers
	glm::vec3 vB[8];
	for (int x = 0; x < 2; x++) {
		for (int y = 0; y < 2; y++) {
			for (int z = 0; z < 2; z++) {
				vB[(x << 2) | (y << 1) | z] = glm::vec3(x - .5f, y - .5f, z - .5f);
			}
		}
	}

	uint16_t iB[12 * 3] = { 0,1,2, 1,3,2, 0,4,1, 1,4,5, 0,2,4, 2,6,4, 2,3,6, 3,7,6, 4,6,5, 5,6,7, 1,5,7, 1,7,3 };

	void* d;
	vkMapMemory(device, constantMemory, 0, sizeof(vB), 0, &d);
	memcpy(d, vB, sizeof(vB));
	vkUnmapMemory(device, constantMemory);
	vkMapMemory(device, constantMemory, indexBufferOffset, sizeof(iB), 0, &d);
	memcpy(d, iB, sizeof(iB));
	vkUnmapMemory(device, constantMemory);
}

void View3d::createPipeline()
{
	VkShaderModule shaderModules[5] = {};
	//the vertex shader for the pipeline
	std::vector<char> vertexBytes = PCUtil::readByteFile(vertPath);
	shaderModules[0] = VkUtil::createShaderModule(device, vertexBytes);
	//the fragment shader for the pipeline
	std::vector<char> fragmentBytes = PCUtil::readByteFile(fragPath);
	shaderModules[4] = VkUtil::createShaderModule(device, fragmentBytes);


	//Description for the incoming vertex attributes
	VkVertexInputBindingDescription bindingDescripiton = {};		//describes how big the vertex data is and how to read the data
	bindingDescripiton.binding = 0;
	bindingDescripiton.stride = sizeof(glm::vec3);
	bindingDescripiton.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputAttributeDescription attributeDescription = {};	//describes the attribute of the vertex. If more than 1 attribute is used this has to be an array
	attributeDescription.binding = 0;
	attributeDescription.location = 0;
	attributeDescription.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	attributeDescription.offset = offsetof(glm::vec3, x);

	VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.pVertexBindingDescriptions = &bindingDescripiton;
	vertexInputInfo.vertexAttributeDescriptionCount = 1;
	vertexInputInfo.pVertexAttributeDescriptions = &attributeDescription;

	//vector with the dynamic states
	std::vector<VkDynamicState> dynamicStates;
	dynamicStates.push_back(VK_DYNAMIC_STATE_VIEWPORT);
	dynamicStates.push_back(VK_DYNAMIC_STATE_SCISSOR);

	//Rasterizer Info
	VkPipelineRasterizationStateCreateInfo rasterizer = {};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizer.lineWidth = 1.0f;
	rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT;
	rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizer.depthBiasEnable = VK_FALSE;
	rasterizer.depthBiasClamp = 0.0f;
	rasterizer.depthBiasConstantFactor = 0.0f;
	rasterizer.depthBiasSlopeFactor = 0.0f;

	//multisampling info
	VkPipelineMultisampleStateCreateInfo multisampling = {};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	multisampling.minSampleShading = 1.0f;
	multisampling.pSampleMask = nullptr;
	multisampling.alphaToCoverageEnable = VK_FALSE;
	multisampling.alphaToOneEnable = VK_FALSE;

	//blendInfo
	VkUtil::BlendInfo blendInfo;

	VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_FALSE;
	colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
	colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
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

	blendInfo.blendAttachment = colorBlendAttachment;
	blendInfo.createInfo = colorBlending;

	//creating the descriptor set layout
	VkDescriptorSetLayoutBinding uboLayoutBinding = {};
	uboLayoutBinding.binding = 0;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uboLayoutBinding.descriptorCount = 1;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_ALL;
	std::vector<VkDescriptorSetLayoutBinding> bindings;
	bindings.push_back(uboLayoutBinding);

	VkUtil::createDescriptorSetLayout(device, bindings, &descriptorSetLayout);
	std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
	descriptorSetLayouts.push_back(descriptorSetLayout);

	VkUtil::createPcPlotRenderPass(device, VkUtil::PASS_TYPE_COLOR16_OFFLINE, &renderPass);

	VkUtil::createPipeline(device, &vertexInputInfo, imageWidth, imageHeight, dynamicStates, shaderModules, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, &rasterizer, &multisampling, nullptr, &blendInfo, descriptorSetLayouts, &renderPass, &pipelineLayout, &pipeline);
}

void View3d::createDescriptorSets()
{
	std::vector<VkDescriptorSetLayout> layouts;
	layouts.push_back(descriptorSetLayout);
	VkUtil::createDescriptorSets(device, layouts, descriptorPool, &descriptorSet);

	VkUtil::updateDescriptorSet(device, uniformBuffer, sizeof(UniformBuffer), 0, descriptorSet);
}