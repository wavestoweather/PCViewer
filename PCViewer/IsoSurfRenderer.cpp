#include "IsoSurfRenderer.h"

char IsoSurfRenderer::vertPath[]= "shader/isoSurfVert.spv";
char IsoSurfRenderer::fragPath[]= "shader/isoSurfFrag.spv";
char IsoSurfRenderer::computePath[] = "shader/isoSurfComp.spv";

IsoSurfRenderer::IsoSurfRenderer(uint32_t height, uint32_t width, VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool)
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
	for (int i = 0; i < AMTOF3DTEXTURES; ++i) {
		image3d[i] = VK_NULL_HANDLE;
		image3dView[i] = VK_NULL_HANDLE;
	}
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
	computePipeline = VK_NULL_HANDLE;
	computePipelineLayout = VK_NULL_HANDLE;
	computeDescriptorSetLayout = VK_NULL_HANDLE;


	camPos = glm::vec3(2, 2, 2);
	lightDir = glm::vec3(-1, -1, -1);

	//setting up graphic resources
	
	createBuffer();
	createPipeline();
	createDescriptorSets();
	resize(width, height);

	
	const int w = 100, h = 5, de = 1;
	glm::vec4 d[w * h * de] = {};
	d[0] = glm::vec4(1, 0, 0, 1);
	d[1] = glm::vec4(1, 0, 0, 1);
	d[2] = glm::vec4(1, 0, 0, 1);
	d[3] = glm::vec4(1, 0, 0, 1);
	d[8] = glm::vec4(0, 1, 0, .5f);
	d[26] = glm::vec4(0, 0, 1, .1f);
	/*for (int i = 1; i < 27; i+=3) {
		d[4 * i] = i / 27.0f;
		d[4 * i + 1] = 1 - (i / 27.0f);
		d[4 * i + 2] = 0;
		d[4 * i + 3] = .1f;
	}*/
	//update3dImage(w, h, de, (float*)d);
	resizeBox(1.5f, 1, 1.5f);
}

IsoSurfRenderer::~IsoSurfRenderer()
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
	for (int i = 0; i < AMTOF3DTEXTURES; ++i) {
		if (image3d[i]) {
			vkDestroyImage(device, image3d[i], nullptr);
		}
		if (image3dView[i]) {
			vkDestroyImageView(device, image3dView[i], nullptr);
		}
	}
	if (image3dSampler) {
		vkDestroySampler(device, image3dSampler, nullptr);
	}
	if (frameBuffer) {
		vkDestroyFramebuffer(device, frameBuffer, nullptr);
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
	if (computePipeline) {
		vkDestroyPipeline(device, computePipeline, nullptr);
	}
	if (computePipelineLayout) {
		vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);
	}
	if (computeDescriptorSetLayout) {
		vkDestroyDescriptorSetLayout(device, computeDescriptorSetLayout, nullptr);
	}
}

void IsoSurfRenderer::resize(uint32_t width, uint32_t height)
{
	if (imageWidth == width && imageHeight == height) {
		return;
	}
	imageWidth = width;
	imageHeight = height;
	
	check_vk_result(vkDeviceWaitIdle(device));

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

	if (image3dSampler) {
		updateCommandBuffer();
	}
}

void IsoSurfRenderer::resizeBox(float width, float height, float depth)
{
	boxWidth = width;
	boxHeight = height;
	boxDepth = depth;
	render();
}

void IsoSurfRenderer::update3dDensities(uint32_t width, uint32_t height, uint32_t depth, uint32_t amtOfAttributes, std::vector<uint32_t>& densityAttributes, std::vector<std::pair<float, float>>& densityAttributesMinMax, uint32_t amtOfIndices, VkBuffer indices, VkBuffer data)
{
	VkResult err;
	uint32_t required3dImages = amtOfIndices / 4 + ((amtOfIndices & 3) == 0) ? 0 : 1;

	//safety check if the amoutn fo indices is supported
	if (required3dImages <= AMTOF3DTEXTURES) {
		std::cout << "Too much attributes for Iso surface rendering. The maximum of attributes is " << AMTOF3DTEXTURES * 4 << " Attributes." << std::endl;
		return;
	}

	image3dWidth = width;
	image3dHeight = height;
	image3dDepth = depth;

	//destroying old resources
	if (image3dMemory) {
		vkFreeMemory(device, image3dMemory, nullptr);
	}
	for (int i = 0; i < AMTOF3DTEXTURES; ++i) {
		if (image3d[i]) {
			vkDestroyImage(device, image3d[i], nullptr);
			image3d[i] = VK_NULL_HANDLE;
		}
		if (image3dView[i]) {
			vkDestroyImageView(device, image3dView[i], nullptr);
			image3dView[i] = VK_NULL_HANDLE;
		}
	}
	if (!image3dSampler) {
		VkUtil::createImageSampler(device,VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,VK_FILTER_NEAREST,16,1,&image3dSampler);
	}

	//creating new resources
	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	uint32_t memoryTypeBits = 0;
	VkMemoryRequirements memRequirements;
	for (int i = 0; i < required3dImages; ++i) {
		image3dOffsets[i] = allocInfo.allocationSize;
		VkUtil::create3dImage(device, image3dWidth, image3dHeight, image3dDepth, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, &image3d[i]);

		vkGetImageMemoryRequirements(device, image3d[i], &memRequirements);

		allocInfo.allocationSize += memRequirements.size;
		memoryTypeBits |= memRequirements.memoryTypeBits;
		
		VkUtil::create3dImageView(device, image3d[i], VK_FORMAT_R8G8B8A8_UNORM, 1, &image3dView[i]);
	}

	allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memRequirements.memoryTypeBits, 0);
	err = vkAllocateMemory(device, &allocInfo, nullptr, &image3dMemory);
	check_vk_result(err);
	for (int i = 0; i < required3dImages; ++i) {
		vkBindImageMemory(device, image3d[i], image3dMemory, image3dOffsets[i]);
	}
	
	std::vector<VkDescriptorSetLayout> layouts;
	layouts.push_back(descriptorSetLayout);
	if (!descriptorSet) {
		VkUtil::createDescriptorSets(device, layouts, descriptorPool, &descriptorSet);
	}

	for (int i = 0; i < required3dImages; ++i) {
		VkUtil::updateImageDescriptorSet(device, image3dSampler, image3dView[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, i + 1, descriptorSet);
	}
	


	if (!descriptorSet) {
		resize(1, 1);
		return;
	}

	updateCommandBuffer();
	render();
}

void IsoSurfRenderer::updateCameraPos(float* mouseMovement)
{
	//rotation matrix for height adjustment
	glm::mat4 vertical;
	vertical = glm::rotate(glm::mat4(1.0f), mouseMovement[1] * VERTICALPANSPEED, glm::normalize(glm::cross(camPos, glm::vec3(0, 1, 0))));
	glm::vec3 temp = vertical * glm::vec4(camPos, 1);
	if (dot(temp, glm::vec3(1, 0, 0)) * dot(camPos, glm::vec3(1, 0, 0)) < 0 || dot(temp, glm::vec3(0, 0, 1)) * dot(camPos, glm::vec3(0, 0, 1)) < 0)
		vertical = glm::mat4(1.0f);
	//rotation matrix for horizontal adjustment
	glm::mat4 horizontal = glm::rotate(glm::mat4(1.0f), mouseMovement[0] * HORIZONTALPANSPEED, glm::vec3(0, 1, 0));
	camPos = horizontal * vertical * glm::vec4(camPos,1);

	//adding zooming
	glm::vec3 zoomDir = -camPos;
	camPos += ZOOMSPEED * zoomDir * mouseMovement[2];
}

void IsoSurfRenderer::render()
{
	VkResult err;

	//uploading the uniformBuffer
	UniformBuffer ubo;
	ubo.mvp = glm::perspective(glm::radians(45.0f), (float)imageWidth / (float)imageHeight, 0.1f, 100.0f);;
	ubo.mvp[1][1] *= -1;
	glm::mat4 look = glm::lookAt(camPos, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
	float max = glm::max(glm::max(image3dWidth, image3dHeight), image3dDepth);
	glm::mat4 scale = glm::scale(glm::mat4(1.0f),glm::vec3(boxWidth,boxHeight,boxDepth));
	ubo.mvp = ubo.mvp * look *scale;
	ubo.camPos = glm::inverse(scale) * glm::vec4(camPos,1);

	ubo.faces.x = float(ubo.camPos.x > 0) - .5f;
	ubo.faces.y = float(ubo.camPos.y > 0) - .5f;
	ubo.faces.z = float(ubo.camPos.z > 0) - .5f;

	ubo.lightDir = lightDir;
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

void IsoSurfRenderer::setImageDescriptorSet(VkDescriptorSet descriptor)
{
	imageDescriptorSet = descriptor;
}

VkDescriptorSet IsoSurfRenderer::getImageDescriptorSet()
{
	return imageDescriptorSet;
}

VkSampler IsoSurfRenderer::getImageSampler()
{
	return sampler;
}

VkImageView IsoSurfRenderer::getImageView()
{
	return imageView;
}

void IsoSurfRenderer::createPrepareImageCommandBuffer()
{
	VkUtil::createCommandBuffer(device, commandPool, &prepareImageCommand);
	VkUtil::transitionImageLayout(prepareImageCommand, image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	vkEndCommandBuffer(prepareImageCommand);
}

void IsoSurfRenderer::createImageResources()
{
	VkResult err;
	
	VkUtil::createImage(device, imageWidth, imageHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT, &image);

	VkMemoryRequirements memReq = {};
	vkGetImageMemoryRequirements(device, image, &memReq);
	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memReq.size;
	allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, 0);
	err = vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory);
	check_vk_result(err);

	vkBindImageMemory(device, image, imageMemory, 0);

	VkUtil::createImageView(device, image, VK_FORMAT_R8G8B8A8_UNORM, 1, VK_IMAGE_ASPECT_COLOR_BIT, &imageView);
	std::vector<VkImageView> views;
	views.push_back(imageView);
	VkUtil::createFrameBuffer(device, renderPass, views, imageWidth, imageHeight, &frameBuffer);
	VkUtil::createImageSampler(device, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR, 16, 1, &sampler);
}

void IsoSurfRenderer::createBuffer()
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

void IsoSurfRenderer::createPipeline()
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
	colorBlendAttachment.blendEnable = VK_TRUE;
	colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
	colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
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

	uboLayoutBinding.binding = 1;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	bindings.push_back(uboLayoutBinding);

	VkUtil::createDescriptorSetLayout(device, bindings, &descriptorSetLayout);
	std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
	descriptorSetLayouts.push_back(descriptorSetLayout);

	VkUtil::createRenderPass(device, VkUtil::PASS_TYPE_COLOR_OFFLINE, &renderPass);

	VkUtil::createPipeline(device, &vertexInputInfo, imageWidth, imageHeight, dynamicStates, shaderModules, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, &rasterizer, &multisampling, nullptr, &blendInfo, descriptorSetLayouts, &renderPass, &pipelineLayout, &pipeline);

	// creating the compute pipeline to fill the density images --------------------------------------------------
	VkShaderModule computeModule = VkUtil::createShaderModule(device, PCUtil::readByteFile(computePath));

	bindings.clear();
	VkDescriptorSetLayoutBinding binding = {};
	binding.descriptorCount = 1;
	binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	for (int i = 0; i < AMTOF3DTEXTURES; ++i) {
		binding.binding = i + 1;
		bindings.push_back(binding);
	}
	VkUtil::createDescriptorSetLayout(device, bindings, &computeDescriptorSetLayout);
	std::vector<VkDescriptorSetLayout>layouts;
	layouts.push_back(computeDescriptorSetLayout);

	VkUtil::createComputePipeline(device, computeModule, layouts, &computePipelineLayout, &computePipeline);
}

void IsoSurfRenderer::createDescriptorSets()
{
	std::vector<VkDescriptorSetLayout> layouts;
	layouts.push_back(descriptorSetLayout);
	if (!descriptorSet) {
		VkUtil::createDescriptorSets(device, layouts, descriptorPool, &descriptorSet);
	}

	VkUtil::updateDescriptorSet(device, uniformBuffer, sizeof(UniformBuffer), 0, descriptorSet);
}

void IsoSurfRenderer::updateCommandBuffer()
{
	vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);

	VkResult err;
	VkUtil::createCommandBuffer(device, commandPool, &commandBuffer);
	VkUtil::transitionImageLayout(commandBuffer, image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	std::vector<VkClearValue> clearValues;
	clearValues.push_back({ .1f,.1f,.1f,1 });
	VkUtil::beginRenderPass(commandBuffer, clearValues, renderPass, frameBuffer, { imageWidth,imageHeight });

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, offsets);
	vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);
	VkDescriptorSet sets[1] = { descriptorSet };
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, sets, 0, nullptr);

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

	VkUtil::transitionImageLayout(commandBuffer, image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	err = vkEndCommandBuffer(commandBuffer);
	check_vk_result(err);

	err = vkDeviceWaitIdle(device);
	check_vk_result(err);
}
