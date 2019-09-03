#include "NodeViewer.h"

char NodeViewer::vertPath[] = "shader/nodeVert.spv";
char NodeViewer::fragPath[] = "shader/nodeFrag.spv";

NodeViewer::NodeViewer(uint32_t width, uint32_t height, VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool)
{
	imageWidth = 0;
	imageHeight = 0;

	this->physicalDevice = physicalDevice;
	this->device = device;
	this->descriptorPool = descriptorPool;
	this->queue = queue;
	this->commandPool = commandPool;

	renderCommands = VK_NULL_HANDLE;
	imageMemory = VK_NULL_HANDLE;
	image = VK_NULL_HANDLE;
	depthImageMemory = VK_NULL_HANDLE;
	depthImage = VK_NULL_HANDLE;
	imageView = VK_NULL_HANDLE;
	depthImageView = VK_NULL_HANDLE;
	framebuffer = VK_NULL_HANDLE;
	imageSampler = VK_NULL_HANDLE;
	imageDescSet = VK_NULL_HANDLE;
	pipeline = VK_NULL_HANDLE;
	pipelineLayout = VK_NULL_HANDLE;
	renderPass = VK_NULL_HANDLE;
	descriptorSetLayout = VK_NULL_HANDLE;
	bufferMemory = VK_NULL_HANDLE;
	sphereVertexBuffer = VK_NULL_HANDLE;
	sphereIndexBuffer = VK_NULL_HANDLE;
	cylinderVertexBuffer = VK_NULL_HANDLE;
	cylinderIndexBuffer = VK_NULL_HANDLE;

	ubo = VK_NULL_HANDLE;
	uboMem = VK_NULL_HANDLE;
	uboSet = VK_NULL_HANDLE;

	cameraPos = glm::vec3(2, 2, 2);
	setupBuffer();
	setupRenderPipeline();
	resizeImage(width, height);
	setupUbo();
	recordRenderCommands();
	render();
}

NodeViewer::~NodeViewer()
{
	if (imageMemory) {
		vkFreeMemory(device, imageMemory, nullptr);
	}
	if (image) {
		vkDestroyImage(device, image, nullptr);
	}
	if (depthImageMemory) {
		vkFreeMemory(device, depthImageMemory, nullptr);
	}
	if (depthImage) {
		vkDestroyImage(device, depthImage, nullptr);
	}
	if (imageView) {
		vkDestroyImageView(device, imageView, nullptr);
	}
	if (depthImageView) {
		vkDestroyImageView(device, depthImageView, nullptr);
	}
	if (framebuffer) {
		vkDestroyFramebuffer(device, framebuffer, nullptr);
	}
	if (imageSampler) {
		vkDestroySampler(device, imageSampler, nullptr);
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
	if (descriptorSetLayout) {
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
	}
	if (bufferMemory) {
		vkFreeMemory(device, bufferMemory, nullptr);
	}
	if (sphereVertexBuffer) {
		vkDestroyBuffer(device, sphereVertexBuffer, nullptr);
	}
	if (sphereIndexBuffer) {
		vkDestroyBuffer(device, sphereIndexBuffer, nullptr);
	}
	if (cylinderVertexBuffer) {
		vkDestroyBuffer(device, cylinderVertexBuffer, nullptr);
	}
	if (cylinderIndexBuffer) {
		vkDestroyBuffer(device, cylinderIndexBuffer, nullptr);
	}

	if (ubo) {
		vkDestroyBuffer(device, ubo, nullptr);
	}
	if (uboMem) {
		vkFreeMemory(device, uboMem, nullptr);
	}
}

void NodeViewer::resizeImage(uint32_t width, uint32_t height)
{
	VkResult err;

	if (!imageSampler) {
		VkUtil::createImageSampler(device, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_FILTER_LINEAR, 16, 1, &imageSampler);
	}
	if (width != imageWidth || height != imageHeight) {
		imageWidth = width;
		imageHeight = height;

		//recreating image resources
		if (imageMemory) {
			vkFreeMemory(device, imageMemory, nullptr);
		}
		if (image) {
			vkDestroyImage(device, image, nullptr);
		}
		if (imageView) {
			vkDestroyImageView(device, imageView, nullptr);
		}
		if (depthImageMemory) {
			vkFreeMemory(device, depthImageMemory, nullptr);
		}
		if (depthImage) {
			vkDestroyImage(device, depthImage, nullptr);
		}
		if (depthImageView) {
			vkDestroyImageView(device, depthImageView, nullptr);
		}
		if (framebuffer) {
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}

		VkUtil::createImage(device, imageWidth, imageHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &image);
		VkUtil::createImage(device, imageWidth, imageHeight, VK_FORMAT_D16_UNORM, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, &depthImage);

		VkMemoryRequirements memReq;
		vkGetImageMemoryRequirements(device, image, &memReq);

		VkMemoryAllocateInfo memAlloc = {};
		memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memAlloc.allocationSize = memReq.size;
		uint32_t memBits = memReq.memoryTypeBits;
		memAlloc.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memBits, 0);

		err = vkAllocateMemory(device, &memAlloc, nullptr, &imageMemory);
		check_vk_result(err);
		
		vkGetImageMemoryRequirements(device, depthImage, &memReq);
		memAlloc.allocationSize = memReq.size;
		memBits = memReq.memoryTypeBits;
		memAlloc.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memBits, 0);

		err = vkAllocateMemory(device, &memAlloc, nullptr, &depthImageMemory);
		check_vk_result(err);

		vkBindImageMemory(device, image, imageMemory, 0);
		vkBindImageMemory(device, depthImage, depthImageMemory, 0);

		VkUtil::createImageView(device, image, VK_FORMAT_R8G8B8A8_UNORM, 1, VK_IMAGE_ASPECT_COLOR_BIT, &imageView);
		VkUtil::createImageView(device, depthImage, VK_FORMAT_D16_UNORM, 1, VK_IMAGE_ASPECT_DEPTH_BIT, &depthImageView);
		std::vector<VkImageView> views;
		views.push_back(imageView);
		views.push_back(depthImageView);
		VkUtil::createFrameBuffer(device, renderPass, views, imageWidth, imageHeight, &framebuffer);

		//transforming the imagelayout to be readable
		VkCommandBuffer command;
		VkUtil::createCommandBuffer(device, commandPool, &command);
		VkUtil::transitionImageLayout(command, image, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		VkUtil::transitionImageLayout(command, depthImage, VK_FORMAT_D16_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
		vkEndCommandBuffer(command);

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.signalSemaphoreCount = 0;
		submitInfo.waitSemaphoreCount = 0;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &command;

		err = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
		check_vk_result(err);
	}
}

void NodeViewer::addSphere(float r, glm::vec4 color, glm::vec3 pos)
{
	gSphere cur;
	cur.radius = r;
	cur.color = color;
	cur.pos = pos;
	//TODO: create descset

	spheres.push_back(cur);
	recordRenderCommands();
}

void NodeViewer::addCylinder(float r, float length, glm::vec4 color, glm::vec3 pos)
{
	gCylinder cur;
	cur.radius = r;
	cur.length = length;
	cur.color = color;
	cur.pos = pos;
	//TODO: create descriptor set

	cylinders.push_back(cur);
	recordRenderCommands();
}

void NodeViewer::render()
{
	VkResult err;

	Ubo ubo;
	ubo.cameraPos = cameraPos;
	ubo.color = glm::vec4(.8f, .1f, .4f, 1);
	glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)imageWidth / (float)imageHeight, 0.1f, 100.0f);
	proj[1][1] *= -1;
	glm::mat4 view = glm::lookAt(cameraPos, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
	ubo.mvp = proj * view;
	ubo.worldNormals = glm::mat4(1.0f);
	
	void* d;
	vkMapMemory(device, uboMem, 0, sizeof(Ubo), 0, &d);
	memcpy(d, &ubo, sizeof(Ubo));
	vkUnmapMemory(device, uboMem);

	//submitting the command buffer
	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.signalSemaphoreCount = 0;
	submitInfo.waitSemaphoreCount = 0;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &renderCommands;

	err = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	check_vk_result(err);
}

void NodeViewer::updateCameraPos(float* mouseMovement)
{
	//rotation matrix for height adjustment
	glm::mat4 vertical;
	vertical = glm::rotate(glm::mat4(1.0f), mouseMovement[1] * VERTICALROTSPEED, glm::normalize(glm::cross(cameraPos, glm::vec3(0, 1, 0))));
	//rotation matrix for horizontal adjustment
	glm::mat4 horizontal = glm::rotate(glm::mat4(1.0f), mouseMovement[0] * HORIZONTALROTSPEED, glm::vec3(0, 1, 0));
	cameraPos = horizontal * vertical * glm::vec4(cameraPos, 1);

	//adding zooming
	glm::vec3 zoomDir = -cameraPos;
	cameraPos += ZOOOMSPEED * zoomDir * mouseMovement[2];
}

VkSampler NodeViewer::getImageSampler()
{
	return imageSampler;
}

VkImageView NodeViewer::getImageView()
{
	return imageView;
}

void NodeViewer::setImageDescSet(VkDescriptorSet desc)
{
	imageDescSet = desc;
}

VkDescriptorSet NodeViewer::getImageDescSet()
{
	return imageDescSet;
}

void NodeViewer::setupRenderPipeline()
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
	bindingDescripiton.stride = sizeof(Shape::Vertex);
	bindingDescripiton.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputAttributeDescription attributeDescriptions[2] = {};	//describes the attribute of the vertex. If more than 1 attribute is used this has to be an array
	attributeDescriptions[0].binding = 0;
	attributeDescriptions[0].location = 0;
	attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
	attributeDescriptions[0].offset = offsetof(Shape::Vertex, position);

	attributeDescriptions[1].binding = 0;
	attributeDescriptions[1].location = 1;
	attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
	attributeDescriptions[1].offset = offsetof(Shape::Vertex, normal);

	VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.pVertexBindingDescriptions = &bindingDescripiton;
	vertexInputInfo.vertexAttributeDescriptionCount = 2;
	vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions;

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
	rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
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

	VkPipelineDepthStencilStateCreateInfo depthStencil = {};
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = VK_TRUE;
	depthStencil.depthWriteEnable = VK_TRUE;
	depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
	depthStencil.depthBoundsTestEnable = VK_FALSE;
	depthStencil.minDepthBounds = 0;
	depthStencil.maxDepthBounds = 1.0f;

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

	VkUtil::createPcPlotRenderPass(device, VkUtil::PASS_TYPE_DEPTH_OFFLINE, &renderPass);

	VkUtil::createPipeline(device, &vertexInputInfo, imageWidth, imageHeight, dynamicStates, shaderModules, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, &rasterizer, &multisampling, &depthStencil, &blendInfo, descriptorSetLayouts, &renderPass, &pipelineLayout, &pipeline);
}

void NodeViewer::setupBuffer()
{
	VkResult err;

	Sphere s(1, 20);
	Cylinder c(1, 2, 20);

	amtOfIdxSphere = s.getIndexBuffer(0).size();
	amtOfIdxCylinder = c.getIndexBuffer(0).size();

	//creating the buffer
	VkUtil::createBuffer(device, s.getVertexBuffer().size() * sizeof(Shape::Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &sphereVertexBuffer);
	VkUtil::createBuffer(device, s.getIndexBuffer(0).size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &sphereIndexBuffer);
	VkUtil::createBuffer(device, c.getVertexBuffer().size() * sizeof(Shape::Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &cylinderVertexBuffer);
	VkUtil::createBuffer(device, c.getIndexBuffer(0).size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &cylinderIndexBuffer);

	//getting memory requirements and allocating memory
	VkMemoryRequirements memReq;
	vkGetBufferMemoryRequirements(device, sphereVertexBuffer, &memReq);

	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memReq.size;
	uint32_t memBits = memReq.memoryTypeBits;
	uint32_t sphereIndBufOffset = memReq.size;

	vkGetBufferMemoryRequirements(device, sphereIndexBuffer, &memReq);
	allocInfo.allocationSize += memReq.size;
	memBits |= memReq.memoryTypeBits;
	uint32_t cylinderVertBufOffset = allocInfo.allocationSize;

	vkGetBufferMemoryRequirements(device, cylinderVertexBuffer, &memReq);
	allocInfo.allocationSize += memReq.size;
	memBits |= memReq.memoryTypeBits;
	uint32_t cylinderIndBufOffset = allocInfo.allocationSize;

	vkGetBufferMemoryRequirements(device, cylinderIndexBuffer, &memReq);
	allocInfo.allocationSize += memReq.size;
	allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memBits | memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

	err = vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);
	check_vk_result(err);

	//binding buffer to memory
	vkBindBufferMemory(device, sphereVertexBuffer, bufferMemory, 0);
	vkBindBufferMemory(device, sphereIndexBuffer, bufferMemory, sphereIndBufOffset);
	vkBindBufferMemory(device, cylinderVertexBuffer, bufferMemory, cylinderVertBufOffset);
	vkBindBufferMemory(device, cylinderIndexBuffer, bufferMemory, cylinderIndBufOffset);

	//uploading the vertex and indexbuffers to the graphicscard
	char* data = new char[allocInfo.allocationSize];
	memcpy(data, s.getVertexBuffer().data(), s.getVertexBuffer().size() * sizeof(Shape::Vertex));
	memcpy(data + sphereIndBufOffset, s.getIndexBuffer(0).data(), s.getIndexBuffer(0).size() * sizeof(uint32_t));
	memcpy(data + cylinderVertBufOffset, c.getVertexBuffer().data(), c.getVertexBuffer().size() * sizeof(Shape::Vertex));
	memcpy(data + cylinderIndBufOffset, c.getIndexBuffer(0).data(), c.getIndexBuffer(0).size() * sizeof(uint32_t));

	void* d;
	vkMapMemory(device, bufferMemory, 0, allocInfo.allocationSize, 0, &d);
	memcpy(d, data, allocInfo.allocationSize);
	vkUnmapMemory(device, bufferMemory);

	delete[] data;
}

void NodeViewer::recordRenderCommands()
{
	VkResult err;
	VkUtil::createCommandBuffer(device, commandPool, &renderCommands);

	VkUtil::transitionImageLayout(renderCommands, image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

	std::vector<VkClearValue> clearValues;
	clearValues.push_back({ 0,0,0,1 });
	clearValues.push_back({ 1.0f });
	VkUtil::beginRenderPass(renderCommands, clearValues, renderPass, framebuffer, { imageWidth,imageHeight });

	vkCmdBindPipeline(renderCommands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize offsets[1] = { 0 };
	vkCmdBindVertexBuffers(renderCommands, 0, 1, &sphereVertexBuffer, offsets);
	vkCmdBindIndexBuffer(renderCommands, sphereIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
	vkCmdBindDescriptorSets(renderCommands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &uboSet, 0, nullptr);

	VkViewport viewport = {};					//description for our viewport for transformation operation after rasterization
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = imageWidth;
	viewport.height = imageHeight;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	vkCmdSetViewport(renderCommands, 0, 1, &viewport);

	VkRect2D scissor = {};						//description for cutting the rendered result if wanted
	scissor.offset = { 0, 0 };
	scissor.extent = { (uint32_t)imageWidth,(uint32_t)imageHeight };
	vkCmdSetScissor(renderCommands, 0, 1, &scissor);

	//TODO: replace this with more useful code
	for (gSphere s : spheres) {
		vkCmdDrawIndexed(renderCommands, amtOfIdxSphere, 1, 0, 0, 0);
	}

	vkCmdBindVertexBuffers(renderCommands, 0, 1, &cylinderVertexBuffer, offsets);
	vkCmdBindIndexBuffer(renderCommands, cylinderIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
	for (gCylinder c : cylinders) {
		vkCmdDrawIndexed(renderCommands, amtOfIdxCylinder, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(renderCommands);

	VkUtil::transitionImageLayout(renderCommands, image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	err = vkEndCommandBuffer(renderCommands);
	check_vk_result(err);
}

void NodeViewer::setupUbo()
{
	VkResult err;

	VkUtil::createBuffer(device, sizeof(Ubo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &ubo);

	VkMemoryRequirements memReq;
	vkGetBufferMemoryRequirements(device, ubo, &memReq);
	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memReq.size;
	allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
	err = vkAllocateMemory(device, &allocInfo, nullptr, &uboMem);
	check_vk_result;

	vkBindBufferMemory(device, ubo, uboMem, 0);

	std::vector<VkDescriptorSetLayout> layouts;
	layouts.push_back(descriptorSetLayout);
	VkUtil::createDescriptorSets(device, layouts, descriptorPool, &uboSet);

	VkUtil::updateDescriptorSet(device, ubo, sizeof(Ubo), 0, uboSet);
}