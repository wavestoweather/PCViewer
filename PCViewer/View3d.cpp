#include "View3d.h"

char View3d::vertPath[]= "shader/3dVert.spv";
char View3d::fragPath[]= "shader/3dFrag.spv";
char View3d::computePath[]= "shader/3dComp.spv";

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
	densityFillPipeline = VK_NULL_HANDLE;
	dimensionCorrectionMemory = VK_NULL_HANDLE;
	dimensionCorrectionImages[0] = VK_NULL_HANDLE;
	dimensionCorrectionImages[1] = VK_NULL_HANDLE;
	dimensionCorrectionImages[2] = VK_NULL_HANDLE;
	dimensionCorrectionViews = std::vector<VkImageView>(3, VK_NULL_HANDLE);


	camPos = glm::vec3(1, 0, 1);
	camRot = glm::vec2(0, .78f);
	flySpeed = .5f;
	fastFlyMultiplier = 2.5f;
	rotationSpeed = .15f;
	lightDir = glm::vec3(-1, -1, -1);
	lightDir = glm::vec3(-1, -1, -1);

	//setting up graphic resources
	
	createBuffer();
	createPipeline();
	createDescriptorSets();
	resize(width, height);

	resizeBox(1.5f, 1, 1.5f);
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
		vkDestroyImage(device, image3d, nullptr);
	}
	if (image3dView) {
		vkDestroyImageView(device, image3dView, nullptr);
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
	if (densityFillPipeline)
		vkDestroyPipeline(device, densityFillPipeline, nullptr);
	if (densityFillPipelineLayout)
		vkDestroyPipelineLayout(device, densityFillPipelineLayout, nullptr);
	if (densityFillDescriptorLayout)
		vkDestroyDescriptorSetLayout(device, densityFillDescriptorLayout, nullptr);
	if (dimensionCorrectionMemory) {
		vkFreeMemory(device, dimensionCorrectionMemory, nullptr);
	}
	if (dimensionCorrectionImages[0]) {
		vkDestroyImage(device, dimensionCorrectionImages[0], nullptr);
		vkDestroyImage(device, dimensionCorrectionImages[1], nullptr);
		vkDestroyImage(device, dimensionCorrectionImages[2], nullptr);
		vkDestroyImageView(device, dimensionCorrectionViews[0], nullptr);
		vkDestroyImageView(device, dimensionCorrectionViews[1], nullptr);
		vkDestroyImageView(device, dimensionCorrectionViews[2], nullptr);
	}
}

void View3d::resize(uint32_t width, uint32_t height)
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

void View3d::resizeBox(float width, float height, float depth)
{
	boxWidth = width;
	boxHeight = height;
	boxDepth = depth;
	render();
}

void View3d::update3dImage(const std::vector<float>& xDim, const std::vector<float>& yDim, const std::vector<float>& zDim, bool linAxis[3], const uint32_t posIndices[3], uint32_t densityIndex, const float minMax[2], VkBuffer data, uint32_t dataByteSize, VkBuffer indices, uint32_t indicesSize, uint32_t amtOfAttributes)
{
	if (!descriptorSet) {
		std::vector<VkDescriptorSetLayout> layouts;
		layouts.push_back(descriptorSetLayout);
		VkUtil::createDescriptorSets(device, layouts, descriptorPool, &descriptorSet);
	}
	VkResult err;
	int width = xDim.size(), height = yDim.size(), depth = zDim.size();
	dimensionCorrectionLinearDim[0] = linAxis[0];
	dimensionCorrectionLinearDim[1] = linAxis[1];
	dimensionCorrectionLinearDim[2] = linAxis[2];
	updateDimensionImages(xDim, yDim, zDim);

	bool imageUpdated = false;
	if ((width != image3dWidth) || (height != image3dHeight) || (depth != image3dDepth)) {
		image3dWidth = width;
		image3dHeight = height;
		image3dDepth = depth;
		imageUpdated = true;

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
		if (!image3dSampler) {
			VkUtil::createImageSampler(device,VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,VK_FILTER_LINEAR,16,1,&image3dSampler);
		}

		VkUtil::create3dImage(device, image3dWidth, image3dHeight, image3dDepth, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &image3d);

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image3d, &memRequirements);

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memRequirements.memoryTypeBits, 0);
		err = vkAllocateMemory(device, &allocInfo, nullptr, &image3dMemory);
		check_vk_result(err);
		vkBindImageMemory(device, image3d, image3dMemory, 0);

		VkUtil::create3dImageView(device, image3d, VK_FORMAT_R8_UNORM, 1, &image3dView);
		VkUtil::updateImageDescriptorSet(device, image3dSampler, image3dView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, descriptorSet);
	}
	
	//filling the 3d image via the compute pipeline
	ComputeUBO ubo{};
	ubo.posIndices[0] = posIndices[0];
	ubo.posIndices[1] = posIndices[1];
	ubo.posIndices[2] = posIndices[2];
	ubo.linearAxes = (uint32_t(dimensionCorrectionLinearDim[0])) | (uint32_t(dimensionCorrectionLinearDim[1]) << 1) | (uint32_t(dimensionCorrectionLinearDim[2]) << 2);
	ubo.densityAttribute = densityIndex;
	ubo.amtOfIndices = indicesSize;
	ubo.amtOfAttributes = amtOfAttributes;
	ubo.xMin = xDim.front();
	ubo.xMax = xDim.back();
	ubo.yMin = yDim.front();
	ubo.yMax = yDim.back();
	ubo.zMin = zDim.front();
	ubo.zMax = zDim.back();
	ubo.dimX = xDim.size();
	ubo.dimY = yDim.size();
	ubo.dimZ = zDim.size();
	ubo.minValue = minMax[0];
	ubo.maxValue = minMax[1];
	uint32_t uboByteSize = sizeof(ComputeUBO);
	VkBuffer buffer;
	VkDeviceMemory bufferMemory;
	VkUtil::createBuffer(device, uboByteSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &buffer);
	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	VkMemoryRequirements memReq = {};
	vkGetBufferMemoryRequirements(device, buffer, &memReq);
	allocInfo.allocationSize = memReq.size;
	allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
	vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);
	vkBindBufferMemory(device, buffer, bufferMemory, 0);
	VkUtil::uploadData(device, bufferMemory, 0, uboByteSize, &ubo);

	//create graphics buffer for dimension values
	uint32_t dimValsByteSize = (4 + xDim.size() + yDim.size() + zDim.size()) * sizeof(float);
	float* dimValsBytes = new float[dimValsByteSize];
	VkBuffer dimValsBuffer;
	VkDeviceMemory dimValsMemory;
	VkUtil::createBuffer(device, dimValsByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &dimValsBuffer);
	vkGetBufferMemoryRequirements(device, dimValsBuffer, &memReq);
	allocInfo.allocationSize = memReq.size;
	allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
	vkAllocateMemory(device, &allocInfo, nullptr, &dimValsMemory);
	vkBindBufferMemory(device, dimValsBuffer, dimValsMemory, 0);
	int offset = 0;
	dimValsBytes[offset++] = xDim.size();
	dimValsBytes[offset++] = yDim.size();
	dimValsBytes[offset++] = zDim.size();
	dimValsBytes[offset++] = 0; //padding
	for (float f : xDim) {
		dimValsBytes[offset++] = f;
	}
	for (float f : yDim) {
		dimValsBytes[offset++] = f;
	}
	for (float f : zDim) {
		dimValsBytes[offset++] = f;
	}
	assert(offset * sizeof(float) == dimValsByteSize);
	VkUtil::uploadData(device, dimValsMemory, 0, dimValsByteSize, dimValsBytes);
	delete[] dimValsBytes;

	VkDescriptorSet commandSet;
	std::vector<VkDescriptorSetLayout> sets{densityFillDescriptorLayout};
	VkUtil::createDescriptorSets(device, sets, descriptorPool, &commandSet);
	VkUtil::updateDescriptorSet(device, buffer, sizeof(ComputeUBO), 0, commandSet);
	VkUtil::updateDescriptorSet(device, indices, indicesSize * sizeof(uint32_t), 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, commandSet);
	VkUtil::updateDescriptorSet(device, data, dataByteSize, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, commandSet);
	VkUtil::updateStorageImageDescriptorSet(device, image3dView, VK_IMAGE_LAYOUT_GENERAL, 3, commandSet);
	VkUtil::updateDescriptorSet(device, dimValsBuffer, dimValsByteSize, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, commandSet);

	VkCommandBuffer commands;
	VkUtil::createCommandBuffer(device, commandPool, &commands);
	if (imageUpdated) {
		VkUtil::transitionImageLayout(commands, image3d, VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
	}
	else {
		VkUtil::transitionImageLayout(commands, image3d, VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
	}
	vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, densityFillPipeline);
	vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, densityFillPipelineLayout, 0, 1, &commandSet, 0, { 0 });
	uint32_t patchAmount = indicesSize / LOCALSIZE;
	patchAmount += (indicesSize % LOCALSIZE) ? 1 : 0;
	vkCmdDispatch(commands, patchAmount, 1, 1);
	VkUtil::transitionImageLayout(commands, image3d, VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	VkUtil::commitCommandBuffer(queue, commands);
	check_vk_result(vkQueueWaitIdle(queue));

	//std::vector<uint8_t> dow(xDim.size() * yDim.size() * zDim.size());
	//VkUtil::downloadImageData(device, physicalDevice, commandPool, queue, image3d, VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, xDim.size(), yDim.size(), zDim.size(), dow.data(), dow.size());

	vkDestroyBuffer(device, buffer, nullptr);
	vkDestroyBuffer(device, dimValsBuffer, nullptr);
	vkFreeMemory(device, bufferMemory, nullptr);
	vkFreeMemory(device, dimValsMemory, nullptr);
	vkFreeDescriptorSets(device, descriptorPool, 1, &commandSet);
	vkFreeCommandBuffers(device, commandPool, 1, &commands);

	if (!descriptorSet) {
		resize(1, 1);
		return;
	}

	updateCommandBuffer();
	render();
}

void View3d::updateCameraPos(const CamNav::NavigationInput& input, float deltaT)
{
	//first do the rotation, as the user has a more inert feeling when the fly direction matches the view direction instantly
	if (input.mouseDeltaX) {
		camRot.y -= rotationSpeed * input.mouseDeltaX * .02f;
	}
	if (input.mouseDeltaY) {
		camRot.x -= rotationSpeed * input.mouseDeltaY * .02f;
	}

	glm::mat4 rot = glm::eulerAngleYX(camRot.y, camRot.x);
	if (input.a) {	//fly left
		glm::vec4 left = rot * glm::vec4(-1, 0, 0, 0) * flySpeed * ((input.shift) ? fastFlyMultiplier : 1) * deltaT;
		camPos += glm::vec3(left.x, left.y, left.z);
	}
	if (input.d) {	//fly right
		glm::vec4 right = rot * glm::vec4(1, 0, 0, 0) * flySpeed * ((input.shift) ? fastFlyMultiplier : 1) * deltaT;
		camPos += glm::vec3(right.x, right.y, right.z);
	}
	if (input.s) {	//fly backward
		glm::vec4 back = rot * glm::vec4(0, 0, 1, 0) * flySpeed * ((input.shift) ? fastFlyMultiplier : 1) * deltaT;
		camPos += glm::vec3(back.x, back.y, back.z);
	}
	if (input.w) {	//fly forward
		glm::vec4 front = rot * glm::vec4(0, 0, -1, 0) * flySpeed * ((input.shift) ? fastFlyMultiplier : 1) * deltaT;
		camPos += glm::vec3(front.x, front.y, front.z);
	}
	if (input.q) {	//fly down
		camPos += glm::vec3(0, -1, 0) * flySpeed * ((input.shift) ? fastFlyMultiplier : 1) * deltaT;
	}
	if (input.e) {	//fly up
		camPos += glm::vec3(0, 1, 0) * flySpeed * ((input.shift) ? fastFlyMultiplier : 1) * deltaT;
	}
}

void View3d::render()
{
	if (commandBuffer == VK_NULL_HANDLE) return;
	VkResult err;

	//uploading the uniformBuffer
	UniformBuffer ubo;
	ubo.mvp = glm::perspective(glm::radians(45.0f), (float)imageWidth / (float)imageHeight, 0.1f, 100.0f);;
	ubo.mvp[1][1] *= -1;
	glm::mat4 view = glm::transpose(glm::eulerAngleY(camRot.y) * glm::eulerAngleX(camRot.x)) * glm::translate(glm::mat4(1.0), -camPos);;
	glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(boxWidth, boxHeight, boxDepth));
	ubo.mvp = ubo.mvp * view * scale;
	ubo.camPos = glm::inverse(scale) * glm::vec4(camPos,1);

	ubo.faces.x = float(ubo.camPos.x > 0) - .5f;
	ubo.faces.y = float(ubo.camPos.y > 0) - .5f;
	ubo.faces.z = float(ubo.camPos.z > 0) - .5f;
	ubo.linearAxes = (uint32_t(dimensionCorrectionLinearDim[0])) | (uint32_t(dimensionCorrectionLinearDim[1]) << 1) | (uint32_t(dimensionCorrectionLinearDim[2]) << 2);

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
	VkUtil::transitionImageLayout(prepareImageCommand, image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	vkEndCommandBuffer(prepareImageCommand);
}

void View3d::createImageResources()
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

	uboLayoutBinding.binding = 2;
	uboLayoutBinding.descriptorCount = 3;
	bindings.push_back(uboLayoutBinding);

	VkUtil::createDescriptorSetLayout(device, bindings, &descriptorSetLayout);
	std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
	descriptorSetLayouts.push_back(descriptorSetLayout);

	VkUtil::createRenderPass(device, VkUtil::PASS_TYPE_COLOR_OFFLINE, &renderPass);

	VkUtil::createPipeline(device, &vertexInputInfo, imageWidth, imageHeight, dynamicStates, shaderModules, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, &rasterizer, &multisampling, nullptr, &blendInfo, descriptorSetLayouts, &renderPass, &pipelineLayout, &pipeline);

	//creating the fill copute pipeline
	VkShaderModule computeModule = VkUtil::createShaderModule(device, PCUtil::readByteFile(computePath));

	bindings.clear();
	VkDescriptorSetLayoutBinding binding = {};
	binding.descriptorCount = 1;
	binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	binding.binding = 0;								//compute infos
	binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	bindings.push_back(binding);

	binding.binding = 1;								//indices buffer
	binding.descriptorCount = 1;
	binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	bindings.push_back(binding);

	binding.binding = 2;								//data buffer
	bindings.push_back(binding);

	binding.binding = 3;								//density image
	binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	bindings.push_back(binding);

	binding.binding = 4;								//dimension values
	binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	bindings.push_back(binding);

	VkUtil::createDescriptorSetLayout(device, bindings, &densityFillDescriptorLayout);
	std::vector<VkDescriptorSetLayout>layouts;
	layouts.push_back(densityFillDescriptorLayout);

	VkUtil::createComputePipeline(device, computeModule, layouts, &densityFillPipelineLayout, &densityFillPipeline);
}

void View3d::createDescriptorSets()
{
	std::vector<VkDescriptorSetLayout> layouts;
	layouts.push_back(descriptorSetLayout);
	if (!descriptorSet) {
		VkUtil::createDescriptorSets(device, layouts, descriptorPool, &descriptorSet);
	}

	VkUtil::updateDescriptorSet(device, uniformBuffer, sizeof(UniformBuffer), 0, descriptorSet);
}

void View3d::updateCommandBuffer()
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

bool View3d::updateDimensionImages(const std::vector<float>& xDim, const std::vector<float>& yDim, const std::vector<float>& zDim)
{
	if (!dimensionCorrectionMemory) {
		VkUtil::create1dImage(device, dimensionCorrectionSize, dimensionCorrectionFormat, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, dimensionCorrectionImages);
		VkUtil::create1dImage(device, dimensionCorrectionSize, dimensionCorrectionFormat, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, dimensionCorrectionImages + 1);
		VkUtil::create1dImage(device, dimensionCorrectionSize, dimensionCorrectionFormat, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, dimensionCorrectionImages + 2);
		VkMemoryRequirements memReq;
		VkMemoryAllocateInfo alloc{};
		vkGetImageMemoryRequirements(device, dimensionCorrectionImages[0], &memReq);
		alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		alloc.allocationSize += memReq.size;
		vkGetImageMemoryRequirements(device, dimensionCorrectionImages[1], &memReq);
		alloc.allocationSize += memReq.size;
		vkGetImageMemoryRequirements(device, dimensionCorrectionImages[2], &memReq);
		alloc.allocationSize += memReq.size;
		alloc.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, 0);
		check_vk_result(vkAllocateMemory(device, &alloc, nullptr, &dimensionCorrectionMemory));
		vkBindImageMemory(device, dimensionCorrectionImages[0], dimensionCorrectionMemory, 0);
		vkBindImageMemory(device, dimensionCorrectionImages[1], dimensionCorrectionMemory, alloc.allocationSize / 3);
		vkBindImageMemory(device, dimensionCorrectionImages[2], dimensionCorrectionMemory, alloc.allocationSize / 3 * 2);
		VkUtil::create1dImageView(device, dimensionCorrectionImages[0], dimensionCorrectionFormat, 1, dimensionCorrectionViews.data());
		VkUtil::create1dImageView(device, dimensionCorrectionImages[1], dimensionCorrectionFormat, 1, dimensionCorrectionViews.data() + 1);
		VkUtil::create1dImageView(device, dimensionCorrectionImages[2], dimensionCorrectionFormat, 1, dimensionCorrectionViews.data() + 2);

		VkCommandBuffer command;
		VkUtil::createCommandBuffer(device, commandPool, &command);
		VkUtil::transitionImageLayout(command, dimensionCorrectionImages[0], dimensionCorrectionFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		VkUtil::transitionImageLayout(command, dimensionCorrectionImages[1], dimensionCorrectionFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		VkUtil::transitionImageLayout(command, dimensionCorrectionImages[2], dimensionCorrectionFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		VkUtil::commitCommandBuffer(queue, command);
		vkQueueWaitIdle(queue);
		vkFreeCommandBuffers(device, commandPool, 1, &command);
	}


	if (!PCUtil::vectorEqual(xDim, dimensionCorrectionArrays[0]) || !PCUtil::vectorEqual(yDim, dimensionCorrectionArrays[1]) || !PCUtil::vectorEqual(zDim, dimensionCorrectionArrays[2])) {
		dimensionCorrectionArrays[0] = std::vector<float>(xDim);
		dimensionCorrectionArrays[1] = std::vector<float>(yDim);
		dimensionCorrectionArrays[2] = std::vector<float>(zDim);
		std::vector<float> correction(dimensionCorrectionSize);
		float alpha = 0;
		if (!dimensionCorrectionLinearDim[0]) {
			for (int i = 0; i < dimensionCorrectionSize; ++i) {
				alpha = i / float(dimensionCorrectionSize - 1);
				float axisVal = alpha * xDim.back() + (1 - alpha) * xDim.front();
				correction[i] = PCUtil::getVectorIndex(xDim, axisVal) / (xDim.size() - 1);
			}
			VkUtil::uploadImageData(device, physicalDevice, commandPool, queue, dimensionCorrectionImages[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, dimensionCorrectionFormat, dimensionCorrectionSize, 1, 1, correction.data(), correction.size() * sizeof(float));
		}
		if (!dimensionCorrectionLinearDim[1]) {
			for (int i = 0; i < dimensionCorrectionSize; ++i) {
				alpha = i / float(dimensionCorrectionSize - 1);
				float axisVal = alpha * yDim.back() + (1 - alpha) * yDim.front();
				correction[i] = PCUtil::getVectorIndex(yDim, axisVal) / (yDim.size() - 1);
			}
			VkUtil::uploadImageData(device, physicalDevice, commandPool, queue, dimensionCorrectionImages[1], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, dimensionCorrectionFormat, dimensionCorrectionSize, 1, 1, correction.data(), correction.size() * sizeof(float));
		}
		if (!dimensionCorrectionLinearDim[2]) {
			for (int i = 0; i < dimensionCorrectionSize; ++i) {
				alpha = i / float(dimensionCorrectionSize - 1);
				float axisVal = alpha * zDim.back() + (1 - alpha) * zDim.front();
				correction[i] = PCUtil::getVectorIndex(zDim, axisVal) / (zDim.size() - 1);
			}
			VkUtil::uploadImageData(device, physicalDevice, commandPool, queue, dimensionCorrectionImages[2], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, dimensionCorrectionFormat, dimensionCorrectionSize, 1, 1, correction.data(), correction.size() * sizeof(float));
		}
		std::vector<VkSampler> samplers(3, sampler);
		std::vector<VkImageLayout> layouts = std::vector<VkImageLayout>(3, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		VkUtil::updateImageArrayDescriptorSet(device, samplers, dimensionCorrectionViews, layouts, 2, descriptorSet);
	}

	return true;
}
