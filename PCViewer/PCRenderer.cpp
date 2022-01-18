#define NOSTATICS
#include "PCRenderer.hpp"
#undef NOSTATICS
#include "PCUtil.h"

PCRenderer::PCRenderer(const VkUtil::Context& context, uint32_t width, uint32_t height, VkDescriptorSetLayout uniformLayout, VkDescriptorSetLayout dataLayout):
_pipelineInstance(PipelineSingleton::getInstance(context, {width, height, uniformLayout, dataLayout}))
{
    //creating the render resources
    VkFormat intermediateFormat = VK_FORMAT_R32_SFLOAT;
	VkFormat plotFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkUtil::createImage(context.device, width, height, intermediateFormat, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &_intermediateImage);
    VkUtil::createImage(context.device, width, height, plotFormat, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, &_plotImage);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    VkUtil::addImageToAllocInfo(context.device, _intermediateImage, allocInfo);
	uint32_t plotImageOffset = allocInfo.allocationSize;
    VkUtil::addImageToAllocInfo(context.device, _plotImage, allocInfo);
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(context.physicalDevice, allocInfo.memoryTypeIndex, 0);
    VkResult res = vkAllocateMemory(context.device, &allocInfo, nullptr, &_imageMemory); check_vk_result(res);
	res = vkBindImageMemory(context.device, _intermediateImage, _imageMemory, 0);
	res = vkBindImageMemory(context.device, _plotImage, _imageMemory, plotImageOffset);

    VkUtil::createImageView(context.device, _intermediateImage, intermediateFormat, 1, VK_IMAGE_ASPECT_COLOR_BIT, &_intermediateView);
    VkUtil::createImageView(context.device, _plotImage, plotFormat, 1, VK_IMAGE_ASPECT_COLOR_BIT, &_plotView);
    
    std::vector<VkImageView> attachments{_intermediateView};
    VkUtil::createFrameBuffer(context.device, _pipelineInstance.renderPass, attachments, width, height, &_framebuffer);

	VkUtil::transitionImageLayoutDirect(context.device, context.commandPool, context.queue, {_intermediateImage, _plotImage}, {intermediateFormat, plotFormat}, {VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED}, {VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL});

	//VkUtil::createDescriptorSets(context.device, {_pipelineInstance.pipelineInfo.descriptorSetLayout}, context.descriptorPool, &_intermediateSet);
	//VkUtil::updateStorageImageDescriptorSet(context.device, _intermediateView, VK_IMAGE_LAYOUT_GENERAL, 0, _intermediateSet);

	VkUtil::createDescriptorSets(context.device, {_pipelineInstance.computeInfo.descriptorSetLayout}, context.descriptorPool, &_computeSet);
	VkUtil::updateStorageImageDescriptorSet(context.device, _intermediateView, VK_IMAGE_LAYOUT_GENERAL, 0, _computeSet);
	VkUtil::updateStorageImageDescriptorSet(context.device, _plotView, VK_IMAGE_LAYOUT_GENERAL, 1, _computeSet);
}

PCRenderer::~PCRenderer(){
    PipelineSingleton::notifyInstanceShutdown(_pipelineInstance);
    auto device = _pipelineInstance.context.device;
    if(_framebuffer) vkDestroyFramebuffer(device, _framebuffer, nullptr);
    if(_intermediateImage) vkDestroyImage(device, _intermediateImage, nullptr);
    if(_plotImage) vkDestroyImage(device, _plotImage, nullptr);
    if(_intermediateView) vkDestroyImageView(device, _intermediateView, nullptr);
    if(_plotView) vkDestroyImageView(device, _plotView, nullptr);
    if(_imageMemory) vkFreeMemory(device, _imageMemory, nullptr);
	if(_intermediateSet) vkFreeDescriptorSets(device, _pipelineInstance.context.descriptorPool, 1, &_intermediateSet);
	if(_computeSet) vkFreeDescriptorSets(device, _pipelineInstance.context.descriptorPool, 1, &_computeSet);
}

void PCRenderer::renderPCPlots(std::list<DrawList>& drawlists, const GlobalPCSettings& globalSettings){
	PCUtil::Stopwatch stopwatch(std::cout, "PCRenderer::renderPCPlots(...)");
	
	// renaming the all things so that they can be accessed in the later code more efficiently and without having to rewrite most of the code
	auto& attributes = globalSettings.attributes;
	bool* attributeEnabled = globalSettings.attributeEnabled;
	std::vector<int>& attributeOrder = globalSettings.attributeOrder;
	auto device = _pipelineInstance.context.device;

	auto placeOfInd = [&](int ind, bool countDisabled = false){
		int place = 0;
		for (int i : attributeOrder) {
			if (i == ind)
				break;
			if (attributeEnabled[i] || countDisabled)
				place++;
		}
		return place;
	};
	// end renaming ----------------------------------------------------------------
    VkResult err;

	err = vkQueueWaitIdle(_pipelineInstance.context.queue);
	check_vk_result(err);

	//beginning the command buffer
	VkCommandBuffer command_buffer;
	VkUtil::createCommandBuffer(_pipelineInstance.context.device, _pipelineInstance.context.commandPool, &command_buffer);

	//now using the memory barrier to transition image state
	VkImageMemoryBarrier use_barrier[1] = {};
	use_barrier[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	use_barrier[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
	use_barrier[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	use_barrier[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	use_barrier[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	use_barrier[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	use_barrier[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	use_barrier[0].image = _intermediateImage;
	use_barrier[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	use_barrier[0].subresourceRange.levelCount = 1;
	use_barrier[0].subresourceRange.layerCount = 1;
	//vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, use_barrier);

	//drawing via copying the indeces into the index buffer
	//the indeces have to have just the right ordering for the vertices
	int amtOfIndeces = 0;
	for (int i = 0; i < attributes.size(); i++) {
		if (attributeEnabled[i])
			amtOfIndeces++;
	}


	//filling the uniform buffer and copying it into the end of the uniformbuffer
	UniformBufferObject ubo = {};
	ubo.amtOfVerts = amtOfIndeces;
	ubo.amtOfAttributes = attributes.size();
	ubo.color = { 1,1,1,1 };
	ubo.vertTransformations.resize(ubo.amtOfAttributes);
	ubo.padding = 0;

	int c = 0;

	for (int i : attributeOrder) {
		ubo.vertTransformations[i].x = c;
		if (attributeEnabled[i])
			c++;
		ubo.vertTransformations[i].y = attributes[i].min;
		ubo.vertTransformations[i].z = attributes[i].max;
	}

	std::vector<std::pair<int, int>> order;
	for (int i = 0; i < attributes.size(); i++) {
		if (attributeEnabled[i]) {
			order.push_back(std::pair<int, int>(i, placeOfInd(i)));
		}
	}

	std::sort(order.begin(), order.end(), [](std::pair<int, int>a, std::pair<int, int>b) {return a.second < b.second; });

	//copying the uniform buffer
	void* da;
	c = 0;
	for (DrawList& ds : drawlists) {
		uint32_t uboSize = sizeof(UniformBufferObject) - sizeof(UniformBufferObject::vertTransformations);
		uint32_t trafoSize = sizeof(ubo.vertTransformations[0]) * ubo.vertTransformations.size();
		std::vector<uint8_t> bits(uboSize + trafoSize);
		ubo.vertTransformations[0].w = 0;
		ubo.color = ds.color;
		std::copy_n(reinterpret_cast<uint8_t*>(&ubo), uboSize, bits.data());
		std::copy_n(reinterpret_cast<uint8_t*>(ubo.vertTransformations.data()), trafoSize, bits.data() + uboSize);
		vkMapMemory(device, ds.dlMem, 0, bits.size(), 0, &da);
		memcpy(da, bits.data(), bits.size());
		vkUnmapMemory(device, ds.dlMem);

		ubo.vertTransformations[0].w = 0;
		ubo.color = ds.medianColor;
		std::copy_n(reinterpret_cast<uint8_t*>(&ubo), uboSize, bits.data());
		std::copy_n(reinterpret_cast<uint8_t*>(ubo.vertTransformations.data()), trafoSize, bits.data() + uboSize);
		vkMapMemory(device, ds.dlMem, ds.medianUboOffset, bits.size(), 0, &da);
		memcpy(da, bits.data(), bits.size());
		vkUnmapMemory(device, ds.dlMem);

		c++;
	}

	//vector of command buffers for batch rendering
	std::vector<VkCommandBuffer> line_batch_commands;
	int max_amt_of_lines = 0;
	for (auto drawList = drawlists.begin(); drawList != drawlists.end(); ++drawList) {
		if(drawList->show)
			max_amt_of_lines += drawList->indices.size();
	}
	int curIndex = 0;

	//counting the amount of active drawLists for histogramm rendering
	int activeDrawLists = 0;

	VkRenderPassBeginInfo passInfo{};
	passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	passInfo.renderPass = _pipelineInstance.renderPass;
	passInfo.framebuffer = _framebuffer;
	uint32_t width = _pipelineInstance.context.screenSize[0], height = _pipelineInstance.context.screenSize[1];
	passInfo.renderArea = {0, 0, width, height};
	passInfo.clearValueCount = 1;
	VkClearValue passClear{};
	passInfo.pClearValues = &passClear;

	//clearing the final color image
	VkClearColorValue clear{};
	VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	vkCmdClearColorImage(command_buffer, _plotImage, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
	for (auto drawList = drawlists.rbegin(); drawlists.rend() != drawList; ++drawList) {
		vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 0, {}, 0, {}, 0, {});
		vkCmdBeginRenderPass(command_buffer, &passInfo, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipelineInstance.pipelineInfo.pipeline);

		if (!drawList->show)
			continue;

		VkDeviceSize offsets[] = { 0 };
		//vkCmdBindVertexBuffers(g_PcPlotCommandBuffer, 0, 1, &drawList->buffer, offsets);
		vkCmdBindIndexBuffer(command_buffer, drawList->indexBuffer, 0, VK_INDEX_TYPE_UINT32);

		//binding the right ubo
		VkDescriptorSet descSets[2]{drawList->uboDescSet, drawList->dataDescriptorSet};
		if (globalSettings.renderSplines)
			vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipelineInstance.pipelineInfo.pipelineLayout, 0, 2, descSets, 0, nullptr);
		else
			vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipelineInstance.pipelineInfo.pipelineLayout, 0, 2, descSets, 0, nullptr);

		vkCmdSetLineWidth(command_buffer, 1.0f);

		//ready to draw with draw indexed
		uint32_t amtOfI = drawList->indices.size() * (order.size() + 3);
		vkCmdDrawIndexed(command_buffer, amtOfI, 1, 0, 0, 0);
		vkCmdEndRenderPass(command_buffer);

		VkUtil::transitionImageLayout(command_buffer, _intermediateImage, VK_FORMAT_R32_SFLOAT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);

		//resolving to the final image
		descSets[0] = _computeSet;
		descSets[1] = drawList->uboDescSet;
		vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineInstance.computeInfo.pipeline);
		vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineInstance.computeInfo.pipelineLayout, 0, 2, descSets, 0, nullptr);

		const uint32_t workWidth = 256, workHeight = 1;
		vkCmdDispatch(command_buffer, (width + workWidth - 1) / workWidth, (height + workHeight - 1) / workHeight, 1);

#ifdef PRINTRENDERTIME
		uint32_t boolSize;
		for (DataSet& ds : g_PcPlotDataSets) {
			if (ds.name == drawList->parentDataSet) {
				boolSize = ds.data.size();
				break;
			}
		}
		bool* active = new bool[boolSize];
		VkUtil::downloadData(g_Device, drawList->dlMem, drawList->activeIndicesBufferOffset, boolSize * sizeof(bool), active);
		for (int i = 0; i < boolSize; ++i) {
			if (active[i]) ++amtOfLines;
		}
		delete[] active;
		//amtOfLines += drawList->activeInd.size();
#endif
	}

	VkUtil::commitCommandBuffer(_pipelineInstance.context.queue, command_buffer);
	err = vkQueueWaitIdle(_pipelineInstance.context.queue); check_vk_result(err);

	vkFreeCommandBuffers(device, _pipelineInstance.context.commandPool, 1, &command_buffer);
}

PCRenderer::PipelineSingleton::PipelineSingleton(const VkUtil::Context& inContext, const PipelineInput& input){
    context = inContext;
	context.screenSize[0] = input.width;
	context.screenSize[1] = input.height;
    //----------------------------------------------------------------------------------------------
	//creating the pipeline for spline rendering
	//----------------------------------------------------------------------------------------------
    VkShaderModule shaderModules[5]{};
    auto vertexBytes = PCUtil::readByteFile(_vertexShader);
    shaderModules[0] = VkUtil::createShaderModule(context.device, vertexBytes);
    auto geometryBytes = PCUtil::readByteFile(_geometryShader);
    shaderModules[3] = VkUtil::createShaderModule(context.device, geometryBytes);
    auto fragmentBytes = PCUtil::readByteFile(_fragmentShader);
    shaderModules[4] = VkUtil::createShaderModule(context.device, fragmentBytes);

    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(float);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attributeDescription{};
    attributeDescription.binding = 0;
    attributeDescription.location = 0;
    attributeDescription.format = VK_FORMAT_UNDEFINED;
    attributeDescription.offset = 0;

    VkPipelineVertexInputStateCreateInfo vertexInfo{};
    vertexInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
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

    VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	multisampling.minSampleShading = 1.0f;
	multisampling.pSampleMask = nullptr;
	multisampling.alphaToCoverageEnable = VK_FALSE;
	multisampling.alphaToOneEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
	colorBlendAttachment.blendEnable = VK_TRUE;
	colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
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
    VkUtil::BlendInfo blendInfo;
    blendInfo.blendAttachment = colorBlendAttachment;
    blendInfo.createInfo = colorBlending;

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings.push_back(uboLayoutBinding);

    //VkUtil::createDescriptorSetLayout(context.device, bindings, &pipelineInfo.descriptorSetLayout);

    std::vector<VkDescriptorSetLayout> descriptorSetLayouts{input.uniformLayout, input.dataLayout};
    
    std::vector<VkDynamicState> dynamicStateVec{VK_DYNAMIC_STATE_LINE_WIDTH};

    VkUtil::createRenderPass(context.device, VkUtil::PASS_TYPE_FLOAT, &renderPass);

    VkUtil::createPipeline(context.device, &vertexInfo, input.width, input.height, dynamicStateVec, shaderModules, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY, &rasterizer, &multisampling, nullptr, &blendInfo, descriptorSetLayouts, &renderPass, &pipelineInfo.pipelineLayout, &pipelineInfo.pipeline);

	//----------------------------------------------------------------------------------------------
	//creating resolve compute pipeline
	//----------------------------------------------------------------------------------------------
	auto computeBytes = PCUtil::readByteFile(_computeShader);
	auto computeModule = VkUtil::createShaderModule(context.device, computeBytes);
	bindings.clear();
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	bindings.push_back(uboLayoutBinding);

	uboLayoutBinding.binding = 1;
	bindings.push_back(uboLayoutBinding);

	VkUtil::createDescriptorSetLayout(context.device, bindings, &computeInfo.descriptorSetLayout);

	descriptorSetLayouts = {computeInfo.descriptorSetLayout, input.uniformLayout};

	VkUtil::createComputePipeline(context.device, computeModule, descriptorSetLayouts, &computeInfo.pipelineLayout, &computeInfo.pipeline);
}

int PCRenderer::PipelineSingleton::_usageCount = 0;