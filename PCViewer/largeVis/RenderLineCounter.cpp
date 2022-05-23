#include "RenderLineCounter.hpp"
#include "../PCUtil.h"

RenderLineCounter::RenderLineCounter(const CreateInfo& info):
    _vkContext(info.context), _aBins(1 << 6), _bBins(1 << 6)
{
    //----------------------------------------------------------------------------------------------
	// creating the pipeline for line counting
	//----------------------------------------------------------------------------------------------
    VkShaderModule shaderModules[5]{};
    auto vertexBytes = PCUtil::readByteFile(_vertexShader);
    shaderModules[0] = VkUtil::createShaderModule(info.context.device, vertexBytes);
    auto fragmentBytes = PCUtil::readByteFile(_fragmentShader);
    shaderModules[4] = VkUtil::createShaderModule(info.context.device, fragmentBytes);

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    VkDescriptorSetLayoutBinding b{};
    // infos
    b.binding = 0;
    b.descriptorCount = 1;
    b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    bindings.push_back(b);

    VkUtil::createDescriptorSetLayout(info.context.device, bindings, &_countPipeInfo.descriptorSetLayout);

    VkVertexInputBindingDescription bindingDescription[2]{};
    bindingDescription[0].binding = 0;
    bindingDescription[0].stride = sizeof(uint16_t);
    bindingDescription[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    bindingDescription[1].binding = 1;
    bindingDescription[1].stride = sizeof(uint16_t);
    bindingDescription[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attributeDescription[2]{};
    attributeDescription[0].binding = 0;
    attributeDescription[0].location = 0;
    attributeDescription[0].format = VK_FORMAT_R16_SFLOAT;
    attributeDescription[0].offset = 0;
    attributeDescription[1].binding = 1;
    attributeDescription[1].location = 1;
    attributeDescription[1].format = VK_FORMAT_R16_SFLOAT;
    attributeDescription[1].offset = 0;

    VkPipelineVertexInputStateCreateInfo vertexInfo{};
    vertexInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInfo.vertexAttributeDescriptionCount = sizeof(attributeDescription) / sizeof(attributeDescription[0]);
    vertexInfo.pVertexAttributeDescriptions = attributeDescription;
    vertexInfo.vertexBindingDescriptionCount = sizeof(bindingDescription) / sizeof(bindingDescription[0]);
    vertexInfo.pVertexBindingDescriptions = bindingDescription;

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
    
    std::vector<VkDynamicState> dynamicStateVec{};

    VkUtil::createImage(info.context.device, _aBins, _bBins, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &_countImage);
    VkMemoryRequirements memReq{};
    vkGetImageMemoryRequirements(info.context.device, _countImage, &memReq);
    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, memReq.size, VkUtil::findMemoryType(info.context.physicalDevice, memReq.memoryTypeBits, 0)};
    _imageMemSize = memReq.size;
    vkAllocateMemory(info.context.device, &allocInfo, nullptr, &_countImageMem);
    vkBindImageMemory(info.context.device, _countImage, _countImageMem, 0);
    VkUtil::createImageView(info.context.device, _countImage, VK_FORMAT_R32_SFLOAT, 1, VK_IMAGE_ASPECT_COLOR_BIT, &_countImageView);

    VkUtil::createRenderPass(info.context.device, VkUtil::PassType::PASS_TYPE_FLOAT, &_renderPass);
    VkUtil::createFrameBuffer(info.context.device, _renderPass, {_countImageView}, _aBins, _bBins, &_framebuffer);

    VkUtil::createPipeline(info.context.device, &vertexInfo, _aBins, _bBins, dynamicStateVec, shaderModules, VK_PRIMITIVE_TOPOLOGY_POINT_LIST, &rasterizer, &multisampling, nullptr, &blendInfo, {_countPipeInfo.descriptorSetLayout}, &_renderPass, &_countPipeInfo.pipelineLayout, &_countPipeInfo.pipeline);
}

void RenderLineCounter::countLines(VkCommandBuffer commands, const CountLinesInfo& info){
    // test counting
    const uint32_t size = 1 << 30;  // 2^30
    const uint32_t runs = 1;   //amount of separate draw calls to do the work
    std::vector<uint16_t> a1(size), a2(size);
    VkBuffer vA, vB, infos;
    VkDeviceMemory mA, mB, mOther;
    VkUtil::createBuffer(_vkContext.device, size * sizeof(uint16_t), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vA);
    VkUtil::createBuffer(_vkContext.device, size * sizeof(uint16_t), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vB);
    VkUtil::createBuffer(_vkContext.device, 4 * sizeof(uint32_t), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &infos);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    VkMemoryRequirements memReq{};

    vkGetBufferMemoryRequirements(_vkContext.device, vA, &memReq);
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(_vkContext.physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkAllocateMemory(_vkContext.device, &allocInfo, nullptr, &mA);
    vkBindBufferMemory(_vkContext.device, vA, mA, 0);

    vkGetBufferMemoryRequirements(_vkContext.device, vB, &memReq);
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(_vkContext.physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkAllocateMemory(_vkContext.device, &allocInfo, nullptr, &mB);
    vkBindBufferMemory(_vkContext.device, vB, mB, 0);

    vkGetBufferMemoryRequirements(_vkContext.device, infos, &memReq);
    allocInfo.allocationSize += memReq.size;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(_vkContext.physicalDevice, allocInfo.memoryTypeIndex | memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkAllocateMemory(_vkContext.device, &allocInfo, nullptr, &mOther);
    vkBindBufferMemory(_vkContext.device, infos, mOther, 0);

    struct Infos{
        uint32_t amtofDataPoints, aBins, bBins, padding;
    }cpuInfos {size, _aBins, _bBins, 0};
    VkUtil::uploadData(_vkContext.device, mOther, 0, sizeof(Infos), &cpuInfos);
    //filling with random numbers
    std::srand(std::time(nullptr));
    for(auto& e: a1) e = std::rand() & std::numeric_limits<uint16_t>::max();
    for(auto& e: a2) e = std::rand() & std::numeric_limits<uint16_t>::max();
    VkUtil::uploadData(_vkContext.device, mA, 0, a1.size() * sizeof(a1[0]), a1.data());
    VkUtil::uploadData(_vkContext.device, mB, 0, a2.size() * sizeof(a2[0]), a2.data());

    if(!_descSet)
        VkUtil::createDescriptorSets(_vkContext.device, {_countPipeInfo.descriptorSetLayout}, _vkContext.descriptorPool, &_descSet);

    VkUtil::updateDescriptorSet(_vkContext.device, infos, sizeof(Infos), 0, _descSet);

    VkClearValue clear;
    clear.color = {0,0,0,0};
    clear.depthStencil = {0, 0};
    VkUtil::beginRenderPass(commands, {clear}, _renderPass, _framebuffer, VkExtent2D{_aBins, _bBins});
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, _countPipeInfo.pipeline);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, _countPipeInfo.pipelineLayout, 0, 1, &_descSet, 0, nullptr);
    VkBuffer vertexBuffers[2]{vA, vB};
    VkDeviceSize offsets[2]{0, 0};
    vkCmdBindVertexBuffers(commands, 0, 2, vertexBuffers, offsets);
    for(int i = 0; i < runs; ++i){
        vkCmdDraw(commands, size / runs, 1, size_t(i * size) / runs, 0);
    }
    //vkCmdDraw(commands, size, 1, 0, 0);
    vkCmdEndRenderPass(commands);

    // done filling hte command buffer.
    // execution is done outside
}

void RenderLineCounter::countLinesPair(size_t dataSize, VkBuffer aData, VkBuffer bData, uint32_t aIndices, uint32_t bIndices, VkBuffer counts, bool clearCounts) const{
    VkUtil::updateDescriptorSet(_vkContext.device, _pairUniform, sizeof(PairInfos), 0, _pairSet);

    PairInfos infos{};
    infos.amtofDataPoints = dataSize;
    infos.aBins = aIndices;
    infos.bBins = bIndices;
    VkUtil::uploadData(_vkContext.device, _pairUniformMem, 0, sizeof(infos), &infos);

    VkCommandBuffer commands;
    VkUtil::createCommandBuffer(_vkContext.device, _vkContext.commandPool, &commands);
    if(clearCounts)
        vkCmdFillBuffer(commands, counts, 0, aIndices * bIndices * sizeof(uint32_t), 0);
    VkClearValue clear;
    clear.color = {0,0,0,0};
    clear.depthStencil = {0, 0};
    VkUtil::beginRenderPass(commands, {clear}, _renderPass, _framebuffer, VkExtent2D{_aBins, _bBins});
    VkBuffer vertexBuffers[2]{aData, bData};
    VkDeviceSize offsets[2]{0, 0};
    vkCmdBindVertexBuffers(commands, 0, 2, vertexBuffers, offsets);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _countPipeInfo.pipelineLayout, 0, 1, &_pairSet, 0, {});
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _countPipeInfo.pipeline);
    vkCmdDraw(commands, dataSize, 1, 0, 0);
    vkCmdEndRenderPass(commands);

    //TODO: conversion pipeline from float image to uint buffer

    VkUtil::commitCommandBuffer(_vkContext.queue, commands);
    auto res = vkQueueWaitIdle(_vkContext.queue); check_vk_result(res);

    vkFreeCommandBuffers(_vkContext.device, _vkContext.commandPool, 1, &commands);
}

RenderLineCounter* RenderLineCounter::_singleton = nullptr;    // init to nullptr

RenderLineCounter* RenderLineCounter::acquireReference(const CreateInfo& info){
    if(!_singleton)
        _singleton = new RenderLineCounter(info);
    _singleton->_refCount++;
    return _singleton;
}

void RenderLineCounter::tests(const CreateInfo& info){
    // small method to perform tests;
    
    // pipeline creation test
    auto t = acquireReference(info);

    VkCommandBuffer commands;
    VkUtil::createCommandBuffer(info.context.device, info.context.commandPool, &commands);
    // record commands
    t->countLines(commands, {});

    // commit and wait (includes timing of everything)
    {
        PCUtil::Stopwatch stopwatch(std::cout, "Render line counter runtime");
        VkUtil::commitCommandBuffer(info.context.queue, commands);
        vkQueueWaitIdle(info.context.queue);
    }

    //downloading results
    std::vector<float> res(t->_imageMemSize / sizeof(float));
    VkUtil::downloadImageData(t->_vkContext.device, t->_vkContext.physicalDevice, t->_vkContext.commandPool, t->_vkContext.queue, t->_countImage, VK_FORMAT_R32_SFLOAT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, t->_aBins, t->_bBins, 1, res.data(), t->_imageMemSize / sizeof(float));
    size_t count{};
    for(auto f: res){
        count += f;
    }

    t->release();
}

RenderLineCounter::~RenderLineCounter() 
{
    _countPipeInfo.vkDestroy(_vkContext);
    if(_descSet)
        vkFreeDescriptorSets(_vkContext.device, _vkContext.descriptorPool, 1, &_descSet);
    if(_pairSet)
        vkFreeDescriptorSets(_vkContext.device, _vkContext.descriptorPool, 1, &_pairSet);
    if(_pairUniform)
        vkDestroyBuffer(_vkContext.device, _pairUniform, nullptr);
    if(_pairUniformMem)
        vkFreeMemory(_vkContext.device, _pairUniformMem, nullptr);
}

void RenderLineCounter::release(){
    assert(_refCount > 0);
    if(--_refCount == 0){
        delete _singleton;
        _singleton = nullptr;
    }
}