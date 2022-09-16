#include "Renderer.hpp"
#include "../PCUtil.h"
#include "../range.hpp"
#include <numeric>

namespace compression{
Renderer::Renderer(const CreateInfo& info) :
    _vkContext(info.context),
    _renderPass(info.renderPass),
    _framebuffer(info.framebuffer)
{
    updatePipeline(info);
}

Renderer::~Renderer(){
    _polyPipeInfo.vkDestroy(_vkContext);
    _splinePipeInfo.vkDestroy(_vkContext);
    _histogramPipeInfo.vkDestroy(_vkContext);
    if(_heatmapSetLayout)
        vkDestroyDescriptorSetLayout(_vkContext.device, _heatmapSetLayout, nullptr);
    for(auto b: _indexBuffers)
        vkDestroyBuffer(_vkContext.device, b, nullptr);
    if(_indexBufferMem)
        vkFreeMemory(_vkContext.device, _indexBufferMem, nullptr);
    if(_fence)
        vkDestroyFence(_vkContext.device, _fence, nullptr);
}
void Renderer::updatePipeline(const CreateInfo& info){
    _renderPass = info.renderPass;
    _framebuffer = info.framebuffer;
    _polyPipeInfo.vkDestroy(_vkContext);
    _splinePipeInfo.vkDestroy(_vkContext);
    _infoDescSets.clear();

    //----------------------------------------------------------------------------------------------
	//creating the pipeline for polyline rendering
	//----------------------------------------------------------------------------------------------
    VkShaderModule shaderModules[5]{};
    auto vertexBytes = PCUtil::readByteFile(_vertexShader);
    shaderModules[0] = VkUtil::createShaderModule(info.context.device, vertexBytes);
    auto geometryBytes = PCUtil::readByteFile(_geometryShader);
    shaderModules[3] = VkUtil::createShaderModule(info.context.device, geometryBytes);
    auto fragmentBytes = PCUtil::readByteFile(_fragmentShader);
    shaderModules[4] = VkUtil::createShaderModule(info.context.device, fragmentBytes);

    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(float);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attributeDescription{};
    attributeDescription.binding = 0;
    attributeDescription.location = 0;
    attributeDescription.format = VK_FORMAT_R32_UINT;
    attributeDescription.offset = 0;

    VkPipelineVertexInputStateCreateInfo vertexInfo{};
    vertexInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInfo.vertexAttributeDescriptionCount = 1;
    vertexInfo.pVertexAttributeDescriptions = &attributeDescription;
    vertexInfo.vertexBindingDescriptionCount = 1;
    vertexInfo.pVertexBindingDescriptions = &bindingDescription;

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
	multisampling.rasterizationSamples = info.sampleCount;
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
    VkUtil::BlendInfo blendInfo;
    blendInfo.blendAttachment = colorBlendAttachment;
    blendInfo.createInfo = colorBlending;

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    bindings.push_back(uboLayoutBinding);

    VkUtil::createDescriptorSetLayout(info.context.device, bindings, &_polyPipeInfo.descriptorSetLayout);

    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    VkUtil::createDescriptorSetLayout(info.context.device, bindings, &_heatmapSetLayout);

    std::vector<VkDescriptorSetLayout> descriptorSetLayouts{_polyPipeInfo.descriptorSetLayout, _heatmapSetLayout};
    
    std::vector<VkDynamicState> dynamicStateVec{};

    std::vector<VkPushConstantRange> pushConstants{};
    pushConstants.push_back({VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants)});    // attribute a and b
    //pushConstants.push_back({VK_SHADER_STAGE_VERTEX_BIT, 4, 4});    // attribute b

    VkUtil::createPipeline(info.context.device, &vertexInfo, info.context.screenSize[0], info.context.screenSize[1], dynamicStateVec, shaderModules, VK_PRIMITIVE_TOPOLOGY_POINT_LIST, &rasterizer, &multisampling, nullptr, &blendInfo, descriptorSetLayouts, &_renderPass, &_polyPipeInfo.pipelineLayout, &_polyPipeInfo.pipeline, pushConstants);

    //----------------------------------------------------------------------------------------------
	//creating the pipeline for spline rendering
	//----------------------------------------------------------------------------------------------
    vertexBytes = PCUtil::readByteFile(_vertexShader);
    shaderModules[0] = VkUtil::createShaderModule(info.context.device, vertexBytes);
    geometryBytes = PCUtil::readByteFile(_geometryShader);
    shaderModules[3] = VkUtil::createShaderModule(info.context.device, geometryBytes);
    fragmentBytes = PCUtil::readByteFile(_fragmentShader);
    shaderModules[4] = VkUtil::createShaderModule(info.context.device, fragmentBytes);

    VkUtil::createPipeline(info.context.device, &vertexInfo, info.context.screenSize[0], info.context.screenSize[1], dynamicStateVec, shaderModules, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, &rasterizer, &multisampling, nullptr, &blendInfo, descriptorSetLayouts, &_renderPass, &_splinePipeInfo.pipelineLayout, &_splinePipeInfo.pipeline, pushConstants);

    VkUtil::createDescriptorSets(info.context.device, {_heatmapSetLayout}, info.context.descriptorPool, &_heatmapSet);
    VkUtil::updateImageDescriptorSet(info.context.device, info.heatmapSampler, info.heatmapView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, _heatmapSet);

    //----------------------------------------------------------------------------------------------
	//creating the pipeline for spline rendering
	//----------------------------------------------------------------------------------------------
    vertexBytes = PCUtil::readByteFile(_histogrammVertexShader);
    shaderModules[0] = VkUtil::createShaderModule(info.context.device, vertexBytes);
    shaderModules[3] = {};
    fragmentBytes = PCUtil::readByteFile(_fragmentShader);
    shaderModules[4] = VkUtil::createShaderModule(info.context.device, fragmentBytes);

    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;

    vertexInfo.vertexAttributeDescriptionCount = 0;
    vertexInfo.vertexBindingDescriptionCount = 0;

    pushConstants = {{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(HistogramPushConstants)}};
    VkUtil::createPipeline(info.context.device, &vertexInfo, info.context.screenSize[0], info.context.screenSize[1], dynamicStateVec, shaderModules, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, &rasterizer, &multisampling, nullptr, &blendInfo, {}, &_renderPass, &_histogramPipeInfo.pipelineLayout, &_histogramPipeInfo.pipeline, pushConstants);

    _fence = VkUtil::createFence(_vkContext.device, 0);
}

void Renderer::render(const RenderInfo& renderInfo) 
{
	//creating the descriptor set for rendering if not available
    if(_infoDescSets.count(renderInfo.drawListId) == 0){
        VkUtil::createDescriptorSets(_vkContext.device, {_polyPipeInfo.descriptorSetLayout}, _vkContext.descriptorPool, &_infoDescSets[renderInfo.drawListId]);
        VkUtil::updateDescriptorSet(_vkContext.device, renderInfo.attributeInformation, VK_WHOLE_SIZE, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _infoDescSets[renderInfo.drawListId]);
    }
    // render pass will clear the standard attachement if in renderInfo the clear bool is set
    VkCommandBuffer commands = renderInfo.renderCommands;
    //VkUtil::createCommandBuffer(_vkContext.device, _vkContext.commandPool, &commands);
    //VkUtil::beginRenderPass(commands, {}, _renderPass, _framebuffer, VkExtent2D{_vkContext.screenSize[0], _vkContext.screenSize[1]});
    if(renderInfo.clear){
        VkClearAttachment attachment{};
        attachment.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        attachment.clearValue.color.float32[0] = .0;
        attachment.clearValue.color.float32[1] = .0;
        attachment.clearValue.color.float32[2] = .0;
        attachment.clearValue.color.float32[3] = .0;
        VkClearRect clearRect{};
        clearRect.baseArrayLayer = 0;
        clearRect.layerCount = 1;
        clearRect.rect.extent.width = _vkContext.screenSize[0];
        clearRect.rect.extent.height = _vkContext.screenSize[1];
        //vkCmdClearAttachments(commands, 1, &attachment, 1, &clearRect);
    }
    
    switch(renderInfo.renderType){
    case RenderType::Polyline:{
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, _polyPipeInfo.pipeline);
        for(int i: irange(renderInfo.counts)){
            auto aAxis = renderInfo.axes[i].first, bAxis = renderInfo.axes[i].second;
            if(!renderInfo.attributeActive[aAxis] || !renderInfo.attributeActive[bAxis])
                continue;
            //assert(renderInfo.attributeAxisSizes * renderInfo.attributeAxisSizes == renderInfo.countSizes || "Somethings wrong with the bins");
            PushConstants pc{aAxis, bAxis, renderInfo.attributeAxisSizes[i], renderInfo.attributeAxisSizes[i]};
            vkCmdPushConstants(commands, _polyPipeInfo.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
            std::array<VkDescriptorSet, 2> descSets{_infoDescSets[renderInfo.drawListId], _heatmapSet};
            vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, _polyPipeInfo.pipelineLayout, 0, descSets.size(), descSets.data(), 0, {});
            VkDeviceSize offsets[1]{0};
            vkCmdBindVertexBuffers(commands, 0, 1, renderInfo.counts.data() + i, offsets);
            vkCmdDraw(commands, pc.aSize * pc.bSize, 1, 0, 0);
        }
        break;
        }
    case RenderType::Spline: {
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, _splinePipeInfo.pipeline);
        for(int i: irange(renderInfo.counts)){
            auto aAxis = renderInfo.axes[i].first, bAxis = renderInfo.axes[i].second;
            PushConstants pc{aAxis, bAxis};
            vkCmdPushConstants(commands, _splinePipeInfo.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
            vkCmdBindVertexBuffers(commands, 0, 1, renderInfo.counts.data() + i, {});

        }
        break;
        }
    case RenderType::PriorityPolyline: {
        if(_indexBuffers.empty() ||renderInfo.attributeAxisSizes[0] * renderInfo.attributeAxisSizes[1] > _prevIndexSize){
            //std::cout << "Attempted render with prev size " << _prevIndexSize << " and current size " << renderInfo.attributeAxisSizes[0] * renderInfo.attributeAxisSizes[1]  << std::endl; 
            return;
        }
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, _polyPipeInfo.pipeline);
        for(int i: irange(renderInfo.counts)){
            auto aAxis = renderInfo.axes[i].first, bAxis = renderInfo.axes[i].second;
            if(!renderInfo.attributeActive[aAxis] || !renderInfo.attributeActive[bAxis])
                continue;
            //assert(renderInfo.attributeAxisSizes * renderInfo.attributeAxisSizes == renderInfo.countSizes || "Somethings wrong with the bins");
            PushConstants pc{aAxis, bAxis, renderInfo.attributeAxisSizes[i], renderInfo.attributeAxisSizes[i]};
            vkCmdPushConstants(commands, _polyPipeInfo.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
            std::array<VkDescriptorSet, 2> descSets{_infoDescSets[renderInfo.drawListId], _heatmapSet};
            vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, _polyPipeInfo.pipelineLayout, 0, descSets.size(), descSets.data(), 0, {});
            VkDeviceSize offsets[1]{0};
            vkCmdBindVertexBuffers(commands, 0, 1, renderInfo.counts.data() + i, offsets);
            vkCmdBindIndexBuffer(commands, _indexBuffers[i], 0, VK_INDEX_TYPE_UINT32);
            assert(pc.aSize * pc.bSize <= _prevIndexSize);
            vkCmdDrawIndexed(commands, pc.aSize * pc.bSize, 1, 0, 0, 0);
        }
        break;
        }
    case RenderType::PrioritySpline: {
        // todo implement
        }
    default:{
        throw std::runtime_error{"LargeVis::Renderer::render(...) unknown render type requested: " + std::to_string(int(renderInfo.renderType))};
        }
    }

    //vkCmdEndRenderPass(commands);
    //PCUtil::Stopwatch renderWatch(std::cout, "compression::Renderer::render()");
    //VkUtil::commitCommandBuffer(_vkContext.queue, commands);
    //auto res = vkQueueWaitIdle(_vkContext.queue); check_vk_result(res);
//
    //vkFreeCommandBuffers(_vkContext.device, _vkContext.commandPool, 1, &commands);
}

void Renderer::renderHistogram(const HistogramRenderInfo& info){
    HistogramPushConstants pc{};
    pc.histValues = info.histValues;
    pc.histValuesCount = info.histValuesCount;
    pc.yLow = info.yLow;
    pc.yHigh = info.yHigh;
    pc.xStart = info.xStart;
    pc.xEnd = info.xEnd;
    pc.alpha = info.alpha;
    vkCmdPushConstants(info.renderCommands, _histogramPipeInfo.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
    vkCmdBindPipeline(info.renderCommands, VK_PIPELINE_BIND_POINT_GRAPHICS, _histogramPipeInfo.pipeline);
    vkCmdDraw(info.renderCommands, info.histValuesCount * 2, 1, 0, 0);
}

void Renderer::updatePriorityIndexlists(const IndexlistUpdateInfo& info){
    //std::cout << "[update indexlist] " << PCUtil::toReadableString(info.countSizes) << std::endl;
    const uint32_t countByteSize = info.countSizes[0] * sizeof(uint32_t);
    // creating the index buffers
    if(info.countSizes[0] > _prevIndexSize){
        for(auto b: _indexBuffers)
            vkDestroyBuffer(_vkContext.device, b, nullptr);
        if(_indexBufferMem)
            vkFreeMemory(_vkContext.device, _indexBufferMem, nullptr);

        std::vector<size_t> sizes(info.counts.size(), countByteSize);
        std::vector<VkBufferUsageFlags> usages(info.counts.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        std::tie(_indexBuffers, _indexBufferOffsets, _indexBufferMem) = VkUtil::createMultiBufferBound(_vkContext, sizes, usages, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    }

    PCUtil::Stopwatch stopwatch(std::cout, "update priority indexlist");
    // creating a staging buffer
    auto [b, o, m] = VkUtil::createMultiBufferBound(_vkContext, {countByteSize}, {VK_BUFFER_USAGE_TRANSFER_DST_BIT}, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    void* gpuMem;
    vkMapMemory(_vkContext.device, m, 0, countByteSize, 0, &gpuMem);
    //downlaoding the counts 1 by 1 and sorting them, creating the indexlist
    std::vector<uint32_t> counts(info.countSizes[0]);
    for(int i: irange(info.counts)){
        {
            VkCommandBuffer commands;
            std::scoped_lock lock(*_vkContext.queueMutex);
            VkUtil::createCommandBuffer(_vkContext.device, _vkContext.commandPool, &commands);
            VkUtil::copyBuffer(commands, info.counts[i], b[0], countByteSize, 0, 0);
            VkUtil::commitCommandBuffer(_vkContext.queue, commands, _fence);
            check_vk_result(vkWaitForFences(_vkContext.device, 1, &_fence, VK_TRUE, 5e9));
            vkResetFences(_vkContext.device, 1, &_fence);
            vkFreeCommandBuffers(_vkContext.device, _vkContext.commandPool, 1, &commands);
        }
        // able to download data and sort
        std::memcpy(counts.data(), gpuMem, countByteSize);
        std::vector<uint32_t> indices(counts.size());
        std::iota(indices.begin(), indices.end(), 0);       // filling with increasing indices
        std::sort(indices.begin(), indices.end(), [&](uint32_t l, uint32_t r){return counts[l] < counts[r];});  // sorting according to the counts (low counts are drawn in the beginning)
        // upload to the indexbuffer
        VkUtil::uploadData(_vkContext.device, _indexBufferMem, _indexBufferOffsets[i], countByteSize, indices.data());
    }
    vkUnmapMemory(_vkContext.device, m);
    vkDestroyBuffer(_vkContext.device, b[0], nullptr);
    vkFreeMemory(_vkContext.device, m, nullptr);
    _prevIndexSize = info.countSizes[0];
}

void Renderer::updateFramebuffer(VkFramebuffer framebuffer, uint32_t newWidth, uint32_t newHeight){
    _framebuffer = framebuffer;
    _vkContext.screenSize[0] = newWidth;
    _vkContext.screenSize[1] = newHeight;
}

void Renderer::release() 
{
    assert(_refCount > 0);  // we have a problem if release is called without having a reference
    if(--_refCount == 0){
        delete _singleton;
        _singleton = nullptr;
    }
}

// initialized to a null pointer
Renderer* Renderer::_singleton = {};

Renderer* Renderer::acquireReference(const CreateInfo& info)
{
    if(!_singleton)
        _singleton = new Renderer(info);

    _singleton->_refCount++;
    return _singleton;
}
}