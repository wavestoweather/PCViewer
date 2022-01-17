#define NOSTATICS
#include "PCRenderer.hpp"
#undef NOSTATICS
#include "PCUtil.h"

PCRenderer::PCRenderer(const VkUtil::Context& context, uint32_t width, uint32_t height):
_pipelineInstance(PipelineSingleton::getInstance(context, {width, height}))
{
    //creating the render resources
    VkFormat intermediateFormat = VK_FORMAT_R32_UINT;
    VkUtil::createImage(context.device, width, height, intermediateFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &_intermediateImage);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    VkUtil::addImageToAllocInfo(context.device, _intermediateImage, allocInfo);
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(context.physicalDevice, allocInfo.memoryTypeIndex, 0);
    VkResult res = vkAllocateMemory(context.device, &allocInfo, nullptr, &_imageMemory); check_vk_result(res);

    VkUtil::createImageView(context.device, _intermediateImage, intermediateFormat, 1, VK_IMAGE_ASPECT_COLOR_BIT, &_intermediateView);
    
    std::vector<VkImageView> attachments{_intermediateView};
    VkUtil::createFrameBuffer(context.device, pipelineInstance.renderPass, attachments, width, height, &_framebuffer);

}

PCRenderer::~PCRenderer(){
    PipelineSingleton::notifyInstanceShutdown(_pipelineInstance);
    auto device = _pipelineInstance.context.device;
    if(_framebuffer) vkDestroyFramebuffer(device, _framebuffer, nullptr);
}

PCRenderer::PipelineSingleton::PipelineSingleton(const VkUtil::Context& inContext, const PipelineInput& input){
    context = inContext;
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
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    bindings.push_back(uboLayoutBinding);

    uboLayoutBinding.binding = 1;
    bindings.push_back(uboLayoutBinding);

    uboLayoutBinding.binding = 2;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings.push_back(uboLayoutBinding);
    VkUtil::createDescriptorSetLayout(context.device, bindings, &pipelineInfo.descriptorSetLayout);

    bindings.resize(1);
    VkUtil::createDescriptorSetLayout(context.device, bindings, &storageLayout);

    std::vector<VkDescriptorSetLayout> descriptorSetLayouts{pipelineInfo.descriptorSetLayout, storageLayout};
    
    std::vector<VkDynamicState> dynamicStateVec{VK_DYNAMIC_STATE_LINE_WIDTH};

    VkUtil::createRenderPass(context.device, VkUtil::PASS_TYPE_UINT32, &renderPass);

    VkUtil::createPipeline(context.device, &vertexInfo, input.width, input.height, dynamicStateVec, shaderModules, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY, &rasterizer, &multisampling, nullptr, &blendInfo, descriptorSetLayouts, &renderPass, &pipelineInfo.pipelineLayout, &pipelineInfo.pipeline);
}

int PCRenderer::PipelineSingleton::_usageCount = 0;