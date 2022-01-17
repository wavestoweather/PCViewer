#define NOSTATICS
#include "PCRenderer.hpp"
#undef NOSTATICS
#include "PCUtil.h"

PCRenderer::PCRenderer(const VkUtil::Context& context, uint32_t width, uint32_t height):
pipelineInstance(PipelineSingleton::getInstance(context))
{
    //creating the render resources
    

}

PCRenderer::PipelineSingleton::PipelineSingleton(const VkUtil::Context& inContext){
    context = inContext;
    //TODO implement
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

    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_ALL;

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

    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
}

int PCRenderer::PipelineSingleton::_usageCount = 0;