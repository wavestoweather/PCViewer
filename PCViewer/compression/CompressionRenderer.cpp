#define NOSTATICS
#include "CompressionRenderer.hpp"
#undef NOSTATICS

CompressionRenderer::CompressionRenderer(const CreateInfo& info) :
    _vkContext(info.context),
    _renderPass(info.renderPass),
    _dataLayout(info.dataLayout),
    _uniformLayout(info.uniformLayout)
{
    //----------------------------------------------------------------------------------------------
	//creating the pipeline for polyline rendering
	//----------------------------------------------------------------------------------------------
    VkShaderModule shaderModules[5]{};
    auto vertexBytes = PCUtil::readByteFile(_vertexShader);
    shaderModules[0] = VkUtil::createShaderModule(info.context.device, vertexBytes);
    auto fragmentBytes = PCUtil::readByteFile(_fragmentShader);
    shaderModules[4] = VkUtil::createShaderModule(info.context.device, fragmentBytes);

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
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings.push_back(uboLayoutBinding);

    //VkUtil::createDescriptorSetLayout(info.context.device, bindings, &pipelineInfo.descriptorSetLayout);

    std::vector<VkDescriptorSetLayout> descriptorSetLayouts{info.uniformLayout, info.dataLayout};
    
    std::vector<VkDynamicState> dynamicStateVec{VK_DYNAMIC_STATE_LINE_WIDTH};

    VkUtil::createPipeline(info.context.device, &vertexInfo, info.context.screenSize[0], info.context.screenSize[1], dynamicStateVec, shaderModules, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, &rasterizer, &multisampling, nullptr, &blendInfo, descriptorSetLayouts, &_renderPass, &_polyPipeInfo.pipelineLayout, &_polyPipeInfo.pipeline);

    //----------------------------------------------------------------------------------------------
	//creating the pipeline for spline rendering
	//----------------------------------------------------------------------------------------------
    vertexBytes = PCUtil::readByteFile(_vertexShader);
    shaderModules[0] = VkUtil::createShaderModule(info.context.device, vertexBytes);
    auto geometryBytes = PCUtil::readByteFile(_geometryShader);
    shaderModules[3] = VkUtil::createShaderModule(info.context.device, geometryBytes);
    fragmentBytes = PCUtil::readByteFile(_fragmentShader);
    shaderModules[4] = VkUtil::createShaderModule(info.context.device, fragmentBytes);

    VkUtil::createPipeline(info.context.device, &vertexInfo, info.context.screenSize[0], info.context.screenSize[1], dynamicStateVec, shaderModules, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, &rasterizer, &multisampling, nullptr, &blendInfo, descriptorSetLayouts, &_renderPass, &_splinePipeInfo.pipelineLayout, &_splinePipeInfo.pipeline);
}

CompressionRenderer::~CompressionRenderer(){
    _polyPipeInfo.vkDestroy(_vkContext);
    _splinePipeInfo.vkDestroy(_vkContext);
}

void CompressionRenderer::render(VkCommandBuffer commands, std::shared_ptr<HierarchyLoadManager> loadManager) 
{
    
}

void CompressionRenderer::release() 
{
    assert(_refCount > 0);  // we have a problem if release is called without having a reference
    if(--_refCount == 0){
        delete _singleton;
        _singleton = nullptr;
    }
}

// initialized to a null pointer
CompressionRenderer* CompressionRenderer::_singleton = {};

CompressionRenderer* CompressionRenderer::acquireReference(const CreateInfo& info)
{
    if(!_singleton)
        _singleton = new CompressionRenderer(info);

    _singleton->_refCount++;
    return _singleton;
}
