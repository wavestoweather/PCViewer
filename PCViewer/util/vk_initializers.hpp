/*
* Initializers for Vulkan structures and objects used by the examples
* Saves lot of VK_STRUCTURE_TYPE assignments
* Some initializers are parameterized for convenience
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de,
* Copyright (C) 2022 by Josef Stumpfegger
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/


#pragma once
#include <vulkan/vulkan.h>

namespace util{
namespace vk{
namespace initializers{
inline VkMemoryAllocateInfo memoryAllocateInfo()
{
    VkMemoryAllocateInfo memAllocInfo {};
    memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    return memAllocInfo;
}

inline VkMappedMemoryRange mappedMemoryRange()
{
    VkMappedMemoryRange mappedMemoryRange {};
    mappedMemoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    return mappedMemoryRange;
}

inline VkCommandBufferAllocateInfo commandBufferAllocateInfo(
    VkCommandPool commandPool, 
    VkCommandBufferLevel level, 
    uint32_t bufferCount)
{
    VkCommandBufferAllocateInfo commandBufferAllocateInfo {};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = level;
    commandBufferAllocateInfo.commandBufferCount = bufferCount;
    return commandBufferAllocateInfo;
}

inline VkCommandPoolCreateInfo commandPoolCreateInfo(uint32_t queueFamilyIndex = 0)
{
    VkCommandPoolCreateInfo cmdPoolCreateInfo {};
    cmdPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
    return cmdPoolCreateInfo;
}

inline VkCommandBufferBeginInfo commandBufferBeginInfo()
{
    VkCommandBufferBeginInfo cmdBufferBeginInfo {};
    cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    return cmdBufferBeginInfo;
}

inline VkCommandBufferInheritanceInfo commandBufferInheritanceInfo()
{
    VkCommandBufferInheritanceInfo cmdBufferInheritanceInfo {};
    cmdBufferInheritanceInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
    return cmdBufferInheritanceInfo;
}

inline VkRenderPassBeginInfo renderPassBeginInfo(VkRenderPass renderPass = {}, VkFramebuffer framebuffer = {}, VkRect2D renderArea = {}, util::memory_view<const VkClearValue> clearValues = {})
{
    VkRenderPassBeginInfo renderPassBeginInfo {};
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.framebuffer = framebuffer;
    renderPassBeginInfo.renderArea = renderArea;
    renderPassBeginInfo.clearValueCount = clearValues.size();
    renderPassBeginInfo.pClearValues = clearValues.data();
    return renderPassBeginInfo;
}

inline VkRenderPassCreateInfo renderPassCreateInfo(const memory_view<VkAttachmentDescription> attachments = {}, const memory_view<VkSubpassDescription> subpasses = {}, const memory_view<VkSubpassDependency> dependencies = {})
{
    VkRenderPassCreateInfo renderPassCreateInfo {};
    renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassCreateInfo.attachmentCount = attachments.size();
    renderPassCreateInfo.pAttachments = attachments.data();
    renderPassCreateInfo.subpassCount = subpasses.size();
    renderPassCreateInfo.pSubpasses = subpasses.data();
    renderPassCreateInfo.dependencyCount = dependencies.size();
    renderPassCreateInfo.pDependencies = dependencies.data();
    return renderPassCreateInfo;
}

/** @brief Initialize an image memory barrier with no image transfer ownership */
inline VkImageMemoryBarrier imageMemoryBarrier(VkImage image, VkImageSubresourceRange range = {}, VkAccessFlags srcAccess = {}, VkAccessFlags dstAccess = {}, VkImageLayout oldLayout = {}, VkImageLayout newLayout = {}, uint32_t srcQueueIndex = VK_QUEUE_FAMILY_IGNORED, uint32_t dstQueueIndex = VK_QUEUE_FAMILY_IGNORED)
{
    VkImageMemoryBarrier imageMemoryBarrier {};
    imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageMemoryBarrier.srcAccessMask = srcAccess;
    imageMemoryBarrier.dstAccessMask = dstAccess;
    imageMemoryBarrier.oldLayout = oldLayout;
    imageMemoryBarrier.newLayout = newLayout;
    imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageMemoryBarrier.image = image;
    imageMemoryBarrier.subresourceRange = range;
    return imageMemoryBarrier;
}

/** @brief Initialize a buffer memory barrier with no image transfer ownership */
inline VkBufferMemoryBarrier bufferMemoryBarrier()
{
    VkBufferMemoryBarrier bufferMemoryBarrier {};
    bufferMemoryBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    return bufferMemoryBarrier;
}

inline VkMemoryBarrier memoryBarrier()
{
    VkMemoryBarrier memoryBarrier {};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    return memoryBarrier;
}

inline VkImageCreateInfo imageCreateInfo(VkFormat format, VkExtent3D extent, VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VkImageType type = VK_IMAGE_TYPE_2D, uint32_t mipLevels = 1, uint32_t arrayLayers = 1, VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT, VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL)
{
    VkImageCreateInfo imageCreateInfo {};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.format = format;
    imageCreateInfo.extent = extent;
    imageCreateInfo.usage = usage;
    imageCreateInfo.imageType = type;
    imageCreateInfo.mipLevels = mipLevels;
    imageCreateInfo.arrayLayers = arrayLayers;
    imageCreateInfo.samples = samples;
    imageCreateInfo.tiling = tiling;
    return imageCreateInfo;
}

inline VkSamplerCreateInfo samplerCreateInfo(VkFilter filter = VK_FILTER_NEAREST)
{
    VkSamplerCreateInfo samplerCreateInfo {};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = filter;
    samplerCreateInfo.minFilter = filter;
    samplerCreateInfo.maxAnisotropy = 1.0f;
    return samplerCreateInfo;
}

inline VkImageViewCreateInfo imageViewCreateInfo(VkImage image = {}, VkImageViewType viewType = {}, VkFormat format = {})
{
    VkImageViewCreateInfo imageViewCreateInfo {};
    imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewCreateInfo.image = image;
    imageViewCreateInfo.viewType = viewType;
    imageViewCreateInfo.format = format;
    imageViewCreateInfo.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    return imageViewCreateInfo;
}

inline VkFramebufferCreateInfo framebufferCreateInfo(VkRenderPass renderPass = {}, const memory_view<VkImageView> attachments = {}, uint32_t width = {}, uint32_t height = {}, uint32_t layers = {})
{
    VkFramebufferCreateInfo framebufferCreateInfo {};
    framebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferCreateInfo.renderPass = renderPass;
    framebufferCreateInfo.attachmentCount = attachments.size();
    framebufferCreateInfo.pAttachments = attachments.data();
    framebufferCreateInfo.width = width;
    framebufferCreateInfo.height = height;
    framebufferCreateInfo.layers = layers;
    return framebufferCreateInfo;
}

inline VkSemaphoreCreateInfo semaphoreCreateInfo()
{
    VkSemaphoreCreateInfo semaphoreCreateInfo {};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    return semaphoreCreateInfo;
}

inline VkFenceCreateInfo fenceCreateInfo(VkFenceCreateFlags flags = 0)
{
    VkFenceCreateInfo fenceCreateInfo {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = flags;
    return fenceCreateInfo;
}

inline VkEventCreateInfo eventCreateInfo()
{
    VkEventCreateInfo eventCreateInfo {};
    eventCreateInfo.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
    return eventCreateInfo;
}

inline VkSubmitInfo submitInfo()
{
    VkSubmitInfo submitInfo {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return submitInfo;
}

inline VkViewport viewport(
    float width,
    float height,
    float minDepth,
    float maxDepth)
{
    VkViewport viewport {};
    viewport.width = width;
    viewport.height = height;
    viewport.minDepth = minDepth;
    viewport.maxDepth = maxDepth;
    return viewport;
}

inline VkRect2D rect2D(
    int32_t width,
    int32_t height,
    int32_t offsetX,
    int32_t offsetY)
{
    VkRect2D rect2D {};
    rect2D.extent.width = width;
    rect2D.extent.height = height;
    rect2D.offset.x = offsetX;
    rect2D.offset.y = offsetY;
    return rect2D;
}

inline VkBufferCreateInfo bufferCreateInfo()
{
    VkBufferCreateInfo bufCreateInfo {};
    bufCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    return bufCreateInfo;
}

inline VkBufferCreateInfo bufferCreateInfo(
    VkBufferUsageFlags usage,
    VkDeviceSize size)
{
    VkBufferCreateInfo bufCreateInfo {};
    bufCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCreateInfo.usage = usage;
    bufCreateInfo.size = size;
    return bufCreateInfo;
}

inline VkDescriptorPoolCreateInfo descriptorPoolCreateInfo(
    const util::memory_view<VkDescriptorPoolSize> poolSizes,
    uint32_t maxSets, VkDescriptorPoolCreateFlags flags = 0)
{
    VkDescriptorPoolCreateInfo descriptorPoolInfo{};
    descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolInfo.flags = flags;
    descriptorPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    descriptorPoolInfo.pPoolSizes = poolSizes.data();
    descriptorPoolInfo.maxSets = maxSets;
    return descriptorPoolInfo;
}

inline VkDescriptorPoolSize descriptorPoolSize(
    VkDescriptorType type,
    uint32_t descriptorCount)
{
    VkDescriptorPoolSize descriptorPoolSize {};
    descriptorPoolSize.type = type;
    descriptorPoolSize.descriptorCount = descriptorCount;
    return descriptorPoolSize;
}

inline VkDescriptorSetLayoutBinding descriptorSetLayoutBinding(
    VkDescriptorType type,
    VkShaderStageFlags stageFlags,
    uint32_t binding = 0,
    uint32_t descriptorCount = 1)
{
    VkDescriptorSetLayoutBinding setLayoutBinding {};
    setLayoutBinding.descriptorType = type;
    setLayoutBinding.stageFlags = stageFlags;
    setLayoutBinding.binding = binding;
    setLayoutBinding.descriptorCount = descriptorCount;
    return setLayoutBinding;
}

inline VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo(
    const util::memory_view<VkDescriptorSetLayoutBinding> bindings)
{
    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo {};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.pBindings = bindings.data();
    descriptorSetLayoutCreateInfo.bindingCount = bindings.size();
    return descriptorSetLayoutCreateInfo;
}

inline VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo(
    const util::memory_view<VkDescriptorSetLayout> descSetLayouts = {}, const util::memory_view<VkPushConstantRange> pushConstantRanges = {}, VkPipelineLayoutCreateFlags createFlags = {})
{
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo {};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.flags = createFlags;
    pipelineLayoutCreateInfo.setLayoutCount = descSetLayouts.size();
    pipelineLayoutCreateInfo.pSetLayouts = descSetLayouts.data();
    pipelineLayoutCreateInfo.pushConstantRangeCount = pushConstantRanges.size();
    pipelineLayoutCreateInfo.pPushConstantRanges = pushConstantRanges.data();
    return pipelineLayoutCreateInfo;
}

inline VkDescriptorSetAllocateInfo descriptorSetAllocateInfo(
    VkDescriptorPool descriptorPool,
    util::memory_view<const VkDescriptorSetLayout> layouts)
{
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo {};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = layouts.size();
    descriptorSetAllocateInfo.pSetLayouts = layouts.data();
    return descriptorSetAllocateInfo;
}

inline VkDescriptorImageInfo descriptorImageInfo(VkSampler sampler, VkImageView imageView, VkImageLayout imageLayout)
{
    VkDescriptorImageInfo descriptorImageInfo {};
    descriptorImageInfo.sampler = sampler;
    descriptorImageInfo.imageView = imageView;
    descriptorImageInfo.imageLayout = imageLayout;
    return descriptorImageInfo;
}

inline VkWriteDescriptorSet writeDescriptorSet(
    VkDescriptorSet dstSet,
    VkDescriptorType type,
    uint32_t binding,
    VkDescriptorBufferInfo* bufferInfo,
    uint32_t descriptorCount = 1)
{
    VkWriteDescriptorSet writeDescriptorSet {};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet = dstSet;
    writeDescriptorSet.descriptorType = type;
    writeDescriptorSet.dstBinding = binding;
    writeDescriptorSet.pBufferInfo = bufferInfo;
    writeDescriptorSet.descriptorCount = descriptorCount;
    return writeDescriptorSet;
}

inline VkWriteDescriptorSet writeDescriptorSet(
    VkDescriptorSet dstSet,
    VkDescriptorType type,
    uint32_t binding,
    VkDescriptorImageInfo *imageInfo,
    uint32_t descriptorCount = 1)
{
    VkWriteDescriptorSet writeDescriptorSet {};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet = dstSet;
    writeDescriptorSet.descriptorType = type;
    writeDescriptorSet.dstBinding = binding;
    writeDescriptorSet.pImageInfo = imageInfo;
    writeDescriptorSet.descriptorCount = descriptorCount;
    return writeDescriptorSet;
}

inline VkVertexInputBindingDescription vertexInputBindingDescription(
    uint32_t binding,
    uint32_t stride,
    VkVertexInputRate inputRate)
{
    VkVertexInputBindingDescription vInputBindDescription {};
    vInputBindDescription.binding = binding;
    vInputBindDescription.stride = stride;
    vInputBindDescription.inputRate = inputRate;
    return vInputBindDescription;
}

inline VkVertexInputAttributeDescription vertexInputAttributeDescription(
    uint32_t binding,
    uint32_t location,
    VkFormat format,
    uint32_t offset)
{
    VkVertexInputAttributeDescription vInputAttribDescription {};
    vInputAttribDescription.location = location;
    vInputAttribDescription.binding = binding;
    vInputAttribDescription.format = format;
    vInputAttribDescription.offset = offset;
    return vInputAttribDescription;
}

inline VkPipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo(
    const memory_view<VkVertexInputBindingDescription> vertexBindingDescriptions = {},
    const memory_view<VkVertexInputAttributeDescription> vertexAttributeDescriptions = {}
)
{
    VkPipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo{};
    pipelineVertexInputStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    pipelineVertexInputStateCreateInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexBindingDescriptions.size());
    pipelineVertexInputStateCreateInfo.pVertexBindingDescriptions = vertexBindingDescriptions.data();
    pipelineVertexInputStateCreateInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributeDescriptions.size());
    pipelineVertexInputStateCreateInfo.pVertexAttributeDescriptions = vertexAttributeDescriptions.data();
    return pipelineVertexInputStateCreateInfo;
}

inline VkPipelineInputAssemblyStateCreateInfo pipelineInputAssemblyStateCreateInfo(
    VkPrimitiveTopology topology,
    VkPipelineInputAssemblyStateCreateFlags flags,
    VkBool32 primitiveRestartEnable)
{
    VkPipelineInputAssemblyStateCreateInfo pipelineInputAssemblyStateCreateInfo {};
    pipelineInputAssemblyStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    pipelineInputAssemblyStateCreateInfo.topology = topology;
    pipelineInputAssemblyStateCreateInfo.flags = flags;
    pipelineInputAssemblyStateCreateInfo.primitiveRestartEnable = primitiveRestartEnable;
    return pipelineInputAssemblyStateCreateInfo;
}

inline VkPipelineRasterizationStateCreateInfo pipelineRasterizationStateCreateInfo(
    VkPolygonMode polygonMode,
    VkCullModeFlags cullMode,
    VkFrontFace frontFace,
    VkPipelineRasterizationStateCreateFlags flags = 0)
{
    VkPipelineRasterizationStateCreateInfo pipelineRasterizationStateCreateInfo {};
    pipelineRasterizationStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    pipelineRasterizationStateCreateInfo.polygonMode = polygonMode;
    pipelineRasterizationStateCreateInfo.cullMode = cullMode;
    pipelineRasterizationStateCreateInfo.frontFace = frontFace;
    pipelineRasterizationStateCreateInfo.flags = flags;
    pipelineRasterizationStateCreateInfo.depthClampEnable = VK_FALSE;
    pipelineRasterizationStateCreateInfo.lineWidth = 1.0f;
    return pipelineRasterizationStateCreateInfo;
}

inline VkPipelineColorBlendAttachmentState pipelineColorBlendAttachmentState(
    VkColorComponentFlags colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |VK_COLOR_COMPONENT_A_BIT,
    VkBool32 blendEnable = VK_FALSE)
{
    VkPipelineColorBlendAttachmentState pipelineColorBlendAttachmentState {};
    pipelineColorBlendAttachmentState.colorWriteMask = colorWriteMask;
    pipelineColorBlendAttachmentState.blendEnable = blendEnable;
    return pipelineColorBlendAttachmentState;
}

inline VkPipelineColorBlendAttachmentState pipelineColorBlendAttachmentStateStandardAlphaBlend(
    VkColorComponentFlags colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |VK_COLOR_COMPONENT_A_BIT)
{
    VkPipelineColorBlendAttachmentState pipelineColorBlendAttachmentState {};
    pipelineColorBlendAttachmentState.colorWriteMask = colorWriteMask;
    pipelineColorBlendAttachmentState.blendEnable = VK_TRUE;
    pipelineColorBlendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    pipelineColorBlendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    pipelineColorBlendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
    pipelineColorBlendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    pipelineColorBlendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    pipelineColorBlendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
    return pipelineColorBlendAttachmentState;
}

inline VkPipelineColorBlendStateCreateInfo pipelineColorBlendStateCreateInfo(
    const memory_view<VkPipelineColorBlendAttachmentState> attachments)
{
    VkPipelineColorBlendStateCreateInfo pipelineColorBlendStateCreateInfo {};
    pipelineColorBlendStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    pipelineColorBlendStateCreateInfo.attachmentCount = attachments.size();
    pipelineColorBlendStateCreateInfo.pAttachments = attachments.data();
    return pipelineColorBlendStateCreateInfo;
}

inline VkPipelineDepthStencilStateCreateInfo pipelineDepthStencilStateCreateInfo(
    VkBool32 depthTestEnable,
    VkBool32 depthWriteEnable,
    VkCompareOp depthCompareOp = VK_COMPARE_OP_LESS)
{
    VkPipelineDepthStencilStateCreateInfo pipelineDepthStencilStateCreateInfo {};
    pipelineDepthStencilStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    pipelineDepthStencilStateCreateInfo.depthTestEnable = depthTestEnable;
    pipelineDepthStencilStateCreateInfo.depthWriteEnable = depthWriteEnable;
    pipelineDepthStencilStateCreateInfo.depthCompareOp = depthCompareOp;
    pipelineDepthStencilStateCreateInfo.back.compareOp = VK_COMPARE_OP_ALWAYS;
    return pipelineDepthStencilStateCreateInfo;
}

inline VkPipelineViewportStateCreateInfo pipelineViewportStateCreateInfo(
    uint32_t viewportCount,
    uint32_t scissorCount,
    VkPipelineViewportStateCreateFlags flags = 0)
{
    VkPipelineViewportStateCreateInfo pipelineViewportStateCreateInfo {};
    pipelineViewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    pipelineViewportStateCreateInfo.viewportCount = viewportCount;
    pipelineViewportStateCreateInfo.scissorCount = scissorCount;
    pipelineViewportStateCreateInfo.flags = flags;
    return pipelineViewportStateCreateInfo;
}

inline VkPipelineMultisampleStateCreateInfo pipelineMultisampleStateCreateInfo(
    VkSampleCountFlagBits rasterizationSamples,
    VkPipelineMultisampleStateCreateFlags flags = 0)
{
    VkPipelineMultisampleStateCreateInfo pipelineMultisampleStateCreateInfo {};
    pipelineMultisampleStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    pipelineMultisampleStateCreateInfo.rasterizationSamples = rasterizationSamples;
    pipelineMultisampleStateCreateInfo.flags = flags;
    return pipelineMultisampleStateCreateInfo;
}

inline VkPipelineDynamicStateCreateInfo pipelineDynamicStateCreateInfo(
    const memory_view<VkDynamicState> dynamicStates,
    VkPipelineDynamicStateCreateFlags flags = 0)
{
    VkPipelineDynamicStateCreateInfo pipelineDynamicStateCreateInfo{};
    pipelineDynamicStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    pipelineDynamicStateCreateInfo.pDynamicStates = dynamicStates.data();
    pipelineDynamicStateCreateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    pipelineDynamicStateCreateInfo.flags = flags;
    return pipelineDynamicStateCreateInfo;
}

inline VkPipelineTessellationStateCreateInfo pipelineTessellationStateCreateInfo(uint32_t patchControlPoints)
{
    VkPipelineTessellationStateCreateInfo pipelineTessellationStateCreateInfo {};
    pipelineTessellationStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
    pipelineTessellationStateCreateInfo.patchControlPoints = patchControlPoints;
    return pipelineTessellationStateCreateInfo;
}

inline VkGraphicsPipelineCreateInfo graphicsPipelineCreateInfo(
    const memory_view<VkPipelineShaderStageCreateInfo> shaderStages = {},
    VkPipelineLayout layout = {},
    VkRenderPass renderPass = {},
    VkPipelineCreateFlags flags = {})
{
    VkGraphicsPipelineCreateInfo pipelineCreateInfo {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.layout = layout;
    pipelineCreateInfo.renderPass = renderPass;
    pipelineCreateInfo.flags = flags;
    pipelineCreateInfo.stageCount = shaderStages.size();
    pipelineCreateInfo.pStages = shaderStages.data();
    pipelineCreateInfo.basePipelineIndex = -1;
    pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
    return pipelineCreateInfo;
}

inline VkComputePipelineCreateInfo computePipelineCreateInfo(
    VkPipelineLayout layout, 
    VkPipelineShaderStageCreateInfo shader_stage_create_info = {},
    VkPipelineCreateFlags flags = 0)
{
    VkComputePipelineCreateInfo computePipelineCreateInfo {};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.layout = layout;
    computePipelineCreateInfo.flags = flags;
    computePipelineCreateInfo.stage = shader_stage_create_info;
    return computePipelineCreateInfo;
}

inline VkPushConstantRange pushConstantRange(
    VkShaderStageFlags stageFlags,
    uint32_t size,
    uint32_t offset)
{
    VkPushConstantRange pushConstantRange {};
    pushConstantRange.stageFlags = stageFlags;
    pushConstantRange.offset = offset;
    pushConstantRange.size = size;
    return pushConstantRange;
}

inline VkBindSparseInfo bindSparseInfo()
{
    VkBindSparseInfo bindSparseInfo{};
    bindSparseInfo.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
    return bindSparseInfo;
}

/** @brief Initialize a map entry for a shader specialization constant */
inline VkSpecializationMapEntry specializationMapEntry(uint32_t constantID, uint32_t offset, size_t size)
{
    VkSpecializationMapEntry specializationMapEntry{};
    specializationMapEntry.constantID = constantID;
    specializationMapEntry.offset = offset;
    specializationMapEntry.size = size;
    return specializationMapEntry;
}

/** @brief Initialize a specialization constant info structure to pass to a shader stage */
inline VkSpecializationInfo specializationInfo(util::memory_view<VkSpecializationMapEntry> mapEntries, util::memory_view<uint8_t> data)
{
    VkSpecializationInfo specializationInfo{};
    specializationInfo.mapEntryCount = static_cast<uint32_t>(mapEntries.size());
    specializationInfo.pMapEntries = mapEntries.data();
    specializationInfo.dataSize = data.size();
    specializationInfo.pData = data.data();
    return specializationInfo;
}

// Ray tracing related
inline VkAccelerationStructureGeometryKHR accelerationStructureGeometryKHR()
{
    VkAccelerationStructureGeometryKHR accelerationStructureGeometryKHR{};
    accelerationStructureGeometryKHR.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    return accelerationStructureGeometryKHR;
}

inline VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfoKHR()
{
    VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfoKHR{};
    accelerationStructureBuildGeometryInfoKHR.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    return accelerationStructureBuildGeometryInfoKHR;
}

inline VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfoKHR()
{
    VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfoKHR{};
    accelerationStructureBuildSizesInfoKHR.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    return accelerationStructureBuildSizesInfoKHR;
}

inline VkRayTracingShaderGroupCreateInfoKHR rayTracingShaderGroupCreateInfoKHR()
{
    VkRayTracingShaderGroupCreateInfoKHR rayTracingShaderGroupCreateInfoKHR{};
    rayTracingShaderGroupCreateInfoKHR.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    return rayTracingShaderGroupCreateInfoKHR;
}

inline VkRayTracingPipelineCreateInfoKHR rayTracingPipelineCreateInfoKHR()
{
    VkRayTracingPipelineCreateInfoKHR rayTracingPipelineCreateInfoKHR{};
    rayTracingPipelineCreateInfoKHR.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    return rayTracingPipelineCreateInfoKHR;
}

inline VkWriteDescriptorSetAccelerationStructureKHR writeDescriptorSetAccelerationStructureKHR()
{
    VkWriteDescriptorSetAccelerationStructureKHR writeDescriptorSetAccelerationStructureKHR{};
    writeDescriptorSetAccelerationStructureKHR.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    return writeDescriptorSetAccelerationStructureKHR;
}

inline VkShaderModuleCreateInfo shaderModuleCreateInfo(const util::memory_view<uint32_t> shader_bytes, VkShaderModuleCreateFlags create_flags = {}){
    VkShaderModuleCreateInfo shaderModuleCreateInfo{};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.codeSize = shader_bytes.byte_size();
    shaderModuleCreateInfo.pCode = shader_bytes.data();
    shaderModuleCreateInfo.flags = create_flags;
    return shaderModuleCreateInfo;
}

inline VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo(VkShaderStageFlagBits stage, VkShaderModule module, const VkSpecializationInfo* specialization_info = {}, VkPipelineShaderStageCreateFlags create_flags = {}, std::string_view entry_name = {"main"}){
    VkPipelineShaderStageCreateInfo  pipelineShaderStageCreateInfo{};
    pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineShaderStageCreateInfo.flags = create_flags;
    pipelineShaderStageCreateInfo.stage = stage;
    pipelineShaderStageCreateInfo.module = module;
    pipelineShaderStageCreateInfo.pName = entry_name.data();
    pipelineShaderStageCreateInfo.pSpecializationInfo = specialization_info;
    return pipelineShaderStageCreateInfo;
}

inline VkAttachmentDescription attachmentDescription(VkFormat format, VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT, VkAttachmentLoadOp loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, VkAttachmentStoreOp storeOp = VK_ATTACHMENT_STORE_OP_STORE, VkAttachmentLoadOp stencilLoad = VK_ATTACHMENT_LOAD_OP_DONT_CARE, VkAttachmentStoreOp stencilStore = VK_ATTACHMENT_STORE_OP_STORE, VkImageLayout initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, VkImageLayout finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR){
    VkAttachmentDescription attachmentDescription{};
    attachmentDescription.format = format;
    attachmentDescription.samples = samples;
    attachmentDescription.loadOp = loadOp;
    attachmentDescription.storeOp = storeOp;
    attachmentDescription.stencilLoadOp = stencilLoad;
    attachmentDescription.stencilStoreOp = stencilStore;
    attachmentDescription.initialLayout = initialLayout;
    attachmentDescription.finalLayout = finalLayout;
    return attachmentDescription;
}

inline VkSubpassDescription subpassDescription(VkPipelineBindPoint bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, const memory_view<VkAttachmentReference> inputAttechments = {}, const memory_view<VkAttachmentReference> colorAttachments = {}, const memory_view<VkAttachmentReference> resolveAttachments = {}, const memory_view<VkAttachmentReference> depthStencilAttachment = {}, const memory_view<uint32_t> preserveAttachments = {}){
    VkSubpassDescription subpassDescription{};
    subpassDescription.pipelineBindPoint = bindPoint;
    subpassDescription.inputAttachmentCount = inputAttechments.size();
    subpassDescription.pInputAttachments = inputAttechments.data();
    subpassDescription.colorAttachmentCount = colorAttachments.size();
    subpassDescription.pColorAttachments = colorAttachments.data();
    subpassDescription.pResolveAttachments = resolveAttachments.data();
    subpassDescription.pDepthStencilAttachment = depthStencilAttachment.data();
    subpassDescription.preserveAttachmentCount = preserveAttachments.size();
    subpassDescription.pPreserveAttachments = preserveAttachments.data();
    return subpassDescription;
}

inline VkPhysicalDeviceFeatures2 physicalDeviceFeatures2(){
    VkPhysicalDeviceFeatures2 physicalDeviceFeatures2{};
    physicalDeviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    return physicalDeviceFeatures2;
}

inline VkPhysicalDeviceVulkan11Features physicalDeviceVulkan11Features(){
    VkPhysicalDeviceVulkan11Features physicalDeviceVulkan11Features{};
    physicalDeviceVulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    return physicalDeviceVulkan11Features;
}

inline VkPhysicalDeviceVulkan12Features physicalDeviceVulkan12Features(){
    VkPhysicalDeviceVulkan12Features physicalDeviceVulkan12Features{};
    physicalDeviceVulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    return physicalDeviceVulkan12Features;
}

inline VkPhysicalDeviceVulkan13Features physicalDeviceVulkan13Features(){
    VkPhysicalDeviceVulkan13Features physicalDeviceVulkan13Features{};
    physicalDeviceVulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    return physicalDeviceVulkan13Features;
}

inline VkDebugUtilsObjectNameInfoEXT debugUtilsObjectNameInfoEXT(VkObjectType objectType, uint64_t objectHandle, std::string_view objectName){
    VkDebugUtilsObjectNameInfoEXT debugUtilsObjectNameInfoEXT{};
    debugUtilsObjectNameInfoEXT.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    debugUtilsObjectNameInfoEXT.objectType = objectType;
    debugUtilsObjectNameInfoEXT.objectHandle = objectHandle;
    debugUtilsObjectNameInfoEXT.pObjectName = objectName.data();
    return debugUtilsObjectNameInfoEXT;
}
}
}
}