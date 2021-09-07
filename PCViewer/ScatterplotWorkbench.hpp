#pragma once

#include<vulkan/vulkan.h>
#include<vector>
#include<string>
#include"imgui/imgui.h"
#include"VkUtil.h"
#include"PCUtil.h"
#include"Structures.hpp"

// workbench for scatter plots
// all scatterplots are scatterplot matrices which can be reduced to only show wanted parameter combinations
class ScatterplotWorkbenchk{
public:
    struct ScatterPlot{
        int curWidth = 0, curHeight = 0;

        VkUtil::Context context;
        VkImage resultImage{};
        VkImageView resultImageView{};
        VkSampler sampler{};
        VkFramebuffer framebuffer{};
        VkDeviceMemory imageMemory{};
        VkBuffer uniformBuffer{};
        VkDeviceMemory uniformMemory{};
        VkDescriptorSet descriptorSet{};
        std::vector<bool> activeAttributes;
        std::vector<Attribute>& attributes;

        //external vulkan resources (dont have to be destroyed)
        VkRenderPass renderPass;

        ScatterPlot(VkUtil::Context context, int width, int height, VkRenderPass renderPass, VkDescriptorSetLayout descriptorSetLayout, VkBuffer data, VkBufferView activePoints, std::vector<Attribute>& attributes): 
        context(context), 
        renderPass(renderPass),
        activeAttributes(attributes.size(), true), 
        attributes(attributes){
            VkUtil::createDescriptorSets(context.device, {descriptorSetLayout}, context.descriptorPool, &descriptorSet);
            VkUtil::updateDescriptorSet(context.device, data, VK_WHOLE_SIZE, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorSet);
            VkUtil::updateTexelBufferDescriptorSet(context.device, activePoints, 1, descriptorSet);
            resizeImage(width, height);
            createBuffer();
        }
        ~ScatterPlot(){
            if(resultImage) vkDestroyImage(context.device, resultImage, nullptr);
            if(resultImageView) vkDestroyImageView(context.device, resultImageView, nullptr);
            if(imageMemory) vkFreeMemory(context.device, imageMemory, nullptr);
            if(framebuffer) vkDestroyFramebuffer(context.device, framebuffer, nullptr);
            if(sampler) vkDestroySampler(context.device, sampler, nullptr);
            if(uniformBuffer) vkDestroyBuffer(context.device, uniformBuffer, nullptr);
            if(uniformMemory) vkFreeMemory(context.device, uniformMemory, nullptr);
            if(descriptorSet) vkFreeDescriptorSets(context.device, context.descriptorPool, 1, &descriptorSet);
        }
        void resizeImage(int width, int height){
            VkResult res;
            if(!sampler){
                VkUtil::createImageSampler(context.device, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_FILTER_LINEAR, 1, 1, &sampler);
            }
            if(width != curWidth || height != curHeight){
                curWidth = width;
                curHeight = height;
                if(resultImage) vkDestroyImage(context.device, resultImage, nullptr);
                if(resultImageView) vkDestroyImageView(context.device, resultImageView, nullptr);
                if(imageMemory) vkFreeMemory(context.device, imageMemory, nullptr);
                if(framebuffer) vkDestroyFramebuffer(context.device, framebuffer, nullptr);
                VkUtil::createImage(context.device, width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &resultImage);
                
                VkMemoryRequirements memReq;
                vkGetImageMemoryRequirements(context.device, resultImage, &memReq);

                VkMemoryAllocateInfo memAlloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                memAlloc.allocationSize = memReq.size;
                uint32_t memBits = memReq.memoryTypeBits;
                memAlloc.memoryTypeIndex = VkUtil::findMemoryType(context.physicalDevice, memBits, 0);

                res = vkAllocateMemory(context.device, &memAlloc, nullptr, &imageMemory); check_vk_result(res);

                vkBindImageMemory(context.device, resultImage, imageMemory, 0);
                VkUtil::createImageView(context.device, resultImage, VK_FORMAT_R8G8B8A8_UNORM, 1, VK_IMAGE_ASPECT_COLOR_BIT, &resultImageView);
                
                VkUtil::createFrameBuffer(context.device, renderPass, {resultImageView}, width, height, &framebuffer);

                //transform image layout form undefined to shader read only optimal
                VkCommandBuffer command;
                VkUtil::createCommandBuffer(context.device, context.commandPool, &command);
                VkUtil::transitionImageLayout(command, resultImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                vkEndCommandBuffer(command);
                VkUtil::commitCommandBuffer(context.queue, command);
                res = vkQueueWaitIdle(context.queue);
                vkFreeCommandBuffers(context.device, context.commandPool, 1, &command);
            }
        }
        
        void createBuffer(){
            if(uniformBuffer) return;

            int uboByteSize;
            VkUtil::updateDescriptorSet(context.device, uniformBuffer, uboByteSize, 2, descriptorSet);
        }
        
        void draw(int index){
            ImGui::BeginChild(("Scatterplot" + std::to_string(index)).c_str());

            ImGui::EndChild();
        };

        void updateRender(VkCommandBuffer commandBuffer){
            
        };
    };

    ScatterplotWorkbenchk(VkUtil::Context context): context(context){
        createPipeline();
    }

    void addPlot(){
        VkBuffer tmp;
        VkBufferView view;
        std::vector<Attribute> tmpAttributes;
        scatterPlots.emplace_back(context, defaultWidth, defaultHeight, renderPass, descriptorSetLayout, tmp, view, tmpAttributes);
    }

    ~ScatterplotWorkbenchk(){
        if(pipeline) vkDestroyPipeline(context.device, pipeline, nullptr);
        if(pipelineLayout) vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
        if(descriptorSetLayout) vkDestroyDescriptorSetLayout(context.device, descriptorSetLayout, nullptr);
        if(renderPass) vkDestroyRenderPass(context.device, renderPass, nullptr);
    }

    void draw(){
        if(!active) return;
        ImGui::Begin("Scatterplot Workbench");
        int c = 0;
        for(ScatterPlot& s: scatterPlots){
            s.draw(c++);
        }
        ImGui::End();
    }

    void updateRenders(const std::vector<int>& attrIndices){
        VkCommandBuffer commandBuffer;
        for(ScatterPlot& s: scatterPlots){
            bool change = false;
            for(int i: attrIndices) change|=  s.activeAttributes[i];
            if(change) s.updateRender(commandBuffer);
        }
    }

    bool active = false;
    std::vector<ScatterPlot> scatterPlots;
    int defaultWidth = 800, defaultHeight = 800;
protected:
    VkPipeline pipeline{};
    VkPipelineLayout pipelineLayout{};
    VkDescriptorSetLayout descriptorSetLayout{};
    VkRenderPass renderPass{};

    VkUtil::Context context;

    void createPipeline(){
        VkShaderModule shaderModules[5] = {};
	    //the vertex shader for the pipeline
	    std::vector<char> vertexBytes = PCUtil::readByteFile("shader/scatter.vert.spv");
	    shaderModules[0] = VkUtil::createShaderModule(context.device, vertexBytes);
	    //the fragment shader for the pipeline
	    std::vector<char> fragmentBytes = PCUtil::readByteFile("shader/scatter.frag.spv");
	    shaderModules[4] = VkUtil::createShaderModule(context.device, fragmentBytes);

        VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount = 0;
	vertexInputInfo.pVertexBindingDescriptions = nullptr;
	vertexInputInfo.vertexAttributeDescriptionCount = 0;
	vertexInputInfo.pVertexAttributeDescriptions = nullptr;

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
	depthStencil.depthTestEnable = VK_FALSE;
	depthStencil.depthWriteEnable = VK_FALSE;
	depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
	depthStencil.depthBoundsTestEnable = VK_FALSE;
	depthStencil.minDepthBounds = 0;
	depthStencil.maxDepthBounds = 1.0f;

	//creating the descriptor set layout
	VkDescriptorSetLayoutBinding uboLayoutBinding = {};
	uboLayoutBinding.binding = 0;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	uboLayoutBinding.descriptorCount = 1;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	std::vector<VkDescriptorSetLayoutBinding> bindings;
	bindings.push_back(uboLayoutBinding);

	uboLayoutBinding.binding = 1;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
	bindings.push_back(uboLayoutBinding);

	uboLayoutBinding.binding = 2;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.stageFlags |= VK_SHADER_STAGE_FRAGMENT_BIT;
	bindings.push_back(uboLayoutBinding);

	VkUtil::createDescriptorSetLayout(context.device, bindings, &descriptorSetLayout);
	std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
	descriptorSetLayouts.push_back(descriptorSetLayout);

	VkUtil::createRenderPass(context.device, VkUtil::PASS_TYPE_COLOR_OFFLINE, &renderPass);
    VkUtil::createPipeline(context.device, &vertexInputInfo, 100, 100, dynamicStates, shaderModules, VK_PRIMITIVE_TOPOLOGY_POINT_LIST, &rasterizer, &multisampling, &depthStencil, &blendInfo, descriptorSetLayouts, &renderPass, &pipelineLayout, &pipeline);
    }
};