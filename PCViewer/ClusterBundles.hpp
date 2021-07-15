#pragma once
#include <vector>
#include <string>
#include <vulkan/vulkan.h>
#include <assert.h>

#include "Data.hpp"
#include "HistogramManager.h"
#include "VkUtil.h"
#include "PCUtil.h"
#include "TemplateList.hpp"

class ClusterBundles{
    //epsilon value for equal min max values
    float eps = 2e-2;
public:
    struct BundleVertex{
        float min, avg, max;        //axis is not need, as it can be induced by the vertex index
    };

    const Data* data;
    const float* baseColor;
    float haloWidth = 1;
    float haloColor[4]= {1.0f,1.0f,1.0f,1.0f};          //standard halo color is white
    std::vector<BundleVertex> bundlesData;              //for each axis an array of bundle data exists
    std::vector<float> alphaValues;                     //for each band the alpha value is stored

    ClusterBundles(): vkContext({}){};

    // A new Line Bundle should be created if histogram changes or the atribute boundaries are updated(Everything where the line bundles at the axis change)
    ClusterBundles(const VkUtil::Context& context, VkRenderPass renderPass, VkFramebuffer framebuffer, const std::string& drawList, const Data* data, const std::vector<std::pair<std::string, std::pair<float, float>>>& attributes, const std::vector<std::pair<uint32_t, bool>>& attributeOrder, const float* color, const std::vector<TemplateList*>& templateLists): 
    vkContext(context), vkRenderPass(renderPass), vkFrameBuffer(framebuffer), data(data), baseColor(color)
    {
        //going through each template list and calculating its alpha values and bundle vertices
        for(auto tlP: templateLists){
            alphaValues.push_back(100.0f * float(tlP->indices.size()) / data->size());
            for(int a = 0; a < attributes.size(); ++a){
                bundlesData.push_back({tlP->minMax[a].first, .5f * tlP->minMax[a].first +  .5f *  tlP->minMax[a].second, tlP->minMax[a].second});
                if(bundlesData.back().min == bundlesData.back().max){
                    bundlesData.back().min -= eps;
                    bundlesData.back().max += eps;
                }
            }
        }

        createVulkanPipeline();
        updateAttributeOrdering(attributeOrder);
    }

    ~ClusterBundles(){
        if(!vkContext.device) return; //empty line bundles object
        if(vkData) 
            vkDestroyBuffer(vkContext.device, vkData, nullptr);
        if(vkVertexBuffer) 
            vkDestroyBuffer(vkContext.device, vkVertexBuffer, nullptr);
        if(vkIndexBuffer) 
            vkDestroyBuffer(vkContext.device, vkIndexBuffer, nullptr);
        if(vkMemory)
            vkFreeMemory(vkContext.device, vkMemory, nullptr);
        if(vkDescriptorSet)
            vkFreeDescriptorSets(vkContext.device, vkContext.descriptorPool, 1, &vkDescriptorSet);
        if(--pipelineCounter != 0) return;      //pipeline is only cleand up if not used
        if(vkPipeline)
            vkDestroyPipeline(vkContext.device, vkPipeline, nullptr);
        if(vkPipelineLayout)
            vkDestroyPipelineLayout(vkContext.device, vkPipelineLayout, nullptr);
        if(vkDescriptorSetLayout)
            vkDestroyDescriptorSetLayout(vkContext.device, vkDescriptorSetLayout, nullptr);
    };

    ClusterBundles(const ClusterBundles&) = delete;
    ClusterBundles& operator=(const ClusterBundles&) = delete;

    // Can be used to reuse current axis grouping to save calculation
    void updateAttributeOrdering(const std::vector<std::pair<uint32_t, bool>>& attributeOrder){
        
        recreateVulkanBuffer(attributeOrder);
    }

    void recordDrawBundles(VkCommandBuffer buffer){
        if(!vkContext.device) return;         //empty LineBundles object
        std::vector<VkClearValue> clearValues;
        std::vector<float> data(9);
        std::copy(baseColor, baseColor + 4, data.data());
        data[4] = haloWidth;
        std::copy(haloColor, haloColor + 4, data.data() + 5);
        VkUtil::uploadData(vkContext.device, vkMemory, 0, 9 * sizeof(float), (void*)data.data());
        VkUtil::beginRenderPass(buffer, clearValues, vkRenderPass, vkFrameBuffer, {vkContext.screenSize[0], vkContext.screenSize[1]});
        vkCmdBindPipeline(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vkPipeline);
        vkCmdBindDescriptorSets(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vkPipelineLayout, 0, 1, &vkDescriptorSet, 0, nullptr);
        VkDeviceSize offsets[1]{0};
        vkCmdBindVertexBuffers(buffer, 0, 1, &vkVertexBuffer, offsets);
        vkCmdBindIndexBuffer(buffer, vkIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(buffer, indexCount, 1, 0, 0, 0);
        vkCmdEndRenderPass(buffer);
    }

    void setAxisInfosBuffer(VkBuffer buffer, uint32_t bufferSize){
        VkUtil::updateDescriptorSet(vkContext.device, buffer, bufferSize, 1, vkDescriptorSet);
    }

private:
    // All vulkan resources are handled in private are automatically allocated and deallocated
    uint32_t indexCount = 0;
    //vkData :  holds vertex information like axis ordering...
    //  structure: float4 Color, float HaloWidth, float4 haloColor, alphaValues[]
    //  alphaValues: float[] alpha vals
    //vkVertexBuffer: holds the min, mean and max points on each axis as a vertex
    //vkIndexBuffer: holds the index buffer for rendering
    VkBuffer vkData = 0, vkVertexBuffer = 0, vkIndexBuffer = 0;
    VkDeviceMemory vkMemory = 0;
    static uint32_t pipelineCounter;
    static VkPipeline vkPipeline;
    static VkPipelineLayout vkPipelineLayout;
    static VkDescriptorSetLayout vkDescriptorSetLayout;
    VkDescriptorSet vkDescriptorSet = 0;
    VkRenderPass vkRenderPass;
    VkFramebuffer vkFrameBuffer;

    VkUtil::Context vkContext = {};

    char* vertPath = "shader/cluster_band.vert.spv";
    char* geomPath = "shader/cluster_band.geom.spv";
    char* fragPath = "shader/band.frag.spv";

    //recreates vulkan data for rendering if already existing, else simply creates
    void recreateVulkanBuffer(const std::vector<std::pair<uint32_t, bool>>& attributeOrder){   
        if(vkData) 
            vkDestroyBuffer(vkContext.device, vkData, nullptr);
        if(vkVertexBuffer) 
            vkDestroyBuffer(vkContext.device, vkVertexBuffer, nullptr);
        if(vkIndexBuffer) 
            vkDestroyBuffer(vkContext.device, vkIndexBuffer, nullptr);
        if(vkMemory)
            vkFreeMemory(vkContext.device, vkMemory, nullptr);

        uint32_t dataSize = 0;      //alpha data and background data
        dataSize = alphaValues.size();
        dataSize *= sizeof(float);
        dataSize += sizeof(float) * 4; //color of colorband
        dataSize += sizeof(float) * 5; //halo color and halo width
        std::vector<uint8_t> gpuData(dataSize);
        *reinterpret_cast<float*>(&gpuData[0]) = baseColor[0];
        *reinterpret_cast<float*>(&gpuData[4]) = baseColor[1];
        *reinterpret_cast<float*>(&gpuData[8]) = baseColor[2];
        *reinterpret_cast<float*>(&gpuData[12]) = baseColor[3];
        *reinterpret_cast<float*>(&gpuData[16]) = haloWidth;
        *reinterpret_cast<float*>(&gpuData[20]) = haloColor[0];
        *reinterpret_cast<float*>(&gpuData[24]) = haloColor[1];
        *reinterpret_cast<float*>(&gpuData[28]) = haloColor[2];
        *reinterpret_cast<float*>(&gpuData[32]) = haloColor[3];
        uint32_t curPos = 36;
        for(float f: alphaValues){
            *reinterpret_cast<float*>(&gpuData[curPos]) = f;
            curPos+=4;
        }
        assert(curPos == gpuData.size());

        VkUtil::createBuffer(vkContext.device, gpuData.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &vkData);

        VkUtil::createBuffer(vkContext.device, bundlesData.size() * sizeof(BundleVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vkVertexBuffer);

        std::vector<uint32_t> indexData;
        uint32_t primitiveRestart = 0xFFFFFFFF;
        for(int l = 0; l < bundlesData.size() / attributeOrder.size(); ++l){
            bool first = true;
            uint32_t last = 0;
            for(auto& att: attributeOrder){
                if(!att.second) continue;
                if(first){
                    indexData.push_back(l * attributeOrder.size() + att.first);
                    first =  false;
                }
                last = l * attributeOrder.size() + att.first;
                indexData.push_back(last);
            }
            indexData.push_back(last);
            indexData.push_back(primitiveRestart);
        }
        

        VkUtil::createBuffer(vkContext.device, indexData.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &vkIndexBuffer);
        indexCount = indexData.size();

        VkMemoryRequirements memReq;
        VkMemoryAllocateInfo allocInfo{};
        uint32_t memFlags = 0;
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;

        vkGetBufferMemoryRequirements(vkContext.device, vkData, &memReq);
        allocInfo.allocationSize += memReq.size;
        memFlags |= memReq.memoryTypeBits;
        uint32_t vertexBufferOffset = allocInfo.allocationSize;
        vkGetBufferMemoryRequirements(vkContext.device, vkVertexBuffer, &memReq);
        allocInfo.allocationSize += memReq.size;
        memFlags |= memReq.memoryTypeBits;
        uint32_t indexBufferOffset = allocInfo.allocationSize;
        vkGetBufferMemoryRequirements(vkContext.device, vkIndexBuffer, &memReq);
        allocInfo.allocationSize += memReq.size;
        memFlags |= memReq.memoryTypeBits;
        allocInfo.memoryTypeIndex = VkUtil::findMemoryType(vkContext.physicalDevice, memFlags, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(vkContext.device, &allocInfo, nullptr, &vkMemory);

        vkBindBufferMemory(vkContext.device, vkData, vkMemory, 0);
        vkBindBufferMemory(vkContext.device, vkVertexBuffer, vkMemory, vertexBufferOffset);
        vkBindBufferMemory(vkContext.device, vkIndexBuffer, vkMemory, indexBufferOffset);

        VkUtil::uploadData(vkContext.device, vkMemory, 0, gpuData.size() * sizeof(gpuData[0]), gpuData.data());
        VkUtil::uploadData(vkContext.device, vkMemory, vertexBufferOffset, bundlesData.size() * sizeof(bundlesData[0]), bundlesData.data());
        VkUtil::uploadData(vkContext.device, vkMemory, indexBufferOffset, indexData.size() * sizeof(uint32_t), indexData.data());

        if(!vkDescriptorSet){
            std::vector<VkDescriptorSetLayout> layouts{vkDescriptorSetLayout};
            VkUtil::createDescriptorSets(vkContext.device, layouts, vkContext.descriptorPool, &vkDescriptorSet);
        }

        VkUtil::updateDescriptorSet(vkContext.device, vkData, dataSize, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, vkDescriptorSet);
    }

    // creates the pipeline including all necessary vulkan resources for rendering such as the descriptor set layout, the descriptor set...
    void createVulkanPipeline(){
        ++pipelineCounter;

        VkShaderModule shaderModules[5] = {};
	    //the vertex shader for the pipeline
	    std::vector<char> vertexBytes = PCUtil::readByteFile(vertPath);
	    shaderModules[0] = VkUtil::createShaderModule(vkContext.device, vertexBytes);
	    //the geometry shader for the pipeline
	    std::vector<char> geometryBytes = PCUtil::readByteFile(geomPath);
	    shaderModules[3] = VkUtil::createShaderModule(vkContext.device, geometryBytes);
	    //the fragment shader for the pipeline
	    std::vector<char> fragmentBytes = PCUtil::readByteFile(fragPath);
	    shaderModules[4] = VkUtil::createShaderModule(vkContext.device, fragmentBytes);

        std::vector<VkDescriptorSetLayoutBinding> bindings{{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL_GRAPHICS, nullptr}
                                                            ,{1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr}};
        VkUtil::createDescriptorSetLayout(vkContext.device, bindings, &vkDescriptorSetLayout);

        std::vector<VkVertexInputAttributeDescription> attributeDescriptions{{0,0,VK_FORMAT_R32G32B32_SFLOAT,0}};

        std::vector<VkVertexInputBindingDescription> bindingDescriptions{{0, 3 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX}};

        VkPipelineVertexInputStateCreateInfo vertexInfo{};
        vertexInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInfo.vertexAttributeDescriptionCount = attributeDescriptions.size();
        vertexInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
        vertexInfo.vertexBindingDescriptionCount = bindingDescriptions.size();
        vertexInfo.pVertexBindingDescriptions = bindingDescriptions.data();

        VkPipelineRasterizationStateCreateInfo rasterizationInfo{};
        rasterizationInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizationInfo.lineWidth = 1;
        rasterizationInfo.polygonMode = VK_POLYGON_MODE_FILL;

        VkPipelineMultisampleStateCreateInfo multisampleInfo{};
        multisampleInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampleInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo stencilInfo{};        //leave disabled
        stencilInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

        VkUtil::BlendInfo blendInfo{};
        blendInfo.blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blendInfo.blendAttachment.blendEnable = VK_TRUE;
        blendInfo.blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blendInfo.blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendInfo.blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        blendInfo.blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blendInfo.blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendInfo.createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        blendInfo.createInfo.logicOpEnable = VK_FALSE;
        blendInfo.createInfo.attachmentCount = 1;
        blendInfo.createInfo.pAttachments = &blendInfo.blendAttachment;
        
        VkUtil::createPipeline(vkContext.device, &vertexInfo, vkContext.screenSize[0], vkContext.screenSize[1], {}, shaderModules, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY, &rasterizationInfo, &multisampleInfo, &stencilInfo,
                            &blendInfo, {vkDescriptorSetLayout}, &vkRenderPass, &vkPipelineLayout, &vkPipeline);
    }
};

VkDescriptorSetLayout ClusterBundles::vkDescriptorSetLayout = 0;
VkPipeline ClusterBundles::vkPipeline = 0;
VkPipelineLayout ClusterBundles::vkPipelineLayout = 0;
uint32_t ClusterBundles::pipelineCounter = 0;