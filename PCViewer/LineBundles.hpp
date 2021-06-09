#pragma once
#include <vector>
#include <string>
#include <vulkan/vulkan.h>
#include <assert.h>

#include "Data.hpp"
#include "HistogramManager.h"
#include "VkUtil.h"
#include "PCUtil.h"

class LineBundles{
public:
    enum BundlingType{
        AxisBundling,
        HirachicalBundling
    };

    struct BundleData{
        float min, avg, max;
        uint32_t axis, groupNr;
    };

    const Data* data;
    const float* baseColor;
    std::vector<std::vector<BundleData>> bundlesData;       //for each axis an array of bundle data exists
    std::vector<std::vector<float>> alphaValues;            //for each pair of Bundles which have to be shown, an alpha value is stored here

    LineBundles(): vkContext({}){};

    // A new Line Bundle should be created if histogram changes or the atribute boundaries are updated(Everything where the line bundles at the axis change)
    // The histogram has to be added prior to creating the LineBundles object, otherwise an empty lineBundles object is returned
    // For attribute order change, attribute activation change try to use "updateAttributeOrdering".
    LineBundles(const VkUtil::Context& context, const std::string& drawList, const Data* data, HistogramManager* histManager, const std::vector<std::pair<std::string, std::pair<float, float>>>& attributes, const std::vector<std::pair<uint32_t, bool>>& attributeOrder, const float* color): vkContext(context), data(data), baseColor(color){
        //first group data at axes, then create vulkan instances for rendering
        assert(histManager->containsHistogram(drawList));
        HistogramManager::Histogram& hist = histManager->getHistogram(drawList);
        for(uint32_t a = 0; a < attributes.size(); ++a){
            bundlesData.push_back({});
            bool increasing = false;
            int groupStartIndex = -1;
            float avg = 0;
            float min = attributes[a].second.first;
            float max = 0;
            float avgValueCount = 0;
            float prevValue = 0;
            uint32_t groupNr = 0;
            for(int b = 0; b < hist.bins[a].size(); ++b){
                if(avg == 0 && hist.bins[a][b] == 0) continue;  //ignore 0 bins at the beginning
                if(!increasing && prevValue < hist.bins[a][b]){ //changing from decreasing to increasing
                    max = (float(b) / hist.bins[a].size()) * (attributes[a].second.second - attributes[a].second.first) + attributes[a].second.first;
                    avg /= avgValueCount;
                    bundlesData.back().push_back(BundleData{min, avg, max, a, groupNr++});
                    min = max;
                    avg = 0;
                    avgValueCount = 0;
                }
                if(increasing && prevValue > hist.bins[a][b]){  //changing from increasing to decreasing
                    increasing = false;
                }
                prevValue = hist.bins[a][b];
                avg += ((float(b) / hist.bins[a].size()) * (attributes[a].second.second - attributes[a].second.first) + attributes[a].second.first) * prevValue;
                avgValueCount += prevValue;
            }
            //adding eventually the last gropu which is not recognized if the values were decreasing all the time to the top
            if(avg != 0){
                avg /= avgValueCount;
                bundlesData.back().push_back({min, avg, attributes[a].second.second, a, groupNr++});
            }
        }
        updateAttributeOrdering(attributeOrder);
    }

    ~LineBundles(){
        if(!vkContext.device) return;
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
        if(vkRenderPass)
            vkDestroyRenderPass(vkContext.device, vkRenderPass, nullptr);
        if(vkDescriptorSetLayout)
            vkDestroyDescriptorSetLayout(vkContext.device, vkDescriptorSetLayout, nullptr);
    };

    // Can be used to reuse current axis grouping to save calculation
    void updateAttributeOrdering(const std::vector<std::pair<uint32_t, bool>>& attributeOrder){
        alphaValues.clear();
        uint32_t a = 0, b = 0;
        bool go = false;
        for(auto it = attributeOrder.begin(); it != attributeOrder.end(); ++it){
            if(!it->second) continue;
            if(!go){
                a = it->first;
                go = true;
            }
            else{
                b = it->first;
                alphaValues.push_back({});
                alphaValues.back().resize(bundlesData[a].size() * bundlesData[b].size(), 0);
                //a and b now hold the two adjacent axes where i have to cross product between all groups and check for line counts
                for(uint64_t i = 0; i < data->size(); ++i){     //go through all lines and check
                    float valA = (*data)(i,a);
                    float valB = (*data)(i,b);
                    int bundleIndexA = getBundleIndex(valA, a);
                    int bundleIndexB = getBundleIndex(valB, b);
                    int alphaIndex = bundleIndexA * bundlesData[b].size() + bundleIndexB;
                    alphaValues.back()[alphaIndex] += 1.0 / data->size();
                }
                a = b;
            }
        }

        recreateVulkanBuffer(attributeOrder);
    }

    void renderBundles(VkCommandBuffer buffer){
        if(!vkContext.device) return;         //empty LineBundles object
        vkCmdBindPipeline(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vkPipeline);
        vkCmdBindDescriptorSets(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vkPipelineLayout, 0, 1, &vkDescriptorSet, 0, nullptr);
        VkDeviceSize offsets[1]{0};
        vkCmdBindVertexBuffers(buffer, 0, 1, &vkVertexBuffer, offsets);
        vkCmdBindIndexBuffer(buffer, vkIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(buffer, indexCount, 1, 0, 0, 0);
    }

private:
    // All vulkan resources are handled in private are automatically allocated and deallocated
    uint32_t indexCount;
    VkBuffer vkData, vkVertexBuffer, vkIndexBuffer;
    VkDeviceMemory vkMemory;
    static uint32_t pipelineCounter;
    static VkPipeline vkPipeline;
    static VkPipelineLayout vkPipelineLayout;
    static VkRenderPass vkRenderPass;
    static VkDescriptorSetLayout vkDescriptorSetLayout;
    VkDescriptorSet vkDescriptorSet;

    const VkUtil::Context vkContext;

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

        uint32_t dataSize = 0;
        for(auto& pair: attributeOrder) if(pair.first > dataSize) dataSize = pair.first;
        dataSize *= sizeof(uint32_t);
        dataSize += sizeof(float) * 3; //color of colorband
        std::vector<uint8_t> gpuData(dataSize);
        *reinterpret_cast<float*>(&gpuData[0]) = baseColor[0];
        *reinterpret_cast<float*>(&gpuData[4]) = baseColor[1];
        *reinterpret_cast<float*>(&gpuData[8]) = baseColor[2];
        uint32_t curPos = 12, dataPos = gpuData.size(), axisIndex = 0;
        for(int i = 0; i < attributeOrder.size(); ++i){
            *reinterpret_cast<uint32_t*>(&gpuData[curPos]) = dataPos;
            curPos += 4;
            
            if(attributeOrder[i].second){
                gpuData.resize(gpuData.size() + alphaValues[axisIndex].size() * sizeof(float));
                for(float val: alphaValues[axisIndex]){
                    *reinterpret_cast<float*>(&gpuData[dataPos]) = val;
                    dataPos += 4;
                }
                ++axisIndex;
            }
        }

        VkUtil::createBuffer(vkContext.device, gpuData.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &vkData);
        std::vector<uint32_t> vertexDataOffsets(bundlesData.size(), 0);
        std::vector<uint8_t> vertexData;
        uint32_t curOffset = 0, index = 0;
        for(auto& bundle: bundlesData){
            vertexDataOffsets[index++] = curOffset / sizeof(BundleData);
            vertexData.resize(vertexData.size() + bundle.size() * sizeof(BundleData));
            std::copy(bundle.begin(), bundle.end(), reinterpret_cast<BundleData*>(&vertexData[curOffset]));
            curOffset += bundle.size() * sizeof(BundleData);
        }
        VkUtil::createBuffer(vkContext.device, vertexData.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vkVertexBuffer);

        std::vector<uint32_t> indexData;
        uint32_t primitiveRestart = 0xFFFFFFFF;
        uint32_t a = 0, b = 0;
        bool go = false;
        for(auto it = attributeOrder.begin(); it != attributeOrder.end(); ++it){
            if(!it->second) continue;
            if(!go){
                a = it->first;
                go = true;
            }
            else{
                b = it->first;
                //a and b now hold the two adjacent axes where i have to cross product between all groups and check for line counts
                for(int x = 0; x < bundlesData[a].size(); ++x){
                    for(int y = 0; y < bundlesData[b].size(); ++y){
                        indexData.push_back(vertexDataOffsets[a] + x);
                        indexData.push_back(vertexDataOffsets[a] + x);
                        indexData.push_back(vertexDataOffsets[b] + y);
                        indexData.push_back(vertexDataOffsets[b] + y);
                        indexData.push_back(primitiveRestart);
                    }
                }
            }
        }
        VkUtil::createBuffer(vkContext.device, indexData.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &vkIndexBuffer);

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

        VkUtil::uploadData(vkContext.device, vkMemory, 0, gpuData.size(), gpuData.data());
        VkUtil::uploadData(vkContext.device, vkMemory, vertexBufferOffset, vertexData.size(), vertexData.data());
        VkUtil::uploadData(vkContext.device, vkMemory, indexBufferOffset, indexData.size() * sizeof(uint32_t), indexData.data());

        if(!vkDescriptorSet){
            std::vector<VkDescriptorSetLayout> layouts{vkDescriptorSetLayout};
            VkUtil::createDescriptorSets(vkContext.device, layouts, vkContext.descriptorPool, &vkDescriptorSet);
        }

        VkUtil::updateDescriptorSet(vkContext.device, vkData, gpuData.size(), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, vkDescriptorSet);
    }

    int getBundleIndex(float val, uint32_t axis) const{
        int index = 0;
        for(auto& bundle: bundlesData[axis]){
            if(val >= bundle.max && val <= bundle.min) break;
            ++index;
        }
        return index;
    }

    // creates the pipeline including all necessary vulkan resources for rendering such as the descriptor set layout, the descriptor set...
    void createVulkanPipeline(){
        ++pipelineCounter;

        std::vector<VkDescriptorSetLayoutBinding> bindings{{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL_GRAPHICS, nullptr}};
        VkUtil::createDescriptorSetLayout(vkContext.device, bindings, &vkDescriptorSetLayout);

        std::vector<VkVertexInputAttributeDescription> attributeDescriptions{{0,0,VK_FORMAT_R32G32B32_SFLOAT,0}, {1,0,VK_FORMAT_R32G32_UINT, 3 * sizeof(float)}};

        std::vector<VkVertexInputBindingDescription> bindingDescriptions{{0, 3 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX}, {0, 2 * sizeof(uint32_t), VK_VERTEX_INPUT_RATE_VERTEX}};

        VkPipelineVertexInputStateCreateInfo vertexInfo{};
        vertexInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInfo.vertexAttributeDescriptionCount = attributeDescriptions.size();
        vertexInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
        vertexInfo.vertexBindingDescriptionCount = bindingDescriptions.size();
        vertexInfo.pVertexBindingDescriptions = bindingDescriptions.data();

        VkPipelineRasterizationStateCreateInfo rasterizationInfo{};
        rasterizationInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizationInfo.lineWidth = 1;
        rasterizationInfo.polygonMode = VK_POLYGON_MODE_LINE;

        VkPipelineMultisampleStateCreateInfo multisampleInfo{};
        multisampleInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;

        VkPipelineDepthStencilStateCreateInfo stencilInfo{};        //leave disabled
        stencilInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

        VkUtil::BlendInfo blendInfo;    //TODO:: change this
        
        VkUtil::createRenderPass(vkContext.device, VkUtil::PASS_TYPE_COLOR16_OFFLINE_NO_CLEAR, &vkRenderPass);

        VkUtil::createPipeline(vkContext.device, &vertexInfo, vkContext.screenSize[0], vkContext.screenSize[1], {}, nullptr, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY, &rasterizationInfo, &multisampleInfo, &stencilInfo,
                            &blendInfo, {vkDescriptorSetLayout}, &vkRenderPass, &vkPipelineLayout, &vkPipeline);
    }
};