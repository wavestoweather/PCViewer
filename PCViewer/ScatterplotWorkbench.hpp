#pragma once

#include<vulkan/vulkan.h>
#include<vector>
#include<string>
#include<map>
#include"imgui/imgui.h"
#include"imgui/imgui_impl_vulkan.h"
#include"VkUtil.h"
#include"PCUtil.h"
#include"Structures.hpp"
#include"LassoBrush.hpp"

// workbench for scatter plots
// all scatterplots are scatterplot matrices which can be reduced to only show wanted parameter combinations
class ScatterplotWorkbench{
public:
    ImVec2 static pixelPosToParameterPos(const ImVec2& mousePos, const ImVec2& borderMin, const ImVec2& borderMax, int attr1, int attr2, const std::vector<Attribute>& pcAttributes);
    ImVec2 static parameterPosToPixelPos(const ImVec2& paramPos, const ImVec2& borderMin, const ImVec2& borderMax, int attr1, int attr2, const std::vector<Attribute>& pcAttributes);
    float static distance2(const ImVec2& a, const ImVec2& b);
    float static distance(const ImVec2& a, const ImVec2& b);

    // the scatter plot workbench can have multiple scatter plots
    struct ScatterPlot{
        // each drawlist which is assigned to the scatterplot has stored its information in the DrawListInstance struct
        struct DrawListInstance{
            VkUtil::Context context{};
            bool active{true};
            std::string drawListName{};
            VkBuffer data{};
            VkBuffer indices{};
            VkBufferView activeData{};
            //private vulkan ressources which have to be destroyed
            struct UBO{
                float spacing{.01f};
                float radius{1};
                uint32_t showInactivePoints{1};
                uint32_t matrixSize;
                float color[4]{1,0,0,.3f};
                float inactiveColor[4]{.1f,.1f,.1f,.1f};
            }uniformBuffer;
            VkDescriptorSet descSet{};
            VkDescriptorSetLayout descSetLayout{};
            VkBuffer ubo{};
            VkDeviceMemory uboMemory{};
            uint32_t indicesSize;
            const std::vector<Attribute>& attributes;

            DrawListInstance(VkUtil::Context context, const DrawList& drawList, VkBuffer data, VkBufferView activeData, VkBuffer indices, uint32_t indicesSize, VkDescriptorSetLayout descriptorSetLayout, const std::vector<Attribute>& attributes);
            DrawListInstance(const DrawListInstance& other);
            DrawListInstance(DrawListInstance&& other);

            DrawListInstance& operator=(const DrawListInstance& other);
            DrawListInstance& operator=(DrawListInstance&& other);

            void setupUniformBuffer();

            void updateUniformBufferData(const std::vector<Attribute>& attributes);

            ~DrawListInstance(){
                if(descSet) vkFreeDescriptorSets(context.device, context.descriptorPool, 1, &descSet);
                if(ubo) vkDestroyBuffer(context.device, ubo, nullptr);
                if(uboMemory) vkFreeMemory(context.device, uboMemory, nullptr);
            }
        };

        int curWidth = 0, curHeight = 0;
        int id;

        const std::list<DrawList>& drawlists;
        const std::vector<int>& selected_drawlists;
        VkUtil::Context context;
        VkImage resultImage{};
        VkImageView resultImageView{};
        VkSampler sampler{};
        VkFramebuffer framebuffer{};
        VkDeviceMemory imageMemory{};
        VkDescriptorSet resultImageSet{};
        uint32_t activeAttributesCount{};
        std::vector<uint8_t> activeAttributes;
        std::vector<Attribute>& attributes;
        std::vector<DrawListInstance> dls;
        float matrixSpacing{.05f};
        float matrixBorderWidth{1};
        ImVec4 matrixBorderColor{1,1,1,1};

        //external vulkan resources (dont have to be destroyed)
        VkRenderPass renderPass;
        VkDescriptorSetLayout descriptorSetLayout;
        VkPipeline pipeline;
        VkPipelineLayout pipelineLayout;

        ScatterPlot(VkUtil::Context context, int width, int height, VkRenderPass renderPass, VkDescriptorSetLayout descriptorSetLayout, VkPipeline pipeline, VkPipelineLayout pipelineLayout, std::vector<Attribute>& attributes, const std::vector<int>& selected_dls, const std::list<DrawList>& dls);
        ~ScatterPlot();
        void resizeImage(int width, int height);

        void draw(int index);

        void updatePlot();

        void updateRender(VkCommandBuffer commandBuffer);
        void addDrawList(const DrawList& dl, std::vector<Attribute>& attr);
    };

    ScatterplotWorkbench(VkUtil::Context context, std::vector<Attribute>& attributes, const std::vector<int>& selected_dl, const std::list<DrawList>& dls): context(context), attributes(attributes), selected_drawlists(selected_dl), drawlists(dls){
        createPipeline();
    }

    void addPlot(){
        scatterPlots.emplace_back(context, defaultWidth, defaultHeight, renderPass, descriptorSetLayout, pipeline, pipelineLayout, attributes, selected_drawlists, drawlists);
    }

    ~ScatterplotWorkbench(){
        if(pipeline) vkDestroyPipeline(context.device, pipeline, nullptr);
        if(pipelineLayout) vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
        if(descriptorSetLayout) vkDestroyDescriptorSetLayout(context.device, descriptorSetLayout, nullptr);
        if(renderPass) vkDestroyRenderPass(context.device, renderPass, nullptr);
    }

    void draw(){
        if(!active) return;
        ImGui::Begin("Scatterplot Workbench", &active);
        int c = 0;
        for(ScatterPlot& s: scatterPlots){
            s.draw(c++);
        }
        if(ImGui::Button("+", ImVec2{500,0})){
            addPlot();
        }
        ImGui::End();
    }

    void updateRenders(const std::vector<int>& attrIndices){
        VkCommandBuffer commandBuffer;
        VkUtil::createCommandBuffer(context.device, context.commandPool, &commandBuffer);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        for(ScatterPlot& s: scatterPlots){
            bool change = false;
            if(attrIndices.size() != s.activeAttributes.size()) continue;
            for(int i: attrIndices) change |= s.activeAttributes[i];
            change |= attrIndices.empty();
            if(change) s.updateRender(commandBuffer);
        }
        VkUtil::commitCommandBuffer(context.queue, commandBuffer);
        VkResult res = vkQueueWaitIdle(context.queue); check_vk_result(res);
        vkFreeCommandBuffers(context.device, context.commandPool, 1, &commandBuffer);
    }

    bool active = false;
    std::vector<ScatterPlot> scatterPlots;
    int defaultWidth = 800, defaultHeight = 800;
    bool showInactivePoints{true};
    
    static std::map<std::string, Polygons> lassoSelections;
    static std::vector<std::string> updatedDrawlists;
    static int scatterPlotCounter;
protected:
    VkPipeline pipeline{};
    VkPipelineLayout pipelineLayout{};
    VkDescriptorSetLayout descriptorSetLayout{};
    VkRenderPass renderPass{};

    VkUtil::Context context;
    std::vector<Attribute>& attributes;
    const std::list<DrawList>& drawlists;
    const std::vector<int>& selected_drawlists;

    struct PushConstant{
        uint32_t posX;
        uint32_t posY;
        uint32_t xAttr;
        uint32_t yAttr;
        uint32_t discard;
    };

    void createPipeline();
};