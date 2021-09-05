#pragma once

#include<vulkan/vulkan.h>
#include<vector>
#include<string>
#include"imgui/imgui.h"

// workbench for scatter plots
// all scatterplots are scatterplot matrices which can be reduced to only show wanted parameter combinations
class ScatterplotWorkbenchk{
public:
    struct ScatterPlot{

    };

    ScatterplotWorkbenchk(){}

    void draw(){
        if(!active) return;
        ImGui::Begin("Scatterplot Workbench");
        int c = 0;
        for(ScatterPlot& s: scatterPlots){
            ImGui::BeginChild(("Scatterplot" + std::to_string(c++)).c_str());

            ImGui::EndChild();
        }
        ImGui::End();
    }

    bool active = false;
    std::vector<ScatterPlot> scatterPlots;
protected:
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkDescriptorSetLayout descriptorSetLayout;

    VkImage resultImage;
    VkImageView resultImageView;
    VkSampler sampler;
};