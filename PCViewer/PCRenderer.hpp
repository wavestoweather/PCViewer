#pragma once
#include "VkUtil.h"
#include "Structures.hpp"
#include <inttypes.h>

class PCRenderer{
public:
    struct GlobalPCSettings;    //forward declare struct
    PCRenderer(PCRenderer const&) = delete;
    void operator=(PCRenderer const&) = delete;
    
    PCRenderer(const VkUtil::Context& context, uint32_t width, uint32_t height, VkDescriptorSetLayout uniformLayout, VkDescriptorSetLayout dataLayout);
    ~PCRenderer();

    void renderPCPlots(std::list<DrawList>& drawlists, const GlobalPCSettings& globalSettings);

    struct GlobalPCSettings{
        std::vector<Attribute>& attributes;
        bool* attributeEnabled;
        std::vector<int>& attributeOrder;
        bool renderSplines;
        float medianLineWidth;
    };
private:
    class PipelineSingleton;    //forward declare
    PipelineSingleton& _pipelineInstance;    //holds a reference to the pipeline singleton for all rendering pipelines

    //vulkan resources for the output image
    VkFramebuffer   _framebuffer{};
    VkImage         _intermediateImage{}, _plotImage{}; //intermediat image holds the uint32 iamge with the counts
    VkImageView     _intermediateView{}, _plotView{};
    VkDeviceMemory  _imageMemory{};                   //meory for all images
    VkDescriptorSet _intermediateSet{};
    VkDescriptorSet _computeSet{};

    class PipelineSingleton{    //provides a safe pipeline singleton
    public:
        struct PipelineInput{
            uint32_t width;
            uint32_t height;
            VkDescriptorSetLayout uniformLayout, dataLayout;
        };

        //singleton access
        static PipelineSingleton& getInstance(const VkUtil::Context& context, const PipelineInput& input){
            static PipelineSingleton instance(context, input);
            ++_usageCount;
            return instance;
        }
        // notify no use
        static void notifyInstanceShutdown(PipelineSingleton& singleton){
            --_usageCount;
            if(_usageCount == 0){
                singleton.pipelineInfo.vkDestroy(singleton.context);
                singleton.computeInfo.vkDestroy(singleton.context);
                if(singleton.renderPass) vkDestroyRenderPass(singleton.context.device, singleton.renderPass, nullptr);
            }
        }

        // deletion of standard constructor and copy operator to avoid copies
        PipelineSingleton(PipelineSingleton const&) = delete;
        void operator=(PipelineSingleton const&) = delete;

        //publicly available info
        VkUtil::PipelineInfo pipelineInfo{};
        VkUtil::PipelineInfo computeInfo{};
        VkRenderPass renderPass{};
        VkUtil::Context context{};
    private:
        const std::string _vertexShader = "shader/vert.spv";    //standard vertex shader to transform line vertices
        const std::string _geometryShader = "shader/geom.spv";  //optional geometry shader for spline rendering
        const std::string _fragmentShader = "shader/fragUint.spv";  //fragment shader to output line count to framebuffer

        const std::string _computeShader = "shader/pcResolve.comp.spv";   //shader which resolves density values to true color
        static int _usageCount;
        PipelineSingleton(const VkUtil::Context& inContext, const PipelineInput& input);
    };
};