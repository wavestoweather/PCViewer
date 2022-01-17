#pragma once
#include "VkUtil.h"
#include "Structures.hpp"
#include <inttypes.h>

class PCRenderer{
public:
    struct GlobalPCSettings;    //forward declare struct
    PCRenderer(PCRenderer const&) = delete;
    void operator=(PCRenderer const&) = delete;
    
    PCRenderer(const VkUtil::Context& context, uint32_t width, uint32_t height);

    VkResult renderPCPlots(std::list<DrawList>& drawlists, const GlobalPCSettings& globalSettings);

    struct GlobalPCSettings{

    };
private:
    class PipelineSingleton;    //forward declare
    PipelineSingleton& pipelineInstance;    //holds a reference to the pipeline singleton


    class PipelineSingleton{    //provides a safe pipeline singleton
    public:
        //singleton access
        static PipelineSingleton& getInstance(const VkUtil::Context& context){
            static PipelineSingleton instance(context);
            ++_usageCount;
            return instance;
        }
        // notify no use
        static void notifyInstanceShutdown(PipelineSingleton& singleton){
            --_usageCount;
            if(_usageCount == 0){
                singleton.pipelineInfo.vkDestroy(singleton.context);
            }
        }

        // deletion of standard constructor and copy operator to avoid copies
        PipelineSingleton(PipelineSingleton const&) = delete;
        void operator=(PipelineSingleton const&) = delete;

        //publicly available info
        VkUtil::PipelineInfo pipelineInfo;
        VkUtil::Context context;
    private:
        const std::string _vertexShader = "";    //standard vertex shader to transform line vertices
        const std::string _geometryShader = "";  //optional geometry shader for spline rendering
        const std::string _fragmentShader = "";  //fragment shader to output line count to framebuffer

        const std::string _computeShader = "";   //shader which resolves density values to true color
        static int _usageCount;
        PipelineSingleton(const VkUtil::Context& inContext);
    };
};