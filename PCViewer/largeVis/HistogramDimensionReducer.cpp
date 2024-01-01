#include "HistogramDimensionReducer.hpp"
#include "../PCUtil.h"

HistogramDimensionReducer* HistogramDimensionReducer::_singleton{};

HistogramDimensionReducer::HistogramDimensionReducer(const VkUtil::Context& context):
    _vkContext(context){
    // pipeline creation

    auto shaderModule = VkUtil::createShaderModule(context.device, PCUtil::readByteFile(_shaderPath));

    std::vector<VkPushConstantRange> pushConstants{{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)}};

    VkUtil::createComputePipeline(context.device, shaderModule, {}, &_pipeline.pipelineLayout, &_pipeline.pipeline, {}, pushConstants);
}

HistogramDimensionReducer::~HistogramDimensionReducer(){
    _pipeline.vkDestroy(_vkContext);
}

void HistogramDimensionReducer::reduceHistogram(reduceInfo& info){
    PushConstants pc{};
    pc.histogramWidth = info.histogramWidth;
    pc.xReduce = info.xReduce;
    pc.srcHistogram = info.srcHistogram;
    pc.dstHistogram = info.dstHistogram;
    vkCmdBindPipeline(info.commands, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline.pipeline);
    vkCmdPushConstants(info.commands, _pipeline.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);
    uint32_t dispatchX = 1; // In the primary direction (x if xReduce == 1 else y) reduction is done in a single workgroup (256 parallel execution cores)
    uint32_t dispatchY = info.histogramWidth;
    vkCmdDispatch(info.commands, dispatchX, dispatchY, 1);
}  
