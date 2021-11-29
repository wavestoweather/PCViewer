#include "GpuRadixSorter.hpp"
#include "PCUtil.h"

GpuRadixSorter::GpuRadixSorter(const VkUtil::Context& context):
_vkContext(context)
{
    if(context.device){
        VkShaderModule shaderModule = VkUtil::createShaderModule(context.device, PCUtil::readByteFile(_shaderPath));
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        VkUtil::createDescriptorSetLayout(context.device, bindings, &_pipelineInfo.descriptorSetLayout);
        VkUtil::createComputePipeline(context.device, shaderModule, {_pipelineInfo.descriptorSetLayout}, &_pipelineInfo.pipelineLayout, &_pipelineInfo.pipeline);
    }
}

GpuRadixSorter::~GpuRadixSorter(){
    _pipelineInfo.vkDestroy(_vkContext);
}