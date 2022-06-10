#include "GpuInstance.hpp"


namespace vkCompress
{
    GpuInstance::GpuInstance(VkUtil::Context context) 
    {
        
    }
    
    GpuInstance::~GpuInstance() 
    {
        Histogram.pipelineInfo.vkDestroy(vkContext);
        HuffmanTable.pipelineInfo.vkDestroy(vkContext);
        RunLength.pipelineInfo.vkDestroy(vkContext);
        DWT.pipelineInfo.vkDestroy(vkContext);
        Quantization.pipelineInfo.vkDestroy(vkContext);
    }
}