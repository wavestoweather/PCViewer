#include "LineCounter.hpp"
#include <assert.h>
#include "../PCUtil.h"
#include "../VkUtil.h"

LineCounter::LineCounter(const CreateInfo& info):
    _vkContext(info.context)
{
    //----------------------------------------------------------------------------------------------
	// creating the pipeline for line counting
	//----------------------------------------------------------------------------------------------
    auto compBytes = PCUtil::readByteFile(_computeShader);
    auto shaderModule = VkUtil::createShaderModule(info.context.device, compBytes);

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    // TODO: fill bindings map

    VkUtil::createDescriptorSetLayout(info.context.device, bindings, &_countPipeInfo.descriptorSetLayout);

    VkUtil::createComputePipeline(info.context.device, shaderModule, {_countPipeInfo.descriptorSetLayout}, &_countPipeInfo.pipelineLayout, &_countPipeInfo.pipeline);
}

void LineCounter::countLines(VkCommandBuffer commands, const CountLinesInfo& info){
    
}

LineCounter* LineCounter::_singleton = nullptr;    // init to nullptr

LineCounter* LineCounter::acquireReference(const CreateInfo& info){
    if(!_singleton)
        _singleton = new LineCounter(info);
    _singleton->_refCount++;
    return _singleton;
}

void LineCounter::tests(const CreateInfo& info){
    // small method to perform tests;
    auto t = acquireReference(info);
    t->release();
}

LineCounter::~LineCounter() 
{
    _countPipeInfo.vkDestroy(_vkContext);
}

void LineCounter::release(){
    assert(_refCount > 0);
    if(--_refCount == 0){
        delete _singleton;
        _singleton = nullptr;
    }
}