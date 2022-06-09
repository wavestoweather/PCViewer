#include "ComputeBrusher.hpp"
#include "../PCUtil.h"

ComputeBrusher::ComputeBrusher(const CreateInfo& info):
    _vkContext(info.context)
{
    //----------------------------------------------------------------------------------------------
	// creating the pipeline for line brushing
	//----------------------------------------------------------------------------------------------
    auto compBytes = PCUtil::readByteFile(_computeShader);
    auto shaderModule = VkUtil::createShaderModule(info.context.device, compBytes);

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    VkDescriptorSetLayoutBinding b{};
    
    b.binding = 0;              // info buffer (includes brush infos)
    b.descriptorCount = 1;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings.push_back(b);

    b.binding = 1;              // data addresses buffer
    bindings.push_back(b);

    VkUtil::createDescriptorSetLayout(info.context.device, bindings, &_brushPipelineInfo.descriptorSetLayout);

    VkUtil::createComputePipeline(info.context.device, shaderModule, {_brushPipelineInfo.descriptorSetLayout}, &_brushPipelineInfo.pipelineLayout, &_brushPipelineInfo.pipeline);

    VkUtil::createDescriptorSets(_vkContext.device, {_brushPipelineInfo.descriptorSetLayout}, _vkContext.descriptorPool, &_descSet);
}
    
ComputeBrusher::~ComputeBrusher(){
    _brushPipelineInfo.vkDestroy(_vkContext);
    if(_descSet)
        vkFreeDescriptorSets(_vkContext.device, _vkContext.descriptorPool, 1, &_descSet);
    if(_infoBuffer)
        vkDestroyBuffer(_vkContext.device, _infoBuffer, nullptr);
    if(_dataBuffer)
        vkDestroyBuffer(_vkContext.device, _dataBuffer, nullptr);
    if(_infoMemory)
        vkFreeMemory(_vkContext.device, _infoMemory, nullptr);
}

ComputeBrusher* ComputeBrusher::_singleton = nullptr;   // init null

ComputeBrusher* ComputeBrusher::acquireReference(const CreateInfo& info){
    if(!_singleton)
        _singleton = new ComputeBrusher(info);
    _singleton->_refCount++;
    return _singleton;
}

void ComputeBrusher::release(){
    assert(_refCount > 0);
    if(--_refCount == 0){
        delete _singleton;
        _singleton = nullptr;
    }
}

void ComputeBrusher::updateActiveIndices(const std::vector<brushing::RangeBrush>& brushes, const Polygons& lassoBrushes, const std::vector<VkBuffer>& dataBuffer, VkBuffer indexActivations, bool andBrushes = false){
    // ensuring size of info buffer (this includes updating the descriptor set)
    uint32_t brushInfoBytes = sizeof(BrushInfos) + 2; //TODO: add brush size to the infos
    createOrUpdateBuffer(dataBuffer.size(), brushInfoBytes);
    
    // getting the buffer device addreses for the data and uploading them to the _dataBuffer
    std::vector<VkDeviceAddress> deviceAddresses(dataBuffer.size());
    for(int i: irange(dataBuffer)){
        VkBufferDeviceAddressInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        info.buffer = dataBuffer[i];
        deviceAddresses[i] = vkGetBufferDeviceAddress(_vkContext.device, &info);
    }
    VkUtil::uploadData(_vkContext.device, _infoMemory, _dataOffset, deviceAddresses.size() * sizeof(deviceAddresses[0]), deviceAddresses.data());

    // converting the brush data to a linearized array and uplaoding it to the gpu for execution
    
}

void ComputeBrusher::createOrUpdateBuffer(uint32_t amtOfAttributes, uint32_t infoByteSize){
    if(amtOfAttributes <= _dataSize && infoByteSize <= _infoByteSize)   // if buffer are large enough skip
        return;

    // destroy old resources
    if(_infoBuffer)
        vkDestroyBuffer(_vkContext.device, _infoBuffer, nullptr);
    if(_dataBuffer)
        vkDestroyBuffer(_vkContext.device, _dataBuffer, nullptr);
    if(_infoMemory)
        vkFreeMemory(_vkContext.device, _infoMemory, nullptr);

    // create new resources
    VkUtil::createBuffer(_vkContext.device, infoByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &_infoBuffer);
    VkUtil::createBuffer(_vkContext.device, amtOfAttributes * sizeof(VkDeviceAddress), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &_dataBuffer);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    VkMemoryRequirements memReq{};
    vkGetBufferMemoryRequirements(_vkContext.device, _infoBuffer, &memReq);
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(_vkContext.physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    _dataOffset = allocInfo.allocationSize;
    vkGetBufferMemoryRequirements(_vkContext.device, _dataBuffer, &memReq);
    allocInfo.allocationSize += memReq.size;
    vkAllocateMemory(_vkContext.device, &allocInfo, nullptr, &_infoMemory);
    vkBindBufferMemory(_vkContext.device, _infoBuffer, _infoMemory, 0); 
    vkBindBufferMemory(_vkContext.device, _dataBuffer, _infoMemory, _dataOffset); 

    // updating descriptor set
    VkUtil::updateDescriptorSet(_vkContext.device, _infoBuffer, VK_WHOLE_SIZE, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _descSet);
    VkUtil::updateDescriptorSet(_vkContext.device, _dataBuffer, VK_WHOLE_SIZE, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _descSet);
}
