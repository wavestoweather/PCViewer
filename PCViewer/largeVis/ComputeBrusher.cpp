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

    b.binding = 2;                // activation buffer
    bindings.push_back(b);

    VkUtil::createDescriptorSetLayout(info.context.device, bindings, &_brushPipelineInfo.descriptorSetLayout);

    VkUtil::createComputePipeline(info.context.device, shaderModule, {_brushPipelineInfo.descriptorSetLayout}, &_brushPipelineInfo.pipelineLayout, &_brushPipelineInfo.pipeline);

    VkUtil::createDescriptorSets(_vkContext.device, {_brushPipelineInfo.descriptorSetLayout}, _vkContext.descriptorPool, &_descSet);

    _brushEvent = VkUtil::createEvent(_vkContext.device, 0);
    vkSetEvent(_vkContext.device, _brushEvent); // has to be set to signal that the pipeline is ready for the next update

    VkFenceCreateInfo fInfo{};
    fInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(_vkContext.device, &fInfo, nullptr, &_brushFence);
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
    if(_brushEvent)
        vkDestroyEvent(_vkContext.device, _brushEvent, nullptr);
    if(_brushFence)
        vkDestroyFence(_vkContext.device, _brushFence, nullptr);
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

VkEvent ComputeBrusher::updateActiveIndices(size_t amtDatapoints, const std::vector<brushing::RangeBrush>& brushes, const Polygons& lassoBrushes, const std::vector<VkBuffer>& dataBuffer, VkBuffer indexActivations, size_t indexOffset, bool andBrushes, VkEvent prevPipeEvent, TimingInfo timingInfo){
    auto err = vkWaitForFences(_vkContext.device, 1, &_brushFence, VK_TRUE, 5e9); check_vk_result(err); // maximum wait for 1 sec
    assert(err == VK_SUCCESS);
    vkResetFences(_vkContext.device, 1, &_brushFence);

    assert(vkGetEventStatus(_vkContext.device, _brushEvent) == VK_EVENT_SET);
    vkResetEvent(_vkContext.device, _brushEvent);
    
    std::scoped_lock<std::mutex> lock(*_vkContext.queueMutex);
    if(_commands)
        vkFreeCommandBuffers(_vkContext.device, _vkContext.commandPool, 1, &_commands);
    VkUtil::createCommandBuffer(_vkContext.device, _vkContext.commandPool, &_commands);

    // wait for previous event/pipeline to finish
    if(prevPipeEvent)
        vkCmdWaitEvents(_commands, 1, &prevPipeEvent, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, 0, {}, 0, {});
    
    if(timingInfo.queryPool){
        vkCmdResetQueryPool(_commands, timingInfo.queryPool, timingInfo.startIndex, 2);
        vkCmdWriteTimestamp(_commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timingInfo.queryPool, timingInfo.startIndex); 
    }
    // if no brushes are active, set all points active
    if(brushes.empty() && lassoBrushes.empty()){
        vkCmdFillBuffer(_commands, indexActivations, indexOffset / 32, (amtDatapoints / 32 + 3) / 4 * 4, uint32_t(-1));
        
        if(timingInfo.queryPool)
            vkCmdWriteTimestamp(_commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timingInfo.queryPool, timingInfo.endIndex); 
        vkCmdSetEvent(_commands, _brushEvent, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkUtil::commitCommandBuffer(_vkContext.queue, _commands, _brushFence);
        //auto res = vkQueueWaitIdle(_vkContext.queue); check_vk_result(res);

        return _brushEvent;
    }
    // converting the range brushes to properly be able to check activation
    struct MM{float min, max;};
    std::vector<std::map<int, std::vector<MM>>> axisBrushes(brushes.size());
    std::vector<bool> axisActive(dataBuffer.size(), false);
    for(int i: irange(brushes)){
        auto& b = axisBrushes[i];
        for(const auto& range: brushes[i]){
            b[range.axis].push_back({range.min, range.max});
            axisActive[range.axis] = true;
        }
    }

    // converting the brush data to a linearized array and uplaoding it to the gpu for execution
    // priorty for linearising the brush data: brushes, axismap, ranges
    // the float array following the brushing info has the following layout:
    // vector<float> brushOffsets, vector<Brush> brushes;   // where brush offests describe the index in the float array from which the brush at index i is positioned
    // with Brush = {flaot nAxisMaps, vector<float> axisOffsets, vector<AxisMap> axisMaps} // axisOffsets same as brushOffsetsf for the axisMap
    // with AxisMap = {float nrRanges, fl_brushEventoat axis, vector<floatt> rangeOffsets, vector<Range> ranges}
    // with Range = {float, float} // first is min, second is max
    std::vector<float> brushData;
    brushData.resize(axisBrushes.size());  // space for brushOffsets
    for(int brush: irange(axisBrushes)){
        // adding a brush
        brushData[brush] = brushData.size();        // inserting the correct offset
        brushData.push_back(axisBrushes[brush].size()); // number of axis maps
        size_t axisOffsetsIndex = brushData.size(); // safing index for fast access later
        brushData.resize(brushData.size() + axisBrushes[brush].size());   // reserving space for axisOffsets
        // adding the Axis Maps
        int curMap = 0;
        for(const auto [axis, ranges]: axisBrushes[brush]){
            brushData[axisOffsetsIndex + curMap] = brushData.size();    // correct offset for the current axis map
            brushData.push_back(ranges.size());                         // ranges size
            brushData.push_back(axis);
            size_t rangesOffsetsIndex = brushData.size();
            brushData.resize(brushData.size() + ranges.size());
            //adding the ranges
            int curRange = 0;
            for(const auto range: ranges){
                brushData[rangesOffsetsIndex + curRange] = brushData.size();
                brushData.push_back(range.min);
                brushData.push_back(range.max);
                ++curRange;
            }       
            ++curMap;
        }
    }
    
    // ensuring size of info buffer (this includes updating the descriptor set)
    uint32_t brushInfoBytes = sizeof(BrushInfos) + brushData.size() * sizeof(brushData[0]); //TODO: add brush size to the infos
    createOrUpdateBuffer(dataBuffer.size(), brushInfoBytes);
    // uploading the brush infos
    BrushInfos bI{};
    bI.amtOfAttributes = dataBuffer.size();
    bI.amtofDataPoints = amtDatapoints;
    bI.amtOfBrushes = axisBrushes.size();
    bI.andBrushes = andBrushes;
    bI.outputOffset = indexOffset / 32; // divide by 32 to convert from bit offset to uint32_t offset
    // todo lasso brushes have to be added
    VkUtil::uploadData(_vkContext.device, _infoMemory, 0, sizeof(BrushInfos), &bI);
    VkUtil::uploadData(_vkContext.device, _infoMemory, sizeof(BrushInfos), brushData.size() * sizeof(brushData[0]), brushData.data());

    // getting the buffer device addreses for the data and uploading them to the _dataBuffer
    std::vector<VkDeviceAddress> deviceAddresses(dataBuffer.size());
    for(int i: irange(dataBuffer)){
        VkBufferDeviceAddressInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        info.buffer = dataBuffer[i];
        deviceAddresses[i] = vkGetBufferDeviceAddress(_vkContext.device, &info);
    }
    VkUtil::uploadData(_vkContext.device, _infoMemory, _dataOffset, deviceAddresses.size() * sizeof(deviceAddresses[0]), deviceAddresses.data());

    // adding the activations buffer to the desc set
    VkUtil::updateDescriptorSet(_vkContext.device, indexActivations, VK_WHOLE_SIZE, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _descSet);

    const uint32_t localSize = 256; // has to be the same as in the largeVisBrush.comp shader

    vkCmdBindPipeline(_commands, VK_PIPELINE_BIND_POINT_COMPUTE, _brushPipelineInfo.pipeline);
    vkCmdBindDescriptorSets(_commands, VK_PIPELINE_BIND_POINT_COMPUTE, _brushPipelineInfo.pipelineLayout, 0, 1, &_descSet, 0, {});
    vkCmdDispatch(_commands, (amtDatapoints + 32 * localSize - 1) / (32 * localSize), 1, 1);
   
    if(timingInfo.queryPool)
        vkCmdWriteTimestamp(_commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timingInfo.queryPool, timingInfo.endIndex); 
    
    vkCmdSetEvent(_commands, _brushEvent, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
    
    VkUtil::commitCommandBuffer(_vkContext.queue, _commands, _brushFence);
    
    //vkFreeCommandBuffers(_vkContext.device, _vkContext.commandPool, 1, &commands);
    return _brushEvent;
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
