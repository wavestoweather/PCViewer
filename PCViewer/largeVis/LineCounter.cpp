#include "LineCounter.hpp"
#include <assert.h>
#include "../PCUtil.h"
#include "../VkUtil.h"
#include "../range.hpp"

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
    VkDescriptorSetLayoutBinding b{};
    // attr a values
    b.binding = 0;
    b.descriptorCount = 1;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // has to be texel buffer to support 16 bit readout
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings.push_back(b);

    // attr b values
    b.binding = 1;
    bindings.push_back(b);

    // line counts
    b.binding = 2;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings.push_back(b);

    b.binding = 3;
    b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings.push_back(b);

    VkUtil::createDescriptorSetLayout(info.context.device, bindings, &_countPipeInfo.descriptorSetLayout);

    VkUtil::createComputePipeline(info.context.device, shaderModule, {_countPipeInfo.descriptorSetLayout}, &_countPipeInfo.pipelineLayout, &_countPipeInfo.pipeline);

    VkUtil::createDescriptorSets(_vkContext.device, {_countPipeInfo.descriptorSetLayout}, _vkContext.descriptorPool, &_pairSet);
    VkUtil::createBuffer(_vkContext.device, sizeof(PairInfos), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &_pairUniform);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    VkMemoryRequirements memReq{};
    vkGetBufferMemoryRequirements(_vkContext.device, _pairUniform, &memReq);
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(_vkContext.physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkAllocateMemory(_vkContext.device, &allocInfo, nullptr, &_pairUniformMem);
    vkBindBufferMemory(_vkContext.device, _pairUniform, _pairUniformMem, 0);

    // second pipeline to count all bins at once
    compBytes = PCUtil::readByteFile(_computeAllShader);
    shaderModule = VkUtil::createShaderModule(info.context.device, compBytes);
    
    bindings[0].descriptorCount = maxAttributes;
    bindings[1].descriptorCount = maxAttributes;
    bindings[2].descriptorCount = maxAttributes - 1; // always one as the 2d bins are always between 2 attributes
    bindings[1] = bindings[3]; bindings.pop_back();     // we can drop the second binding

    VkUtil::createDescriptorSetLayout(info.context.device, bindings, &_countAllPipeInfo.descriptorSetLayout);

    VkUtil::createComputePipeline(info.context.device, shaderModule, {_countAllPipeInfo.descriptorSetLayout}, &_countAllPipeInfo.pipelineLayout, &_countAllPipeInfo.pipeline);

    VkUtil::createDescriptorSets(_vkContext.device, {_countAllPipeInfo.descriptorSetLayout}, _vkContext.descriptorPool, &_allSet);    
}

void LineCounter::countLines(VkCommandBuffer commands, const CountLinesInfo& info){
    // test counting
    const uint32_t size = (1 << 27);  // 2^30
    const uint32_t aBins = 1 << 10, bBins = 1 << 10;
    const uint32_t iterations = 1;
    std::vector<uint16_t> a1(size), a2(size);
    VkBuffer vA, vB, counts, infos;
    VkDeviceMemory mA, mB, mOther;
    VkUtil::createBuffer(_vkContext.device, size * sizeof(uint16_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &vA);
    VkUtil::createBuffer(_vkContext.device, size * sizeof(uint16_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &vB);
    VkUtil::createBuffer(_vkContext.device, (aBins * bBins) * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &counts);
    VkUtil::createBuffer(_vkContext.device, 4 * sizeof(uint32_t), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &infos);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    VkMemoryRequirements memReq{};

    vkGetBufferMemoryRequirements(_vkContext.device, vA, &memReq);
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(_vkContext.physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkAllocateMemory(_vkContext.device, &allocInfo, nullptr, &mA);
    vkBindBufferMemory(_vkContext.device, vA, mA, 0);

    vkGetBufferMemoryRequirements(_vkContext.device, vB, &memReq);
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(_vkContext.physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkAllocateMemory(_vkContext.device, &allocInfo, nullptr, &mB);
    vkBindBufferMemory(_vkContext.device, vB, mB, 0);

    vkGetBufferMemoryRequirements(_vkContext.device, counts, &memReq);
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memReq.memoryTypeBits;
    vkGetBufferMemoryRequirements(_vkContext.device, infos, &memReq);
    uint32_t infoOffset = allocInfo.allocationSize;
    allocInfo.allocationSize += memReq.size;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(_vkContext.physicalDevice, allocInfo.memoryTypeIndex | memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkAllocateMemory(_vkContext.device, &allocInfo, nullptr, &mOther);
    vkBindBufferMemory(_vkContext.device, counts, mOther, 0);
    vkBindBufferMemory(_vkContext.device, infos, mOther, infoOffset);

    struct Infos{
        uint32_t amtofDataPoints, aBins, bBins, padding;
    }cpuInfos {size, aBins, bBins, 0};
    std::vector<uint32_t> zeros(aBins * bBins);
    VkUtil::uploadData(_vkContext.device, mOther, infoOffset, sizeof(Infos), &cpuInfos);
    VkUtil::uploadData(_vkContext.device, mOther, 0, zeros.size() * sizeof(zeros[0]), zeros.data());

    //filling with random numbers
    std::srand(std::time(nullptr));
    //for(auto& e: a1) e = std::rand() & std::numeric_limits<uint16_t>::max();
    //for(auto& e: a2) e = std::rand() & std::numeric_limits<uint16_t>::max();
    VkUtil::uploadData(_vkContext.device, mA, 0, a1.size() * sizeof(a1[0]), a1.data());
    VkUtil::uploadData(_vkContext.device, mB, 0, a2.size() * sizeof(a2[0]), a2.data());

    if(!_descSet)
        VkUtil::createDescriptorSets(_vkContext.device, {_countPipeInfo.descriptorSetLayout}, _vkContext.descriptorPool, &_descSet);

    VkUtil::updateDescriptorSet(_vkContext.device, vA, VK_WHOLE_SIZE, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _descSet);
    VkUtil::updateDescriptorSet(_vkContext.device, vB, VK_WHOLE_SIZE, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _descSet);
    VkUtil::updateDescriptorSet(_vkContext.device, counts, (aBins * bBins) * sizeof(uint32_t), 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _descSet);
    VkUtil::updateDescriptorSet(_vkContext.device, infos, sizeof(Infos), 3, _descSet);

    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _countPipeInfo.pipeline);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _countPipeInfo.pipelineLayout, 0, 1, &_descSet, 0, nullptr);
    for(int i = 0; i < iterations; ++i)
        vkCmdDispatch(commands, size / 256, 1, 1);

    _bins = mOther;
    _binsSize = (aBins * bBins) * sizeof(uint32_t);
    // done filling hte command buffer.
    // execution is done outside
}

void LineCounter::countLinesPair(size_t dataSize, VkBuffer aData, VkBuffer bData, uint32_t aIndices, uint32_t bIndices, VkBuffer counts, bool clearCounts) const{
    VkUtil::updateDescriptorSet(_vkContext.device, aData, VK_WHOLE_SIZE, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _pairSet);
    VkUtil::updateDescriptorSet(_vkContext.device, bData, VK_WHOLE_SIZE, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _pairSet);
    VkUtil::updateDescriptorSet(_vkContext.device, counts, VK_WHOLE_SIZE, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _pairSet);
    VkUtil::updateDescriptorSet(_vkContext.device, _pairUniform, sizeof(PairInfos), 3, _pairSet);

    PairInfos infos{};
    infos.amtofDataPoints = dataSize;
    infos.aBins = aIndices;
    infos.bBins = bIndices;
    VkUtil::uploadData(_vkContext.device, _pairUniformMem, 0, sizeof(infos), &infos);

    assert(_vkContext.queueMutex);  // debug check that the optional value is set
	std::lock_guard<std::mutex> queueGuard(*_vkContext.queueMutex);	// locking the queue submission
    VkCommandBuffer commands;
    VkUtil::createCommandBuffer(_vkContext.device, _vkContext.commandPool, &commands);
    if(clearCounts)
        vkCmdFillBuffer(commands, counts, 0, aIndices * bIndices * sizeof(uint32_t), 0);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _countPipeInfo.pipelineLayout, 0, 1, &_pairSet, 0, {});
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _countPipeInfo.pipeline);
    vkCmdDispatch(commands, dataSize / 256, 1, 1);


    PCUtil::Stopwatch stop(std::cout, "Gpu Pairwise");
    VkUtil::commitCommandBuffer(_vkContext.queue, commands);
    auto res = vkQueueWaitIdle(_vkContext.queue); check_vk_result(res);

    vkFreeCommandBuffers(_vkContext.device, _vkContext.commandPool, 1, &commands);
}

void LineCounter::countLinesAll(size_t dataSize, std::vector<VkBuffer> data, uint32_t binAmt, std::vector<VkBuffer> counts, std::vector<uint32_t> activeIndices, bool clearCounts = false) const{
    assert(aData.size == bData.size() && aData.size() == counts.size() - 1);
    for(auto a: irange(data))
        VkUtil::updateArrayDescriptorSet(_vkContext.device, data[a], VK_WHOLE_SIZE, 0, a, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _allSet);
    for(auto c: irange(counts))
        VkUtil::updateArrayDescriptorSet(_vkContext.device, counts[c], VK_WHOLE_SIZE, 2, c, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _allSet);
    VkUtil::updateDescriptorSet(_vkContext.device, _pairUniform, sizeof(PairInfos), 3, _allSet);

    PairInfos infos{};
    infos.amtofDataPoints = dataSize;
    infos.aBins = binAmt;
    for(int i: irange(activeIndices.size() - 1)){
        infos.bBins |= int(activeIndices[i] > activeIndices[i + 1]) << i;
    }
    infos.padding = counts.size();
    VkUtil::uploadData(_vkContext.device, _pairUniformMem, 0, sizeof(infos), &infos);

    assert(_vkContext.queueMutex);  // debug check that the optional value is set
	std::lock_guard<std::mutex> queueGuard(*_vkContext.queueMutex);	// locking the queue submission
    VkCommandBuffer commands;
    VkUtil::createCommandBuffer(_vkContext.device, _vkContext.commandPool, &commands);
    if(clearCounts){
        for(auto count: counts)
            vkCmdFillBuffer(commands, count, 0, binAmt * binAmt * sizeof(uint32_t), 0);
    }
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _countAllPipeInfo.pipelineLayout, 0, 1, &_allSet, 0, {});
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _countAllPipeInfo.pipeline);
    vkCmdDispatch(commands, dataSize / 256, 1, 1);

    PCUtil::Stopwatch stop(std::cout, "Gpu Compute All");
    VkUtil::commitCommandBuffer(_vkContext.queue, commands);
    auto res = vkQueueWaitIdle(_vkContext.queue); check_vk_result(res);

    vkFreeCommandBuffers(_vkContext.device, _vkContext.commandPool, 1, &commands);
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
    
    // pipeline creation test
    auto t = acquireReference(info);

    VkCommandBuffer commands;
    VkUtil::createCommandBuffer(info.context.device, info.context.commandPool, &commands);
    // record commands
    t->countLines(commands, {});

    // commit and wait (includes timing of everything)
    {
        PCUtil::Stopwatch stopwatch(std::cout, "Line counter runtime");
        VkUtil::commitCommandBuffer(info.context.queue, commands);
        vkQueueWaitIdle(info.context.queue);
    }
    //check for count sum
    std::vector<uint32_t> counts(t->_binsSize / 4);
    VkUtil::downloadData(t->_vkContext.device, t->_bins, 0, t->_binsSize, counts.data());
    size_t sum = 0;
    for(auto i: counts)
        sum += i;
    
    std::cout << "Counts sum: " << sum << std::endl;
    t->release();
}

LineCounter::~LineCounter() 
{
    _countPipeInfo.vkDestroy(_vkContext);
    _countAllPipeInfo.vkDestroy(_vkContext);
    if(_descSet)
        vkFreeDescriptorSets(_vkContext.device, _vkContext.descriptorPool, 1, &_descSet);
    if(_pairSet)
        vkFreeDescriptorSets(_vkContext.device, _vkContext.descriptorPool, 1, &_pairSet);
    if(_pairUniform)
        vkDestroyBuffer(_vkContext.device, _pairUniform, nullptr);
    if(_pairUniformMem)
        vkFreeMemory(_vkContext.device, _pairUniformMem, nullptr);
}

void LineCounter::release(){
    assert(_refCount > 0);
    if(--_refCount == 0){
        delete _singleton;
        _singleton = nullptr;
    }
}