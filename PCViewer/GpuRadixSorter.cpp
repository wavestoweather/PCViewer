#include "GpuRadixSorter.hpp"
#include "PCUtil.h"
#include <algorithm>
#include <chrono>
#include <numeric>
#include <set>

GpuRadixSorter::GpuRadixSorter(const VkUtil::Context &context) : _vkContext(context)
{
    sortStats.resize(_timestepCount, 0);
    if (context.device)
    {
        // query subgroup size to set the specialization constant correctly
        VkPhysicalDeviceSubgroupProperties subgroupProperties;
        subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        subgroupProperties.pNext = NULL;
        VkPhysicalDeviceProperties2 physicalDeviceProperties;
        physicalDeviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        physicalDeviceProperties.pNext = &subgroupProperties;
        vkGetPhysicalDeviceProperties2(_vkContext.physicalDevice, &physicalDeviceProperties);
        if (subgroupProperties.subgroupSize < 16)
        { // we need at least 16 wide subgroups for global scan
            std::cout << "Gpu has only " << subgroupProperties.subgroupSize << " wide subgroups available, however 16 wide subgroups are needed." << std::endl;
            std::cout << "Gpu sorter falls back to c++ <algorithm>std::sort for sorting" << std::endl;
            return;
        }
        _timeMultiplier =  physicalDeviceProperties.properties.limits.timestampPeriod;

        std::vector<VkDescriptorSetLayoutBinding> bindings;
        VkDescriptorSetLayoutBinding binding;
        binding.binding = 0; // keys
        binding.descriptorCount = 2;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(binding);

        binding.binding = 1; // groupInfos
        binding.descriptorCount = 1;
        bindings.push_back(binding);

        binding.binding = 2; // Dispatch infos
        bindings.push_back(binding);

        binding.binding = 3; // global uniform info
        binding.descriptorCount = 2;
        bindings.push_back(binding);

        binding.binding = 4;
        binding.descriptorCount = 1;
        bindings.push_back(binding);

        std::vector<VkSpecializationMapEntry> entries{{0, 0, sizeof(uint32_t)}}; // subgroup size
        std::vector<uint32_t> values{subgroupProperties.subgroupSize};
        VkSpecializationInfo specializationConstants{
            static_cast<uint32_t>(entries.size()),
            entries.data(),
            sizeof(uint32_t),
            values.data()};

        VkShaderModule shaderModule = VkUtil::createShaderModule(context.device, PCUtil::readByteFile(_shaderLocalSortPath));
        VkUtil::createDescriptorSetLayout(context.device, bindings, &_localSortPipeline.descriptorSetLayout);
        VkUtil::createComputePipeline(context.device, shaderModule, {_localSortPipeline.descriptorSetLayout}, &_localSortPipeline.pipelineLayout, &_localSortPipeline.pipeline, &specializationConstants);

        shaderModule = VkUtil::createShaderModule(context.device, PCUtil::readByteFile(_shaderHistogramPath));
        VkUtil::createComputePipeline(context.device, shaderModule, {_localSortPipeline.descriptorSetLayout}, &_histogramPipeline.pipelineLayout, &_histogramPipeline.pipeline, &specializationConstants);

        shaderModule = VkUtil::createShaderModule(context.device, PCUtil::readByteFile(_shaderGlobalScanPath));
        VkUtil::createComputePipeline(context.device, shaderModule, {_localSortPipeline.descriptorSetLayout}, &_globalScanPipeline.pipelineLayout, &_globalScanPipeline.pipeline, &specializationConstants);

        shaderModule = VkUtil::createShaderModule(context.device, PCUtil::readByteFile(_shaderScatteringPath));
        VkUtil::createComputePipeline(context.device, shaderModule, {_localSortPipeline.descriptorSetLayout}, &_scatterPipeline.pipelineLayout, &_scatterPipeline.pipeline, &specializationConstants);

        shaderModule = VkUtil::createShaderModule(context.device, PCUtil::readByteFile(_shaderControlPath));
        VkUtil::createComputePipeline(context.device, shaderModule, {_localSortPipeline.descriptorSetLayout}, &_controlPipeline.pipelineLayout, &_controlPipeline.pipeline, &specializationConstants);

        shaderModule = VkUtil::createShaderModule(context.device, PCUtil::readByteFile(_shaderDispatchPath));
        VkUtil::createComputePipeline(context.device, shaderModule, {_localSortPipeline.descriptorSetLayout}, &_dispatchPipeline.pipelineLayout, &_dispatchPipeline.pipeline, &specializationConstants);
    }
}

GpuRadixSorter::~GpuRadixSorter()
{
    _controlPipeline.vkDestroy(_vkContext);
    _dispatchPipeline.vkDestroy(_vkContext);
    _localSortPipeline.vkDestroy(_vkContext);
    _histogramPipeline.vkDestroy(_vkContext);
    _globalScanPipeline.vkDestroy(_vkContext);
    _scatterPipeline.vkDestroy(_vkContext);
}

void GpuRadixSorter::sort(std::vector<uint32_t> &v)
{
    if (!_histogramPipeline.pipeline)
    {
        std::cout << "Gpu sorting not available, using standard c++ sort" << std::endl;
        std::sort(v.begin(), v.end());
        return;
    }
    const uint32_t controlSize = 64;
    const uint32_t KPB = 384 * 18; // keys per block/workgroup
    uint32_t firstPassGroups = (v.size() + KPB - 1) / KPB;
    uint32_t uniformBufferSize = std::max(.1f * v.size() * sizeof(uint32_t), 500.f * sizeof(uint32_t)); // current max amount of histograms (not histogram mergin currently possible)
    uint32_t groupInfoSize = uniformBufferSize;                     // in worst case for each global histogram a single worker...
    uint32_t localInfoSize = uniformBufferSize;
    uint32_t dispatchSize = 9;
    uint32_t timestampAmt = _timestepCount;

    // Buffer creation and filling -------------------------------------------------------------------------------------
    VkQueryPool qPool;
    VkQueryPoolCreateInfo qPoolInfo{};
    qPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qPoolInfo.queryCount = 20;
    VkResult t = vkCreateQueryPool(_vkContext.device, &qPoolInfo, nullptr, &qPool);
    
    VkBuffer dispatch, uniformFront, uniformBack, keysFront, keysBack, groupInfos, localInfos;
    VkDeviceMemory memory;
    VkUtil::createBuffer(_vkContext.device, controlSize * sizeof(uint32_t), VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &dispatch);
    VkUtil::createBuffer(_vkContext.device, v.size() * sizeof(v[0]), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &keysFront);
    VkUtil::createBuffer(_vkContext.device, v.size() * sizeof(v[0]), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &keysBack);
    VkUtil::createBuffer(_vkContext.device, uniformBufferSize * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &uniformFront);
    VkUtil::createBuffer(_vkContext.device, uniformBufferSize * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &uniformBack);
    VkUtil::createBuffer(_vkContext.device, groupInfoSize * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &groupInfos);
    VkUtil::createBuffer(_vkContext.device, localInfoSize * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &localInfos);
    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    VkMemoryRequirements memReq;
    uint32_t memBits{};
    vkGetBufferMemoryRequirements(_vkContext.device, dispatch, &memReq);
    allocInfo.allocationSize = memReq.size;
    memBits |= memReq.memoryTypeBits;

    uint32_t uniformFrontOffset = allocInfo.allocationSize;
    vkGetBufferMemoryRequirements(_vkContext.device, uniformFront, &memReq);
    allocInfo.allocationSize += memReq.size;
    memBits |= memReq.memoryTypeBits;

    uint32_t uniformBackOffset = allocInfo.allocationSize;
    vkGetBufferMemoryRequirements(_vkContext.device, uniformBack, &memReq);
    allocInfo.allocationSize += memReq.size;
    memBits |= memReq.memoryTypeBits;

    uint32_t keysFrontOffset = allocInfo.allocationSize;
    vkGetBufferMemoryRequirements(_vkContext.device, keysFront, &memReq);
    allocInfo.allocationSize += memReq.size;
    memBits |= memReq.memoryTypeBits;

    uint32_t keysBackOffset = allocInfo.allocationSize;
    vkGetBufferMemoryRequirements(_vkContext.device, keysBack, &memReq);
    allocInfo.allocationSize += memReq.size;
    memBits |= memReq.memoryTypeBits;

    uint32_t groupInfosOffset = allocInfo.allocationSize;
    vkGetBufferMemoryRequirements(_vkContext.device, groupInfos, &memReq);
    allocInfo.allocationSize += memReq.size;
    memBits |= memReq.memoryTypeBits;

    uint32_t localInfosOffset = allocInfo.allocationSize;
    vkGetBufferMemoryRequirements(_vkContext.device, localInfos, &memReq);
    allocInfo.allocationSize += memReq.size;
    memBits |= memReq.memoryTypeBits;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(_vkContext.physicalDevice, memBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkAllocateMemory(_vkContext.device, &allocInfo, nullptr, &memory);
    vkBindBufferMemory(_vkContext.device, dispatch, memory, 0);
    vkBindBufferMemory(_vkContext.device, uniformFront, memory, uniformFrontOffset);
    vkBindBufferMemory(_vkContext.device, uniformBack, memory, uniformBackOffset);
    vkBindBufferMemory(_vkContext.device, keysFront, memory, keysFrontOffset);
    vkBindBufferMemory(_vkContext.device, keysBack, memory, keysBackOffset);
    vkBindBufferMemory(_vkContext.device, groupInfos, memory, groupInfosOffset);
    vkBindBufferMemory(_vkContext.device, localInfos, memory, localInfosOffset);

    VkUtil::uploadData(_vkContext.device, memory, keysFrontOffset, v.size() * sizeof(v[0]), v.data());
    VkUtil::uploadData(_vkContext.device, memory, keysBackOffset, v.size() * sizeof(v[0]), v.data());
    std::vector<uint32_t> cpuUniform(5, 0);
    cpuUniform[0] = 0; // always start with pass nr 0
    cpuUniform[1] = 1; // 1 global histogram in the beginning
    cpuUniform[2] = firstPassGroups;
    cpuUniform[3] = 0;        // beginning of the list to sort
    cpuUniform[4] = v.size(); // end of the list to sort
    VkUtil::uploadData(_vkContext.device, memory, uniformFrontOffset, cpuUniform.size() * sizeof(cpuUniform[0]), cpuUniform.data());
    cpuUniform[1] = 0;  // global amt of histograms has to be set to 0
    cpuUniform[2] = 0;  // group count has also to be rest
    VkUtil::uploadData(_vkContext.device, memory, uniformBackOffset, cpuUniform.size() * sizeof(cpuUniform[0]), cpuUniform.data());
    std::vector<uint32_t> controlBytes(controlSize, 1);
    controlBytes[0] = firstPassGroups;
    controlBytes[3] = 1; // one reduction of the global histogram
    controlBytes[6] = 1;
    controlBytes[9] = 0; // by default no local sorts have to be done
    VkUtil::uploadData(_vkContext.device, memory, 0, controlBytes.size() * sizeof(uint32_t), controlBytes.data());
    std::vector<uint32_t> cpuGroupInfos(groupInfoSize, 0);
    for (int i = 0; i < firstPassGroups; ++i)
    {
        uint32_t baseOffset = i * 258;
        cpuGroupInfos[baseOffset + 1] = i * KPB;
    }
    VkUtil::uploadData(_vkContext.device, memory, groupInfosOffset, cpuGroupInfos.size() * sizeof(uint32_t), cpuGroupInfos.data());

    // Descriptorset setup filling -------------------------------------------------------------------------------------
    VkDescriptorSet descSet;
    VkUtil::createDescriptorSets(_vkContext.device, {_localSortPipeline.descriptorSetLayout}, _vkContext.descriptorPool, &descSet);
    VkUtil::updateDescriptorSet(_vkContext.device, keysFront, v.size() * sizeof(uint32_t), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateArrayDescriptorSet(_vkContext.device, keysBack, v.size() * sizeof(uint32_t), 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateDescriptorSet(_vkContext.device, groupInfos, groupInfoSize * sizeof(uint32_t), 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateDescriptorSet(_vkContext.device, dispatch, controlSize * sizeof(uint32_t), 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateDescriptorSet(_vkContext.device, uniformFront, uniformBufferSize * sizeof(uint32_t), 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateArrayDescriptorSet(_vkContext.device, uniformBack, uniformBufferSize * sizeof(uint32_t), 3, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateDescriptorSet(_vkContext.device, localInfos, localInfoSize * sizeof(uint32_t), 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    

    // Creating and submitting commands -------------------------------------------------------------------------------------
    VkMemoryBarrier memBarrier{};
    memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    VkMemoryBarrier memBarrierIndirect{};
    memBarrierIndirect.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memBarrierIndirect.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrierIndirect.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    VkCommandBuffer commands;
    VkUtil::createCommandBuffer(_vkContext.device, _vkContext.commandPool, &commands);
    vkCmdResetQueryPool(commands, qPool, 0, timestampAmt);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _localSortPipeline.pipelineLayout, 0, 1, &descSet, 0, nullptr);
    int query = 0;
    const int passes = 4;
    for (int i = 0; i < passes; ++i)
    {
        vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qPool, query++);
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _histogramPipeline.pipeline);
        vkCmdDispatchIndirect(commands, dispatch, 0);
        vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0, {}, 0, {});
        
        vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qPool, query++);
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _scatterPipeline.pipeline);
        vkCmdDispatchIndirect(commands, dispatch, 0);
        vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0, {}, 0, {});
        if (i < passes - 1)
        {
            vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qPool, query++);
            vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _controlPipeline.pipeline);
            vkCmdDispatchIndirect(commands, dispatch, 24);
            vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0, {}, 0, {});
            
            vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qPool, query++);
            vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _dispatchPipeline.pipeline);
            vkCmdDispatch(commands, 1, 1, 1);
            vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &memBarrierIndirect, 0, {}, 0, {});
        }
    }
    // last but not least submit local sorts
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qPool, query++);    
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0, {}, 0, {});
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _localSortPipeline.pipeline);
    vkCmdDispatchIndirect(commands, dispatch, 36);
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qPool, query++);    
    auto t1 = std::chrono::high_resolution_clock::now();
    VkUtil::commitCommandBuffer(_vkContext.queue, commands);
    VkResult res = vkQueueWaitIdle(_vkContext.queue);
    check_vk_result(res);
    auto t2 = std::chrono::high_resolution_clock::now();
    VkUtil::downloadData(_vkContext.device, memory, keysFrontOffset, v.size() * sizeof(v[0]), v.data());
    
    // debugging code.... ---------------------------------------------------------------------------------------------------------
    //std::vector<uint32_t> cpuUniformInfoFront(uniformBufferSize);
    //std::vector<uint32_t> cpuUniformInfoBack(uniformBufferSize);
    //std::vector<uint32_t> cpuLocalInfo(localInfoSize);
    //if(passes & 1)
    //    VkUtil::downloadData(_vkContext.device, memory, keysBackOffset, v.size() * sizeof(v[0]), v.data());
    //else
    //    VkUtil::downloadData(_vkContext.device, memory, keysFrontOffset, v.size() * sizeof(v[0]), v.data());
    //std::vector<int> places0;
    //for(int i = 0; i < v.size(); ++i)if(!v[i]) places0.push_back(i);
    //std::vector<int> placesWrong;
    //std::vector<uint32_t> missingNumbers;
    //for(int i = 1; i < v.size(); ++i)if(v[i] >> ((4 - passes) * 8) < v[i - 1] >> ((4 - passes) * 8)) placesWrong.push_back(i);
    //std::vector<bool> alreadyFound(v.size(), false);
    //for(int i = 0; i < v.size(); ++i){
    //    if(origV.find(v[i]) == origV.end()) missingNumbers.push_back(i);
    //}
    //std::sort(placesWrong.begin(), placesWrong.end());
    //std::sort(missingNumbers.begin(), missingNumbers.end());
    //VkUtil::downloadData(_vkContext.device, memory, 0, controlBytes.size() * sizeof(uint32_t), controlBytes.data());
    //VkUtil::downloadData(_vkContext.device, memory, groupInfosOffset, cpuGroupInfos.size() * sizeof(uint32_t), cpuGroupInfos.data());
    //VkUtil::downloadData(_vkContext.device, memory, uniformFrontOffset, cpuUniformInfoFront.size() * sizeof(uint32_t), cpuUniformInfoFront.data());
    //VkUtil::downloadData(_vkContext.device, memory, uniformBackOffset, cpuUniformInfoBack.size() * sizeof(uint32_t), cpuUniformInfoBack.data());
    //VkUtil::downloadData(_vkContext.device, memory, localInfosOffset, cpuLocalInfo.size() * sizeof(uint32_t), cpuLocalInfo.data());
    //std::vector<std::pair<uint32_t, uint32_t>> bucketBounds;
    //std::vector<std::pair<uint32_t, uint32_t>> groupI;
    //std::vector<uint32_t> bucketControl(256, 0);
    //auto getBucket = [&](uint32_t val){return (val >> ((4-passes) * 8)) & 0xff;};
    //while(cpuUniformInfoFront[bucketBounds.size() * 258 + 4] != 0){
    //    bucketBounds.push_back({cpuUniformInfoFront[bucketBounds.size() * 258 + 3], cpuUniformInfoFront[bucketBounds.size() * 258 + 4]});
    //}
    //for(int i = 0; i < controlBytes[0]; ++i){
    //    groupI.push_back({cpuGroupInfos[i * 258], cpuGroupInfos[i * 258 + 1]});
    //    //check group info
    //    int groupStart = groupI.back().second;
    //    int histIndex = groupI.back().first;
    //    if(groupStart < bucketBounds[histIndex].first || groupStart >= bucketBounds[histIndex].second){
    //        bool nope = true;
    //    }
    //    if(i > 1 && groupStart - groupI[i - 1].second < KPB && histIndex == groupI[i - 1].first){
    //        bool hell = true;
    //    }
    //}
    //for(auto val: v) if(val >> ((5 - passes) * 8) == 138)bucketControl[getBucket(val)]++;
    //for(int i = 1; i < 256; ++i) bucketControl[i] += bucketControl[i - 1];
    std::vector<uint32_t> timestamps(timestampAmt);
    vkGetQueryPoolResults(_vkContext.device, qPool, 0, timestampAmt, timestamps.size() * sizeof(uint32_t), timestamps.data(), sizeof(uint32_t), 0);
    for(int i = 0; i < timestampAmt - 1; ++i){
        float a = _runs / (_runs + 1.0f);
        sortStats[i] = a * sortStats[i] + (1.0f - a) * (timestamps[i + 1] - timestamps[i]) * _timeMultiplier / 1e6;
    }
    _runs++;
#ifdef ENABLE_TEST_SORT
    std::cout << "Sorting " << v.size() << " elements took " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms" << std::endl;
    std::cout << "Timing differences in the pipelines itself were:" << std::endl;
    for(int i = 0; i < timestampAmt - 1; ++i){
        std::cout << sortStats[i] << std::endl;
    }
#endif

    // Cleaning everything up -------------------------------------------------------------------------------------
    vkDestroyBuffer(_vkContext.device, dispatch, nullptr);
    vkDestroyBuffer(_vkContext.device, uniformFront, nullptr);
    vkDestroyBuffer(_vkContext.device, uniformBack, nullptr);
    vkDestroyBuffer(_vkContext.device, keysFront, nullptr);
    vkDestroyBuffer(_vkContext.device, keysBack, nullptr);
    vkDestroyBuffer(_vkContext.device, groupInfos, nullptr);
    vkDestroyBuffer(_vkContext.device, localInfos, nullptr);
    vkFreeMemory(_vkContext.device, memory, nullptr);
    vkFreeDescriptorSets(_vkContext.device, _vkContext.descriptorPool, 1, &descSet);
    vkFreeCommandBuffers(_vkContext.device, _vkContext.commandPool, 1, &commands);
    vkDestroyQueryPool(_vkContext.device, qPool, nullptr);
}

void GpuRadixSorter::sort(std::vector<float> &v)
{
    if (!_histogramPipeline.pipeline)
    {
        std::cout << "Gpu sorting not available, using standard c++ sort" << std::endl;
        std::sort(v.begin(), v.end());
        return;
    }
    // sorting floats simply by mappgin them to uints and then running sortUints
    auto mapFloat2Uint = [](float n)
    {uint32_t f = *reinterpret_cast<uint32_t*>(&n); uint32_t mask = -int(f >> 31) | 0x80000000; return f ^ mask; };
    auto mapUint2Float = [](uint32_t n)
    {uint32_t mask = ((n >> 31) - 1) | 0x80000000; n ^= mask; return *reinterpret_cast<float*>(&n); };
    std::vector<uint32_t> m(v.size());
    for (uint32_t i = 0; i < v.size(); ++i)
        m[i] = mapFloat2Uint(v[i]);
    sort(m);
    for (uint32_t i = 0; i < m.size(); ++i)
        v[i] = mapUint2Float(m[i]);
}

bool GpuRadixSorter::checkLocalSort()
{
    bool correct = true;
    const uint32_t threadsPerLocalSort = 512, keysPerThread = 16;
    const uint32_t keysPerLocalSort = threadsPerLocalSort * keysPerThread;
    const uint32_t uinformBufferSize = 64;

    VkDescriptorSet descSet;
    VkUtil::createDescriptorSets(_vkContext.device, {_localSortPipeline.descriptorSetLayout}, _vkContext.descriptorPool, &descSet);
    VkBuffer keys, uniformBuffer;
    VkDeviceMemory memory;
    VkUtil::createBuffer(_vkContext.device, keysPerLocalSort * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &keys);
    VkUtil::createBuffer(_vkContext.device, uinformBufferSize * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &uniformBuffer);
    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    VkMemoryRequirements memReq;
    uint32_t memBits{};
    vkGetBufferMemoryRequirements(_vkContext.device, keys, &memReq);
    allocInfo.allocationSize = memReq.size;
    memBits |= memReq.memoryTypeBits;
    vkGetBufferMemoryRequirements(_vkContext.device, uniformBuffer, &memReq);
    allocInfo.allocationSize += memReq.size;
    memBits |= memReq.memoryTypeBits;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(_vkContext.physicalDevice, memBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkAllocateMemory(_vkContext.device, &allocInfo, nullptr, &memory);
    vkBindBufferMemory(_vkContext.device, uniformBuffer, memory, 0);
    vkBindBufferMemory(_vkContext.device, keys, memory, uinformBufferSize * sizeof(uint32_t));

    VkUtil::updateDescriptorSet(_vkContext.device, keys, keysPerLocalSort * sizeof(uint32_t), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateArrayDescriptorSet(_vkContext.device, keys, keysPerLocalSort * sizeof(uint32_t), 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateDescriptorSet(_vkContext.device, uniformBuffer, uinformBufferSize * sizeof(uint32_t), 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateArrayDescriptorSet(_vkContext.device, uniformBuffer, uinformBufferSize * sizeof(uint32_t), 3, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);

    std::vector<uint32_t> cpuKeys(keysPerLocalSort);
    for (auto &n : cpuKeys)
    {
        int r = rand() & 0xff;
        n = *reinterpret_cast<uint32_t *>(&r);
    }
    std::vector<uint32_t> orig = cpuKeys;
    std::vector<uint32_t> c = cpuKeys;
    std::vector<uint32_t> cpuUniform{1}; // pass number 1 to save in front buffer
    VkUtil::uploadData(_vkContext.device, memory, 0, cpuUniform.size() * sizeof(cpuUniform[0]), cpuUniform.data());
    VkCommandBuffer commands;
    VkUtil::createCommandBuffer(_vkContext.device, _vkContext.commandPool, &commands);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _localSortPipeline.pipelineLayout, 0, 1, &descSet, 0, nullptr);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _localSortPipeline.pipeline);
    vkCmdDispatch(commands, 1, 1, 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    VkUtil::uploadData(_vkContext.device, memory, uinformBufferSize * sizeof(uint32_t), cpuKeys.size() * sizeof(cpuKeys[0]), cpuKeys.data());
    VkUtil::commitCommandBuffer(_vkContext.queue, commands);
    VkResult res = vkQueueWaitIdle(_vkContext.queue);
    check_vk_result(res);
    VkUtil::downloadData(_vkContext.device, memory, uinformBufferSize * sizeof(uint32_t), cpuKeys.size() * sizeof(cpuKeys[0]), cpuKeys.data());
    auto t2 = std::chrono::high_resolution_clock::now();

    // test sorting on cpu
    auto t3 = std::chrono::high_resolution_clock::now();
    std::sort(c.begin(), c.end());
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "GPU: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << std::endl;
    std::cout << "CPU: " << std::chrono::duration<double, std::milli>(t4 - t2).count() << std::endl;
    // check for number loss
    std::vector<uint32_t> secondPart(&cpuKeys[1000], &cpuKeys[2000]);
    for (uint32_t n : orig)
    {
        if (std::find(cpuKeys.begin(), cpuKeys.end(), n) == cpuKeys.end())
        {
            // std::cout << "Lost " << n << std::endl;
        }
    }
    bool doubled = false;
    for (int i = 0; i < cpuKeys.size(); ++i)
    {
        if (c[i] != cpuKeys[i])
        {
            // std::cout << "[" << i << "]:" << c[i] << " - " << cpuKeys[i] << std::endl;
            correct = false;
        }
        if (i != 0 && c[i] == c[i - 1])
            doubled = true;
    }

    vkDestroyBuffer(_vkContext.device, uniformBuffer, nullptr);
    vkDestroyBuffer(_vkContext.device, keys, nullptr);
    vkFreeMemory(_vkContext.device, memory, nullptr);

    return correct;
}

bool GpuRadixSorter::checkHistogram()
{
    bool correct = true;
    const uint32_t threadsPerBlock = 384, keysPerThread = 18;
    const uint32_t uinformBufferSize = 320;
    const uint32_t keysSize = threadsPerBlock * keysPerThread;
    const uint32_t numBuckets = 256;
    const uint32_t numWorkGroups = threadsPerBlock / 32;
    const uint32_t pass = 3;

    // Buffer creation and filling -------------------------------------------------------------------------------------
    VkBuffer uniform, keysFront, keysBack, groupInfos;
    VkDeviceMemory memory;
    VkUtil::createBuffer(_vkContext.device, keysSize * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &keysFront);
    VkUtil::createBuffer(_vkContext.device, keysSize * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &keysBack);
    VkUtil::createBuffer(_vkContext.device, uinformBufferSize * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &uniform);
    VkUtil::createBuffer(_vkContext.device, numBuckets * numWorkGroups * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &groupInfos);
    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    VkMemoryRequirements memReq;
    uint32_t memBits{};
    vkGetBufferMemoryRequirements(_vkContext.device, keysFront, &memReq);
    allocInfo.allocationSize = memReq.size;
    memBits |= memReq.memoryTypeBits;
    vkGetBufferMemoryRequirements(_vkContext.device, keysBack, &memReq);
    allocInfo.allocationSize += memReq.size;
    memBits |= memReq.memoryTypeBits;
    vkGetBufferMemoryRequirements(_vkContext.device, uniform, &memReq);
    allocInfo.allocationSize += memReq.size;
    memBits |= memReq.memoryTypeBits;
    vkGetBufferMemoryRequirements(_vkContext.device, groupInfos, &memReq);
    allocInfo.allocationSize += memReq.size;
    memBits |= memReq.memoryTypeBits;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(_vkContext.physicalDevice, memBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkAllocateMemory(_vkContext.device, &allocInfo, nullptr, &memory);
    vkBindBufferMemory(_vkContext.device, uniform, memory, 0);
    vkBindBufferMemory(_vkContext.device, keysFront, memory, uinformBufferSize * sizeof(uint32_t));
    vkBindBufferMemory(_vkContext.device, keysBack, memory, uinformBufferSize * sizeof(uint32_t) + keysSize * sizeof(uint32_t));
    vkBindBufferMemory(_vkContext.device, groupInfos, memory, uinformBufferSize * sizeof(uint32_t) + 2 * keysSize * sizeof(uint32_t));
    std::vector<uint32_t> cpuKeys(keysSize);
    for (auto &n : cpuKeys)
    {
        int r = rand() & 0xff;
        n = *reinterpret_cast<uint32_t *>(&r);
    }
    VkUtil::uploadData(_vkContext.device, memory, uinformBufferSize * sizeof(uint32_t) + keysSize * sizeof(uint32_t), cpuKeys.size() * sizeof(cpuKeys[0]), cpuKeys.data());
    std::vector<uint32_t> cpuUniform(uinformBufferSize, 0);
    cpuUniform[0] = pass;
    cpuUniform[1] = 1; // 1 global histogram
    VkUtil::uploadData(_vkContext.device, memory, 0, cpuUniform.size() * sizeof(cpuUniform[0]), cpuUniform.data());

    // Descriptorset setup filling -------------------------------------------------------------------------------------
    VkDescriptorSet descSet;
    VkUtil::createDescriptorSets(_vkContext.device, {_localSortPipeline.descriptorSetLayout}, _vkContext.descriptorPool, &descSet);
    VkUtil::updateDescriptorSet(_vkContext.device, keysFront, keysSize * sizeof(uint32_t), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateArrayDescriptorSet(_vkContext.device, keysBack, keysSize * sizeof(uint32_t), 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateDescriptorSet(_vkContext.device, uniform, uinformBufferSize * sizeof(uint32_t), 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateDescriptorSet(_vkContext.device, groupInfos, numBuckets * numWorkGroups * sizeof(uint32_t), 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);

    // Creating and submitting commands -------------------------------------------------------------------------------------
    VkCommandBuffer commands;
    VkUtil::createCommandBuffer(_vkContext.device, _vkContext.commandPool, &commands);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _localSortPipeline.pipelineLayout, 0, 1, &descSet, 0, nullptr);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _histogramPipeline.pipeline);
    vkCmdDispatch(commands, 1, 1, 1);
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, {}, 0, {}, 0, {});
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _globalScanPipeline.pipeline);
    vkCmdDispatch(commands, 1 + 1, 1, 1); // 1 local work group has to be reduced as well
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, {}, 0, {}, 0, {});
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _scatterPipeline.pipeline);
    vkCmdDispatch(commands, 1, 1, 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    VkUtil::commitCommandBuffer(_vkContext.queue, commands);
    VkResult res = vkQueueWaitIdle(_vkContext.queue);
    check_vk_result(res);
    auto t2 = std::chrono::high_resolution_clock::now();
    VkUtil::downloadData(_vkContext.device, memory, 0, uinformBufferSize * sizeof(uint32_t), cpuUniform.data());
    std::vector<uint32_t> sortedGpu(keysSize);
    VkUtil::downloadData(_vkContext.device, memory, (uinformBufferSize) * sizeof(uint32_t), keysSize * sizeof(uint32_t), sortedGpu.data());

    // checking the results --------------------------------------------------------------------------------------------
    std::vector<uint32_t> hist(256);
    std::vector<uint32_t> prefix(256, 0);
    std::vector<uint32_t> sorted(keysSize);
    auto getBin = [](uint32_t n, uint32_t pass)
    { return (n >> ((3 - pass) * 8)) & 0xff; };
    uint32_t bin = getBin(255, 3);
    for (auto n : cpuKeys)
        hist[getBin(n, pass)]++;
    uint32_t cpuBinCount = 0, gpuBinCount = 0;
    if (false)
    { // histogram computation
        for (int i = 0; i < hist.size(); ++i)
        {
            cpuBinCount += hist[i];
            gpuBinCount += cpuUniform[i + 1];
        }
        for (int i = 0; i < hist.size(); ++i)
        {
            if (hist[i] != cpuUniform[1 + i])
            {
                std::cout << "Miss: CPU " << hist[i] << " | GPU " << cpuUniform[1 + i] << std::endl;
                correct = false;
            }
        }
    }
    if (false)
    { // exclusive summation
        for (int i = 1; i < hist.size(); ++i)
        {
            prefix[i] = prefix[i - 1] + hist[i - 1];
        }
        for (int i = 0; i < prefix.size(); ++i)
        {
            if (prefix[i] != cpuUniform[1 + i])
            {
                std::cout << "Miss: CPU " << prefix[i] << " | GPU " << cpuUniform[1 + i] << std::endl;
                correct = false;
            }
        }
    }
    if (true)
    { // key scattering
        for (int i = 1; i < hist.size(); ++i)
        {
            prefix[i] = prefix[i - 1] + hist[i - 1];
        }
        for (auto n : cpuKeys)
        {
            sorted[prefix[getBin(n, pass)]++] = n;
        }
        for (int i = 0; i < sorted.size(); ++i)
        {
            if (sorted[i] != sortedGpu[i])
            {
                std::cout << "Miss: CPU " << sorted[i] << " | GPU " << sortedGpu[i] << std::endl;
                correct = false;
            }
        }
    }

    std::cout << "GPU: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << std::endl;

    // Freeing resources -------------------------------------------------------------------------------------
    vkDestroyBuffer(_vkContext.device, uniform, nullptr);
    vkDestroyBuffer(_vkContext.device, keysFront, nullptr);
    vkDestroyBuffer(_vkContext.device, keysBack, nullptr);
    vkFreeMemory(_vkContext.device, memory, nullptr);

    return correct;
}