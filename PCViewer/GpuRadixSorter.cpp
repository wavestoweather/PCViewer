#include "GpuRadixSorter.hpp"
#include "PCUtil.h"
#include <algorithm>
#include <chrono>
#include <numeric>

GpuRadixSorter::GpuRadixSorter(const VkUtil::Context& context):
_vkContext(context)
{
    if(context.device){
        //query subgroup size to set the specialization constant correctly
        VkPhysicalDeviceSubgroupProperties subgroupProperties;
        subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        subgroupProperties.pNext = NULL;
        VkPhysicalDeviceProperties2 physicalDeviceProperties;
        physicalDeviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        physicalDeviceProperties.pNext = &subgroupProperties;
        vkGetPhysicalDeviceProperties2(_vkContext.physicalDevice, &physicalDeviceProperties);

        std::vector<VkDescriptorSetLayoutBinding> bindings;
        VkDescriptorSetLayoutBinding binding;
        binding.binding = 0;
        binding.descriptorCount = 2;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(binding);

        binding.binding = 1;
        binding.descriptorCount = 1;
        bindings.push_back(binding);
        
        binding.binding = 2;
        bindings.push_back(binding);

        binding.binding = 3;
        bindings.push_back(binding);

        std::vector<VkSpecializationMapEntry> entries{{0,0,sizeof(uint32_t)}};   // subgroup size
        std::vector<uint32_t> values{subgroupProperties.subgroupSize};
        VkSpecializationInfo specializationConstants{
            static_cast<uint32_t>(entries.size()),
            entries.data(),
            sizeof(uint32_t),
            values.data()
        };
        
        //VkShaderModule shaderModule = VkUtil::createShaderModule(context.device, PCUtil::readByteFile(_shaderPath));
        //VkUtil::createDescriptorSetLayout(context.device, bindings, &_pipelineInfo.descriptorSetLayout);
        //VkUtil::createComputePipeline(context.device, shaderModule, {_pipelineInfo.descriptorSetLayout}, &_pipelineInfo.pipelineLayout, &_pipelineInfo.pipeline);

        VkShaderModule shaderModule = VkUtil::createShaderModule(context.device, PCUtil::readByteFile(_shaderLocalSortPath));
        VkUtil::createDescriptorSetLayout(context.device, bindings, &_localSortPipeline.descriptorSetLayout);
        VkUtil::createComputePipeline(context.device, shaderModule, {_localSortPipeline.descriptorSetLayout}, &_localSortPipeline.pipelineLayout, &_localSortPipeline.pipeline, &specializationConstants);
    
        shaderModule = VkUtil::createShaderModule(context.device, PCUtil::readByteFile(_shaderHistogramPath));
        VkUtil::createComputePipeline(context.device, shaderModule, {_localSortPipeline.descriptorSetLayout}, & _histogramPipeline.pipelineLayout, &_histogramPipeline.pipeline);

    }
}

GpuRadixSorter::~GpuRadixSorter(){
    _pipelineInfo.vkDestroy(_vkContext);
    _localSortPipeline.vkDestroy(_vkContext);
    _localSortPipeline.vkDestroy(_vkContext);
}

bool GpuRadixSorter::checkLocalSort(){
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

    std::vector<uint32_t> cpuKeys(keysPerLocalSort);
    for(auto& n: cpuKeys) {int r = rand() & 0xff; n = *reinterpret_cast<uint32_t*>(&r);}
    std::vector<uint32_t> orig = cpuKeys;
    std::vector<uint32_t> c = cpuKeys;
    std::vector<uint32_t> cpuUniform{1};    // pass number 1 to save in front buffer
    VkUtil::uploadData(_vkContext.device, memory, 0, cpuUniform.size() * sizeof(cpuUniform[0]), cpuUniform.data());
    VkCommandBuffer commands;
    VkUtil::createCommandBuffer(_vkContext.device, _vkContext.commandPool, &commands);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _localSortPipeline.pipelineLayout, 0, 1, &descSet, 0, nullptr);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _localSortPipeline.pipeline);
    vkCmdDispatch(commands, 1, 1, 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    VkUtil::uploadData(_vkContext.device, memory, uinformBufferSize * sizeof(uint32_t), cpuKeys.size() * sizeof(cpuKeys[0]), cpuKeys.data());
    VkUtil::commitCommandBuffer(_vkContext.queue, commands);
    VkResult res = vkQueueWaitIdle(_vkContext.queue); check_vk_result(res);
    VkUtil::downloadData(_vkContext.device, memory, uinformBufferSize * sizeof(uint32_t), cpuKeys.size() * sizeof(cpuKeys[0]), cpuKeys.data());
    auto t2 = std::chrono::high_resolution_clock::now();

    //test sorting on cpu
    auto t3 = std::chrono::high_resolution_clock::now();
    std::sort(c.begin(), c.end());
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "GPU: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << std::endl;
    std::cout << "CPU: " << std::chrono::duration<double, std::milli>(t4 - t2).count() << std::endl;
    //check for number loss
    for(uint32_t n: orig){
        if(std::find(cpuKeys.begin(), cpuKeys.end(), n) == cpuKeys.end()){
            std::cout << "Lost " << n << std::endl;
        }
    }
    bool doubled = false;
    for(int i = 0; i < cpuKeys.size(); ++i){
        if(c[i] != cpuKeys[i]){
            std::cout << "[" << i << "]:" << c[i] << " - " << cpuKeys[i] << std::endl;
            correct = false;
        }
        if (i != 0 && c[i] == c[i - 1]) doubled = true;
    }

    vkDestroyBuffer(_vkContext.device, uniformBuffer, nullptr);
    vkDestroyBuffer(_vkContext.device, keys, nullptr);
    vkFreeMemory(_vkContext.device, memory, nullptr);

    return correct;
}

bool GpuRadixSorter::checkHistogram(){
    bool correct = true;
    const uint32_t threadsPerBlock = 384, keysPerThread = 18;
    const uint32_t uinformBufferSize = 320;
    const uint32_t keysSize = threadsPerBlock * keysPerThread;
    const uint32_t pass = 3;

    // Buffer creation and filling -------------------------------------------------------------------------------------
    VkBuffer uniform, keys;
    VkDeviceMemory memory;
    VkUtil::createBuffer(_vkContext.device, keysSize * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &keys);
    VkUtil::createBuffer(_vkContext.device, uinformBufferSize * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &uniform);
    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    VkMemoryRequirements memReq;
    uint32_t memBits{};
    vkGetBufferMemoryRequirements(_vkContext.device, keys, &memReq);
    allocInfo.allocationSize = memReq.size;
    memBits |= memReq.memoryTypeBits;
    vkGetBufferMemoryRequirements(_vkContext.device, uniform, &memReq);
    allocInfo.allocationSize += memReq.size;
    memBits |= memReq.memoryTypeBits;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(_vkContext.physicalDevice, memBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkAllocateMemory(_vkContext.device, &allocInfo, nullptr, &memory);
    vkBindBufferMemory(_vkContext.device, uniform, memory, 0);
    vkBindBufferMemory(_vkContext.device, keys, memory, uinformBufferSize * sizeof(uint32_t));
    std::vector<uint32_t> cpuKeys(keysSize);
    for(auto& n: cpuKeys) {int r = rand() & 0xff; n = *reinterpret_cast<uint32_t*>(&r);}
    VkUtil::uploadData(_vkContext.device, memory, uinformBufferSize * sizeof(uint32_t), cpuKeys.size() * sizeof(cpuKeys[0]), cpuKeys.data());
    std::vector<uint32_t> cpuUniform(uinformBufferSize, 0);
    cpuUniform[0] = pass;
    VkUtil::uploadData(_vkContext.device, memory, 0, cpuUniform.size() * sizeof(cpuUniform[0]), cpuUniform.data());

    // Descriptorset setup filling -------------------------------------------------------------------------------------
    VkDescriptorSet descSet;
    VkUtil::createDescriptorSets(_vkContext.device, {_localSortPipeline.descriptorSetLayout}, _vkContext.descriptorPool, &descSet);
    VkUtil::updateDescriptorSet(_vkContext.device, keys, keysSize * sizeof(uint32_t), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateArrayDescriptorSet(_vkContext.device, keys, keysSize * sizeof(uint32_t), 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateDescriptorSet(_vkContext.device, uniform, uinformBufferSize * sizeof(uint32_t), 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);

    // Creating and submitting commands -------------------------------------------------------------------------------------
    VkCommandBuffer commands;
    VkUtil::createCommandBuffer(_vkContext.device, _vkContext.commandPool, &commands);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _localSortPipeline.pipelineLayout, 0, 1, &descSet, 0, nullptr);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _histogramPipeline.pipeline);
    vkCmdDispatch(commands, 1, 1, 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    VkUtil::commitCommandBuffer(_vkContext.queue, commands);
    VkResult res = vkQueueWaitIdle(_vkContext.queue); check_vk_result(res);
    auto t2 = std::chrono::high_resolution_clock::now();
    VkUtil::downloadData(_vkContext.device, memory, 0 ,uinformBufferSize * sizeof(uint32_t), cpuUniform.data());
    
    //checking the results --------------------------------------------------------------------------------------------
    std::vector<uint> hist(256);
    auto getBin = [](uint32_t n, uint32_t pass){return (n >> ((3 - pass) * 8)) & 0xff;};
    uint32_t bin = getBin(255, 3);
    for(auto n: cpuKeys) hist[getBin(n, pass)]++;
    uint32_t z_count = 0;
    for(auto n: cpuKeys) if(getBin(n, pass) == 0) z_count++;
    uint cpuBinCount = 0, gpuBinCount = 0;
    for(int i = 0; i < hist.size(); ++i){
        cpuBinCount += hist[i];
        gpuBinCount += cpuUniform[i + 1];
    }
    for(int i = 0; i < hist.size(); ++i){
        if(hist[i] != cpuUniform[1 + i]){
            std::cout << "Miss: CPU " << hist[i] << " | GPU " << cpuUniform[1 + i] << std::endl;
            correct = false;
        }
    }
    std::cout << "GPU: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << std::endl;

    // Freeing resources -------------------------------------------------------------------------------------
    vkDestroyBuffer(_vkContext.device, uniform, nullptr);
    vkDestroyBuffer(_vkContext.device, keys, nullptr);
    vkFreeMemory(_vkContext.device, memory, nullptr);

    return correct;
}