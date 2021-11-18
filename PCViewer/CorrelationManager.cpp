#define NOSTATICS
#include "CorrelationManager.hpp"
#undef NOSTATICS
#include <cmath>
#include <algorithm>
#include <numeric>
#include <algorithm>

CorrelationManager::CorrelationManager(const VkUtil::Context& context): _vkContext(context)
{
    if(context.device){
        VkShaderModule shader = VkUtil::createShaderModule(context.device, PCUtil::readByteFile(_pearsonShader));
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorCount = 1;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(binding);
        binding.binding = 1;
        bindings.push_back(binding);
        binding.binding = 3;
        bindings.push_back(binding);
        binding.binding = 2;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
        bindings.push_back(binding);
        VkUtil::createDescriptorSetLayout(context.device, bindings, &_pearsonPipeline.descriptorSetLayout);
        VkUtil::createComputePipeline(context.device, shader, {_pearsonPipeline.descriptorSetLayout}, &_pearsonPipeline.pipelineLayout, &_pearsonPipeline.pipeline);
    
        shader = VkUtil::createShaderModule(context.device, PCUtil::readByteFile(_meanShader));
        VkUtil::createComputePipeline(context.device, shader, {_pearsonPipeline.descriptorSetLayout}, &_meanPipeline.pipelineLayout, &_meanPipeline.pipeline);
        
        shader = VkUtil::createShaderModule(context.device, PCUtil::readByteFile(_spearmanShader));
        VkUtil::createDescriptorSetLayout(context.device, bindings, &_spearmanPipeline.descriptorSetLayout);
        VkUtil::createComputePipeline(context.device, shader, {_spearmanPipeline.descriptorSetLayout}, &_spearmanPipeline.pipelineLayout, &_spearmanPipeline.pipeline);
        
        shader = VkUtil::createShaderModule(context.device, PCUtil::readByteFile(_kendallShader));
        VkUtil::createDescriptorSetLayout(context.device, bindings, &_kendallPipeline.descriptorSetLayout);
        VkUtil::createComputePipeline(context.device, shader, {_kendallPipeline.descriptorSetLayout}, &_kendallPipeline.pipelineLayout, &_kendallPipeline.pipeline);
    }
}

CorrelationManager::~CorrelationManager()
{
    _pearsonPipeline.vkDestroy(_vkContext);
    _spearmanPipeline.vkDestroy(_vkContext);
    _kendallPipeline.vkDestroy(_vkContext);
    _meanPipeline.vkDestroy(_vkContext);
}

void CorrelationManager::calculateCorrelation(const DrawList& dl, CorrelationMetric metric, int baseAttribute, bool useGpu) 
{
    if(baseAttribute < 0) return;
    if(useGpu && _vkContext.screenSize[0] != 0xffffffff)
        _execCorrelationGPU(dl, metric, baseAttribute);
    else 
        _execCorrelationCPU(dl, metric, baseAttribute);
}

void CorrelationManager::_execCorrelationCPU(const DrawList& dl, CorrelationMetric metric, int baseAttribute) 
{
    int amtOfAttributes = dl.data->columns.size();
    DrawlistCorrelations& dlCorrelations = correlations[dl.name];
    dlCorrelations.drawlist = dl.name;
    AttributeCorrelation& curAttCorrelation = dlCorrelations.attributeCorrelations[baseAttribute];
    curAttCorrelation.metric = metric;
    curAttCorrelation.baseAttribute = baseAttribute;
    curAttCorrelation.correlationScores.resize(amtOfAttributes);
    std::vector<uint8_t> activations(dl.data->size());
    VkUtil::downloadData(_vkContext.device, dl.dlMem, dl.activeIndicesBufferOffset, activations.size(), activations.data());
    std::vector<double> means(amtOfAttributes, 0);
    std::vector<double> nominator(amtOfAttributes, 0), denom1(amtOfAttributes, 0), denom2(amtOfAttributes, 0);
    switch(metric){
    case CorrelationMetric::Pearson:
    {
        for(int a = 0; a < amtOfAttributes; ++a){
            int c = 0;
            for(size_t i = 0; i < dl.indices.size(); ++i){
                if(!activations[dl.indices[i]]) continue;
                means[a] += ((*dl.data)(dl.indices[i], a) - means[a]) / (++c);
            }
        }
        for(size_t i = 0; i < dl.indices.size(); ++i){
            for(int a = 0; a < amtOfAttributes; ++a){
                if(!activations[dl.indices[i]]) continue;
                size_t index = dl.indices[i];
                double aDiff = (*dl.data)(index, a) - means[a];
                double bDiff = (*dl.data)(index, baseAttribute) - means[baseAttribute];
                nominator[a] += aDiff * bDiff;
                denom1[a] += aDiff * aDiff;
                denom2[a] += bDiff * bDiff;
            }
        }
        for(int i = 0; i < amtOfAttributes; ++i){
            curAttCorrelation.correlationScores[i] = nominator[i] / std::sqrt((denom1[i] + 1e-5) * (denom2[i] + 1e-5));
            assert(curAttCorrelation.correlationScores[i] >= -1 && curAttCorrelation.correlationScores[i] <= 1);
        }
        break;
    }
    case CorrelationMetric::SpearmanRank:
    {
        std::vector<std::vector<size_t>> sortedIdx(amtOfAttributes);
        // sorts all attributes of data and puts a vector of indices into sortedIdx for each attribute
        // Index vector describes the position of a data point 
        std::vector<size_t> reducedInd;
        for(uint32_t i: dl.indices) if(activations[i]) reducedInd.push_back(i);
        for(int i =0; i < amtOfAttributes; ++i){
            auto& idx = sortedIdx[i];
            idx.resize(reducedInd.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b){return (*dl.data)(reducedInd[a], i) < (*dl.data)(reducedInd[b], i);});
            std::vector<size_t> cpy(idx.size());
            for(size_t i = 0; i < idx.size(); ++i) cpy[idx[i]] = i;
            idx = cpy;
        }
        for(int a = 0; a < amtOfAttributes; ++a) means[a] = (sortedIdx[a].size() - 1) / 2.0;
        for(size_t i = 0; i < reducedInd.size(); ++i){
            for(int a = 0; a < amtOfAttributes; ++a){
                size_t aIdx = i;
                size_t bIdx = i;
                double aDiff = sortedIdx[a][aIdx] - means[a];
                double bDiff = sortedIdx[baseAttribute][bIdx] - means[baseAttribute];
                nominator[a] += aDiff * bDiff;
                denom1[a] += aDiff * aDiff;
                denom2[a] += bDiff * bDiff;
            }
        }
        for(int i = 0; i < amtOfAttributes; ++i){
            curAttCorrelation.correlationScores[i] = nominator[i] / std::sqrt(denom1[i] * denom2[i]);
            assert(curAttCorrelation.correlationScores[i] >= -1 && curAttCorrelation.correlationScores[i] <= 1);
        }
        break;
    }
    case CorrelationMetric::KendallRank:
    {
        double bin = activations[dl.indices[0]];
        for(size_t i = 1; i < dl.indices.size(); ++i){
            bin += activations[dl.indices[i]];
            for(size_t j = 0; j < i; ++j){
                size_t indexA = dl.indices[i], indexB = dl.indices[j];
                if(!activations[indexA] || !activations[indexB]) continue;
                for(int a = 0; a < amtOfAttributes; ++a){
                    double aDiff = (*dl.data)(indexA, a) - (*dl.data)(indexB, a);
                    aDiff = std::signbit(aDiff) * -2.0 + 1;
                    double bDiff = (*dl.data)(indexA, baseAttribute) - (*dl.data)(indexB, baseAttribute);
                    bDiff = std::signbit(bDiff) * -2.0 + 1;
                    nominator[a] += aDiff * bDiff;
                }
            }
        }
        bin = bin * (bin - 1) / 2;
        for(int i = 0; i < amtOfAttributes; ++i){
            assert(nominator[i] / bin >= -1 && nominator[i] / bin <= 1);
            curAttCorrelation.correlationScores[i] = nominator[i] / bin;
        }
        break;
    }
    };
}

void CorrelationManager::_execCorrelationGPU(const DrawList& dl, CorrelationMetric metric, int baseAttribute) 
{
    //TODO implement...
    int amtOfAttributes = dl.data->columns.size();
    DrawlistCorrelations& dlCorrelations = correlations[dl.name];
    dlCorrelations.drawlist = dl.name;
    AttributeCorrelation& curAttCorrelation = dlCorrelations.attributeCorrelations[baseAttribute];
    curAttCorrelation.metric = metric;
    curAttCorrelation.baseAttribute = baseAttribute;
    curAttCorrelation.correlationScores.resize(amtOfAttributes);

    VkDescriptorSet descSet;
    switch(metric){
    case CorrelationMetric::Pearson:{
        uint32_t bufferByteSize = (4 + dl.attributes->size() * 6) * sizeof(float);
        std::vector<float> bufferData(bufferByteSize / sizeof(float));
        reinterpret_cast<uint32_t*>(bufferData.data())[0] = dl.indices.size();
        reinterpret_cast<uint32_t*>(bufferData.data())[1] = dl.attributes->size();
        reinterpret_cast<uint32_t*>(bufferData.data())[2] = static_cast<uint32_t>(baseAttribute);
        reinterpret_cast<uint32_t*>(bufferData.data())[3] = dl.activeLinesAmt; //todo right value
        for(int i = 0; i < 4 * dl.attributes->size(); ++i) 
        {
            bufferData[4 + i] = 0;
            if(i < dl.attributes->size()){
                bufferData[4 + 4 * amtOfAttributes + 2 * i] = dl.attributes->at(i).min;
                bufferData[4 + 4 * amtOfAttributes + 2 * i + 1] = dl.attributes->at(i).max;
            }
        }
        VkBuffer buffer;
        VkUtil::createBuffer(_vkContext.device, bufferByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buffer);
        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        VkMemoryRequirements memReq; vkGetBufferMemoryRequirements(_vkContext.device, buffer, &memReq);
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = VkUtil::findMemoryType(_vkContext.physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory memory;
        VkResult res = vkAllocateMemory(_vkContext.device, &allocInfo, nullptr, &memory); check_vk_result(res);
        res = vkBindBufferMemory(_vkContext.device, buffer, memory, 0); check_vk_result(res);
        VkUtil::uploadData(_vkContext.device, memory, 0, bufferByteSize, bufferData.data());
        VkUtil::createDescriptorSets(_vkContext.device, {_pearsonPipeline.descriptorSetLayout}, _vkContext.descriptorPool, &descSet);
        VkUtil::updateDescriptorSet(_vkContext.device, dl.buffer, VK_WHOLE_SIZE, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
        VkUtil::updateDescriptorSet(_vkContext.device, dl.indexBuffer, VK_WHOLE_SIZE, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
        VkUtil::updateTexelBufferDescriptorSet(_vkContext.device, dl.activeIndicesBufferView, 2, descSet);
        VkUtil::updateDescriptorSet(_vkContext.device, buffer, bufferByteSize, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);

        uint32_t workAmt = (dl.indices.size() + compLocalSize - 1) / compLocalSize;
        VkCommandBuffer commands; 
        VkUtil::createCommandBuffer(_vkContext.device, _vkContext.commandPool, &commands);
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _meanPipeline.pipelineLayout, 0, 1, &descSet, 0, {});
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _meanPipeline.pipeline);
        vkCmdDispatch(commands, workAmt, 1, 1);
        vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, {}, 0, {}, 0, {});
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _pearsonPipeline.pipelineLayout, 0, 1, &descSet, 0, {});
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, _pearsonPipeline.pipeline);
        vkCmdDispatch(commands, workAmt, 1, 1);
        VkUtil::commitCommandBuffer(_vkContext.queue, commands);
        res = vkQueueWaitIdle(_vkContext.queue); check_vk_result(res);

        // get data from gpu and write to correlation vector
        VkUtil::downloadData(_vkContext.device, memory, 0, bufferByteSize, bufferData.data());
        std::vector<float> means(dl.attributes->size());
        for(int i = 0; i < amtOfAttributes; ++i){
            means[i] = bufferData[4 + i] * (dl.attributes->at(i).max - dl.attributes->at(i).min) + dl.attributes->at(i).min;
        }
        for(int i = 0; i < amtOfAttributes; ++i){
            curAttCorrelation.correlationScores[i] = bufferData[4 + amtOfAttributes + i] / std::sqrt((bufferData[4 + 2 * amtOfAttributes + i] + 1e-5) * (bufferData[4 + 3 * amtOfAttributes + i] + 1e-5));
            curAttCorrelation.correlationScores[i] = std::clamp(curAttCorrelation.correlationScores[i], -1.f, 1.f);
            assert(curAttCorrelation.correlationScores[i] >= -1 && curAttCorrelation.correlationScores[i] <= 1);
        }

        vkFreeDescriptorSets(_vkContext.device, _vkContext.descriptorPool, 1, &descSet);
        break;
    }
    case CorrelationMetric::SpearmanRank:{
        break;
    }
    case CorrelationMetric::KendallRank:{
        break;
    }
    }
}
