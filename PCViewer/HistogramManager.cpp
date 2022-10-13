#include "HistogramManager.h"
#include <cmath>
#include <numeric>
#include <list>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

HistogramManager::HistogramManager(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool, uint32_t binsAmount) : device(device), physicalDevice(physicalDevice), commandPool(commandPool), queue(queue), descriptorPool(descriptorPool), numOfBins(binsAmount)
{
    VkShaderModule module = VkUtil::createShaderModule(device, PCUtil::readByteFile(std::string(SHADERPATH)));

    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
    VkDescriptorSetLayoutBinding binding = {};
    binding.binding = 0;
    binding.descriptorCount = 1;        //informations
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBindings.push_back(binding);

    binding.binding = 1;                //indices
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBindings.push_back(binding);

    binding.binding = 2;                //data
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBindings.push_back(binding);

    binding.binding = 3;                //activations
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    layoutBindings.push_back(binding);

    binding.binding = 4;                //bins
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBindings.push_back(binding);

    VkUtil::createDescriptorSetLayout(device, layoutBindings, &descriptorSetLayout);

    std::vector<VkDescriptorSetLayout> layouts;
    layouts.push_back(descriptorSetLayout);
    VkUtil::createComputePipeline(device, module, layouts, &pipelineLayout, &pipeline);

    stdDev = -1;

    ignoreZeroValues = false;
    ignoreZeroBins = false;
    logScale = nullptr;
    adaptMinMaxToBrush = false;
}

HistogramManager::~HistogramManager()
{
    if (pipeline) {
        vkDestroyPipeline(device, pipeline, nullptr);
    }
    if (pipelineLayout) {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    }
    if (descriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    }
    if (logScale) {
        delete[] logScale;
    }
}

void HistogramManager::computeHistogramm(const std::string& name, std::vector<std::pair<float, float>>& minMax, VkBuffer data, uint32_t amtOfData, VkBuffer indices, uint32_t amtOfIndices, VkBufferView indicesActivations)
{
    if (!logScale) {
        logScale = new bool[minMax.size()];
        for (int i = 0; i < minMax.size(); ++i) logScale[i] = false;
    }

    VkResult err;

    uint32_t infosByteSize = (4 + minMax.size() * 2) * sizeof(float);
    char* infosBytes = new char[infosByteSize];
    uint32_t* inf = (uint32_t*)infosBytes;
    inf[0] = numOfBins;
    inf[1] = minMax.size();
    inf[2] = amtOfIndices;
    inf[3] = ignoreZeroValues;
#ifdef _DEBUG
    std::cout << "Bins: " << numOfBins << std::endl << "Amount of attributes: " << minMax.size() << std::endl << "Amount of indices: " << amtOfIndices << std::endl;
#endif
    float* infos = (float*)infosBytes;
    infos += 4;
    for (int i = 0; i < minMax.size(); ++i) {
        infos[2 * i] = minMax[i].first;
        infos[2 * i + 1] = minMax[i].second;
#ifdef _DEBUG
        std::cout << infos[2 * i] << "|" << infos[2 * i + 1] << std::endl;
#endif
    }

    uint32_t binsByteSize = minMax.size() * numOfBins * sizeof(uint32_t);
    char* binsBytes = new char[binsByteSize];
    for (int i = 0; i < minMax.size() * numOfBins; ++i) ((float*)binsBytes)[i] = 0;

    //buffer allocations
    VkUtil::createBuffer(device, infosByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, uboBuffers);
    VkUtil::createBuffer(device, binsByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &uboBuffers[1]);
    uboOffsets[0] = 0;
    uint32_t memType = 0;
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, uboBuffers[0], &memReq);
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize += memReq.size;
    memType |= memReq.memoryTypeBits;

    uboOffsets[1] = allocInfo.allocationSize;
    vkGetBufferMemoryRequirements(device, uboBuffers[1], &memReq);
    allocInfo.allocationSize += memReq.size;
    memType |= memReq.memoryTypeBits;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memType, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(device, &allocInfo, nullptr, &uboMemory);

    vkBindBufferMemory(device, uboBuffers[0], uboMemory, uboOffsets[0]);
    vkBindBufferMemory(device, uboBuffers[1], uboMemory, uboOffsets[1]);

    //upload of data
    VkUtil::uploadData(device, uboMemory, 0, infosByteSize, infosBytes);
    VkUtil::uploadData(device, uboMemory, uboOffsets[1], binsByteSize, binsBytes);

    //creation of the descriptor set
    VkDescriptorSet descSet;
    std::vector<VkDescriptorSetLayout> layouts;
    layouts.push_back(descriptorSetLayout);
    VkUtil::createDescriptorSets(device, layouts, descriptorPool, &descSet);
    VkUtil::updateDescriptorSet(device, uboBuffers[0], infosByteSize, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateDescriptorSet(device, indices, amtOfIndices * sizeof(uint32_t), 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateDescriptorSet(device, data, VK_WHOLE_SIZE, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateTexelBufferDescriptorSet(device, indicesActivations, 3, descSet);
    VkUtil::updateDescriptorSet(device, uboBuffers[1], binsByteSize, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);

    //running the compute pipeline
    VkCommandBuffer commands;
    VkUtil::createCommandBuffer(device, commandPool, &commands);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, {});
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    int patchAmount = amtOfIndices / LOCALSIZE;
    patchAmount += (amtOfIndices % LOCALSIZE) ? 1 : 0;
    vkCmdDispatch(commands, patchAmount, 1, 1);
    VkUtil::commitCommandBuffer(queue, commands);
    err = vkQueueWaitIdle(queue);
    check_vk_result(err);

    //downloading results, analysing and saving them
    VkUtil::downloadData(device, uboMemory, uboOffsets[1], binsByteSize, binsBytes);
    uint32_t* bins = (uint32_t*)binsBytes;
    Histogram histogram = {};
    for (int i = 0; i < minMax.size(); ++i) {
        histogram.bins.push_back({ });            //push back empty vector
        histogram.binsRendered.push_back({ });            //push back empty vector
        histogram.originalBins.push_back({ });
        histogram.maxCount.push_back({ });
        histogram.area.push_back(0);
        histogram.areaShown.push_back(0);
        histogram.areaRendered.push_back(0);
        for (int j = 0; j < numOfBins; ++j) {
            float curVal = 0;
            //int div = 0;
            //int h = .2f * numOfBins;
            //for (int k = -h>>1; k <= h>>1; k += 1) {    //applying a box cernel according to chamers et al.
            //    if (j + k >= 0 && j + k < numOfBins) {
            //        curVal += bins[i * numOfBins + j + k];
            //        div++;
            //    }
            //}
            //curVal /= div;
            //if (curVal > maxVal)maxVal = curVal;
            histogram.bins.back().push_back(curVal);
            histogram.originalBins.back().push_back(bins[i * numOfBins + j]);
            histogram.area.back() += histogram.originalBins.back().back();
        }
    }
    histogram.ranges = minMax;

    // determineSideHist(histogram);

    histograms[name] = histogram;
    updateSmoothedValues();

    vkFreeCommandBuffers(device, commandPool, 1, &commands);
    vkFreeDescriptorSets(device, descriptorPool, 1, &descSet);
    vkDestroyBuffer(device, uboBuffers[0], nullptr);
    vkDestroyBuffer(device, uboBuffers[1], nullptr);
    vkFreeMemory(device, uboMemory, nullptr);
    delete[] infosBytes;
    delete[] binsBytes;
}

HistogramManager::Histogram& HistogramManager::getHistogram(const std::string& name)
{
    return histograms[name];
}

bool HistogramManager::containsHistogram(const std::string& name)
{
    return histograms.find(name) != histograms.end();
}

float HistogramManager::computeHistogramDistance(std::string& nameRep, std::string& name, bool **active,  int mode)
{
    std::map<std::string, Histogram>::iterator histRep = histograms.find(nameRep);
    std::map<std::string, Histogram>::iterator hist = histograms.find(name);

    //std::vector<float> chiSquared;
    float chiSquaredSum = 0;
    float sumOfBinsInRep = 0;

    if (mode == 0){
        // Loop over attributes
        for (unsigned int i = 0; i < histRep->second.originalBins.size(); ++i)
        {
            if (!((*active)[i])){continue;}
            //chiSquared.push_back(0);        // Loop over bins
            for (unsigned int j = 0; j < histRep->second.originalBins.at(i).size(); ++j)
            {
                float valRep = histRep->second.originalBins.at(i).at(j);
                sumOfBinsInRep += valRep;
                float val = hist->second.originalBins.at(i).at(j);
                if (val + valRep > 0){
                chiSquaredSum += (val-valRep)*(val-valRep) / (val + valRep);}
                //chiSquared[i] += (val-valRep)*(val-valRep) / (val + valRep);
            }
        }
    }
    else if (mode == 1){
        // Loop over attributes
        for (unsigned int i = 0; i < histRep->second.binsRendered.size(); ++i)
        {
            if (!((*active)[i])){continue;}
            //chiSquared.push_back(0);        // Loop over bins
            for (unsigned int j = 0; j < histRep->second.binsRendered.at(i).size(); ++j)
            {
                float valRep = histRep->second.binsRendered.at(i).at(j);
                sumOfBinsInRep += valRep;
                float val = hist->second.binsRendered.at(i).at(j);
                if (val + valRep > 0){
                chiSquaredSum += (val-valRep)*(val-valRep) / (val + valRep);}
                //chiSquared[i] += (val-valRep)*(val-valRep) / (val + valRep);
            }
        }
    }

    // Scale by sum of bins in representative member
    chiSquaredSum /= sumOfBinsInRep;

    return chiSquaredSum / 2.0 ;
}

void HistogramManager::setNumberOfBins(uint32_t n)
{
    numOfBins = n;
}

void HistogramManager::setSmoothingKernelSize(float stdDev)
{
    this->stdDev = stdDev;
    
    updateSmoothedValues();
}

void HistogramManager::updateSmoothedValues()
{
    for (auto& hist : histograms) {
        updateSmoothedValues(hist.second);
    }
}

void HistogramManager::updateSmoothedValues(Histogram& hist)
{
    float stdDevSq = 2 * stdDev * stdDev;
    int kSize = (stdDev < 0) ? 0.2 * numOfBins + 1 : stdDev * 3 + 1;    //the plus 1 is there to realise the ceiling function

    //integrated is to 3 sigma standard deviation
    hist.maxGlobalCount = 0;
    float maxVal = 0;
    hist.area = std::vector<float>(hist.area.size(), 0);
    int att = 0;
    for (auto& attribute : hist.originalBins) {
        for (int bin = 0; bin < attribute.size(); ++bin) {
            float divisor = 0;
            float divider = 0;

            for (int k = -kSize; k <= kSize; ++k) {
                if (bin + k >= ((ignoreZeroBins)? 1:0) && bin + k < attribute.size()) {
                    float factor = std::exp(-(k * k) / stdDevSq);
                    divisor += attribute[bin + k] * factor;
                    divider += factor;
                }
            }

            hist.bins[att][bin] = divisor / divider;
            if (logScale[att]) hist.bins[att][bin] = log(hist.bins[att][bin] + 1);
            hist.area[att] += hist.bins[att][bin];
            if (hist.bins[att][bin] > maxVal) maxVal = hist.bins[att][bin];
        }
        hist.maxCount[att] = maxVal;
        if (maxVal > hist.maxGlobalCount) {
            hist.maxGlobalCount = maxVal;
        }
        maxVal = 0;
        ++att;
    }
}


void HistogramManager::determineSideHist(Histogram& hist, bool **active, bool considerBlendingOrder)
{
    std::vector<std::vector<float>> *bins = nullptr;
    std::vector<float> *area = nullptr;
    // If binsRendered is filled, the actual rendered size of the violins is known. Hence, this is taken to compute the side assignement. 
    // If it is empty, the vertical scaling as well as the specific horizontal scaling might be ignored!

    bool bRendered = false;
    if (hist.binsRendered.size() > 0)
    {
        if (hist.binsRendered[0].size() > 0)
        {
            bins = &hist.binsRendered;
            area = &hist.areaRendered;
            for (unsigned int i = 0; i < area->size(); ++i)
            {
                // If less than one pixel is occupied, set the area to plain 0.
                if ((area->at(i) < 1) || std::isnan(area->at(i)))
                {
                    (*area)[i] = 0;
                }
            }
            bRendered = true;
        }
    }

    if (!bRendered) { bins = &hist.bins; area = &hist.area; }

    unsigned int n = bins->size();
    std::vector<std::vector<float>> histOverlaps(n, std::vector<float>(n, 0));
    std::vector<std::vector<float>> histOverlapsPerc(n, std::vector<float>(n, 0));
    std::vector<std::vector<float>> histOverlapsPercMin(n, std::vector<float>(n, 0));
    hist.side.resize(n);

    unsigned int nrBins = 0;
    if (n > 0)
    {
        nrBins = bins->at(0).size();
    }


    std::vector<unsigned int> v(n);
    std::vector<unsigned int> vUsed(0);

    std::iota(v.begin(), v.end(), 0);

    // Remove all inactive attributes from the decision.
    if (active != nullptr)
    {
        for (int i = v.size() - 1; i >= 0; --i)
        {
            if (!((*active)[i]))
            {
                v.erase(v.begin() + i);
            }
        }
    }

    for (int i = v.size() - 1; i >= 0; --i)
    {
        // If the area is too small, don't take the attribute into consideration.
        if (area->at(v[i]) < 0.000001)
        {
            vUsed.push_back(v[i]);
            v.erase(v.begin() + i);
        }
    }


    for (unsigned int i = 0; i < v.size(); ++i)
    {
        for (unsigned int j = 0; j < v.size(); ++j)
        {
            // The number of bins should be the same in order for this to work properly. However, there might be cases when the bins are not initialized completly yet,
            // so prevent the program from crashing. 
            int currNrBins = std::min(bins->at(v[i]).size(), bins->at(v[j]).size());
            for (unsigned int k = 0; k < currNrBins; ++k)
                {
                    // The overlap is the minimum of the bin size between the two bars.
                    histOverlaps[v[i]][v[j]] += std::fmin(bins->at(v[i])[k], bins->at(v[j])[k]);
                }
            
            // Divide the overlap by the total length of bars in histogram 1. 1 means the whole histogram is covered by the other.
            histOverlapsPerc[v[i]][v[j]] = histOverlaps[v[i]][v[j]]/ area->at(v[i]);

            if (v[i] >= v[j])
            {
                histOverlapsPercMin[v[i]][v[j]] = std::fmin(histOverlapsPerc[v[i]][v[j]], histOverlapsPerc[v[j]][v[i]]);
                histOverlapsPercMin[v[j]][v[i]] = std::fmin(histOverlapsPerc[v[i]][v[j]], histOverlapsPerc[v[j]][v[i]]);
            }
        }
    }



    // Now, determin which histogram has to be moved to which side.
    // Separate the most similar ones, i.e. max(min(PercA, PercB))
    while(true)
    {
        float currMax = 0;
        int currIdx1 = -1;
        int currIdx2 = -1;

        for (unsigned int i = 0; i < v.size(); ++i) {
            for (unsigned j = i + 1;j < v.size(); ++j ) {
                if (histOverlapsPercMin[v[i]][v[j]] > currMax) {
                    currMax = histOverlapsPercMin[v[i]][v[j]];
                    currIdx1 = i;
                    currIdx2 = j;
                }
            }
        }

        if ((currIdx1 == -1) || (currIdx2 == -1)) { break; }
        
        // There is a difference in the assignement for the first elements and later ones. The later ones also take into account the overlap with the existing ones, since there are 2 options to distribute them on either side.
        if (vUsed.size() == 0)
        {
            hist.side[v[currIdx1]] = 0;
            hist.side[v[currIdx2]] = 1;
        }
        else
        {
            float sideAPerc = 0;
            float sideBPerc = 0;
            // Check what insert to each side would mean in the worst case concerning percentage overlap

            // arg min_{sides} (  max(overlapA) + max(overlapB) )

            float maxOverlapIdx1Side1 = 0;
            float maxOverlapIdx2Side1 = 0;
            float maxOverlapIdx1Side2 = 0;
            float maxOverlapIdx2Side2 = 0;

            for (unsigned int i = 0; i < vUsed.size(); i += 2)
            {
                float overlap1 = histOverlapsPercMin[vUsed[i]][v[currIdx1]];
                float overlap2 = histOverlapsPercMin[vUsed[i]][v[currIdx2]];

                if (overlap1 > maxOverlapIdx1Side1) { maxOverlapIdx1Side1 = overlap1; }
                if (overlap2 > maxOverlapIdx2Side1) { maxOverlapIdx2Side1 = overlap2; }
            }
            for (unsigned int i = 1; i < vUsed.size(); i += 2)
            {
                float overlap1 = histOverlapsPercMin[vUsed[i]][v[currIdx1]];
                float overlap2 = histOverlapsPercMin[vUsed[i]][v[currIdx2]];

                if (overlap1 > maxOverlapIdx1Side2) { maxOverlapIdx1Side2 = overlap1; }
                if (overlap2 > maxOverlapIdx2Side2) { maxOverlapIdx2Side2 = overlap2; }
            }
            if (maxOverlapIdx1Side1 + maxOverlapIdx2Side2 < maxOverlapIdx1Side2 + maxOverlapIdx2Side1){ hist.side[v[currIdx1]] = 0; hist.side[v[currIdx2]] = 1;}
            else{ hist.side[v[currIdx1]] = 1; hist.side[v[currIdx2]] = 0; }
        }
        
        vUsed.push_back(v[currIdx1]);
        vUsed.push_back(v[currIdx2]);

        // Now, remove the list elements, and put the found indices on opposize sides.
        
        v.erase(v.begin() + currIdx2);
        v.erase(v.begin() + currIdx1);

        if (v.size() == 1)
        {    
            // insert the last element to the right side.
            float maxOverlapIdx1Side1 = 0;
            float maxOverlapIdx1Side2 = 0;
            for (unsigned int i = 0; i < vUsed.size(); i += 2)
            {
                float overlap1 = histOverlapsPercMin[vUsed[i]][v[0]];
                float overlap2 = 0;
                if (vUsed.size() > i + 1)
                {
                    overlap2 = histOverlapsPercMin[vUsed[i + 1]][v[0]];
                }

                if (overlap1 > maxOverlapIdx1Side1) { maxOverlapIdx1Side1 = overlap1; }
                if (overlap2 > maxOverlapIdx1Side2) { maxOverlapIdx1Side2 = overlap2; }
            }
            (maxOverlapIdx1Side1 < maxOverlapIdx1Side2) ?  hist.side[v[0]] = 0 :  hist.side[v[0]] = 1;
            vUsed.push_back(v[0]);
            break;
        }
        else if (v.size() == 0)
        {
            break;
        }
    }

    hist.attributeColorOrderIdx.clear();
    hist.attributeColorOrderIdx = vUsed;

    if (considerBlendingOrder)
    {
        std::vector<int> idxLeftSide;
        std::vector<int> idxRightSide;
        std::vector<int> activeParameters;

        for (int i = 0; i < hist.side.size(); ++i)
        {    
            if (!((*active)[i])) { continue; }
            activeParameters.push_back(i);
            if (hist.side[i] == 0)
            {
                idxLeftSide.push_back(i);

            }
            else
            {
                idxRightSide.push_back(i);
            }
        }

        std::vector<unsigned int> w(idxLeftSide.size());
        std::vector<unsigned int> z(idxRightSide.size());

        // Create a vector with increasing indices. Then sort them with decreasing area for the left side. Then, adapt attributeColorOrderIdx. 
        std::iota(w.begin(), w.end(), 0);
        std::sort(w.begin(), w.end(), [area, idxLeftSide](size_t  a, size_t  b) {return area->at(idxLeftSide[a]) > area->at(idxLeftSide[b]);});

        std::iota(z.begin(), z.end(), 0);
        std::sort(z.begin(), z.end(), [area, idxRightSide](size_t  a, size_t  b) {return area->at(idxRightSide[a]) > area->at(idxRightSide[b]);});

        int iLeft = 0;
        int iRight = 0;
        for (int i = 0; i < hist.attributeColorOrderIdx.size(); ++i)
        {
            if (hist.side[activeParameters[i]] == 0) { hist.attributeColorOrderIdx[i] = idxLeftSide[w[iLeft++]]; };

            if (hist.side[activeParameters[i]] == 1) { hist.attributeColorOrderIdx[i] = idxRightSide[z[iRight++]]; };
        }

    }


}

