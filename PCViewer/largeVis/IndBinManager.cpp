#define NOSTATICS
#include "IndBinManager.hpp"
#include "../Structures.hpp"
#undef NOSTATICS
#include "../range.hpp"
#include "CpuLineCounter.hpp"
#include "RoaringCounter.hpp"
#include "../Brushing.hpp"
#include <math.h>
#include <filesystem>

IndBinManager::IndBinManager(const CreateInfo& info) :
    compressedData(info.compressedData), 
    _vkContext(info.context), 
    _hierarchyFolder(info.hierarchyFolder), 
    columnBins(info.context.screenSize[1])
{
    // --------------------------------------------------------------------------------
    // creating the index activation buffer resource and setting up the cpu side activation vector
    // --------------------------------------------------------------------------------
    uint32_t dataSize = compressedData.dataSize;
    indexActivation = std::vector<uint8_t>((dataSize + 7) / 8, 0xff);    // set all to active on startup
    VkUtil::createBuffer(_vkContext.device, indexActivation.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &_indexActivation);
    VkMemoryRequirements memReq{};
    vkGetBufferMemoryRequirements(_vkContext.device, _indexActivation, &memReq);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(_vkContext.physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkAllocateMemory(_vkContext.device, &allocInfo, nullptr, &_indexActivationMemory);
    vkBindBufferMemory(_vkContext.device, _indexActivation, _indexActivationMemory, 0);
    VkUtil::uploadData(_vkContext.device, _indexActivationMemory, 0, indexActivation.size(), indexActivation.data());

    // --------------------------------------------------------------------------------
    // getting the handles for the counter pipelines
    // --------------------------------------------------------------------------------
    _renderLineCounter = RenderLineCounter::acquireReference(RenderLineCounter::CreateInfo{info.context});
    _lineCounter = LineCounter::acquireReference(LineCounter::CreateInfo{info.context});
    _renderer = compression::Renderer::acquireReference(compression::Renderer::CreateInfo{info.context, info.renderPass, info.framebuffer});
    _computeBrusher = ComputeBrusher::acquireReference(ComputeBrusher::CreateInfo{info.context});
}


IndBinManager::~IndBinManager(){
    // releasing the gpu pipeline singletons
    if(_renderLineCounter)
        _renderLineCounter->release();
    if(_lineCounter)
        _lineCounter->release();
    if(_renderer)
        _renderer->release();
    if(_computeBrusher)
        _computeBrusher->release();
    // count resource is destructed automatically by destructor

    if(_indexActivation)
        vkDestroyBuffer(_vkContext.device, _indexActivation, nullptr);

    if(_indexActivationMemory)
        vkFreeMemory(_vkContext.device, _indexActivationMemory, nullptr);

    // joining all threads
    if(_countUpdateThread.joinable())
        _countUpdateThread.join();
    
}

void IndBinManager::notifyAttributeUpdate(const std::vector<int>& attributeOrdering, const std::vector<Attribute>& attributes, bool* attributeActivations){
    _attributeOrdering = attributeOrdering;
    assert(attributes == compressedData.attributes); // debug check
    _attributeActivations = attributeActivations;
    updateCounts();         // updating the counts if there are some missing and pushing a render call
}

void IndBinManager::notifyPriorityCenterUpdate(uint32_t attributeIndex, float attributeValue){

}

void IndBinManager::forceCountUpdate(){
    _currentBrushState.id = _curBrushingId++;   // increasing brush id to force recounting
    updateCounts();
}

void IndBinManager::render(VkCommandBuffer commands,VkBuffer attributeInfos, bool clear){
    std::vector<int> activeIndices; // these are already ordered
    for(auto i: _attributeOrdering){
        if(_attributeActivations[i])
            activeIndices.push_back(i);
    }
    std::vector<VkBuffer> counts;
    std::vector<std::pair<uint32_t, uint32_t>> axes;
    std::vector<uint32_t> binSizes;
    for(int i: irange(activeIndices.size() - 1)){
        uint32_t a = activeIndices[i];
        uint32_t b = activeIndices[i + 1];
        if(a > b)
            std::swap(a, b);
        if(!_countResources.contains({a,b})){

        }
        counts.push_back(_countResources[{a,b}].countBuffer);
        axes.push_back({a,b});
        binSizes.push_back(static_cast<uint32_t>(std::sqrt(_countResources[{a,b}].binAmt)));
    }

    compression::Renderer::RenderInfo renderInfo{
        commands,
        "dummy",
        compression::Renderer::RenderType::Polyline,
        counts,
        axes,
        _attributeOrdering,
        compressedData.attributes,
        _attributeActivations,
        binSizes,
        attributeInfos,
        clear
    };
    
    _renderer->render(renderInfo);
    requestRender = false;
}

void IndBinManager::notifyBrushUpdate(const std::vector<RangeBrush>& rangeBrushes, const Polygons& lassoBrushes){
    _currentBrushState = {rangeBrushes, lassoBrushes, _curBrushingId++};
    updateCounts();
}

void IndBinManager::updateCounts(){
    if(countingMethod == CountingMethod::CpuRoaring)
        columnBins = binsMaxCenterAmt;

    bool prevValue{};

    std::vector<uint32_t> activeIndices; // these are already ordered
    for(auto i: _attributeOrdering){
        if(_attributeActivations[i])
            activeIndices.push_back(i);
    }
    if(activeIndices.size() < 2){
        std::cout << "Less than 2 attribute active, not updating the counts" << std::endl;
        return;
    }

    if(_countUpdateThreadActive.compare_exchange_strong(prevValue, true)){    // trying to block any further incoming notifies
        std::cout << "Starting counting pipeline for " << columnBins << " bins and " << compressedData.columnData[0].cpuData.size() << " data points." << std::endl;
        // making shure the counting images are created and have the right size
        for(int i: irange(activeIndices.size() - 1)){
            uint32_t a = activeIndices[i];
            uint32_t b = activeIndices[i + 1];
            if(a > b)
                std::swap(a, b);
            if(!_countResources[{a,b}].countBuffer || _countResources[{a,b}].binAmt != columnBins * columnBins){
                if(_countResources[{a,b}].countBuffer){
                    vkDestroyBuffer(_vkContext.device, _countResources[{a,b}].countBuffer, nullptr);
                    vkFreeMemory(_vkContext.device, _countResources[{a,b}].countMemory, nullptr);
                }
                // create resource if not yet available
                VkUtil::createBuffer(_vkContext.device, columnBins * columnBins * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &(_countResources[{a,b}].countBuffer));
                VkMemoryAllocateInfo allocInfo{};
                allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                VkMemoryRequirements memReq{};
                vkGetBufferMemoryRequirements(_vkContext.device, _countResources[{a,b}].countBuffer, &memReq);
                allocInfo.allocationSize = memReq.size;
                allocInfo.memoryTypeIndex = VkUtil::findMemoryType(_vkContext.physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
                vkAllocateMemory(_vkContext.device, &allocInfo, nullptr, &_countResources[{a,b}].countMemory);
                vkBindBufferMemory(_vkContext.device, _countResources[{a,b}].countBuffer, _countResources[{a,b}].countMemory, 0);
                _countResources[{a,b}].binAmt = columnBins * columnBins;
                _countResources[{a,b}]._vkDevice = _vkContext.device;
            }
        }
        _countBrushState = _currentBrushState;
        if(_countUpdateThread.joinable())
            _countUpdateThread.join();
        _countUpdateThread = std::thread(execCountUpdate, this, activeIndices);
    }
}

void IndBinManager::execCountUpdate(IndBinManager* t, std::vector<uint32_t> activeIndices){
    PCUtil::Stopwatch totalTime(std::cout, "Total Count update Time");
    // starting with updating the counts if needed to have all information available for the following counting/reduction
    // note: might be changed to be settable by the user if cpu or gpu should be used for counting
    bool gpuDecompression = t->countingMethod > CountingMethod::CpuRoaring && t->compressedData.columnData[0].compressedRLHuffGpu.size();
    long blockSize = gpuDecompression ? t->compressedData.compressedBlockSize: t->compressedData.dataSize;
    if(!t->_gpuDecompressForward)
        blockSize = -blockSize;
    long startOffset = t->_gpuDecompressForward ? 0: t->compressedData.columnData[0].compressedRLHuffGpu.size() - 1 * t->compressedData.compressedBlockSize;
    VkEvent curEvent{};
    std::set<uint32_t> neededIndices(activeIndices.begin(), activeIndices.end()); // getting all needed indices, not just the visible once, but also all brushed ones
    for(auto& b: t->_countBrushState.rangeBrushes){
        for(auto& r: b)
            neededIndices.insert(r.axis);
    }
    for(auto& b: t->_countBrushState.lassoBrushes){
        neededIndices.insert(b.attr1);
        neededIndices.insert(b.attr2);
    }
        
    for(size_t dataOffset = startOffset; dataOffset < t->compressedData.dataSize && dataOffset >= 0; dataOffset += blockSize){
        // if compressed data first decompressing
        if(gpuDecompression)
        {
            uint32_t blockIndex = dataOffset / std::abs(blockSize);

            DecompressManager::CpuColumns cpuColumns(t->compressedData.attributes.size(), {});
            DecompressManager::GpuColumns gpuColumns(cpuColumns.size(), {});
            for(int i: neededIndices){
                // checking if column is needed and adding to decompression if so
                if(std::count(activeIndices.begin(), activeIndices.end(), i)){
                    cpuColumns[i] = &t->compressedData.columnData[i].compressedRLHuffCpu[blockIndex];
                    gpuColumns[i] = &t->compressedData.columnData[i].compressedRLHuffGpu[blockIndex];
                }
            }

            curEvent = t->compressedData.decompressManager->executeBlockDecompression(blockSize, *t->compressedData.gpuInstance, cpuColumns, gpuColumns, t->compressedData.quantizationStep, curEvent);
        }
        
        if(t->countingMethod <= CountingMethod::CpuRoaring){
            // updating cpu activations
            if(t->_indexActivationState != t->_countBrushState.id){
                std::cout << "Updating cpu index activations" << std::endl; std::cout.flush();
                PCUtil::Stopwatch updateWatch(std::cout, "Cpu index activation");
                t->_indexActivationState = t->_countBrushState.id;
                std::vector<const std::vector<half>*> data(t->compressedData.attributes.size());
                const uint32_t amtOfThreads = 12;
                auto execClear = [&](size_t start, size_t end){
                    for(size_t e: irange(start, end)) 
                        t->indexActivation[e] = 0;
                };
                // having to set all activations to 0 as the activation calculation currently uses or
                if(amtOfThreads == 1){
                    execClear(0, t->indexActivation.size());
                }
                else{
                    std::vector<std::thread> threads(amtOfThreads);
                    size_t curStart = 0;
                    size_t size = t->indexActivation.size();
                    for(int i: irange(amtOfThreads)){
                        size_t curEnd = size_t(i + 1) * size / amtOfThreads;
                        threads[i] = std::thread(execClear, curStart, curEnd);
                        curStart = curEnd;
                    }
                    for(auto& t: threads)
                        t.join();
                }

                for(int i: irange(t->compressedData.columnData)){
                    data[i] = &t->compressedData.columnData[i].cpuData;
                }

                brushing::updateIndexActivation(t->_countBrushState.rangeBrushes, t->_countBrushState.lassoBrushes, data, t->indexActivation, 1);
            }
        }
        else{
            // updating gpu activations
            if(t->_gpuIndexActivationState != t->_countBrushState.id){
                std::cout << "Updating gpu index activations" << std::endl; std::cout.flush();
                PCUtil::Stopwatch updateWatch(std::cout, "Gpu index activation");
                t->_gpuIndexActivationState = t->_countBrushState.id;
                
                if(gpuDecompression){
                    auto &dataBuffer = t->compressedData.decompressManager->buffers;
                    curEvent = t->_computeBrusher->updateActiveIndices(t->compressedData.columnData[0].cpuData.size(), t->_countBrushState.rangeBrushes, t->_countBrushState.lassoBrushes, dataBuffer, t->_indexActivation, false, curEvent);
                }
                else{
                    std::vector<VkBuffer> dataBuffer(t->compressedData.attributes.size());
                    for(int i: irange(dataBuffer))
                        dataBuffer[i] = t->compressedData.columnData[i].gpuHalfData;
                    t->_computeBrusher->updateActiveIndices(t->compressedData.columnData[0].cpuData.size(), t->_countBrushState.rangeBrushes, t->_countBrushState.lassoBrushes, dataBuffer, t->_indexActivation);
                }
            }
        }

        // Note: vulkan resources for the count images were already provided by main thread
        switch(t->countingMethod){
        case CountingMethod::CpuGeneric:{
            for(int i: irange(activeIndices.size() - 1)){
                uint32_t a = activeIndices[i];
                uint32_t b = activeIndices[i + 1];
                if(a > b)
                    std::swap(a, b);
                if(t->_countResources.contains({a,b}) && t->_countResources[{a,b}].brushingId == t->_countBrushState.id)
                    continue;
                std::cout << "Counting pairwise (cpu generic) for attribute " << t->compressedData.attributes[a].name << " and " << t->compressedData.attributes[b].name << std::endl;
                auto counts = compression::lineCounterPair(t->compressedData.columnData[a].cpuData, t->compressedData.columnData[b].cpuData, t->columnBins, t->columnBins, t->indexActivation, t->cpuLineCountingAmtOfThreads);
                VkUtil::uploadData(t->_vkContext.device, t->_countResources[{a,b}].countMemory, 0, t->_countResources[{a,b}].binAmt * sizeof(uint32_t), counts.data());
                t->_countResources[{a,b}].brushingId = t->_countBrushState.id;
            }
            break;
        }
        case CountingMethod::CpuMinGeneric:{
            if(t->priorityDistances.empty()){
                // filling with radom numbers to have something interesting to min about
                t->priorityDistances.resize(t->compressedData.columnData[0].cpuData.size());
                for(uint32_t i: irange(t->priorityDistances))
                    t->priorityDistances[i] = double(std::rand()) / RAND_MAX;   // random value in [0, 1);
            }
            for(int i: irange(activeIndices.size() - 1)){
                uint32_t a = activeIndices[i];
                uint32_t b = activeIndices[i + 1];
                if(a > b)
                    std::swap(a, b);
                if(t->_countResources.contains({a,b}) && t->_countResources[{a,b}].brushingId == t->_countBrushState.id)
                    continue;
                std::cout << "ReducingMin distance pairwise (cpu generic) for attribute " << t->compressedData.attributes[a].name << " and " << t->compressedData.attributes[b].name << std::endl;
                auto minDist = compression::lineMinPair(t->compressedData.columnData[a].cpuData, t->compressedData.columnData[b].cpuData, t->columnBins, t->columnBins, t->indexActivation, t->priorityDistances, t->cpuLineCountingAmtOfThreads);
                std::vector<uint32_t> convertedDist(minDist.size());
                for(uint32_t d: irange(minDist))
                    convertedDist[d] = float(minDist[d]) * 255;    // multiplied with 255 as there are only 255 different color values
                VkUtil::uploadData(t->_vkContext.device, t->_countResources[{a,b}].countMemory, 0, t->_countResources[{a,b}].binAmt * sizeof(uint32_t), convertedDist.data());
                t->_countResources[{a,b}].brushingId = t->_countBrushState.id;
            }
            break;
        }
        case CountingMethod::CpuGenericSingleField:{
            for(int i: irange(activeIndices.size() - 1)){
                uint32_t a = activeIndices[i];
                uint32_t b = activeIndices[i + 1];
                if(a > b)
                    std::swap(a, b);
                if(t->_countResources.contains({a,b}) && t->_countResources[{a,b}].brushingId == t->_countBrushState.id)
                    continue;
                std::cout << "Counting pairwise (cpu generic) for attribute " << t->compressedData.attributes[a].name << " and " << t->compressedData.attributes[b].name << std::endl;
                auto counts = compression::lineCounterPairSingleField(t->compressedData.columnData[a].cpuData, t->compressedData.columnData[b].cpuData, t->columnBins, t->columnBins, t->indexActivation, t->cpuLineCountingAmtOfThreads);
                VkUtil::uploadData(t->_vkContext.device, t->_countResources[{a,b}].countMemory, 0, t->_countResources[{a,b}].binAmt * sizeof(uint32_t), counts.data());
                t->_countResources[{a,b}].brushingId = t->_countBrushState.id;
            }
            break;
        }
        case CountingMethod::CpuRoaring:{
            for(int i: irange(activeIndices.size() -1)){
                std::cout << "Cpu roaring currently disabled" << std::endl;
                break;
                uint32_t a = activeIndices[i];
                uint32_t b = activeIndices[i + 1];
                if(a > b)
                    std::swap(a, b);
                if(t->_countResources.contains({a,b}) && t->_countResources[{a,b}].brushingId == t->_countBrushState.id)
                    continue;
                std::vector<uint32_t> aIndexBins, bIndexBins;
                for(int bi: irange(t->_attributeCenters[a])){
                    uint32_t bin = t->_attributeCenters[a][bi].val * t->binsMaxCenterAmt;
                    if(bin >= t->binsMaxCenterAmt)
                        --bin;
                    aIndexBins.push_back(bin);
                }
                for(int bi: irange(t->_attributeCenters[b])){
                    uint32_t bin = t->_attributeCenters[b][bi].val * t->binsMaxCenterAmt;
                    if(bin >= t->binsMaxCenterAmt)
                        --bin;
                    bIndexBins.push_back(bin);
                }
                std::cout << "Counting pairwise (roaring) for attribute " << t->compressedData.attributes[a].name << " and " << t->compressedData.attributes[b].name << std::endl;
                //auto counts = compression::lineCounterRoaring(t->binsMaxCenterAmt, t->ndBuckets[{a}], aIndexBins, t->ndBuckets[{b}], bIndexBins, t->columnBins, t->columnBins, t->cpuLineCountingAmtOfThreads);
                //VkUtil::uploadData(t->_vkContext.device, t->_countResources[{a,b}].countMemory, 0, t->_countResources[{a,b}].binAmt * sizeof(uint32_t), counts.data());
                //t->_countResources[{a,b}].brushingId = t->_countBrushState.id;
            }
            break;
        }
        case CountingMethod::GpuComputeFull:{
            std::vector<VkBuffer> datas(activeIndices.size()), 
                                    counts(activeIndices.size() - 1);
            bool anyUpdate = false;
            uint32_t b{};
            for(int i: irange(activeIndices.size() - 1)){
                uint32_t a = activeIndices[i];
                b = activeIndices[i + 1];
                if(a > b)
                    std::swap(a, b);
                if(t->_countResources.contains({a,b}) && t->_countResources[{a,b}].brushingId != t->_countBrushState.id)
                    anyUpdate = true;
                datas[i] = t->compressedData.columnData[i].gpuHalfData;
                counts[i] = t->_countResources[{a,b}].countBuffer;
            }
            datas.back() = t->compressedData.columnData[activeIndices.back()].gpuHalfData;
            if(!anyUpdate)
                goto finish;
            t->_lineCounter->countLinesAll(t->compressedData.columnData[0].cpuData.size(), datas, t->columnBins, counts, activeIndices, t->_indexActivation, true);
            break;
        }
        case CountingMethod::GpuComputePairwise:{
            for(int i: irange(activeIndices.size() -1)){
                uint32_t a = activeIndices[i];
                uint32_t b = activeIndices[i + 1];
                if(a > b)
                    std::swap(a, b);
                if(t->_countResources.contains({a,b}) && t->_countResources[{a,b}].brushingId == t->_countBrushState.id)
                    continue;
                std::cout << "Counting pairwise (compute pipeline) for attribute " << t->compressedData.attributes[a].name << " and " << t->compressedData.attributes[b].name << std::endl;
                t->_lineCounter->countLinesPair(t->compressedData.columnData[a].cpuData.size(), t->compressedData.columnData[a].gpuHalfData, t->compressedData.columnData[b].gpuHalfData, t->columnBins, t->columnBins, t->_countResources[{a,b}].countBuffer, t->_indexActivation, true);
                t->_countResources[{a,b}].brushingId = t->_countBrushState.id;
            }
            break;
        }
        case CountingMethod::GpuDrawPairwise:{
            for(int i: irange(activeIndices.size() -1)){
                uint32_t a = activeIndices[i];
                uint32_t b = activeIndices[i + 1];
                if(a > b)
                    std::swap(a, b);
                if(t->_countResources.contains({a,b}) && t->_countResources[{a,b}].brushingId == t->_countBrushState.id)
                    continue;
                std::cout << "Counting pairwise (render pipeline) for attribute " << t->compressedData.attributes[a].name << " and " << t->compressedData.attributes[b].name << std::endl;
                t->_renderLineCounter->countLinesPair(t->compressedData.columnData[a].cpuData.size(), t->compressedData.columnData[a].gpuHalfData, t->compressedData.columnData[b].gpuHalfData, t->columnBins, t->columnBins, t->_countResources[{a,b}].countBuffer, t->_indexActivation, true);
                t->_countResources[{a,b}].brushingId = t->_countBrushState.id;
            }
            break;
        }
        case CountingMethod::GpuDrawPairwiseTiled:{
            for(int i: irange(activeIndices.size() -1)){
                uint32_t a = activeIndices[i];
                uint32_t b = activeIndices[i + 1];
                if(a > b)
                    std::swap(a, b);
                if(t->_countResources.contains({a,b}) && t->_countResources[{a,b}].brushingId == t->_countBrushState.id)
                    continue;
                const uint32_t tileAmt = 6;
                std::cout << "Counting pairwise tiled (render pipeline) for attribute " << t->compressedData.attributes[a].name << " and " << t->compressedData.attributes[b].name << " with " << tileAmt * tileAmt << " tiles" << std::endl;
                t->_renderLineCounter->countLinesPairTiled(t->compressedData.columnData[a].cpuData.size(), t->compressedData.columnData[a].gpuHalfData, t->compressedData.columnData[b].gpuHalfData, t->columnBins, t->columnBins, t->_countResources[{a,b}].countBuffer, true, tileAmt);
                t->_countResources[{a,b}].brushingId = t->_countBrushState.id;
            }
            break;
        }
        case CountingMethod::GpuDrawMultiViewport:{
            // ToDo
            break;
        }
        case CountingMethod::HybridRoaringGpuDraw:{
            break;
        }
        };
    }
    finish:
    std::cout.flush();
    t->requestRender = true;                   // ready to update rendering
    t->_countUpdateThreadActive = false;        // releasing the update thread
}
