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
    size_t dataSize = compressedData.uploadManager ? compressedData.uploadManager->stagingBufferSize / sizeof(half): compressedData.dataSize;
    indexActivation = std::vector<uint8_t>(PCUtil::alignedSize((dataSize + 7) / 8, 0x40), 0xff);    // set all to active on startup
    auto [b, o, m] = VkUtil::createMultiBufferBound(info.context, {indexActivation.size()}, {VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT}, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    _indexActivation = b[0];
    _indexActivationMemory = m;
    VkUtil::uploadDataIndirect(_vkContext, _indexActivation, indexActivation.size(), indexActivation.data());

    uint32_t amtOfBlocks = std::max<uint32_t>(compressedData.columnData[0].compressedRLHuffCpu.size(), 1u);
    uint32_t timingAmt = compressedData.columnData.size() * amtOfBlocks * 2;    // timing points for counting
    timingAmt += amtOfBlocks * 2 * 2;   // 2 timing points for decompression, 2 for index activation multiplied by amt Of blocks as for each block this has to be executed
    VkQueryPoolCreateInfo createInfo{}; createInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    createInfo.queryCount = timingAmt;
    _timingAmt = timingAmt;
    createInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    vkCreateQueryPool(_vkContext.device, &createInfo, nullptr, &_timingPool);

    // --------------------------------------------------------------------------------
    // getting the handles for the counter pipelines
    // --------------------------------------------------------------------------------
    _renderLineCounter = RenderLineCounter::acquireReference(RenderLineCounter::CreateInfo{info.context});
    _lineCounter = LineCounter::acquireReference(LineCounter::CreateInfo{info.context});
    _renderer = compression::Renderer::acquireReference(compression::Renderer::CreateInfo{info.context, info.renderPass, info.sampleCount, info.framebuffer, info.heatmapView, info.heatmapSampler});
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

    if(_timingPool)
        vkDestroyQueryPool(_vkContext.device, _timingPool, nullptr);

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

void IndBinManager::render(VkCommandBuffer commands,VkBuffer attributeInfos, compression::Renderer::RenderType renderType, bool clear){
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

    std::stringstream ss;
    ss << attributeInfos;
    //std::cout << ss.str() << std::endl;

    compression::Renderer::RenderInfo renderInfo{
        commands,
        ss.str(),
        renderType,
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
        std::cout << "Starting counting pipeline for " << columnBins << " bins and " << compressedData.dataSize << " data points with counting type: " << countingMethodNames[static_cast<int>(countingMethod)] << std::endl;
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
                VkUtil::createBuffer(_vkContext.device, columnBins * columnBins * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &(_countResources[{a,b}].countBuffer));
                VkMemoryAllocateInfo allocInfo{};
                allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                VkMemoryRequirements memReq{};
                vkGetBufferMemoryRequirements(_vkContext.device, _countResources[{a,b}].countBuffer, &memReq);
                allocInfo.allocationSize = memReq.size;
                allocInfo.memoryTypeIndex = VkUtil::findMemoryType(_vkContext.physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
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
    static std::mutex countMutex;   // only used currently to avoid clashing
    std::scoped_lock updateLock(countMutex);

    PCUtil::Stopwatch totalTime(std::cout, "Total Count update Time");
    // starting with updating the counts if needed to have all information available for the following counting/reduction
    // note: might be changed to be settable by the user if cpu or gpu should be used for counting
    bool gpuDecompression = t->countingMethod > CountingMethod::CpuRoaring && t->compressedData.columnData[0].compressedRLHuffGpu.size();
    bool streamGpuData = t->countingMethod > CountingMethod::CpuRoaring && t->compressedData.uploadManager;
    bool combinedActivationCounting = t->countingMethod >= CountingMethod::GpuComputeFullBrush && t->countingMethod <= CountingMethod::GpuComputeFullBrushPartitionedNoAtomics;
    long blockSize = gpuDecompression ? t->compressedData.compressedBlockSize: streamGpuData ? t->compressedData.uploadManager->stagingBufferSize / sizeof(half): t->compressedData.dataSize;
    uint32_t amtOfBlocks = (t->compressedData.dataSize + blockSize - 1) / blockSize;
    long startOffset = t->_gpuDecompressForward ? 0: (amtOfBlocks - 1) * blockSize;
    if(!t->_gpuDecompressForward)
        blockSize = -blockSize;
    //startOffset = 2 * 264305696;
    VkSemaphore curSemaphore{};
    std::set<uint32_t> neededIndices(activeIndices.begin(), activeIndices.end()); // getting all needed indices, not just the visible once, but also all brushed ones
    for(auto& b: t->_countBrushState.rangeBrushes){
        for(auto& r: b)
            neededIndices.insert(r.axis);
    }
    for(auto& b: t->_countBrushState.lassoBrushes){
        neededIndices.insert(b.attr1);
        neededIndices.insert(b.attr2);
    }

    std::vector<float> timings(2 + t->compressedData.attributes.size(), {});
    VkQueryPool timingPool{};
    if(t->printDeocmpressionTimes && (t->countingMethod == CountingMethod::GpuDrawPairwise || (t->countingMethod >= CountingMethod::GpuComputeFull && t->countingMethod < CountingMethod::HybridRoaringGpuDraw )))
        timingPool = t->_timingPool;
    uint32_t uploadTimingCount{};
    float uploadTimingAverage{};
    uint32_t iteration{};
    for(size_t dataOffset = startOffset; dataOffset < t->compressedData.dataSize && dataOffset >= 0; dataOffset += blockSize, ++iteration){
        uint32_t timingIndex{};
        assert((dataOffset & 31) == 0); //check that dataOffset ist 32 aligned (needed for index activation)
        uint32_t curDataBlockSize = gpuDecompression ? std::min<uint32_t>(t->compressedData.dataSize - dataOffset, t->compressedData.compressedBlockSize): streamGpuData ? std::min<uint32_t>(t->compressedData.dataSize - dataOffset, std::abs(blockSize)): t->compressedData.dataSize;
        std::cout << "Current data offset: " << dataOffset << ", with block size: " << curDataBlockSize <<  std::endl; std::cout.flush();
        //if(curDataBlockSize != blockSize)
        //   break;
        // if compressed data first decompressing
        if(gpuDecompression)
        {
            uint32_t blockIndex = dataOffset / std::abs(blockSize);
            assert(curDataBlockSize <= t->compressedData.columnData[0].compressedSymbolSize[blockIndex]);

            DecompressManager::CpuColumns cpuColumns(t->compressedData.attributes.size(), {});
            DecompressManager::GpuColumns gpuColumns(cpuColumns.size(), {});
            for(int i: neededIndices){
                // checking if column is needed and adding to decompression if so
                if(std::count(activeIndices.begin(), activeIndices.end(), i)){
                    cpuColumns[i] = &t->compressedData.columnData[i].compressedRLHuffCpu[blockIndex];
                    gpuColumns[i] = &t->compressedData.columnData[i].compressedRLHuffGpu[blockIndex];
                }
            }

            //for(int i: irange(cpuColumns)){
            //    if(cpuColumns[i]){
            //        DecompressManager::CpuColumns cpuColumnsSingle(cpuColumns.size(), {});
            //        DecompressManager::GpuColumns gpuColumnsSingle(gpuColumns.size(), {});
            //        cpuColumnsSingle[i] = cpuColumns[i];
            //        gpuColumnsSingle[i] = gpuColumns[i];
            //        t->compressedData.decompressManager->executeBlockDecompression(t->compressedData.columnData[0].compressedSymbolSize[blockIndex], *t->compressedData.gpuInstance, cpuColumnsSingle, gpuColumnsSingle, t->compressedData.quantizationStep, curEvent);
            //        std::scoped_lock<std::mutex> lock(*t->compressedData.gpuInstance->vkContext.queueMutex);
            //        auto err = vkQueueWaitIdle(t->compressedData.gpuInstance->vkContext.queue); check_vk_result(err);
            //    }
            //}
            curSemaphore = t->compressedData.decompressManager->executeBlockDecompression(t->compressedData.columnData[0].compressedSymbolSize[blockIndex], *t->compressedData.gpuInstance, cpuColumns, gpuColumns, t->compressedData.quantizationStep, curSemaphore, {timingPool, timingIndex++, timingIndex++});
            //auto err = vkQueueWaitIdle(t->compressedData.gpuInstance->vkContext.queue); check_vk_result(err);
        }
        else if(streamGpuData){
            // uploading all needed data indices via upload manager
            //if(curSemaphore){
            //    uint64_t values;
            //    VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, {}, 0, 1, &curSemaphore, &values};
            //    check_vk_result(vkWaitSemaphores(t->_vkContext.device, &info, 5e9));
            //}

            PCUtil::AverageWatch upload(uploadTimingAverage, uploadTimingCount);
            bool lastIter = iteration == amtOfBlocks - 1;
            if(lastIter)
                std::cout << "Last iteration, clearing drawlistcounts in the upload manager" << std::endl;
            std::vector<VkBuffer> uploadBuffers(neededIndices.size());
            std::vector<const void*> uploadDatas(neededIndices.size());
            std::vector<int> neededInd(neededIndices.begin(), neededIndices.end());
            for(int i: irange(neededInd)){
                uploadBuffers[i] = t->compressedData.columnData[neededInd[i]].gpuHalfData;
                uploadDatas[i] = reinterpret_cast<const void*>(t->compressedData.columnData[neededInd[i]].cpuData.data() + dataOffset);
            }
            t->compressedData.uploadManager->uploadTaskMulti(uploadDatas, curDataBlockSize * sizeof(half), uploadBuffers, lastIter);
            // kept in for reference -----------------------------------------------------
            //VkFence f;
            //for(int i: neededIndices){
            //    assert((t->compressedData.columnData[i].cpuData.data() + dataOffset)[0] == t->compressedData.columnData[i].cpuData[dataOffset]);
            //    //std::cout << "[task] " <<  reinterpret_cast<const void*>(t->compressedData.columnData[i].cpuData.data()) << "    " << reinterpret_cast<const void*>(t->compressedData.columnData[i].cpuData.data() + dataOffset) << std::endl; std::cout.flush();
            //    f = t->compressedData.uploadManager->uploadTask(reinterpret_cast<const void*>(t->compressedData.columnData[i].cpuData.data() + dataOffset), curDataBlockSize * sizeof(half), t->compressedData.columnData[i].gpuHalfData);
            //}
            //// having to wait until upload task is in idle
            //while(!t->compressedData.uploadManager->idle())
            //    std::this_thread::sleep_for(std::chrono::milliseconds(1));
            //t->compressedData.uploadManager->queueWaitIdle();
        }

        size_t indexOffset = streamGpuData ? 0: dataOffset;
        std::vector<VkBuffer> dataBuffer(t->compressedData.attributes.size());  // vector of the gpu buffer which will contian the column data.
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
            if(!combinedActivationCounting && (gpuDecompression || streamGpuData || t->_gpuIndexActivationState != t->_countBrushState.id)){
                //std::cout << "Updating gpu index activations" << std::endl; std::cout.flush();
                //PCUtil::Stopwatch updateWatch(std::cout, "Gpu index activation");
                t->_gpuIndexActivationState = t->_countBrushState.id;
                
                if(gpuDecompression){
                    dataBuffer = t->compressedData.decompressManager->buffers;
                }
                else{
                    for(int i: irange(dataBuffer))
                        dataBuffer[i] = t->compressedData.columnData[i].gpuHalfData;
                }
                curSemaphore = t->_computeBrusher->updateActiveIndices(curDataBlockSize, t->_countBrushState.rangeBrushes, t->_countBrushState.lassoBrushes, dataBuffer, t->_indexActivation, indexOffset, true, curSemaphore, {timingPool, timingIndex++, timingIndex++});
            }
            else{
                for(int i: irange(dataBuffer))
                    dataBuffer[i] = t->compressedData.columnData[i].gpuHalfData;
            }
        }
        assert(t->countingMethod <= CountingMethod::CpuRoaring || dataBuffer[0]); //chedck for valid gpu buffer

        // Note: vulkan resources for the count images were already provided by main thread
        bool firstIter = dataOffset == startOffset;
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
                VkUtil::uploadDataIndirect(t->_vkContext, t->_countResources[{a,b}].countBuffer, t->_countResources[{a,b}].binAmt * sizeof(uint32_t), counts.data());
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
        case CountingMethod::GpuComputeSubgroupPairwise:
        case CountingMethod::GpuComputeSubgroupPartitionedPairwise:
        case CountingMethod::GpuComputePairwise:{
            for(int i: irange(activeIndices.size() -1)){
                uint32_t a = activeIndices[i];
                uint32_t b = activeIndices[i + 1];
                if(a > b)
                    std::swap(a, b);
                if(t->_countResources.contains({a,b}) && t->_countResources[{a,b}].brushingId == t->_countBrushState.id)
                    continue;
                std::cout << "Counting pairwise for attribute" << t->compressedData.attributes[a].name << " and " << t->compressedData.attributes[b].name << std::endl;
                LineCounter::ReductionTypes reductionType{};
                switch(t->countingMethod){
                    case CountingMethod::GpuComputeSubgroupPairwise: reductionType = LineCounter::ReductionSubgroupAllAdd; break;
                    case CountingMethod::GpuComputeSubgroupPartitionedPairwise: reductionType = LineCounter::ReductionSubgroupAdd; break;
                    case CountingMethod::GpuComputePairwise: reductionType = LineCounter::ReductionAdd; break;
                }
                t->_lineCounter->countLinesPair(curDataBlockSize, t->compressedData.columnData[a].gpuHalfData, t->compressedData.columnData[b].gpuHalfData, t->columnBins, t->columnBins, t->_countResources[{a,b}].countBuffer, t->_indexActivation, firstIter, reductionType);
                t->_countResources[{a,b}].brushingId = t->_countBrushState.id;
            }
            break;
        }
        case CountingMethod::GpuComputeFullSubgroup:
        case CountingMethod::GpuComputeFullPartitioned:
        case CountingMethod::GpuComputeFullMax:
        case CountingMethod::GpuComputeFullPartitionGeneric:
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
                datas[i] = dataBuffer[i];//t->compressedData.columnData[i].gpuHalfData;
                counts[i] = t->_countResources[{a,b}].countBuffer;
                t->_countResources[{a,b}].brushingId = t->_countBrushState.id;
            }
            datas.back() = dataBuffer[activeIndices.back()];//t->compressedData.columnData[activeIndices.back()].gpuHalfData;
            if(!anyUpdate && !gpuDecompression && !streamGpuData)
                continue;
            LineCounter::ReductionTypes reductionType{};
            switch(t->countingMethod){
                case CountingMethod::GpuComputeFullSubgroup:        reductionType = LineCounter::ReductionSubgroupAllAdd; break;
                case CountingMethod::GpuComputeFullPartitioned:     reductionType = LineCounter::ReductionSubgroupAdd; break;
                case CountingMethod::GpuComputeFull:                reductionType = LineCounter::ReductionAdd; break;
                case CountingMethod::GpuComputeFullMax:             reductionType = LineCounter::ReductionSubgroupMax; break;
                case CountingMethod::GpuComputeFullPartitionGeneric:reductionType = LineCounter::ReductionAddPartitionGeneric; break;
            }
            curSemaphore = t->_lineCounter->countLinesAll(curDataBlockSize, datas, t->columnBins, counts, activeIndices, t->_indexActivation, indexOffset, firstIter, reductionType, curSemaphore, {timingPool, timingIndex++, timingIndex++}, t->priorityInfo);
            break;
        }
        case CountingMethod::GpuComputeFullBrush:
        case CountingMethod::GpuComputeFullBrushPartitioned:
        case CountingMethod::GpuComputeFullBrushNoAtomics:
        case CountingMethod::GpuComputeFullBrushPartitionedNoAtomics:{
            std::vector<VkBuffer> counts(activeIndices.size() - 1);
            bool anyUpdate = false;
            uint32_t b{};
            for(int i: irange(activeIndices.size() - 1)){
                uint32_t a = activeIndices[i];
                b = activeIndices[i + 1];
                if(a > b)
                    std::swap(a, b);
                if(t->_countResources.contains({a,b}) && t->_countResources[{a,b}].brushingId != t->_countBrushState.id)
                    anyUpdate = true;
                counts[i] = t->_countResources[{a,b}].countBuffer;
                t->_countResources[{a,b}].brushingId = t->_countBrushState.id;
            }
            if(!anyUpdate && !gpuDecompression && !streamGpuData)
                continue;

            LineCounter::ReductionTypes reductionType{};
            switch(t->countingMethod){
                case CountingMethod::GpuComputeFullBrush:                       reductionType = LineCounter::ReductionAdd; break;
                case CountingMethod::GpuComputeFullBrushPartitioned:            reductionType = LineCounter::ReductionSubgroupAdd; break;
                case CountingMethod::GpuComputeFullBrushNoAtomics:              reductionType = LineCounter::ReductionAddNonAtomic; break;
                case CountingMethod::GpuComputeFullBrushPartitionedNoAtomics:   reductionType = LineCounter::ReductionAddPartitionNonAtomic; break;
            }
            curSemaphore = t->_lineCounter->countBrushLinesAll(curDataBlockSize, dataBuffer, t->columnBins, counts, activeIndices, t->_countBrushState.rangeBrushes, t->_countBrushState.lassoBrushes, true, firstIter, reductionType, curSemaphore, {timingPool, timingIndex++, timingIndex++});
            break;
        }
        case CountingMethod::GpuDrawPairwise:{
            for(int i: irange(activeIndices.size() -1)){
                uint32_t a = activeIndices[i];
                uint32_t b = activeIndices[i + 1];
                if(a > b)
                    std::swap(a, b);
                if(!gpuDecompression && t->_countResources.contains({a,b}) && t->_countResources[{a,b}].brushingId == t->_countBrushState.id)
                    continue;
                //std::cout << "Counting pairwise (render pipeline) for attribute " << t->compressedData.attributes[a].name << " and " << t->compressedData.attributes[b].name << std::endl; std::cout.flush();
                curSemaphore = t->_renderLineCounter->countLinesPair(curDataBlockSize, dataBuffer[a], dataBuffer[b], t->columnBins, t->columnBins, t->_countResources[{a,b}].countBuffer, t->_indexActivation, indexOffset, firstIter, curSemaphore, {timingPool, timingIndex++, timingIndex++});
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
                t->_renderLineCounter->countLinesPairTiled(curDataBlockSize, t->compressedData.columnData[a].gpuHalfData, t->compressedData.columnData[b].gpuHalfData, t->columnBins, t->columnBins, t->_countResources[{a,b}].countBuffer, firstIter, tileAmt);
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
        //break;
        std::vector<uint32_t> timeCounts(timingIndex);
        assert(timingIndex == timeCounts.size());
        if(timingPool){
            vkGetQueryPoolResults(t->_vkContext.device, t->_timingPool, 0, timeCounts.size(), timeCounts.size() * sizeof(timeCounts[0]), timeCounts.data(), sizeof(uint32_t), VK_QUERY_RESULT_WAIT_BIT);
            for(int i: irange(timeCounts.size() / 2)){
                //float a = 1 / float(iteration + 1);
                //timings[i] = (1.f - a) * timings[i] + a * (timeCounts[2 * i + 1] - timeCounts[2 * i]) * 1e-6;
                timings[i] += (timeCounts[2 * i + 1] - timeCounts[2 * i]) * 1e-6;
            }
        }

        {
            std::scoped_lock lock(*t->_vkContext.queueMutex);
            check_vk_result(vkQueueWaitIdle(t->_vkContext.queue));
        }
    }
    // wait for curSemaphore
    if(false && curSemaphore){
        uint64_t val{};
        VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, {}, 0, 1, &curSemaphore, &val};
        check_vk_result(vkWaitSemaphores(t->_vkContext.device, &info, 5e9));
    }

    // printing single pipeline timings
    if(timingPool){
        uint32_t timingIndex{};
        const uint printWidth = 30;
        if(gpuDecompression)
            std::cout << "[timing]" << std::setw(printWidth) << "Decompression: " << timings[timingIndex++] << " ms" << std::endl;
        if(uploadTimingCount)
            std::cout << "[timing]" << std::setw(printWidth) << "Data upload: " << uploadTimingCount * uploadTimingAverage << " ms" << std::endl;
        if(!combinedActivationCounting)
            std::cout << "[timing]" << std::setw(printWidth) << "Index activaiton: " << timings[timingIndex++] << " ms" << std::endl;
        if(t->countingMethod == CountingMethod::GpuDrawPairwise){
            for(int i: irange(timingIndex, timings.size())){
                if(i > t->compressedData.attributes.size())
                    break;
                std::cout << "[timing]" << std::setw(printWidth) << t->compressedData.attributes[i - timingIndex].name << ": " << timings[i] << " ms" << std::endl;
            }
        }
        else{
            std::cout << "[timing]" << std::setw(printWidth) << "Compute count all: " << timings[timingIndex++] << " ms" << std::endl;
        }
    }

    if(gpuDecompression || streamGpuData)
        t->_gpuDecompressForward ^= true;   // switch forward and backward for ping pong

    finish:
    std::cout.flush();
    if(t->priorityInfo.axis >= 0){
        std::vector<VkBuffer> buffers;
        std::vector<size_t> bufferSizes;
        for(int i: irange(activeIndices.size() - 1)){
            uint32_t a = activeIndices[i];
            uint32_t b = activeIndices[i + 1];
            if(a > b)
                std::swap(a, b);
            auto& res = t->_countResources[{a,b}];
            buffers.push_back(res.countBuffer);
            bufferSizes.push_back(res.binAmt);
        }
        t->_renderer->updatePriorityIndexlists({buffers, bufferSizes});
    }
    t->requestRender = true;                   // ready to update rendering
    t->_countUpdateThreadActive = false;        // releasing the update thread
}
