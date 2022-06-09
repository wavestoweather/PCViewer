#define NOSTATICS
#include "IndBinManager.hpp"
#undef NOSTATICS
#include "../range.hpp"
#include "CpuLineCounter.hpp"
#include "RoaringCounter.hpp"
#include "../Brushing.hpp"
#include <math.h>
#include <filesystem>

IndBinManager::IndBinManager(const CreateInfo& info) :
_vkContext(info.context), _hierarchyFolder(info.hierarchyFolder), columnBins(info.context.screenSize[1])
{
    // --------------------------------------------------------------------------------
    // determine kind of stored data (used to convert internally to a unified represantation for further processing)
    // --------------------------------------------------------------------------------
    std::ifstream dataInfo(_hierarchyFolder + "/data.info", std::ios::binary);
    compression::DataStorageBits dataBits;
    uint32_t dataBlockSize;
    dataInfo >> dataBits;
    dataInfo >> dataBlockSize;  // block size for compressed data
    dataInfo.close();

    // --------------------------------------------------------------------------------
    // attribute infos, cluster infos
    // --------------------------------------------------------------------------------
    std::ifstream attributeInfos(_hierarchyFolder + "/attr.info", std::ios_base::binary);
    attributeInfos >> binsMaxCenterAmt; // first line contains the maximum amt of centers/bins
    std::string a; float aMin, aMax;
    while(attributeInfos >> a >> aMin >> aMax){
        attributes.push_back({a, a, {}, {}, aMin, aMax});
    }
    attributeInfos.close();

    // --------------------------------------------------------------------------------
    // center infos
    // --------------------------------------------------------------------------------
    std::ifstream attributeCenterFile(_hierarchyFolder + "/attr.ac", std::ios_base::binary);
    std::vector<compression::ByteOffsetSize> offsetSizes(attributes.size());
    attributeCenterFile.read(reinterpret_cast<char*>(offsetSizes.data()), offsetSizes.size() * sizeof(offsetSizes[0]));
    _attributeCenters.resize(attributes.size());
    for(int i = 0; i < attributes.size(); ++i){
        assert(attributeCenterFile.tellg() == offsetSizes[i].offset);
        _attributeCenters[i].resize(offsetSizes[i].size / sizeof(_attributeCenters[0][0]));
        attributeCenterFile.read(reinterpret_cast<char*>(_attributeCenters[i].data()), offsetSizes[i].size);
    }

    // --------------------------------------------------------------------------------
    // 1d index data either compressed or not (automatic conversion if not compressed)
    // --------------------------------------------------------------------------------
    size_t dataSize = 0;
    if((dataBits & compression::DataStorageBits::RawAttributeBins) != compression::DataStorageBits::None){
        // reading index data
        std::cout << "Loading indexdata..." << std::endl;
        std::vector<std::vector<uint32_t>> attributeIndices;
        attributeIndices.resize(attributes.size());
        for(int i = 0; i < attributes.size(); ++i){
            std::ifstream indicesData(_hierarchyFolder + "/" + std::to_string(i) + ".ids", std::ios_base::binary);
            uint32_t indicesSize = _attributeCenters[i].back().offset + _attributeCenters[i].back().size;
            if(indicesSize > dataSize)
                dataSize = indicesSize;
            attributeIndices[i].resize(indicesSize);
            indicesData.read(reinterpret_cast<char*>(attributeIndices[i].data()), indicesSize * sizeof(attributeIndices[0][0]));
        }
        // compressing index data
        std::cout << "Compressing indexdata..."  << std::endl;
        for(uint32_t compInd: irange(_attributeCenters)){
            ndBuckets[{compInd}].resize(_attributeCenters[compInd].size());
            size_t indexlistSize{attributeIndices[compInd].size() * sizeof(uint32_t)}, compressedSize{};
            for(uint32_t bin: irange(ndBuckets[{compInd}])){
                ndBuckets[{compInd}][bin] = roaring::Roaring64Map(_attributeCenters[compInd][bin].size, attributeIndices[compInd].data() + _attributeCenters[compInd][bin].offset);
                ndBuckets[{compInd}][bin].runOptimize();
                ndBuckets[{compInd}][bin].shrinkToFit();
                compressedSize += ndBuckets[{compInd}][bin].getSizeInBytes();
            }
            std::cout << "Attribute " << attributes[compInd].name << ": Uncompressed Indices take " << indexlistSize / float(1 << 20) << " MByte vs " << compressedSize / float(1 << 20) << " MByte compressed." << "Compression rate 1:" << indexlistSize / float(compressedSize) << std::endl;
        }
    }
    else if((dataBits & compression::DataStorageBits::RoaringAttributeBins) != compression::DataStorageBits::None){
        // compressed indices, can be read out directly
        std::cout << "Loading compressed indexdata..." << std::endl;
        for(uint32_t i: irange(attributes)){
            std::ifstream indicesData(_hierarchyFolder + "/" + std::to_string(i) + ".ids", std::ios_base::binary);
            uint32_t indicesSize = _attributeCenters[i].back().offset + _attributeCenters[i].back().size;   // size is given in bytes
            std::vector<char> indices(indicesSize);
            indicesData.read(indices.data(), indicesSize * sizeof(indices[0]));
            // parse into roaring bitmaps
            // filling only the attribute centers that are available, the other centers are empty
            ndBuckets[{i}].resize(_attributeCenters[i].size());
            size_t curSize{};
            for(uint32_t bin: irange(_attributeCenters[i])){
                ndBuckets[{i}][bin] = roaring::Roaring64Map::readSafe(indices.data() + _attributeCenters[i][bin].offset, _attributeCenters[i][bin].size * sizeof(indices[0]));
                curSize += ndBuckets[{i}][bin].cardinality();
            }
            //std::cout << "Idex size attribute " << attributes[i].name << ": " << curSize << std::endl;
            if(curSize > dataSize)
                dataSize = curSize;
        }
    }

    // --------------------------------------------------------------------------------
    // 1d data either compressed or not (automatic conversion if not compressed, stored currently as 16bit float vec)
    // --------------------------------------------------------------------------------
    columnData.resize(attributes.size());   // for each attribute there is one column
    if((dataBits & compression::DataStorageBits::RawColumnData) != compression::DataStorageBits::None){
        // convert normalized float data automatically to half data
        std::cout << "Loading float column data" << std::endl;
        for(uint32_t i: irange(attributes)){
            std::ifstream data(_hierarchyFolder + "/" + std::to_string(i) + ".col", std::ios_base::binary);
            std::vector<float> dVec(dataSize);
            data.read(reinterpret_cast<char*>(dVec.data()), dVec.size() * sizeof(dVec[0]));
            columnData[i] = {std::vector<half>(dVec.begin(), dVec.end())};  // automatic conversion to half via range constructor
        }
    }
    else if((dataBits & compression::DataStorageBits::HalfColumnData) != compression::DataStorageBits::None){
        // directly parse
        std::cout << "Loading half column data" << std::endl;
        if(dataSize == 0){  //getting the data size from the file size
            dataSize = std::filesystem::file_size(_hierarchyFolder + "/0.col") / 2;
            std::cout << "Data size from col file: " << dataSize << std::endl;
        }
        for(uint32_t i: irange(attributes)){
            std::cout << "Loading half data for attribute " << attributes[i].name << std::endl;
            std::ifstream data(_hierarchyFolder + "/" + std::to_string(i) + ".col", std::ios_base::binary);
            auto& dVec = columnData[i].cpuData;
            dVec.resize(dataSize);
            data.read(reinterpret_cast<char*>(dVec.data()), dVec.size() * sizeof(dVec[0]));
        }
    }
    else if((dataBits & compression::DataStorageBits::CuComColumnData) != compression::DataStorageBits::None){
        // directly parse if same compression block size
        // not yet implemented
        std::cout << "Well thats wrong. Seems like there is some compression in the column data but we cant handle it." << std::endl;
    }
    // currently the uncompressed 16 bit vectors are uploaded. Has to be changed to compressed vectors
    for(auto& d: columnData){
        std::cout << "Creating vulkan buffer" << std::endl;
        // creating the vulkan resources and uploading the data to them
        VkUtil::createBuffer(_vkContext.device, d.cpuData.size() * sizeof(d.cpuData[0]), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &d.gpuData);
        VkMemoryRequirements memReq{};
        vkGetBufferMemoryRequirements(_vkContext.device, d.gpuData, &memReq);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = VkUtil::findMemoryType(_vkContext.physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        vkAllocateMemory(_vkContext.device, &allocInfo, nullptr, &d.gpuMemory);
        vkBindBufferMemory(_vkContext.device, d.gpuData, d.gpuMemory, 0);
        VkUtil::uploadData(_vkContext.device, d.gpuMemory, 0, d.cpuData.size() * sizeof(d.cpuData[0]), d.cpuData.data());
    }

    // --------------------------------------------------------------------------------
    // creating the index activation buffer resource and setting up the cpu side activation vector
    // --------------------------------------------------------------------------------
    indexActivation = std::vector<uint8_t>((dataSize + 7) / 8, 0xff);    // set all to active on startup
    VkUtil::createBuffer(_vkContext.device, dataSize / 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &_indexActivation);
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

    std::cout << "Loaded " << dataSize << " datapoints" << std::endl;
}


IndBinManager::~IndBinManager(){
    // releasing the gpu pipeline singletons
    if(_renderLineCounter)
        _renderLineCounter->release();
    if(_lineCounter)
        _lineCounter->release();
    if(_renderer)
        _renderer->release();
    // count resource is destructed automatically by destructor
    // columnd data info destruction
    for(auto& d: columnData){
        if(d.gpuData)
            vkDestroyBuffer(_vkContext.device, d.gpuData, nullptr);
        if(d.gpuMemory)
            vkFreeMemory(_vkContext.device, d.gpuMemory, nullptr);
    }

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
    assert(attributes == this->attributes); // debug check
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
        attributes,
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

    auto execCountUpdate = [](IndBinManager* t, std::vector<uint32_t> activeIndices){
        // starting with updating the counts if needed to have all information available for the following counting/reduction
        // note: might be changed to be settable by the user if cpu or gpu should be used for counting
        if(t->countingMethod <= CountingMethod::CpuRoaring){
            // updating cpu activations
            if(t->_indexActivationState != t->_countBrushState.id){
                std::cout << "Updating cpu index activations" << std::endl; std::cout.flush();
                PCUtil::Stopwatch updateWatch(std::cout, "Cpu index activation");
                t->_indexActivationState = t->_countBrushState.id;
                std::vector<std::vector<half>*> data(t->attributes.size());
                const uint32_t amtOfThreads = 12;
                auto execClear = [&](size_t start, size_t end){
                    for(size_t e: irange(start, end)) 
                        t->indexActivation[e] = 0;
                };
                // having to set all activations to 0 as the activation calculation currently uses oring
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

                for(int i: irange(t->columnData)){
                    data[i] = &t->columnData[i].cpuData;
                }
                
                brushing::updateIndexActivation(t->_countBrushState.rangeBrushes, t->_countBrushState.lassoBrushes, data, t->indexActivation, 1);
            }
        }
        else{
            // updating gpu activations
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
                std::cout << "Counting pairwise (cpu generic) for attribute " << t->attributes[a].name << " and " << t->attributes[b].name << std::endl;
                auto counts = compression::lineCounterPair(t->columnData[a].cpuData, t->columnData[b].cpuData, t->columnBins, t->columnBins, t->indexActivation, t->cpuLineCountingAmtOfThreads);
                VkUtil::uploadData(t->_vkContext.device, t->_countResources[{a,b}].countMemory, 0, t->_countResources[{a,b}].binAmt * sizeof(uint32_t), counts.data());
                t->_countResources[{a,b}].brushingId = t->_countBrushState.id;
            }
            break;
        }
        case CountingMethod::CpuMinGeneric:{
            if(t->priorityDistances.empty()){
                // filling with radom numbers to have something interesting to min about
                t->priorityDistances.resize(t->columnData[0].cpuData.size());
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
                std::cout << "ReducingMin distance pairwise (cpu generic) for attribute " << t->attributes[a].name << " and " << t->attributes[b].name << std::endl;
                auto minDist = compression::lineMinPair(t->columnData[a].cpuData, t->columnData[b].cpuData, t->columnBins, t->columnBins, t->indexActivation, t->priorityDistances, t->cpuLineCountingAmtOfThreads);
                std::vector<uint32_t> convertedDist(minDist.size());
                for(uint32_t d: irange(minDist))
                    convertedDist[d] = minDist[d] * 255;    // multiplied with 255 as there are only 255 different color values
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
                std::cout << "Counting pairwise (cpu generic) for attribute " << t->attributes[a].name << " and " << t->attributes[b].name << std::endl;
                auto counts = compression::lineCounterPairSingleField(t->columnData[a].cpuData, t->columnData[b].cpuData, t->columnBins, t->columnBins, t->indexActivation, t->cpuLineCountingAmtOfThreads);
                VkUtil::uploadData(t->_vkContext.device, t->_countResources[{a,b}].countMemory, 0, t->_countResources[{a,b}].binAmt * sizeof(uint32_t), counts.data());
                t->_countResources[{a,b}].brushingId = t->_countBrushState.id;
            }
            break;
        }
        case CountingMethod::CpuRoaring:{
            for(int i: irange(activeIndices.size() -1)){
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
                std::cout << "Counting pairwise (roaring) for attribute " << t->attributes[a].name << " and " << t->attributes[b].name << std::endl;
                auto counts = compression::lineCounterRoaring(t->binsMaxCenterAmt, t->ndBuckets[{a}], aIndexBins, t->ndBuckets[{b}], bIndexBins, t->columnBins, t->columnBins, t->cpuLineCountingAmtOfThreads);
                VkUtil::uploadData(t->_vkContext.device, t->_countResources[{a,b}].countMemory, 0, t->_countResources[{a,b}].binAmt * sizeof(uint32_t), counts.data());
                t->_countResources[{a,b}].brushingId = t->_countBrushState.id;
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
                datas[i] = t->columnData[i].gpuData;
                counts[i] = t->_countResources[{a,b}].countBuffer;
            }
            datas.back() = t->columnData[activeIndices.back()].gpuData;
            if(!anyUpdate)
                goto finish;
            t->_lineCounter->countLinesAll(t->columnData[0].cpuData.size(), datas, t->columnBins, counts, activeIndices, t->_indexActivation, true);
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
                std::cout << "Counting pairwise (compute pipeline) for attribute " << t->attributes[a].name << " and " << t->attributes[b].name << std::endl;
                t->_lineCounter->countLinesPair(t->columnData[a].cpuData.size(), t->columnData[a].gpuData, t->columnData[b].gpuData, t->columnBins, t->columnBins, t->_countResources[{a,b}].countBuffer, t->_indexActivation, true);
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
                std::cout << "Counting pairwise (render pipeline) for attribute " << t->attributes[a].name << " and " << t->attributes[b].name << std::endl;
                t->_renderLineCounter->countLinesPair(t->columnData[a].cpuData.size(), t->columnData[a].gpuData, t->columnData[b].gpuData, t->columnBins, t->columnBins, t->_countResources[{a,b}].countBuffer, true);
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
                std::cout << "Counting pairwise tiled (render pipeline) for attribute " << t->attributes[a].name << " and " << t->attributes[b].name << " with " << tileAmt * tileAmt << " tiles" << std::endl;
                t->_renderLineCounter->countLinesPairTiled(t->columnData[a].cpuData.size(), t->columnData[a].gpuData, t->columnData[b].gpuData, t->columnBins, t->columnBins, t->_countResources[{a,b}].countBuffer, true, tileAmt);
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
        finish:
        std::cout.flush();
        t->requestRender = true;                   // ready to update rendering
        t->_countUpdateThreadActive = false;        // releasing the update thread
    };

    if(_countUpdateThreadActive.compare_exchange_strong(prevValue, true)){    // trying to block any further incoming notifies
        std::cout << "Starting counting pipeline for " << columnBins << " bins and " << columnData[0].cpuData.size() << " data points." << std::endl;
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