#define NOSTATICS
#include "IndBinManager.hpp"
#undef NOSTATICS
#include "../range.hpp"
#include "CpuLineCounter.hpp"
#include "RoaringCounter.hpp"

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
    uint32_t maxCenterAmt;
    attributeInfos >> maxCenterAmt; // first line contains the maximum amt of centers/bins
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
            ndBuckets[{i}].resize(maxCenterAmt);
            size_t curSize{};
            for(uint32_t bin: irange(_attributeCenters[i])){
                uint32_t b = _attributeCenters[i][bin].val * maxCenterAmt;
                if(b == maxCenterAmt) --b;
                ndBuckets[{i}][b] = roaring::Roaring64Map::readSafe(indices.data() + _attributeCenters[i][bin].offset, _attributeCenters[i][bin].size * sizeof(indices[0]));
                curSize += ndBuckets[{i}][b].cardinality();
            }
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
        for(uint32_t i: irange(attributes)){
            std::ifstream data(_hierarchyFolder + "/" + std::to_string(i) + ".col", std::ios_base::binary);
            auto& dVec = columnData[i].columnData;
            dVec.resize(dataSize);
            data.read(reinterpret_cast<char*>(dVec.data()), dVec.size() * sizeof(dVec[0]));
        }
    }
    else if((dataBits & compression::DataStorageBits::CuComColumnData) != compression::DataStorageBits::None){
        // directly parse if same compression block size
        // not yet implemented
        std::cout << "Well thats wrong. Seems like there is some compression in the column data but we cant handle it." << std::endl;
    }

    // --------------------------------------------------------------------------------
    // getting the handles for the counter pipelines
    // --------------------------------------------------------------------------------
    _renderLineCounter = RenderLineCounter::acquireReference(RenderLineCounter::CreateInfo{info.context});
    _lineCounter = LineCounter::acquireReference(LineCounter::CreateInfo{info.context});
    _renderer = compression::Renderer::acquireReference(compression::Renderer::CreateInfo{info.context, info.renderPass, info.framebuffer});
}


IndBinManager::~IndBinManager(){
    // releasing the gpu pipeline singletons
    if(_renderLineCounter)
        _renderLineCounter->release();
    if(_lineCounter)
        _lineCounter->release();
    if(_renderer)
        _renderer->release();
    //TODO: release all here created things
}

void IndBinManager::notifyAttributeUpdate(const std::vector<int>& attributeOrdering, const std::vector<Attribute>& attributes, bool* attributeActivations){
    _attributeOrdering = attributeOrdering;
    assert(attributes == this->attributes); // debug check
    _attributeActivations = attributeActivations;
    updateCounts();         // updating the counts if there are some missing and pushing a render call
}

void IndBinManager::render(VkCommandBuffer commands,VkBuffer attributeInfos, bool clear){
    std::vector<int> activeIndices; // these are already ordered
    for(auto i: _attributeOrdering){
        if(_attributeActivations[i])
            activeIndices.push_back(i);
    }
    std::vector<VkBuffer> counts;
    std::vector<std::pair<uint32_t, uint32_t>> axes;
    for(int i: irange(activeIndices.size() - 1)){
        uint32_t a = activeIndices[i];
        uint32_t b = activeIndices[i + 1];
        if(a > b)
            std::swap(a, b);
        if(!_countResources.contains({a,b})){

        }
        counts.push_back(_countResources[{a,b}].countBuffer);
        axes.push_back({a,b});
    }

    compression::Renderer::RenderInfo renderInfo{
        commands,
        "dummy",
        compression::Renderer::RenderType::Polyline,
        counts,
        axes,
        columnBins * columnBins,
        _attributeOrdering,
        attributes,
        _attributeActivations,
        columnBins,
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
    bool prevValue{};

    std::vector<int> activeIndices; // these are already ordered
    for(auto i: _attributeOrdering){
        if(_attributeActivations[i])
            activeIndices.push_back(i);
    }
    if(activeIndices.size() < 2){
        std::cout << "Less than 2 attribute active, not updating the counts" << std::endl;
        return;
    }

    auto execCountUpdate = [](IndBinManager* t, std::vector<int> activeIndices){
        // Note: vulkan resources for the count images were already provided by main thread
        switch(t->countingMethod){
        case CountingMethod::CpuGeneric:{
            for(int i: irange(activeIndices.size() - 1)){
                uint32_t a = activeIndices[i];
                uint32_t b = activeIndices[i + 1];
                if(a > b)
                    std::swap(a, b);
                if(t->_countResources.contains({a,b}) && t->_countResources[{a,b}].brushingId == t->_countBrushState.id)
                    return;
                std::cout << "Counting pairwise (cpu generic) for attribute " << t->attributes[a].name << " and " << t->attributes[b].name << std::endl;
                auto counts = compression::lineCounterPair(t->columnData[a].columnData, t->columnData[b].columnData, t->columnBins, t->columnBins, t->cpuLineCountingAmtOfThreads);
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
                    return;
                std::cout << "Counting pairwise (roaring) for attribute " << t->attributes[a].name << " and " << t->attributes[b].name << std::endl;
                auto counts = compression::lineCounterRoaring(t->ndBuckets[{a}], t->ndBuckets[{b}], t->columnBins, t->columnBins, t->cpuLineCountingAmtOfThreads);
                VkUtil::uploadData(t->_vkContext.device, t->_countResources[{a,b}].countMemory, 0, t->_countResources[{a,b}].binAmt * sizeof(uint32_t), counts.data());
                t->_countResources[{a,b}].brushingId = t->_countBrushState.id;
            }
            break;
        }
        case CountingMethod::GpuComputeFull:{
            //ToDo: implement full pipeline
            break;
        }
        case CountingMethod::GpuComputePairwise:{
            for(int i: irange(activeIndices.size() -1)){
                uint32_t a = activeIndices[i];
                uint32_t b = activeIndices[i + 1];
                if(a > b)
                    std::swap(a, b);
                if(t->_countResources.contains({a,b}) && t->_countResources[{a,b}].brushingId == t->_countBrushState.id)
                    return;
                std::cout << "Counting pairwise (compute pipeline) for attribute " << t->attributes[a].name << " and " << t->attributes[b].name << std::endl;
                t->_lineCounter->countLinesPair(t->columnData[a].columnData.size(), t->columnData[a].gpuData, t->columnData[b].gpuData, t->columnBins, t->columnBins, t->_countResources[{a,b}].countBuffer, true);
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
                    return;
                std::cout << "Counting pairwise (render pipeline) for attribute " << t->attributes[a].name << " and " << t->attributes[b].name << std::endl;
                t->_lineCounter->countLinesPair(t->columnData[a].columnData.size(), t->columnData[a].gpuData, t->columnData[b].gpuData, t->columnBins, t->columnBins, t->_countResources[{a,b}].countBuffer, true);
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
        t->requestRender = true;                   // ready to update rendering
        t->_countUpdateThreadActive = false;        // releasing the update thread
    };

    if(_countUpdateThreadActive.compare_exchange_strong(prevValue, true)){    // trying to block any further incoming notifies
        // making shure the counting images are created and have the right size
        for(int i: irange(activeIndices.size() - 1)){
            uint32_t a = activeIndices[i];
            uint32_t b = activeIndices[i + 1];
            if(a > b)
                std::swap(a, b);
            if(!_countResources[{a,b}].countBuffer || _countResources[{a,b}].binAmt != columnBins * columnBins){
                // create resource if not yet available
                VkUtil::createBuffer(_vkContext.device, columnBins * columnBins * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &(_countResources[{a,b}].countBuffer));
                VkMemoryAllocateInfo allocInfo{};
                allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                VkMemoryRequirements memReq{};
                vkGetBufferMemoryRequirements(_vkContext.device, _countResources[{a,b}].countBuffer, &memReq);
                allocInfo.allocationSize = memReq.size;
                allocInfo.memoryTypeIndex = VkUtil::findMemoryType(_vkContext.physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
                vkAllocateMemory(_vkContext.device, &allocInfo, nullptr, &_countResources[{a,b}].countMemory);
                vkBindBufferMemory(_vkContext.device, _countResources[{a,b}].countBuffer, _countResources[{a,b}].countMemory, 0);
                _countResources[{a,b}].binAmt = columnBins * columnBins;
            }
        }
        _countBrushState = _currentBrushState;
        _countUpdateThread = std::thread(execCountUpdate, this, activeIndices);
    }
}