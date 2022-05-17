#define NOSTATICS
#include "IndBinManager.hpp"
#undef NOSTATICS

IndBinManager::IndBinManager(const CreateInfo& info) :
_hierarchyFolder(info.hierarchyFolder)
{
    // --------------------------------------------------------------------------------
    // determine kind of stored data (used to convert internally to a unified represantation for further processing)
    // --------------------------------------------------------------------------------
    std::ifstream dataInfo(_hierarchyFolder + "/data.info", std::ios::binary);
    compression::DataStorageBits dataBits;
    dataInfo >> dataBits;
    dataInfo.close();

    // --------------------------------------------------------------------------------
    // attribute infos, cluster infos
    // --------------------------------------------------------------------------------
    std::ifstream attributeInfos(_hierarchyFolder + "/attr.info", std::ios_base::binary);
    std::string a; float aMin, aMax;
    while(attributeInfos >> a >> aMin >> aMax){
        attributes.push_back({a, a, {}, {}, aMin, aMax});
        //_attributeDimensions.push_back(PCUtil::fromReadableString<uint32_t>(vec));
    }
    attributeInfos.close();

    // --------------------------------------------------------------------------------
    // center infos
    // --------------------------------------------------------------------------------
    if((dataBits & compression::DataStorageBits::RawAttributeBins) != compression::DataStorageBits::None){
        // uncompressed indices
    }
    else if((dataBits & compression::DataStorageBits::Roaring2dBins) != compression::DataStorageBits::None){
        // compressed indices
    }
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
    if(true){
        _attributeIndices.resize(attributes.size());
        uint32_t dataSize = 0;
        for(int i = 0; i < attributes.size(); ++i){
            std::ifstream indicesData(_hierarchyFolder + "/" + std::to_string(i) + ".ids", std::ios_base::binary);
            uint32_t indicesSize = _attributeCenters[i].back().offset + _attributeCenters[i].back().size;
            if(indicesSize > dataSize)
                dataSize = indicesSize;
            _attributeIndices[i].resize(indicesSize);
            indicesData.read(reinterpret_cast<char*>(_attributeIndices[i].data()), indicesSize * sizeof(_attributeIndices[0][0]));
        }
    }
    else if(true){
        //TODO: simply load
    }

    // test indexcompression
    // testing random indexlist compression with uniformly split clusters
    //{
    //    std::vector<uint32_t> randomInts(_attributeIndices[4].size());
    //    std::iota(randomInts.begin(), randomInts.end(), 0);
    //    //std::random_shuffle(randomInts.begin(), randomInts.end());
    //    size_t curStart{};
    //    size_t binSize = 1 << 14;
    //    size_t indexlistSize{randomInts.size() * sizeof(uint32_t)}, compressedSize{};
    //    for(size_t i = 0; i < binSize; ++i){   //1024 bins
    //        size_t end = (i + 1) * randomInts.size() / binSize;
    //        auto compressed = roaring::Roaring(end - curStart, randomInts.data() + curStart);
    //        compressed.runOptimize();
    //        compressedSize += compressed.getSizeInBytes();
    //        curStart = end;
    //    }
    //    std::cout << "Ordered indices " << binSize << " bins : Uncompressed Indices take " << indexlistSize / float(1 << 20) << " MByte vs " << compressedSize / float(1 << 20) << " MByte compressed." << "Compression rate 1:" << indexlistSize / float(compressedSize) << std::endl;
    //}

    //for(uint32_t compInd = 4; compInd < _attributeCenters.size(); ++compInd){
    //    std::vector<roaring::Roaring> compressed(_attributeCenters[compInd].size());
    //    size_t indexlistSize{_attributeIndices[compInd].size() * sizeof(uint32_t)}, compressedSize{};
    //    for(int i = 0; i < compressed.size(); ++i){
    //        compressed[i] = roaring::Roaring(_attributeCenters[compInd][i].size, _attributeIndices[compInd].data() + _attributeCenters[compInd][i].offset);
    //        compressed[i].runOptimize();
    //        compressedSize += compressed[i].getSizeInBytes();
    //    }
    //    std::cout << "Attribute " << _attributes[compInd].name << ": Uncompressed Indices take " << indexlistSize / float(1 << 20) << " MByte vs " << compressedSize / float(1 << 20) << " MByte compressed." << "Compression rate 1:" << indexlistSize / float(compressedSize) << std::endl;
    //}
    // end test
}

IndBinManager::~IndBinManager(){
    // releasing the gpu pipeline singletons
    if(_renderLineCounter)
        _renderLineCounter->release();
    if(_lineCounter)
        _lineCounter->release();
    if(_renderer)
        _renderer->release();
}