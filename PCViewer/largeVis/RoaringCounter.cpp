#include "RoaringCounter.hpp"
#include <roaring.hh>
#include <atomic>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <thread>
#include <string_view>
#include <fstream>
#include <set>
#include "../range.hpp"
#include "../PCUtil.h"
#include "../compression/HirarchyCreation.hpp"

namespace compression
{
    void testRoaringCounting() 
    {
        std::cout << "Starting testRoaringCounting..." << std::endl;
        const uint32_t size = 1 << 30;
        const uint32_t aBins = 1 << 0, bBins = 1 << 0;
        const uint32_t amtOfThreads = 12;
        const bool randomVals = false;
        std::vector<uint32_t> aIndices(size), bIndices(size);
        std::cout << "Filling the indices lists" << std::endl;
        std::iota(aIndices.begin(), aIndices.end(), 0);
        std::iota(bIndices.begin(), bIndices.end(), 0);
        if constexpr(randomVals){
            std::cout << "Shuffling index lists" << std::endl;
            //std::random_shuffle(aIndices.begin(), aIndices.end());
            //std::random_shuffle(bIndices.begin(), bIndices.end());
        }
        std::cout << "Creating roaring lists perfectly evenly distributed" << std::endl;
        std::vector<roaring::Roaring> aMaps(aBins);
        std::vector<roaring::Roaring> bMaps(bBins);
        size_t reducedSize = 0, fullSize = 2 * size * sizeof(uint32_t);
        for(int i: irange(0, aBins)){
            size_t begin = size_t(i) * size / aBins;
            size_t end = size_t(i + 1) * size / aBins;
            aMaps[i] = roaring::Roaring(end - begin, aIndices.data() + begin);
            aMaps[i].runOptimize();
            aMaps[i].shrinkToFit();
            reducedSize += aMaps[i].getSizeInBytes();
        }
        std::cout << "Creation of roaring lists for a done" << std::endl;
        for(int i: irange(0, bBins)){
            size_t begin = size_t(i) * size / bBins;
            size_t end = size_t(i + 1) * size / bBins;
            bMaps[i] = roaring::Roaring(end - begin, bIndices.data() + begin);
            bMaps[i].runOptimize();
            bMaps[i].shrinkToFit();
            reducedSize += bMaps[i].getSizeInBytes();
        }
        std::cout << "Original byteSize: " << fullSize << " vs reducedSize: " << reducedSize << " -> compression ratio 1:" << double(fullSize) / reducedSize << std::endl;

        std::atomic<uint32_t> curInd{0};
        std::vector<uint32_t> counts(aBins * bBins, 0);
        auto threadExec = [&](){
            uint32_t i;
            while((i = curInd++) < counts.size()){    //let the threads run as long as work is to be done
                uint32_t a = i / bBins, b = i % bBins;
                counts[i] = (aMaps[a] & bMaps[b]).cardinality();
            }
        };
        std::cout << "Setup done, starting to count with " << amtOfThreads << " threads for " << size << " datapoints ..." << std::endl;
        {
        std::vector<std::thread> threads(amtOfThreads);
        PCUtil::Stopwatch stopwatch(std::cout, "CpuLineCounter counting time");
        for(int cur: irange(0, amtOfThreads)){
            threads[cur] = std::thread(threadExec);
        }
        // wait for all threads
        for(auto& t: threads)
            t.join();
        }   ///end stopwatch
        size_t count = 0;
        for(auto i: counts)
            count += i;
        std::cout << "Count size is " << count << " || ref " << size << std::endl;
    }
    
    void testRoaringRealWorld() 
    {
        const std::string _hierarchyFolder{"/run/media/lachei/3d02119e-bc93-4969-9fc5-523f06321708/tt"};
        const int threadAmt = 12;
        // --------------------------------------------------------------------------------
        // loading the data
        // --------------------------------------------------------------------------------
        std::ifstream attributeInfos(_hierarchyFolder + "/attr.info", std::ios_base::binary);
        std::string cacheMethod; attributeInfos >> cacheMethod;
        //std::string vec; attributeInfos >> vec;
        //_dimensionSizes = PCUtil::fromReadableString<uint32_t>(vec);
        assert(cacheMethod == compression::CachingMethodNames[int(compression::CachingMethod::Bundled)]);

        std::string a; float aMin, aMax;
        std::vector<Attribute> _attributes;
        while(attributeInfos >> a >> aMin >> aMax){
            _attributes.push_back({a, a, {}, {}, aMin, aMax});
            //_attributeDimensions.push_back(PCUtil::fromReadableString<uint32_t>(vec));
        }
        attributeInfos.close();

        //std::ifstream clusterInfos(_hierarchyFolder + "/hierarchy.info", std::ios_base::binary);
        //assert(clusterInfos);
        //clusterInfos >> _hierarchyLevels;
        //clusterInfos >> _clusterDim;
        //_dimensionCombinations.resize(_clusterDim);
        //uint32_t clusterLevelSize;
        //for(int l = 0; l < _hierarchyLevels; ++l){
        //    clusterInfos >> clusterLevelSize;
        //    _clusterLevelSizes.push_back(clusterLevelSize);
        //}
        //assert(_clusterLevelSizes.size() == _hierarchyLevels);
        //clusterInfos.close();

        // --------------------------------------------------------------------------------
        // center infos
        // --------------------------------------------------------------------------------
        struct IndexCenterFileDataOld{
            float val, min, max;
            uint32_t offset, size;
        };
        std::vector<std::vector<IndexCenterFileDataOld>> _attributeCenters;
        std::ifstream attributeCenterFile(_hierarchyFolder + "/attr.ac", std::ios_base::binary);
        std::vector<compression::ByteOffsetSize> offsetSizes(_attributes.size());
        attributeCenterFile.read(reinterpret_cast<char*>(offsetSizes.data()), offsetSizes.size() * sizeof(offsetSizes[0]));
        _attributeCenters.resize(_attributes.size());
        for(int i = 0; i < _attributes.size(); ++i){
            assert(attributeCenterFile.tellg() == offsetSizes[i].offset);
            _attributeCenters[i].resize(offsetSizes[i].size / sizeof(_attributeCenters[0][0]));
            attributeCenterFile.read(reinterpret_cast<char*>(_attributeCenters[i].data()), offsetSizes[i].size);
        }

        // --------------------------------------------------------------------------------
        // 1d index data
        // --------------------------------------------------------------------------------
        std::vector<std::vector<uint32_t>> _attributeIndices;
        _attributeIndices.resize(_attributes.size());
        uint32_t dataSize = 0;
        for(int i = 0; i < _attributes.size(); ++i){
            std::ifstream indicesData(_hierarchyFolder + "/" + std::to_string(i) + ".ids", std::ios_base::binary);
            uint32_t indicesSize = _attributeCenters[i].back().offset + _attributeCenters[i].back().size;
            if(indicesSize > dataSize)
                dataSize = indicesSize;
            _attributeIndices[i].resize(indicesSize);
            indicesData.read(reinterpret_cast<char*>(_attributeIndices[i].data()), indicesSize * sizeof(_attributeIndices[0][0]));
        }

        std::cout << "Data loaded, starting to compress" << std::endl;

        std::vector<std::vector<roaring::Roaring>> attributeRoars(_attributes.size());
        for(uint32_t compInd: irange(4, _attributeCenters.size())){
            std::vector<roaring::Roaring> compressed(_attributeCenters[compInd].size());
            size_t indexlistSize{_attributeIndices[compInd].size() * sizeof(uint32_t)}, compressedSize{};
            size_t amtOriginalIndices{}, reducesIndices{};
            for(int i = 0; i < compressed.size(); ++i){
                amtOriginalIndices += _attributeCenters[compInd][i].size;
                compressed[i] = roaring::Roaring(_attributeCenters[compInd][i].size, _attributeIndices[compInd].data() + _attributeCenters[compInd][i].offset);
                compressed[i].runOptimize();
                compressed[i].shrinkToFit();
                compressedSize += compressed[i].getSizeInBytes();
                reducesIndices += compressed[i].cardinality();
            }
            attributeRoars[compInd] = std::move(compressed);
            std::cout << "Attribute " << _attributes[compInd].name << ": Uncompressed Indices take " << indexlistSize / float(1 << 20) << " MByte vs " << compressedSize / float(1 << 20) << " MByte compressed." << "Compression rate 1:" << indexlistSize / float(compressedSize) << std::endl;
            std::cout << "Original index amt: " <<  amtOriginalIndices << " vs reduced " << reducesIndices << std::endl;
        }

        std::atomic<uint32_t> at{0};
        std::vector<std::vector<uint32_t>> attributeCounts(_attributes.size() - 1);
        for(int i: irange(4, _attributes.size()- 1)){
            attributeCounts[i].resize(attributeRoars[i].size() * attributeRoars[i + 1].size());
        }
        auto threadExec = [&](int i){
            uint32_t j;
            uint32_t maxSize = attributeCounts[i].size(); 
            while((j = at++) < maxSize){
                int a  = j / attributeRoars[i + 1].size(), b = j % attributeRoars[i +1].size();
                attributeCounts[i][j] = attributeRoars[i][a].and_cardinality(attributeRoars[i + 1][b]);
            }
        };

        std::cout << "Starting anding" << std::endl;

        uint32_t count{};
        float avg{};
        for(int i: irange(4, _attributes.size() - 1)){
            PCUtil::Stopwatch watch(std::cout, "iteration xxx");
            std::vector<std::thread> threads(threadAmt);
            for(int cur: irange(0, threadAmt)){
                threads[cur] = std::thread(threadExec, i);
            }
            //join
            for(auto& t: threads)
                t.join();
            at = 0;
        }
        std::cout << "Average per attribute " << avg << "ms for " << _attributeIndices[4].size() << " datapoints" << std::endl;
    }
    
    std::vector<uint32_t> lineCounterRoaring(uint32_t maxBins,const std::vector<roaring::Roaring64Map>& aIndices, const std::vector<uint32_t>& aIndexBins, const std::vector<roaring::Roaring64Map>& bIndices, const std::vector<uint32_t>& bIndexBins, uint32_t aBins, uint32_t bBins, uint32_t amtOfThreads) 
    {
        std::vector<uint32_t> counts(aBins * bBins);
        size_t maxSize = aIndices.size() * bIndices.size(), j;
        double fac = counts.size() / double(maxBins * maxBins);
        std::atomic<size_t> at{0};
        auto threadExec = [&](){
            while((j = at++) < maxSize){
                int a  = j / bIndices.size(), b = j % bIndices.size();
                size_t aBin = aIndexBins[a], bBin = bIndexBins[b];
                size_t finalBuck = (aBin * maxBins + bBin) * fac;

                counts[finalBuck] = aIndices[a].and_cardinality(bIndices[b]);
            }
        };
        std::vector<std::thread> threads(amtOfThreads);
        PCUtil::Stopwatch watch(std::cout, "Indexpair roaring time");
        for(int cur: irange(0, amtOfThreads)){
            threads[cur] = std::thread(threadExec);
        }
        //join
        for(auto& t: threads)
            t.join();
        return counts;
    }
}