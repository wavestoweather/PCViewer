#define NOSTATICS
#include "HierarchyBinManager.hpp"
#undef NOSTATICS
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <iostream>
#include "../robin_hood_map/robin_hood.h"

HierarchyBinManager::HierarchyBinManager(const std::string_view& hierarchyFolder, uint32_t maxDrawLines) :
_maxLines(maxDrawLines), _hierarchyFolder(hierarchyFolder)
{
    // loading attribute info and data
    _hierarchyFolder = hierarchyFolder;

    // --------------------------------------------------------------------------------
    // attribute infos, cluster infos
    // --------------------------------------------------------------------------------
    std::ifstream attributeInfos(_hierarchyFolder + "/attr.info", std::ios_base::binary);
    std::string cacheMethod; attributeInfos >> cacheMethod;
    std::string vec; attributeInfos >> vec;
    _dimensionSizes = PCUtil::fromReadableString<uint32_t>(vec);
    assert(cacheMethod == compression::CachingMethodNames[int(compression::CachingMethod::Bundled)]);
    
    std::string a; float aMin, aMax;
    while(attributeInfos >> a >> aMin >> aMax >> vec){
        _attributes.push_back({a, a, {}, {}, aMin, aMax});
        _attributeDimensions.push_back(PCUtil::fromReadableString<uint32_t>(vec));
    }
    attributeInfos.close();

    std::ifstream clusterInfos(_hierarchyFolder + "/hierarchy.info", std::ios_base::binary);
    assert(clusterInfos);
    clusterInfos >> _hierarchyLevels;
    clusterInfos >> _clusterDim;
    _dimensionCombinations.resize(_clusterDim);
    uint32_t clusterLevelSize;
    for(int l = 0; l < _hierarchyLevels; ++l){
        clusterInfos >> clusterLevelSize;
        _clusterLevelSizes.push_back(clusterLevelSize);
    }
    assert(_clusterLevelSizes.size() == _hierarchyLevels);
    clusterInfos.close();

    // --------------------------------------------------------------------------------
    // center infos
    // --------------------------------------------------------------------------------
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
    _attributeIndices.resize(_attributes.size());
    uint32_t dataSize = 0;
    for(int i = 0; _attributes.size(); ++i){
        std::ifstream indicesData(_hierarchyFolder + "/" + std::to_string(i) + ".ids", std::ios_base::binary);
        uint32_t indicesSize = _attributeCenters[i].back().offset + _attributeCenters[i].back().size;
        if(indicesSize > dataSize)
            dataSize = indicesSize;
        indicesData.read(reinterpret_cast<char*>(_attributeIndices[i].data()), indicesSize * sizeof(_attributeIndices[0][0]));
    }

    _indexActivations.resize(dataSize, true);
}

void HierarchyBinManager::notifyBrushUpdate(const std::vector<RangeBrush>& rangeBrushes, const Polygons& lassoBrushes) 
{
    std::cout << "HierarchyImportManager::notifyBrushUpdate(): updating..." << std::endl;
    
    //converting the brushes to local coordinates (the normalized coordinates)
    std::vector<RangeBrush> normalizedBrushes;
    for(auto& b: rangeBrushes){
        RangeBrush rb{};
        for(auto& r: b){
            brushing::AxisRange ar{};
            ar.axis = r.axis;
            double diff = _attributes[r.axis].max - _attributes[r.axis].min;
            ar.min = (r.min - _attributes[r.axis].min) / diff;
            ar.max = (r.max - _attributes[r.axis].min) / diff;
            rb.push_back(ar);
        }
        normalizedBrushes.push_back(rb);
    }
    Polygons normalizedLassos;
    for(auto& l: lassoBrushes){
        Polygon p{};
        p.attr1 = l.attr1;
        p.attr2 = l.attr2;
        for(auto& bp: l.borderPoints){
            ImVec2 v{};
            v.x = bp.x - _attributes[l.attr1].min / (_attributes[l.attr1].max - _attributes[l.attr1].min);
            v.y = bp.y - _attributes[l.attr2].min / (_attributes[l.attr2].max - _attributes[l.attr2].min);
            p.borderPoints.push_back(v);
        }
        normalizedLassos.push_back(p);
    }

    //checking if something changed from last notifyBrushUpdate() call
    bool change = false;
    if(normalizedBrushes.size() != _curRangeBrushes.size())
        change = true;
    for(int i = 0; !change && i < normalizedBrushes.size(); ++i){
        if(normalizedBrushes[i].size() != _curRangeBrushes[i].size()){
            change = true;
            break;
        }
        for(int j = 0;!change && j < normalizedBrushes[i].size(); ++j){
            if(normalizedBrushes[i][j].axis != _curRangeBrushes[i][j].axis ||
                normalizedBrushes[i][j].min != _curRangeBrushes[i][j].min ||
                normalizedBrushes[i][j].max != _curRangeBrushes[i][j].max)
                change = true;
        }
    }
    if(normalizedLassos.size() != _curLassoBrushes.size())
        change = true;
    for(int i = 0;!change && i < normalizedLassos.size(); ++i){
        if(normalizedLassos[i].attr1 != normalizedLassos[i].attr2 || 
            normalizedLassos[i].borderPoints.size() != _curLassoBrushes[i].borderPoints.size()){
            change = true;
            break;
        }
        for(int j = 0; !change && j < normalizedLassos[i].borderPoints.size(); ++j){
            if(normalizedLassos[i].borderPoints[j].x != _curLassoBrushes[i].borderPoints[j].x ||
                normalizedLassos[i].borderPoints[j].y != _curLassoBrushes[i].borderPoints[j].y)
                change = true;
        }
    }
    if(!change)     //don't do any updates if nothing has changed
        return;

    std::cout << "HierarchyImportManager::notifyBrushUpdate(): brushChanged..." << std::endl;

    _curRangeBrushes = normalizedBrushes;
    std::cout << "RangeBrushes:" << std::endl;
    for(auto& b: _curRangeBrushes){
        std::cout << "RangeBrush:" << std::endl;
        for(auto& r: b){
            std::cout<< "    " << r.axis << ":" << r.min << "|" << r.max << std::endl;
        }
    }
    _curLassoBrushes = normalizedLassos;

    //after normalization go through the hierarchy to check the level sizes beginning at the top most level
   
}

void HierarchyBinManager::checkPendingFiles() 
{
    using namespace std::chrono_literals;
    if(!_loadThreadActive && _enqueuedFiles.size() != 0)
        openHierarchyFiles(_enqueuedFiles, _enqueuedBundles);
}

void HierarchyBinManager::openHierarchyFiles(const std::vector<std::string_view>& files, const std::vector<std::vector<size_t>> bundleOffsets){
    using namespace std::chrono_literals;
    // if loading is still active return and dont open new hierarchy files, but cache new files
    bool prevValue = false;
    if(!_loadThreadActive.compare_exchange_strong(prevValue, true)) {   //trying to block
        _enqueuedFiles = files;
        _enqueuedBundles = bundleOffsets;
        return;
    }
    
    auto exec = [](std::vector<std::string_view> files, HierarchyBinManager* m){
        std::vector<Data> dataVec(files.size());
        uint32_t i = 0;
        for(auto& f: files){
            compression::loadAndDecompress(f, dataVec[i++]);
        }
        compression::combineData(dataVec, m->_nextData);
        //denormalizing from [0,1] to [min,max]
        for(int a = 0; a < m->_attributes.size(); ++a){
            float diff = m->_attributes[a].max - m->_attributes[a].min;
            for(i = 0; i < m->_nextData.columns[a].size(); ++i)
                m->_nextData.columns[a][i] = m->_nextData.columns[a][i] * diff + m->_attributes[a].min;
        }
        std::cout << "HierarchyImportManager::openHierarchyFiles() loaded new data with " << m->_nextData.size() << " datapoints" << std::endl;
        m->newDataLoaded = true;
        m->_loadThreadActive = false;   //signaling end of loading
    };

    _dataLoadThread = std::thread(exec, files, this);
}

void HierarchyBinManager::updateLineCombinations(const std::vector<uint32_t> attributeOrder){

}
