#define NOSTATICS
#include "HierarchyLoadManager.hpp"
#undef NOSTATICS
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <iostream>
#include "../robin_hood_map/robin_hood.h"

HierarchyLoadManager::HierarchyLoadManager(const std::string_view& hierarchyFolder, uint32_t maxDrawLines) :
_maxLines(maxDrawLines), _hierarchyFolder(hierarchyFolder)
{
    // loading attribute info and data
    _hierarchyFolder = hierarchyFolder;

    // --------------------------------------------------------------------------------
    // attribute infos, cluster infos
    // --------------------------------------------------------------------------------
    std::ifstream attributeInfos(_hierarchyFolder + "/attr.info", std::ios_base::binary);
    std::string cacheMethod; attributeInfos >> cacheMethod;
    assert(cacheMethod == compression::CachingMethodNames[int(compression::CachingMethod::Bundled)]);
    
    std::string a; float aMin, aMax;
    while(attributeInfos >> a >> aMin >> aMax){
        _attributes.push_back({a, a, {}, {}, aMin, aMax});
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
    // clustering data
    // --------------------------------------------------------------------------------
    // creating an extra maybe delay opening to an extra thread should runtime problems occur
    auto execLoading = [](HierarchyLoadManager* m){
        // getting the subdimensions
        std::ifstream combinationInfo(m->_hierarchyFolder + "/combination.info", std::ios_base::binary);
        std::string firstLine; std::getline(combinationInfo, firstLine);
        m->_clusterDim = std::count(firstLine.begin(), firstLine.end(), ' ');
        combinationInfo.seekg(0);
        std::vector<uint32_t> line(m->_clusterDim);
        while(combinationInfo){
            for(int i = 0; i < line.size(); ++i) 
                combinationInfo >> line[i];
            if(!combinationInfo)
                break;
            for(int i = 0; i < line.size(); ++i)
                m->_dimensionCombinations[i].push_back(line[i]);
        }
        combinationInfo.close();

        // loading the attribute center data
        std::string acFilePath = m->_hierarchyFolder + "/attributeCenters.ac";
        std::ifstream acFile(acFilePath, std::ios_base::binary);
        auto byteSize = std::filesystem::file_size(acFilePath);
        std::vector<uint32_t> binaryData(byteSize / sizeof(uint32_t));
        acFile.read(reinterpret_cast<char*>(binaryData.data()), byteSize);
        acFile.close();

        std::vector<std::vector<std::pair<size_t, size_t>>> attributeCenterOffsets(m->_hierarchyLevels);
        size_t curInd = 0;
        for(int i = 0; i < m->_attributes.size(); ++i){
            for(int l = 0; l < m->_hierarchyLevels; ++l){
                attributeCenterOffsets[l].push_back({binaryData[curInd++] / 4, binaryData[curInd++] / 4});  // instantly converting from byte offset to uint32_t offset
            }
        }
        m->_attributeCenters.resize(m->_hierarchyLevels, std::vector<std::vector<compression::CenterData>>(m->_attributes.size()));
        for(int l = 0; l < m->_hierarchyLevels; ++l){
            for(int a = 0; a < m->_attributes.size(); ++a){
                auto [offset, size] = attributeCenterOffsets[l][a];
                m->_attributeCenters[l][a].resize(size * 4 / sizeof(compression::CenterData));   // * 4 to get to byte size and / sizeof(..) to get to center data counts
                std::memcpy(m->_attributeCenters[l][a].data(), &binaryData[offset], size * 4);
            }
        }
        binaryData.clear();

        // creating map to map bin index on each axis to cluster index
        std::vector<robin_hood::unordered_map<uint32_t, uint32_t>> binToCl(m->_attributes.size());
        for(int a = 0; a < m->_attributes.size(); ++a){
            for(int c = 0; c < m->_attributeCenters.back()[a].size(); ++c){
                uint32_t bin = m->_attributeCenters.back()[a][c].val * m->_clusterLevelSizes.back();
                binToCl[a][bin] = c;
            }
        }

        // loading the center data and converting it to cluster indices (from bin values) plus adding the subdim index
        // everything is stored in a Data object with all subdim indices etc. in one row follwed by rows with other data
        std::ifstream clusterData(m->_hierarchyFolder + "/cluster.cd", std::ios_base::binary);
        std::vector<std::pair<uint32_t, uint32_t>> clusterOffsetSizes(m->_dimensionCombinations[0].size()); //all kept in bytes
        auto& cData = m->_clusterData;
        cData.columns.resize(2 + m->_clusterDim);
        uint32_t paddedDim = ((m->_clusterDim + 1) >> 1) << 1;
        uint32_t paddedDimBytes = paddedDim * sizeof(uint16_t);
        uint32_t clusterBytes = paddedDimBytes + sizeof(uint32_t); // cluster size is sizeof bins + sizeof counter
        for(int i = 0; i < m->_dimensionCombinations[0].size(); ++i){
            clusterData.read(reinterpret_cast<char*>(&clusterOffsetSizes[i]), sizeof(clusterOffsetSizes[0]));
        }
        for(int i = 0; i < clusterOffsetSizes.size(); ++i){
            uint32_t amtOfCluster = clusterOffsetSizes[i].second / (clusterBytes);
            size_t offset = cData.columns[0].size();
            for(int c = 0; c < cData.columns.size(); ++c)
                cData.columns[c].resize(cData.columns[c].size() + amtOfCluster);
            //single read of cluster data
            std::vector<uint32_t> data(clusterOffsetSizes[i].second / sizeof(uint32_t));
            clusterData.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(data[0]));
            std::vector<uint16_t> dimBins(paddedDim);
            for(int c = 0; c < amtOfCluster; ++c){
                uint32_t clusterBase = c * clusterBytes / sizeof(uint32_t);
                std::memcpy(dimBins.data(), &data[clusterBase], paddedDimBytes);
                cData.columns[0][offset + c] = i;   
                for(int d = 0; d < m->_clusterDim; ++d){
                    int dim = m->_dimensionCombinations[d][i];
                    assert(binToCl[dim].contains(dimBins[d]));
                    cData.columns[1 + d][offset + c] = binToCl[dim][dimBins[d]];
                }
                cData.columns[m->_clusterDim + 1][offset + c] = data[clusterBase + paddedDim / 2]; // plus paddedDim / 2 as per uint 2 dimension bins are stored
            }
        }
        clusterData.close();
        // I think i am done ... already tested and it seems pog
    };
    execLoading(this);
}

void HierarchyLoadManager::notifyBrushUpdate(const std::vector<RangeBrush>& rangeBrushes, const Polygons& lassoBrushes) 
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

void HierarchyLoadManager::checkPendingFiles() 
{
    using namespace std::chrono_literals;
    if(!_loadThreadActive && _enqueuedFiles.size() != 0)
        openHierarchyFiles(_enqueuedFiles, _enqueuedBundles);
}

void HierarchyLoadManager::openHierarchyFiles(const std::vector<std::string_view>& files, const std::vector<std::vector<size_t>> bundleOffsets){
    using namespace std::chrono_literals;
    // if loading is still active return and dont open new hierarchy files, but cache new files
    bool prevValue = false;
    if(!_loadThreadActive.compare_exchange_strong(prevValue, true)) {   //trying to block
        _enqueuedFiles = files;
        _enqueuedBundles = bundleOffsets;
        return;
    }
    
    auto exec = [](std::vector<std::string_view> files, HierarchyLoadManager* m){
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
