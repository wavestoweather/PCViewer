#define NOSTATICS
#include "HierarchyBinManager.hpp"
#undef NOSTATICS
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <queue>
#include <numeric>
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
    // ---------------------------------------------------------------------------
    // clustering together all attribute clusters
    // ---------------------------------------------------------------------------
    struct Cluster{
        uint32_t startIndex, endIndex; 
        float span; 
        bool operator>(const Cluster& other) const{return span > other.span;};
    };

    std::vector<std::vector<Cluster>> attributeCluster(_attributes.size()); //contains all cluster for all attributes

    // going through all consecutive attributes and update clusters to get max amount of cluster combinations
    uint32_t dimensionality = 2;
    using  ClusterQueue = PCUtil::PriorityQueue<Cluster, std::vector<Cluster>, std::greater<Cluster>>;
    for(int c = 0; c < _attributeOrdering.size() - 1; ++c){
        std::vector<uint32_t> axes(dimensionality);
        for(int d = std::max(c - d / 2 - 1, 0); d < std::min<uint32_t>(c + d / 2 + 1, _attributeOrdering.size()); ++d){
            axes.push_back(_attributeOrdering[d]);
        }
        std::sort(axes.begin(), axes.end());    // ordering should always be increasing
        std::vector<ClusterQueue> pQs(axes.size());  //for reach axis we are starting with the default cluster
        for(int i = 0; i < axes.size(); ++i){
            uint32_t a = axes[i];    
            pQs[i].push(Cluster{0, static_cast<uint32_t>(_attributeCenters[a].size()), _attributeCenters[a].back().max - _attributeCenters[a].back().min});
        }

        uint32_t curCombinationCount = 1;   // always starts with one combination, as all queues have only a single element
        while(1){
            std::vector<uint32_t> sortedQueues; // indeces of the queues in pQs
            std::iota(sortedQueues.begin(), sortedQueues.end(), 0); // fill with indices
            std::sort(sortedQueues.begin(), sortedQueues.end(), [&](uint32_t a, uint32_t b){return pQs[a].top().span > pQs[b].top().span;});
            int p = 0;
            for(; p < sortedQueues.size(); ++p){
                if(attributeCluster[axes[sortedQueues[p]]].empty() || pQs[sortedQueues[p]].size() < attributeCluster[axes[sortedQueues[p]]].size())
                    break;      // found a priority queue that does not excced axis attribute limit from previouis iterations
            }
            // checking if split is possible accordint to max lines limit or other limits
            if(p >= sortedQueues.size())
                break;  // all attributes are constrained in terms of cluster counts
            uint32_t nextCombCount = curCombinationCount / pQs[sortedQueues[p]].size() * (pQs[sortedQueues[p]].size() + 1);
            if(nextCombCount > _maxLines)
                break;  // the maximum number of lines is reached
            // split the biggest gap for the attribute combination
            auto& pQ = pQs[sortedQueues[p]];
            auto cl = pQ.pop();
            uint32_t clAxis = axes[sortedQueues[p]];
            // finding the best split
            uint32_t bestSplit = 0;
            float maxGap = 0;
            for(int i = cl.startIndex; i < cl.endIndex; ++i){
                float gap = _attributeCenters[clAxis][i + 1].min - _attributeCenters[clAxis][i].max;
                if(gap > maxGap){
                    maxGap = gap;
                    bestSplit = i;
                }
            }
            // pushing the two new clusters
            pQ.push(Cluster{cl.startIndex, bestSplit, _attributeCenters[clAxis][bestSplit].max - _attributeCenters[clAxis][cl.startIndex].min});
            pQ.push(Cluster{bestSplit, cl.endIndex, _attributeCenters[clAxis][cl.endIndex].max - _attributeCenters[clAxis][bestSplit + 1].min});
            // updating the cvurrent combination count
            curCombinationCount = nextCombCount;
        }
        
        // updating the axes cluster
        for(int i = 0; i < axes.size(); ++i){
            attributeCluster[axes[i]] = pQs[i].container();
        }
    }

    // ---------------------------------------------------------------------------
    // updating combination counts
    // --------------------------------------------------------------------------- 
    _clusterData.clear();
    _clusterData.columns.resize(2 * dimensionality + 1 + 1 + 1); // first all dimension combinations(including the x and y positions), then for each cluster a reference index to the dimensioncombinations followed by the active and overall count
    // setting up index to cluster map for all attributes except one for quick lookup times
    std::vector<std::vector<uint32_t>> indexClusterMap(_attributeOrdering.size() - 1, std::vector<uint32_t>(_indexActivations.size()));
    for(int c = 0; c < _attributeOrdering.size() -1; ++c){
        uint32_t axis = _attributeOrdering[c];
        // for each index in each attributeCluster we write the cluster index in the index cluster map
        for(int i = 0; i < attributeCluster[c].size(); ++i){
            const auto& cl = attributeCluster[c][i];
            for(int l = cl.startIndex; l < cl.endIndex; ++l){
                uint32_t indexStart = _attributeCenters[axis][l].offset;
                uint32_t indexSize = _attributeCenters[axis][l].offset;
                for(int ind = indexStart; ind < indexStart + indexSize; ++ind){
                    uint32_t index = _attributeIndices[axis][ind];
                    indexClusterMap[c][index] = i;
                }
            }
        }
    }

    // starting to counting the cluster counts
    std::vector<uint32_t> axisToOrder(_attributes.size());
    for(int a = 0; a < _attributeOrdering.size(); ++a)
        axisToOrder[_attributeOrdering[a]] = a; 
    for(int o = 0; o < _attributeOrdering.size() - 1; ++o){
        std::vector<uint32_t> axes(dimensionality);
        for(int d = std::max(o - d / 2 - 1, 0); d < std::min<uint32_t>(o + d / 2 + 1, _attributeOrdering.size()); ++d){
            axes.push_back(_attributeOrdering[d]);
        }
        std::sort(axes.begin(), axes.end());
        std::vector<uint32_t> curIndices(dimensionality, 0);

        // adding dimension information
        bool overflow = false;
        while(!overflow){
            // creating the intersection of the current cluster
            // intersection is computed by going through the indices from the cluster to the very right of the plot
            // and checking for each index in the cluster if the index on the other axes corresponds to the
            // cluster indicated in axes
            // also instantly checks if the index is active
            uint32_t count{}, countSum{};
            const auto& clusters = attributeCluster[axes.back()];
            for(int c = 0; c < clusters.size(); ++c){
                const auto& cluster = clusters[c];
                const auto& indices = _attributeIndices[axes.back()];
                for(int ind = cluster.startIndex; ind < cluster.endIndex; ++ind){
                    countSum++;
                    uint32_t index = indices[ind];
                    if(_indexActivations[index]){   // only continue counting if index is active
                        // going through other axes and check for the correct cluster
                        for(int other = 0; other < axes.size() - 1; ++other){
                            uint32_t otherCluster = indexClusterMap[axisToOrder[other]][index];
                            if(otherCluster == curIndices[other]){
                                ++count;    // we have another index found
                            }
                        }
                    }
                }
            }
            // adding the data to the final data for rendering
            _clusterData.columns;

            // incrementing the counter and propagating overflow
            curIndices.back()++;

            for(int i = curIndices.size() - 1; i >= 0 && (overflow || i == curIndices.size() - 1); --i){
                if(overflow)
                    curIndices[i]++;
                if(curIndices[i] >= attributeCluster[i].size()){
                    curIndices[i] = 0;  //reset counter to 0 to get cartesiaan product fo all cluster
                }
                else
                    overflow = false;
            }
        }
    }
}
