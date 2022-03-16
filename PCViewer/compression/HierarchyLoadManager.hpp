#pragma once
#include <string_view>
#include <string>
#include <vector>
#include "../LassoBrush.hpp"
#include "../Attribute.hpp"
#include "../Structures.hpp"
#include "../Brushing.hpp"
#include "HirarchyCreation.hpp"
#include <atomic>
#include <future>

//forwad decl
struct DrawList;

class HierarchyLoadManager{
public:
    using RangeBrush = brushing::RangeBrush;

    // maxDrawLines describes the max lines inbetween two attributes
    HierarchyLoadManager(const std::string_view& hierarchyFolder, uint32_t maxDrawLines = 1e6);

    void notifyAttributeOrderUpdate(const std::vector<int>& attributeOrdering);
    void notifyBrushUpdate(const std::vector<RangeBrush>& rangeBrushes, const Polygons& lassoBrushes);
    void updateDrawList(DrawList& dl);
    void checkPendingFiles();
    Data retrieveNewData(){Data t = std::move(_nextData); _nextData = {}; newDataLoaded = false; return std::move(t);};

    // bool which indicates new data was loaded
    std::atomic<bool> newDataLoaded{false};
private:
    bool _hierarchyValid{true};
    uint32_t _baseLevel{0};
    uint32_t _maxLines;
    uint32_t _hierarchyLevels;
    uint32_t _clusterDim;
    std::vector<uint32_t> _clusterLevelSizes{};

    std::string _hierarchyFolder;
    std::vector<float> _levelEpsilons;
    std::vector<Attribute> _attributes;
    std::vector<int> _attributeOrdering;

    std::thread _dataLoadThread;
    std::atomic<bool> _loadThreadActive{false};
    std::thread _prepareDataThread;
    std::atomic<bool> _prepareThreadActive{false};
    std::vector<std::string_view> _enqueuedFiles;  //if a new openHierarchyFiles call is issued while data is loaded, the new files are stored in this vector to be loaded when the previous load is done
    std::vector<std::vector<size_t>> _enqueuedBundles;
    std::vector<std::vector<uint32_t>> _dimensionCombinations;  //stored in column major format. One row is (_dC[0][0], _dC[1][0], ..., _dC[n][0])
    std::vector<std::vector<std::vector<compression::CenterData>>> _attributeCenters; // for each level for all attributes a singel list with the centers exists 
    Data _clusterData;                              //loaded data wich was already preloaded from disk
    Data _nextData;

    std::vector<RangeBrush> _curRangeBrushes;       //these are stored in normalized form, to be able to be reapplied if ther eshoudl ever be the need
    Polygons _curLassoBrushes;


    void openHierarchyFiles(const std::vector<std::string_view>& files, const std::vector<std::vector<size_t>> bundleOffsets = {});
};