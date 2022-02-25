#pragma once
#include <string_view>
#include <string>
#include <vector>
#include "../LassoBrush.hpp"
#include "../Attribute.hpp"
#include "../Structures.hpp"
#include "../Brushing.hpp"
#include "Constants.hpp"
#include "HirarchyCreation.hpp"
#include <atomic>
#include <future>

//forwad decl
struct DrawList;

class HierarchyImportManager{
public:
    using RangeBrush = brushing::RangeBrush;

    HierarchyImportManager(const std::string_view& hierarchyFolder, uint32_t maxDrawLines = 1e6);

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
    std::function<uint32_t (uint32_t, float*, float*, float, float)> _indexFunc;

    std::string _hierarchyFolder;
    compression::CachingMethod _cachingMethod;
    std::vector<std::string> _hierarchyFiles;       //contains all hierarchy files
    std::vector<std::vector<std::string_view>> _levelFiles;    //contains for each hierarchy level the files which are in that level
    std::vector<std::vector<std::string_view>> _levelInfos;
    std::vector<float> _levelEpsilons;
    std::vector<Attribute> _attributes;
    std::vector<Attribute> _reservedAttributes;

    std::future<void> _dataLoadFuture;
    std::vector<std::string_view> _enqueuedFiles;  //if a new openHierarchyFiles call is issued while data is loaded, the new files are stored in this vector to be loaded when the previous load is done
    std::vector<std::vector<size_t>> _enqueuedBundles;
    Data _nextData;

    std::vector<RangeBrush> _curRangeBrushes;       //these are stored in normalized form, to be able to be reapplied if ther eshoudl ever be the need
    Polygons _curLassoBrushes;

    void openHierarchyFiles(const std::vector<std::string_view>& files, const std::vector<std::vector<size_t>> bundleOffsets = {});
};