#pragma once
#include <string_view>
#include <string>
#include <vector>
#include "../LassoBrush.hpp"
#include "../Attribute.hpp"
#include "../Structures.hpp"
#include "Constants.hpp"
#include <atomic>
#include <future>

class HierarchyImportManager{
public:
    struct AxisRange{
        uint32_t axis;
        float min;
        float max;
    };
    using RangeBrush = std::vector<AxisRange>;

    HierarchyImportManager(const std::string_view& hierarchyFolder, uint32_t maxDrawLines = 1e6);

    void notifyBrushUpdate(const std::vector<RangeBrush>& rangeBrushes, const Polygons& lassoBrushes);
    void updateDrawList(DrawList& dl);

    // bool which indicates new data was loaded
    std::atomic<bool> newDataLoaded{false};
private:
    bool _hierarchyValid{true};
    uint32_t _baseLevel{0};
    uint32_t _maxLines;

    std::string _hierarchyFolder;
    std::vector<std::string> _hierarchyFiles;       //contains all hierarchy files
    std::vector<std::vector<std::string_view>> _levelFiles;    //contains for each hierarchy level the files which are in that level
    std::vector<Attribute> _attributes;
    std::vector<Attribute> _reservedAttributes;

    std::future<void> _dataLoadFuture;
    Data _nextData;

    void openHierarchyFiles(const std::vector<std::string_view>& files);
};