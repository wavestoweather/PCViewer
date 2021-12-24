#pragma once

#include <vector>
#include <iostream>
#include <iterator>
#include "imgui/imgui.h"
#include "DataClusterer.hpp"
#include "DataProjector.hpp"
#include "Structures.hpp"
#include "LassoBrush.hpp"
#include <memory>

class ClusteringWorkbench{
public:
    ClusteringWorkbench(VkDevice device, const std::vector<Attribute>& attributes, std::list<DataSet>& datasets, std::list<DrawList>& drawLists);

    //draws a standard imgui window with all functionalyties for the clustering workbench
    void draw();
    bool active = false;
    bool requestPcPlotUpdate = false;
    DrawList* updateDl{};
    int projectionDimension = 2;
    int projectPlotWidth = 500;
    DataProjector::ProjectionSettings projectionSettings{20.0, 1.0, -1, 500, 0, 700, false};
    DataProjector::Method projectorMethod = DataProjector::Method::PCA;
    std::shared_ptr<DataProjector> projector{};
    DataClusterer::ClusterSettings clusterSettings;
    DataClusterer::Method clusterMethod = DataClusterer::Method::KMeans;
    std::shared_ptr<DataClusterer> clusterer{};

    std::vector<ImVec4> colors{{1,1,0,.2f}, {0,1,0,.2f}, {0,1,1,.2f}, {1,0,1,.2f}, {1,0,0,.2f}};
    ImVec4 inactiveColor{.5f, .5f, .5f, .1f};

    Polygon lassoSelection;

protected:
    void updateActiveLines(DrawList& dl);

    VkDevice _device;
    std::list<DataSet>& datasets;
    const std::vector<Attribute>& attributes;
    std::list<DrawList>& _drawLists;
    std::vector<uint8_t> activations;   //attribute activations
    std::vector<uint8_t> _activations;  //drawlist lines activations
};