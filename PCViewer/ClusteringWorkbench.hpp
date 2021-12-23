#pragma once

#include <vector>
#include <iostream>
#include <iterator>
#include "imgui/imgui.h"
#include "DataClusterer.hpp"
#include "DataProjector.hpp"
#include "Structures.hpp"
#include "LassoBrush.hpp"

class ClusteringWorkbench{
public:
    ClusteringWorkbench(const std::vector<Attribute>& attributes, std::list<DataSet>& datasets);

    //draws a standard imgui window with all functionalyties for the clustering workbench
    void draw();
    bool active = false;
    int projectionDimension = 2;
    int projectPlotWidth = 500;
    DataProjector::ProjectionSettings projectionSettings{20.0, 1.0, -1, 100, 0, 700, false};
    DataProjector::Method projectorMethod = DataProjector::Method::PCA;
    DataProjector* projector = 0;
    DataClusterer::ClusterSettings clusterSettings;
    DataClusterer::Method clusterMethod = DataClusterer::Method::KMeans;
    DataClusterer* clusterer = 0;

    std::vector<ImVec4> colors{{1,1,0,.2f}, {0,1,0,.2f}, {0,1,1,.2f}, {1,0,1,.2f}, {1,0,0,.2f}};

    Polygon lassoSelection;

protected:
    std::list<DataSet>& datasets;
    const std::vector<Attribute>& attributes;
    std::vector<uint8_t> activations;
};