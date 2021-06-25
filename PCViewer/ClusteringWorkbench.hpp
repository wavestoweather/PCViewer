#pragma once

#include <vector>
#include <iostream>
#include "imgui/imgui.h"
#include "DataClusterer.hpp"
#include "DataProjector.hpp"
#include "Structures.hpp"

class ClusteringWorkbench{
public:
    ClusteringWorkbench(const std::vector<DataSet> datasets): datasets(datasets){}

    //draws a standard imgui window with all functionalyties for the clustering workbench
    void draw(){
        if(ImGui::Begin("Clustering Workbench", &active, 0)){
            ImGui::Text("Projection Settings:");
            if(ImGui::InputInt("Reduction dimension", &projectionDimension)){std::clamp(projectionDimension, 1, 100)};
            if(ImGui::BeginTabBar("ProjectionTabBar")){
                if(ImGui::BeginTabItem("PCA")){
                    ImGui::Text("No special settings for PCA...");
                    ImGui::EndTabItem();
                }
                if(ImGui::BeginTabItem("t-SNE")){

                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }

            ImGui::Text("Clustering Settings");
            if(ImGui::BeginTabBar("ClusteringTabBar")){
                if(ImGui::BeginTabItem("KMeans")){
                    static char* distanceMetrics[3]{"Norm", "SquaredNorm", "L1Norm"};
                    if(ImGui::BeginCombo("Distance Metric", distanceMetrics[clusterSettings.distanceMetric])){
                        for(int i = 0; i < std::size(distanceMetrics); ++i){
                            if(ImGui::MenuItem(distanceMetrics[i])) clusterSettings.distanceMetric = DataClusterer::DistanceMetric(i);
                        }
                        ImGui::EndCombo();
                    }
                    
                    ImGui::EndTabItem();
                }
                if(ImGui::BeginTabItem("DBScan")){

                    ImGui::EndTabItem();
                }
                if(ImGui::BeginTabItem("HClustering")){

                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }
        }
    }
    bool active = false;
    int projectionDimension = 2;
    DataProjector::ProjectionSettings projectionSettings{};
    DataClusterer::ClusterSettings clusterSettings{};

protected:
    const std::vector<DataSet>& datasets;

    void cluster(){
	    DataProjector projector(datasets.front().data, 2, DataProjector::Method::PCA, {});
	    while(!projector.projected){
	    	std::cout << "\r" << projector.progress << std::flush;
	    }
	    std::cout << std::endl;
	    projector.future.wait();
	    DataClusterer::ClusterSettings settings;
	    settings.maxIterations = 100;
	    settings.distanceMetric = DataClusterer::DistanceMetric::Norm;
	    settings.kmeansClusters = 10;
	    settings.kmeansInitMethod = DataClusterer::InitMethod::PlusPlus;
	    settings.kmeansMethod = DataClusterer::KMethod::Mean;
	    DataClusterer clusterer(projector.projectedPoints, DataClusterer::Method::KMeans, settings);
	    while(!clusterer.clustered) std::cout << "\r" << clusterer.progress << std::flush;
	    clusterer.future.wait();
	    std::cout << std::endl;
    }
};