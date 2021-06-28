#pragma once

#include <vector>
#include <iostream>
#include <iterator>
#include "imgui/imgui.h"
#include "DataClusterer.hpp"
#include "DataProjector.hpp"
#include "Structures.hpp"

class ClusteringWorkbench{
public:
    ClusteringWorkbench(const std::vector<Attribute>& attributes, const std::list<DataSet>& datasets): datasets(datasets), attributes(attributes), activations(attributes.size(), 1){}

    //draws a standard imgui window with all functionalyties for the clustering workbench
    void draw(){
        if(!active) return;
        if(ImGui::Begin("Clustering Workbench", &active, 0)){
            ImGui::Text("Data Projection:");
            static int datasetIndex = 0;
            static char* defaultName = "No datasets available";
            auto ds = datasets.begin(); std::advance(ds, datasetIndex);
            if(ImGui::BeginCombo("Dataset to cluster", datasetIndex < datasets.size() ? ds->name.c_str(): defaultName)){
                int c = 0;
                for(auto& ds: datasets){
                    if(ImGui::MenuItem(ds.name.c_str())){
                        datasetIndex = c;
                    }
                    ++c;
                }
                ImGui::EndCombo();
            }
            if(datasetIndex < datasets.size()){
                activations.resize(attributes.size(), 1);
                ImGui::Text("Attributes active for projection:");
                for(int i = 0; i < activations.size(); ++i){
                    ImGui::Checkbox(attributes[i].name.c_str(), (bool*)&activations[i]);
                    if(i < activations.size() - 1) ImGui::SameLine();
                }
            }
            if(ImGui::InputInt("Reduction dimension", &projectionDimension)) projectionDimension = std::clamp(projectionDimension, 1, 100);
            if(ImGui::BeginTabBar("ProjectionTabBar")){
                if(ImGui::BeginTabItem("PCA")){
                    projectorMethod = DataProjector::Method::PCA;
                    ImGui::Text("No special settings for PCA (But its fast ;) )...");
                    ImGui::EndTabItem();
                }
                if(ImGui::BeginTabItem("t-SNE")){
                    projectorMethod = DataProjector::Method::TSNE;
                    ImGui::Text("NOTE: Very slow for large datasets");
                    if(ImGui::InputInt("Max Iteration", &projectionSettings.maxIter)) projectionSettings.maxIter = std::clamp(projectionSettings.maxIter, 1, 100);
                    ImGui::InputInt("Stop lying iteration", &projectionSettings.stopLyingIter);
                    ImGui::InputInt("Momentum switch iteration", &projectionSettings.momSwitchIter);
                    ImGui::InputDouble("Perplexity", &projectionSettings.perplexity);
                    ImGui::InputDouble("Theta", &projectionSettings.theta);
                    ImGui::Checkbox("Skip Random Init", &projectionSettings.skipRandomInit);
                    ImGui::InputInt("Random Seed", &projectionSettings.randSeed);
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
            bool disabled = projector && !projector->projected;
            if(disabled) ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            if(ImGui::Button("Project")){
                if(projector) delete projector;
                if(clusterer) delete clusterer;
                std::vector<uint32_t> indices;
                int c = 0;
                for(uint8_t n: activations){
                    if(n) indices.push_back(c);
                    ++c;
                }
                projector = new DataProjector(datasets.front().data, 2, projectorMethod, projectionSettings, indices);
            }
            if(disabled) ImGui::PopItemFlag();
            if(projector) ImGui::Text("Projectioin progress: %d%%", int(projector->progress * 100));

            ImGui::Text("Clustering Settings");
            if(ImGui::BeginTabBar("ClusteringTabBar")){
                if(ImGui::BeginTabItem("KMeans")){
                    clusterMethod = DataClusterer::Method::KMeans;
                    static char* distanceMetrics[3]{"Norm", "SquaredNorm", "L1Norm"};
                    if(ImGui::BeginCombo("Distance Metric", distanceMetrics[clusterSettings.distanceMetric])){
                        for(int i = 0; i < std::size(distanceMetrics); ++i){
                            if(ImGui::MenuItem(distanceMetrics[i])) clusterSettings.distanceMetric = DataClusterer::DistanceMetric(i);
                        }
                        ImGui::EndCombo();
                    }

                    if(ImGui::InputInt("Amount of clusters", &clusterSettings.kmeansClusters)) clusterSettings.kmeansClusters = std::clamp(clusterSettings.kmeansClusters, 1, 100);
                    if(ImGui::InputInt("Amount of iterations", &clusterSettings.maxIterations)) clusterSettings.maxIterations = std::clamp(clusterSettings.maxIterations, 1, 100);
                    static char* initMethods[]{"Forgy", "Uniform Random", "Normal Random", "PlusPlus"};
                    if(ImGui::BeginCombo("Init method", initMethods[int(clusterSettings.kmeansInitMethod)])){
                        for(int i = 0; i < 4; ++i){
                            if(ImGui::MenuItem(initMethods[i])) clusterSettings.kmeansInitMethod = DataClusterer::InitMethod(i);
                        }
                        ImGui::EndCombo();
                    }
                    static char* kMethods[]{"Mean", "Median", "Mediod"};
                    if(ImGui::BeginCombo("K Method", kMethods[clusterSettings.kmeansMethod])){
                        for(int i = 0; i < 3; ++i)
                            if(ImGui::MenuItem(kMethods[i])) clusterSettings.kmeansMethod = DataClusterer::KMethod(i);
                        ImGui::EndCombo();
                    }
                    
                    ImGui::EndTabItem();
                }
                if(ImGui::BeginTabItem("DBScan")){
                    clusterMethod = DataClusterer::Method::DBScan;
                    ImGui::Text("Hier könnte jetzt ihre werbung stehen");
                    ImGui::EndTabItem();
                }
                if(ImGui::BeginTabItem("HClustering")){
                    clusterMethod = DataClusterer::Method::Hirarchical;
                    ImGui::Text("Hier könnte jetzt ihre werbung stehen");
                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }
            disabled = clusterer && !clusterer->clustered;
            if(disabled) ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            if(ImGui::Button("Cluster")){
                if(clusterer) delete clusterer;
                clusterer = new DataClusterer(projector->projectedPoints, clusterMethod, clusterSettings);
            }
            if(disabled) ImGui::PopItemFlag();
            if(clusterer) ImGui::Text("Cluster progress: %d%%", int(clusterer->progress * 100));

            // drawing a 2d scatterplot for the projected and clustered points
            if(projector && projector->projected){
                ImGui::Separator();
                ImVec2 min = ImGui::GetCursorScreenPos() , max = min + ImVec2(500,500);
                ImGui::RenderFrame(min, max, ImGui::GetColorU32(ImGuiCol_FrameBg), true, ImGui::GetStyle().FrameRounding);
                //rendering the projected points
                if(!clusterer || !clusterer->clustered){
                    for(int i = 0; i < projector->projectedPoints.rows(); ++i){
                        ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(projector->projectedPoints(i,0) * 500, projector->projectedPoints(i,1) * -500) + ImVec2(0,500) + min, 2, ImGui::ColorConvertFloat4ToU32(colors[0]));
                    }
                }
                else{
                    int clus = 0;
                    for(auto& c: clusterer->clusters){
                        for(auto i: c){
                            ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(projector->projectedPoints(i,0) * 500, projector->projectedPoints(i,1) * -500) + ImVec2(0,500) + min, 2, ImGui::ColorConvertFloat4ToU32(colors[clus % colors.size()]));
                        }
                        ++clus;
                    }
                }

            }
        }
        ImGui::End();
    }
    bool active = false;
    int projectionDimension = 2;
    DataProjector::ProjectionSettings projectionSettings{};
    DataProjector::Method projectorMethod = DataProjector::Method::PCA;
    DataProjector* projector = 0;
    DataClusterer::ClusterSettings clusterSettings{};
    DataClusterer::Method clusterMethod = DataClusterer::Method::KMeans;
    DataClusterer* clusterer = 0;

    std::vector<ImVec4> colors{{1,1,0,.2f}, {0,1,0,.2f}, {0,1,1,.2f}, {1,0,1,.2f}, {1,0,0,.2f}};

protected:
    const std::list<DataSet>& datasets;
    const std::vector<Attribute>& attributes;
    std::vector<uint8_t> activations;

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