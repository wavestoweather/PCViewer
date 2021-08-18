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
    ClusteringWorkbench(const std::vector<Attribute>& attributes, std::list<DataSet>& datasets): datasets(datasets), attributes(attributes), activations(attributes.size(), 1){
        clusterSettings.kmeansClusters = 10;
        clusterSettings.kmeansInitMethod = DataClusterer::InitMethod::PlusPlus;
        clusterSettings.maxIterations = 20;
        
        clusterSettings.dbscanEpsilon = .01f;
        clusterSettings.dbscanMinPoints = 10;
    }

    //draws a standard imgui window with all functionalyties for the clustering workbench
    void draw(){
        if(!active) return;
        if(ImGui::Begin("Clustering Workbench", &active)){
            ImGui::Text("Data Projection:");
            static int datasetIndex = 0;
            static int templateListIndex = 0;
            static char* defaultName = "No datasets available";
            static char* defaultTemplate = "Dataset has to be selected";
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
            std::list<TemplateList>::iterator tl;
            if(datasetIndex < datasets.size()){
                templateListIndex = std::min(templateListIndex, int(ds->drawLists.size() - 1));
                tl = ds->drawLists.begin(); std::advance(tl, templateListIndex);
            }
            if(ImGui::BeginCombo("Templatelist to cluster", datasetIndex < datasets.size() ? tl->name.c_str() : defaultTemplate)){
                int c = 0;
                for(auto& tl: ds->drawLists){
                    if(ImGui::MenuItem(tl.name.c_str())){
                        templateListIndex = c;
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
            if(ImGui::InputInt("Reduction dimension", &projectionDimension)) projectionDimension = std::max(projectionDimension, 1);
            if(ImGui::BeginTabBar("ProjectionTabBar")){
                if(ImGui::BeginTabItem("PCA")){
                    projectorMethod = DataProjector::Method::PCA;
                    ImGui::Text("No special settings for PCA (But its fast ;) )...");
                    ImGui::EndTabItem();
                }
                if(ImGui::BeginTabItem("t-SNE")){
                    projectorMethod = DataProjector::Method::TSNE;
                    ImGui::Text("NOTE: Very slow for large datasets");
                    if(ImGui::InputInt("Max Iteration", &projectionSettings.maxIter)) projectionSettings.maxIter = std::clamp(projectionSettings.maxIter, 1, 10000);
                    ImGui::InputInt("Stop lying iteration (Stop lying about similarity values)", &projectionSettings.stopLyingIter);
                    ImGui::InputInt("Momentum switch iteration (Switching momentum for faster convergence)", &projectionSettings.momSwitchIter);
                    if(ImGui::InputDouble("Perplexity (Variation scale, small value->local variations dominate)", &projectionSettings.perplexity)) std::clamp(projectionSettings.perplexity, 1.0, 100.0);
                    if(ImGui::InputDouble("Theta (Approximation: 0->exact, >>0->less exact)", &projectionSettings.theta)) projectionSettings.theta = std::clamp(projectionSettings.theta, .0, 100.0);
                    ImGui::Checkbox("Skip Random Init", &projectionSettings.skipRandomInit);
                    ImGui::InputInt("Random Seed (negativ for time as seed)", &projectionSettings.randSeed);
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
            bool disabled = datasetIndex >= datasets.size() || (projector && !projector->projected && !projector->interrupted);
            if(disabled){ 
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
            }
            if(ImGui::Button("Project")){
                if(projector) delete projector;
                if(clusterer) delete clusterer;
                std::vector<uint32_t> indices;
                int c = 0;
                for(uint8_t n: activations){
                    if(n) indices.push_back(c);
                    ++c;
                }
                projector = new DataProjector(datasets.front().data, projectionDimension, projectorMethod, projectionSettings, tl->indices, indices);
            }
            if(disabled) {
                ImGui::PopItemFlag();
                ImGui::PopStyleVar();
            }
            if(projector) ImGui::Text("Projectioin progress: %d%%", int(projector->progress * 100));

            ImGui::Separator();

            ImGui::Text("Clustering Settings");
            if(ImGui::BeginTabBar("ClusteringTabBar")){
                if(ImGui::BeginTabItem("KMeans")){
                    clusterMethod = DataClusterer::Method::KMeans;
                    static char* distanceMetrics[3]{"Norm", "SquaredNorm", "L1Norm"};
                    if(ImGui::BeginCombo("Distance Metric", distanceMetrics[(int)clusterSettings.distanceMetric])){
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
                    if(ImGui::BeginCombo("K Method", kMethods[(int)clusterSettings.kmeansMethod])){
                        for(int i = 0; i < 3; ++i)
                            if(ImGui::MenuItem(kMethods[i])) clusterSettings.kmeansMethod = DataClusterer::KMethod(i);
                        ImGui::EndCombo();
                    }
                    
                    ImGui::EndTabItem();
                }
                if(ImGui::BeginTabItem("DBScan")){
                    clusterMethod = DataClusterer::Method::DBScan;
                    if(ImGui::InputFloat("Epsilon(max point distance)", &clusterSettings.dbscanEpsilon,0,0,"%f")) clusterSettings.dbscanEpsilon = std::clamp(clusterSettings.dbscanEpsilon, 1e-6f, 100.0f);
                    if(ImGui::InputInt("Min Points(Minimum number of points for a cluster)", &clusterSettings.dbscanMinPoints)) clusterSettings.dbscanMinPoints = std::clamp(clusterSettings.dbscanMinPoints, 1, 1000);
                    ImGui::EndTabItem();
                }
                if(ImGui::BeginTabItem("HClustering")){
                    clusterMethod = DataClusterer::Method::Hirarchical;
                    if(ImGui::InputInt("ClusterAmt", &clusterSettings.hclusteringClusters)) clusterSettings.hclusteringClusters = std::clamp(clusterSettings.hclusteringClusters, 1, 100);
                    static char* hLinkages[]{"Single", "Complete", "Weighted", "Median", "Average", "Ward", "Centroid"};
                    if(ImGui::BeginCombo("Clustering Linkage", hLinkages[(int)clusterSettings.hclusteringLinkage])){
                        for(int i = 0; i < 7; ++i){
                            if(ImGui::MenuItem(hLinkages[i])) clusterSettings.hclusteringLinkage = DataClusterer::HClusteringLinkage(i);
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }
            disabled = clusterer && !clusterer->clustered || !projector || (!projector->projected || projector->interrupted);
            if(disabled){ 
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
            }
            if(ImGui::Button("Cluster") && projector &&projector->projected){
                if(clusterer) delete clusterer;
                clusterer = new DataClusterer(projector->projectedPoints, clusterMethod, clusterSettings);
            }
            if(disabled) {
                ImGui::PopItemFlag();
                ImGui::PopStyleVar();
            }
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

            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 500 + 5);

            static char clusterName[200]{"Cluster"};
            ImGui::InputText("Cluster base name", clusterName, sizeof(clusterName)); ImGui::SameLine();
            if(ImGui::Button("Convert clusters to indexlist") && clusterer && clusterer->clustered){
                for(int i = 0; i < clusterer->clusters.size(); ++i){
                    //ds->drawLists.push_back(TemplateList{})
                    TemplateList tl{};
                    tl.name = clusterName + std::to_string(i);
                    tl.buffer = ds->buffer.buffer;
                    tl.indices = clusterer->clusters[i];
                    tl.minMax = std::vector<std::pair<float, float>>(attributes.size(), std::pair{std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()});
                    for(int a = 0; a < attributes.size(); ++a){
                        for(auto index: clusterer->clusters[i]){
                            const auto& d = ds->data(index, a);
                            if(d < tl.minMax[a].first) tl.minMax[a].first = d;
                            if(d > tl.minMax[a].second) tl.minMax[a].second = d;
                        }
                    }
                    tl.pointRatio = clusterer->clusters[i].size() / float(ds->data.size());
                    tl.parentDataSetName = ds->name;
                    ds->drawLists.push_back(tl);
                }
            }
        }
        ImGui::End();
    }
    bool active = false;
    int projectionDimension = 2;
    DataProjector::ProjectionSettings projectionSettings{20.0, 1.0, -1, 100, 0, 700, false};
    DataProjector::Method projectorMethod = DataProjector::Method::PCA;
    DataProjector* projector = 0;
    DataClusterer::ClusterSettings clusterSettings;
    DataClusterer::Method clusterMethod = DataClusterer::Method::KMeans;
    DataClusterer* clusterer = 0;

    std::vector<ImVec4> colors{{1,1,0,.2f}, {0,1,0,.2f}, {0,1,1,.2f}, {1,0,1,.2f}, {1,0,0,.2f}};

protected:
    std::list<DataSet>& datasets;
    const std::vector<Attribute>& attributes;
    std::vector<uint8_t> activations;
};