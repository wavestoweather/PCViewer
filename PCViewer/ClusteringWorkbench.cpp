#define NOSTATICS
#include "ClusteringWorkbench.hpp"
#undef NOSTATICS
#include "imgui/imgui_internal.h"

ClusteringWorkbench::ClusteringWorkbench(VkDevice device, const std::vector<Attribute>& attributes, std::list<DataSet>& datasets, std::list<DrawList>& drawLists): 
_device(device), datasets(datasets), attributes(attributes), _drawLists(drawLists), activations(attributes.size(), 1){
    clusterSettings.kmeansClusters = 10;
    clusterSettings.kmeansMethod = DataClusterer::KMethod::Mean;
    clusterSettings.kmeansInitMethod = DataClusterer::InitMethod::PlusPlus;
    clusterSettings.distanceMetric = DataClusterer::DistanceMetric::Norm;
    clusterSettings.maxIterations = 20;
    
    clusterSettings.dbscanEpsilon = .01f;
    clusterSettings.dbscanMinPoints = 10;
}

void ClusteringWorkbench::draw(){
    if(!active) return;
        if(ImGui::Begin("Clustering Workbench", &active)){
            ImGui::Text("Data Projection:");
            static int drawlistIndex = -1;
            static char const* defaultName = "Select drawlist";
            static char const* defaultTemplate = "Dataset has to be selected";
            auto dl = _drawLists.begin(); 
            if(drawlistIndex >= 0){
                std::advance(dl, drawlistIndex);
                _activations.resize(dl->data->size(), 1);
            } 
            if(ImGui::BeginCombo("Select drawlist", drawlistIndex >= 0 ? dl->name.c_str(): defaultName)){
                int i = 0;
                for(auto it = _drawLists.begin(); it != _drawLists.end(); ++it, ++i){
                    if(ImGui::MenuItem(it->name.c_str())){
                        drawlistIndex = i;
                    }
                }
                ImGui::EndCombo();
            }
            if(drawlistIndex < datasets.size() && drawlistIndex >= 0){
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
            bool disabled = drawlistIndex >= _drawLists.size() || drawlistIndex < 0 || (projector && !projector->projected && !projector->interrupted);
            if(disabled){ 
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
            }
            if(ImGui::Button("Project")){
                std::vector<uint32_t> indices;
                int c = 0;
                for(uint8_t n: activations){
                    if(n) indices.push_back(c);
                    ++c;
                }
                projector = std::make_shared<DataProjector>(*dl->data, projectionDimension, projectorMethod, projectionSettings, dl->indices, indices);
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
                    static char const* distanceMetrics[3]{"Norm", "SquaredNorm", "L1Norm"};
                    if(ImGui::BeginCombo("Distance Metric", distanceMetrics[(int)clusterSettings.distanceMetric])){
                        for(int i = 0; i < std::size(distanceMetrics); ++i){
                            if(ImGui::MenuItem(distanceMetrics[i])) clusterSettings.distanceMetric = DataClusterer::DistanceMetric(i);
                        }
                        ImGui::EndCombo();
                    }

                    if(ImGui::InputInt("Amount of clusters", &clusterSettings.kmeansClusters)) clusterSettings.kmeansClusters = std::clamp(clusterSettings.kmeansClusters, 1, 100);
                    if(ImGui::InputInt("Amount of iterations", &clusterSettings.maxIterations)) clusterSettings.maxIterations = std::clamp(clusterSettings.maxIterations, 1, 100);
                    static char const* initMethods[]{"Forgy", "Uniform Random", "Normal Random", "PlusPlus"};
                    if(ImGui::BeginCombo("Init method", initMethods[int(clusterSettings.kmeansInitMethod)])){
                        for(int i = 0; i < 4; ++i){
                            if(ImGui::MenuItem(initMethods[i])) clusterSettings.kmeansInitMethod = DataClusterer::InitMethod(i);
                        }
                        ImGui::EndCombo();
                    }
                    static char const* kMethods[]{"Mean", "Median", "Mediod"};
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
                    static char const* hLinkages[]{"Single", "Complete", "Weighted", "Median", "Average", "Ward", "Centroid"};
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
                clusterer = std::make_shared<DataClusterer>(projector->projectedPoints, clusterMethod, clusterSettings);
            }
            if(disabled) {
                ImGui::PopItemFlag();
                ImGui::PopStyleVar();
            }
            if(clusterer) ImGui::Text("Cluster progress: %d%%", int(clusterer->progress * 100));

            //------------------------------------------------------------------------------------------
            // drawing a 2d scatterplot for the projected and clustered points
            //------------------------------------------------------------------------------------------
            if(projector && projector->projected){
                ImGui::Separator();
                ImVec2 min = ImGui::GetCursorScreenPos() , max = min + ImVec2(projectPlotWidth,projectPlotWidth);
                ImGui::RenderFrame(min, max, ImGui::GetColorU32(ImGuiCol_FrameBg), true, ImGui::GetStyle().FrameRounding);
                //rendering the projected points
                if(!clusterer || !clusterer->clustered){
                    for(int i = 0; i < projector->projectedPoints.rows(); ++i){
                        ImU32 col = ImGui::ColorConvertFloat4ToU32(colors[0]);
                        if(!_activations[dl->indices[i]]) col = ImGui::ColorConvertFloat4ToU32(inactiveColor);
                        ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(projector->projectedPoints(i,0) * projectPlotWidth, projector->projectedPoints(i,1) * -projectPlotWidth) + ImVec2(0,projectPlotWidth) + min, 2, col);
                    }
                }
                else{
                    int clus = 0;
                    for(auto& c: clusterer->clusters){
                        for(auto i: c){
                            ImU32 col = ImGui::ColorConvertFloat4ToU32(colors[clus % colors.size()]);
                            if(!_activations[dl->indices[i]]) col = ImGui::ColorConvertFloat4ToU32(inactiveColor);
                            ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(projector->projectedPoints(i,0) * projectPlotWidth, projector->projectedPoints(i,1) * -projectPlotWidth) + ImVec2(0,projectPlotWidth) + min, 2, col);
                        }
                        ++clus;
                    }
                }
                //manging the lasso selection
                //drawing lasso
                if(lassoSelection.borderPoints.size()){
                    for(int i = 1; i < lassoSelection.borderPoints.size(); ++i){
                        ImVec2 a{lassoSelection.borderPoints[i - 1].x * projectPlotWidth, lassoSelection.borderPoints[i - 1].y * -projectPlotWidth + projectPlotWidth};
                        ImVec2 b{lassoSelection.borderPoints[i].x * projectPlotWidth, lassoSelection.borderPoints[i].y * -projectPlotWidth + projectPlotWidth};
                        a = a + min; b = b + min;
                        ImGui::GetWindowDrawList()->AddLine(a, b, ImGui::GetColorU32({0,0,1,1}), 2);
                    }
                    ImVec2 a{lassoSelection.borderPoints[0].x * projectPlotWidth, lassoSelection.borderPoints[0].y * -projectPlotWidth + projectPlotWidth};
                    ImVec2 b{lassoSelection.borderPoints.back().x * projectPlotWidth, lassoSelection.borderPoints.back().y * -projectPlotWidth + projectPlotWidth};
                    a = a + min; b = b + min;
                    ImGui::GetWindowDrawList()->AddLine(a, b, ImGui::GetColorU32({0,0,1,1}), 2);
                }
                //checking for lasso selection
                ImVec2 mousePos = ImGui::GetMousePos();
                bool inside = mousePos.x > min.x && mousePos.x < max.x && mousePos.y > min.y && mousePos.y < max.y && ImGui::IsWindowFocused();
                static ImVec2 prevPointsPos;
                static bool drawingLasso = false;
                if(ImGui::IsMouseClicked(0) && inside && !drawingLasso){    //begin lasso selection
                    drawingLasso = true;
                    lassoSelection.borderPoints.clear();
                }
                if(drawingLasso){
                    if(!ImGui::IsMouseDown(0)){ //stop lasso selection
                        if(lassoSelection.borderPoints.size() < 3) lassoSelection.borderPoints.clear();
                        drawingLasso = false;
                        //update active lines
                        updateActiveLines(*dl);
                    }
                    else{
                        ImVec2 relativePos{mousePos - min};
                        relativePos.y = projectPlotWidth - relativePos.y;   //invert y position
                        if(lassoSelection.borderPoints.empty()){//first point
                            lassoSelection.borderPoints.push_back({relativePos.x / projectPlotWidth, relativePos.y / projectPlotWidth});
                            prevPointsPos = mousePos;
                        }
                        else if(PCUtil::distance2(mousePos, prevPointsPos) > 25){   //if distance is high enough, spawn new point
                            lassoSelection.borderPoints.push_back({relativePos.x / projectPlotWidth, relativePos.y / projectPlotWidth});
                            prevPointsPos = mousePos;
                        }
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
                    tl.buffer = dl->buffer;
                    tl.indices = clusterer->clusters[i];
                    tl.minMax = std::vector<std::pair<float, float>>(attributes.size(), std::pair{std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()});
                    for(int a = 0; a < attributes.size(); ++a){
                        for(auto index: clusterer->clusters[i]){
                            const auto& d = (*dl->data)(index, a);
                            if(d < tl.minMax[a].first) tl.minMax[a].first = d;
                            if(d > tl.minMax[a].second) tl.minMax[a].second = d;
                        }
                    }
                    tl.pointRatio = clusterer->clusters[i].size() / float(dl->data->size());
                    tl.parentDataSetName = dl->parentDataSet;
                    auto ds = std::find_if(datasets.begin(), datasets.end(), [&](const DataSet& d){return d.name == dl->parentDataSet;});
                    ds->drawLists.push_back(tl);
                }
            }
        }
    ImGui::End();
}

void ClusteringWorkbench::updateActiveLines(DrawList& dl){
    //VkUtil::downloadData(_device, dl.dlMem, dl.activeIndicesBufferOffset, _activations.size(), _activations.data());
    std::fill(_activations.begin(), _activations.end(), 1); //reset all points to active
    dl.activeLinesAmt = 0;
    for(int i = 0; i < dl.indices.size() && lassoSelection.borderPoints.size(); ++i){
        auto row = projector->projectedPoints.row(i);
        //lasso selection check only for first 2 dimensions
        bool inLasso = false;
        for(int a = 0; a  < lassoSelection.borderPoints.size(); ++a){
            ImVec2 aP = lassoSelection.borderPoints[a];
            int b = (a + 1) % lassoSelection.borderPoints.size();
            ImVec2 bP = lassoSelection.borderPoints[b];
            // intersection check via https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
            if( ((aP.y > row[1]) != (bP.y > row[1])) &&
                (row[0] < (bP.x - aP.x) * (row[1] - aP.y) / (bP.y - aP.y) + aP.x)){
                inLasso = !inLasso;
            }
        }
        _activations[dl.indices[i]] &= inLasso;
        dl.activeLinesAmt += _activations[dl.indices[i]];
    }
    requestPcPlotUpdate = true;
    updateDl = &dl;
    VkUtil::uploadData(_device, dl.dlMem, dl.activeIndicesBufferOffset, _activations.size(), _activations.data());
}
