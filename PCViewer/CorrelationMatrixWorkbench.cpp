#define NOSTATICS
#include "CorrelationMatrixWorkbench.hpp"
#undef NOSTATICS
#include "imgui/imgui.h"
#include "ColorMaps.hpp"

CorrelationMatrixWorkbench::CorrelationMatrixWorkbench(const VkUtil::Context& vkContext):vkContext(vkContext)
{
    correlationManager = std::make_shared<CorrelationManager>(vkContext);
}

void CorrelationMatrixWorkbench::draw()
{
    if(active){
        ImGui::Begin("Correlation Matrix");
        
        for(auto& correlationMatrix: correlationMatrices){
            correlationMatrix.draw();
        }

        if(ImGui::Button("+##cormatwork", {500, 150})){
            correlationMatrices.emplace_back(correlationManager);
        }

        ImGui::End();
    }
}

CorrelationMatrixWorkbench::CorrelationMatrix::CorrelationMatrix(std::shared_ptr<CorrelationManager>& CorrelationManager):correlationManager(correlationManager), id(idCounter++)
{
    
}

void CorrelationMatrixWorkbench::CorrelationMatrix::draw() 
{
    if(ImGui::CollapsingHeader("Drawlists")){
        for(auto& drawlistRef: drawlists){
            if(ImGui::Checkbox((drawlistRef.drawlist + "##corMat").c_str(), &drawlistRef.active)){
                // well do some stuff when activatin changes
            }
        }
    }
    // drawing the correlation matrix -------------------------------------
    // the multi correlation matrix itself depicts for each pair of attributes
    // a pie chart, where each pie correspoonds to the correlation for a single drawlist.
    // labels
    int activeAttributesCount = 0;
    for(auto& a: attributes) if(a.active) activeAttributesCount++;
    float curY = ImGui::GetCursorPosY();
    float xSpacing = matrixPixelWidth / (activeAttributesCount - 1);
    float xSpacingMargin = xSpacing * (1.0 - matrixSpacing);
    const int leftSpace = 150;
    int curPlace = 0;
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + xSpacing / 2);
    int startInd = 1; while(!attributes[startInd - 1].active) ++startInd;
    for(int i = startInd; i < attributes.size(); ++i){
        int curAttr = i;
        if(attributes[curAttr].active || curPlace == activeAttributesCount - 1) continue;
        ImGui::Text("%s", attributes[curAttr].attribute.c_str());
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + xSpacing - ImGui::GetTextLineHeightWithSpacing());
        ++curPlace;
    }
    ImGui::SetCursorPosY(curY);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + leftSpace);
    ImVec2 matrixPos = ImGui::GetCursorScreenPos();
    ImVec2 matrixSize{float(matrixPixelWidth), float(matrixPixelWidth)};
    // invisible button to have a drop target to drop the drawlists onto
    ImGui::InvisibleButton(("cm" + std::to_string(id)).c_str(),matrixSize);
    if(ImGui::BeginDragDropTarget()){
        if(const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Drawlist")){
            DrawList* dl = *((DrawList**)payload->Data);
            addDrawlist(*dl);
        }
    }
    // drawing the correlation pies
    float curX = matrixPos.x;
    int curAttr = 1; while(!attributes[curAttr - 1].active) ++curAttr;
    curY = matrixPos.y;
    for(int i = 0; i < activeAttributesCount - 1; ++i){
        while(!attributes[curAttr].active && curAttr < attributes.size()) ++curAttr;
        int curAttr2 = 0;
        for(int j = 0; j <= i; ++j){
            //boxes
            ImGui::GetWindowDrawList()->AddRect({curX, curY}, {curX + xSpacingMargin, curY + xSpacingMargin}, ImGui::GetColorU32(matrixBorderColor), 0, ImDrawCornerFlags_All, matrixBorderWidth);
            //pie
            if(correlationManager->correlations.size());
            std::vector<float> percentages(correlationManager->correlations.size());
            std::vector<ImU32> colors(percentages.size());
            int d = 0;
            for(auto& dlCorr: correlationManager->correlations){
                percentages[d] = 1.0 / correlationManager->correlations.size();
                int idx = (dlCorr.second.attributeCorrelations[curAttr].correlationScores[curAttr2] * .5 + .5) * sizeof(diverging_blue_red_map) / 4;
                colors[d] = ImGui::GetColorU32({diverging_blue_red_map[idx * 4] / 255.0f,
                                                diverging_blue_red_map[idx * 4 + 1] / 255.0f,
                                                diverging_blue_red_map[idx * 4 + 2] / 255.0f,
                                                diverging_blue_red_map[idx * 4 + 3] / 255.0f});
                ++d;
            }
            
            ImGui::GetWindowDrawList()->AddPie(ImVec2{curX + xSpacingMargin / 2, curY + xSpacingMargin / 2}, xSpacingMargin / 2.0, colors.data(), percentages.data(), percentages.size());
        }
    }
}

void CorrelationMatrixWorkbench::CorrelationMatrix::addDrawlist(const DrawList& dl) 
{
    if(attributes.empty()){
        attributes.resize(dl.attributes->size());
        for(int i = 0; i < attributes.size(); ++i){
            attributes[i].attribute = dl.attributes->at(i).name;
            attributes[i].active = true;
        }
    }
    drawlists.push_back({dl.name, true});
    for(int i = 0; i < attributes.size(); ++i) correlationManager->calculateCorrelation(dl, CorrelationManager::CorrelationMetric::Pearson, i);
}

uint32_t CorrelationMatrixWorkbench::CorrelationMatrix::idCounter = 0;