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
        ImGui::Begin("Correlation Matrix", &active);
        int c = 0;
        for(auto& correlationMatrix: correlationMatrices){
            correlationMatrix.draw();
            requestUpdate |= correlationMatrix.requestCorrelationUpdate;
        }

        if(ImGui::Button("+##cormatwork", {500, 50})){
            correlationMatrices.emplace_back(correlationManager);
        } 
        ImGui::End();
    }
}

void CorrelationMatrixWorkbench::updateCorrelationScores(const std::list<DrawList>& dls, bool force)
{
    for(auto& mat: correlationMatrices){
        if(mat.requestCorrelationUpdate || force){
            mat.updateAllCorrelationScores(dls);
            mat.requestCorrelationUpdate = false;
        }
    }
    requestUpdate = false;
}

void CorrelationMatrixWorkbench::updateCorrelationScores(const std::list<DrawList>& dls, const std::vector<std::string>& dlNames){
    for(auto& mat:correlationMatrices){
        mat.updateAllCorrelationScores(dls, dlNames);
    }
}

CorrelationMatrixWorkbench::CorrelationMatrix::CorrelationMatrix(const std::shared_ptr<CorrelationManager>& correlationManager):correlationManager(correlationManager), id(idCounter++)
{
    
}

void CorrelationMatrixWorkbench::CorrelationMatrix::draw() 
{
    ImGui::BeginChild(("Correlation Matrix" + std::to_string(id)).c_str(), {}, true);
    ImGui::Text("Attribute activations:");
    for(int i = 0; i < attributes.size(); ++i){
        if(i != 0) ImGui::SameLine();
        if(ImGui::Checkbox((attributes[i].attribute + "##cor").c_str(), &attributes[i].active)){
            // well do something when attributes are changed
        }
    }
    ImGui::Text("Drawlists");
    if(ImGui::CollapsingHeader("Drawlists")){
        for(auto& drawlistRef: drawlists){
            if(ImGui::Checkbox((drawlistRef.drawlist + "##corMat").c_str(), &drawlistRef.active)){
                // well do some stuff when activatin changes
            }
        }
    }
    static char const* metrics[]{"Pearson", "SpearmanRank", "KendallRank"};
    if(ImGui::BeginCombo("Correlation Metric", metrics[static_cast<int>(currentMetric)])){
        for(int i = 0; i < 3; ++i){
            if(ImGui::MenuItem(metrics[i])) 
            {
                auto prevMetric = currentMetric;
                currentMetric = static_cast<CorrelationManager::CorrelationMetric>(i);
                requestCorrelationUpdate = prevMetric != currentMetric; // only reuqest when the metric changed
            }
        }
        ImGui::EndCombo();
    }
    // drawing the correlation matrix -------------------------------------
    // the multi correlation matrix itself depicts for each pair of attributes
    // a pie chart, where each pie correspoonds to the correlation for a single drawlist.
    // labels
    int activeAttributesCount = 0;
    for(auto& a: attributes) if(a.active) activeAttributesCount++;
    int activeDrawlistCount = 0;
    for(auto& d: drawlists) if(d.active) activeDrawlistCount++;
    float curY = ImGui::GetCursorPosY();
    float xSpacing = matrixPixelWidth / (activeAttributesCount - 1);
    float xSpacingMargin = xSpacing * (1.0 - matrixSpacing);
    const int leftSpace = 150;
    int curPlace = 0;
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + xSpacing / 2);
    int startInd = 1; while(attributes.size() && !attributes[startInd - 1].active) ++startInd;
    for(int i = startInd; i < attributes.size(); ++i){
        int curAttr = i;
        if(!attributes[curAttr].active || curPlace == activeAttributesCount - 1) continue;
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
    int curAttr = 1; while(attributes.size() && !attributes[curAttr - 1].active) ++curAttr;
    curY = matrixPos.y;
    for(int i = 0; i < activeAttributesCount - 1; ++i){
        while(!attributes[curAttr].active && curAttr < attributes.size()) ++curAttr;
        int curAttr2 = 0;
        for(int j = 0; j <= i; ++j){
            //boxes
            ImGui::GetWindowDrawList()->AddRect({curX, curY}, {curX + xSpacingMargin, curY + xSpacingMargin}, ImGui::GetColorU32(matrixBorderColor), 0, ImDrawCornerFlags_All, matrixBorderWidth);
            //pie
            while(!attributes[curAttr2].active && curAttr2 < attributes.size()) ++curAttr2;
            std::vector<float> percentages(activeDrawlistCount);
            std::vector<ImU32> colors(percentages.size());
            int d = 0;
            for(auto& dl: drawlists){
                if(!dl.active) continue;
                auto& corr = correlationManager->correlations[dl.drawlist];
                percentages[d] = 1.0 / activeDrawlistCount;
                int idx = (corr.attributeCorrelations[curAttr2].correlationScores[curAttr] * .5 + .5) * sizeof(diverging_blue_red_map) / 4;
                colors[d] = ImGui::GetColorU32({diverging_blue_red_map[idx * 4] / 255.0f,
                                                diverging_blue_red_map[idx * 4 + 1] / 255.0f,
                                                diverging_blue_red_map[idx * 4 + 2] / 255.0f,
                                                diverging_blue_red_map[idx * 4 + 3] / 255.0f});
                ++d;
            }
            
            ImGui::GetWindowDrawList()->AddPie(ImVec2{curX + xSpacingMargin / 2, curY + xSpacingMargin / 2}, xSpacingMargin / 2.0, colors.data(), percentages.data(), percentages.size());
            curX += xSpacing;
            ++curAttr2;
        }
        curX = matrixPos.x;
        curY += xSpacing;
        ++curAttr;
    }

    float curSpace = xSpacing / 2 + leftSpace;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + xSpacing / 2 + leftSpace);
    bool firstLabel = true;
    curPlace = 0;
    for(int i = 0; attributes.size() && i < attributes.size() - 1; ++i){
        if(!attributes[i].active || curPlace == activeAttributesCount - 1) continue;
        if(!firstLabel) ImGui::SameLine(curSpace); 
        if(firstLabel) firstLabel = false;
        ImGui::Text("%s", attributes[i].attribute.c_str());
        curSpace += xSpacing;
        ++curPlace;
    }

    ImGui::EndChild();
}

void CorrelationMatrixWorkbench::CorrelationMatrix::updateAllCorrelationScores(const std::list<DrawList>& dls, const std::vector<std::string>& drawlistNames)
{
    std::vector<std::string> deletes;
    for(auto& dl: drawlists){
        if(std::find_if(dls.begin(), dls.end(), [&](auto& d){return dl.drawlist == d.name;}) == dls.end()) deletes.push_back(dl.drawlist);
    }
    for(auto& s: deletes){
        drawlists.erase(std::find_if(drawlists.begin(), drawlists.end(), [&](auto& d){return s == d.drawlist;}));
        correlationManager->correlations.erase(s);
    }
    for(auto& dl: drawlists){
        // ignore drawlist if it is not in the drawlistNames vector
        if(drawlistNames.size() && std::find_if(drawlistNames.begin(), drawlistNames.end(), [&](auto& n){return n == dl.drawlist;}) == drawlistNames.end()) continue;
        auto d = std::find_if(dls.begin(), dls.end(), [&](auto& draw){return draw.name == dl.drawlist;});
        for(int i = 0; i < attributes.size(); ++i)
            correlationManager->calculateCorrelation(*d, currentMetric, i);
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
    if(std::find_if(drawlists.begin(), drawlists.end(), [&](auto& d){return d.drawlist == dl.name;}) == drawlists.end())
        drawlists.push_back({dl.name, true});
    for(int i = 0; i < attributes.size(); ++i) correlationManager->calculateCorrelation(dl, CorrelationManager::CorrelationMetric::Pearson, i);
}

uint32_t CorrelationMatrixWorkbench::CorrelationMatrix::idCounter = 0;