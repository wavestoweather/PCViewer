#include "DeriveWorkbench.hpp"
#include "../imgui/imgui.h"
#include "../imgui/imgui_internal.h"
#include "../imgui_nodes/imgui_node_editor.h"
#include "Nodes.hpp"
#include "ExecutionGraph.hpp"

namespace nodes = ax::NodeEditor;
void DeriveWorkbench::show() 
{
    if(!active)
        return;
    ImGui::Begin("DeriveWorkbench", &active);
    nodes::SetCurrentEditor(_editorContext);
    nodes::Begin("DeriveWorkbench");


    auto& editorStyle = nodes::GetStyle();
    static int nodeId = 0;
    const ImVec4 headerColor{.1,.1,.1,1};
    
    auto cursorTopLeft = ImGui::GetCursorStartPos();
    for(auto& [id, node]: _executionGraphs[0].nodes){
        nodes::BeginNode(id);
        // drawing header
        ImGui::Text(node->name.c_str());
        ImGui::Separator();
        
        // drawing middle
        for(int i: irange(node->inputTypes)){
            auto alpha = ImGui::GetStyle().Alpha;

            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
            ax::Widgets::IconType iconType = node->inputTypes[i]->iconType();
            auto c = node->inputTypes[i]->color;
            auto color = ImColor(c[0], c[1], c[2], c[3]);
            nodes::BeginPin();
            ax::Widgets::Icon({pinIconSize, pinIconSize}, iconType, true, color, ImColor(32, 32, 32, alpha));

            nodes::EndPin();

        }

        // drawing end

        nodes::EndNode();
    }

    nodes::End();
    ImGui::End();
}

void DeriveWorkbench::addDataset(std::string_view datasetId) 
{
    
}

void DeriveWorkbench::removeDataset(std::string_view datasetId) 
{
    
}

DeriveWorkbench::DeriveWorkbench() 
{
    _editorContext = ax::NodeEditor::CreateEditor();
    _executionGraphs.resize(1);
}

DeriveWorkbench::~DeriveWorkbench() 
{
    ax::NodeEditor::DestroyEditor(_editorContext);
}
