#include "DeriveWorkbench.hpp"
#include "../imgui/imgui.h"
#include "../imgui_nodes/imgui_node_editor.h"
#include "Nodes.hpp"

void DeriveWorkbench::show() 
{
    if(!active)
        return;
    ImGui::Begin("DeriveWorkbench", &active);
    ImGui::End();
}

void DeriveWorkbench::addDataset(std::string_view datasetId) 
{
    
}

void DeriveWorkbench::removeDataset(std::string_view datasetId) 
{
    
}
