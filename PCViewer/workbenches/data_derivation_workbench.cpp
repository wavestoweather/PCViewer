#include "data_derivation_workbench.hpp"
#include <logger.hpp>
#include "../imgui_nodes/utilities/builders.h"
#include <Nodes.hpp>
#include <datasets.hpp>
#include <imgui_stdlib.h>

namespace workbenches{
namespace nodes = ax::NodeEditor;

data_derivation_workbench::data_derivation_workbench(std::string_view id):
    workbench(id)
{

}

void data_derivation_workbench::show(){
    if(!active)
        return;
    const static std::string_view some_menu_id{};

    ImGui::SetNextWindowSize({800, 800}, ImGuiCond_Once);
    ImGui::Begin(id.c_str(), &active);

    if(ImGui::Button("Execute Graph")){
        try{
            execute_graph(main_execution_graph_id);
        }
        catch(const std::runtime_error& e){
            logger << logging::error_prefix << " data_derivation_workbench::execute_graph() " << e.what() << logging::endl;
        }
    }
    nodes::SetCurrentEditor(_editor_context);
    nodes::Begin("Derivation workbench");

    auto&           editor_style = nodes::GetStyle();
    const ImVec4    header_color{.1,.1,.1,1};
    const float     pin_icon_size = 15;

    auto& [nodes, pint_to_nodes, links, link_to_connection, pint_to_links] = *_execution_graphs[std::string(main_execution_graph_id)];

    nodes::Utilities::BlueprintNodeBuilder builder;
    if(nodes.count(0))
        nodes.erase(0);
    for(auto& [id, node_pins]: nodes){
        assert(id > 0);
        auto& node = node_pins.node;
        builder.Begin(id);
        // header
        if(node->name.size()){
            builder.Header();
            ImGui::Spring(0);
            ImGui::TextUnformatted(node->name.c_str());
            ImGui::Dummy({0,28});
            builder.EndHeader();
        }
        // inputs
        for(int i: irange(node->inputTypes)){
            if(i >= node->inputTypes.size())
                break;
            builder.Input(node_pins.inputIds[i]);
            auto alpha = ImGui::GetStyle().Alpha;
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
            bool is_linked = pint_to_links[node_pins.inputIds[i]].size() > 0;
            ax::Widgets::Icon({pin_icon_size, pin_icon_size}, node->inputTypes[i]->iconType(), is_linked, node->inputTypes[i]->color(), ImColor(32, 32, 32, int(alpha * 255)));
            ImGui::Spring(0);
            if(deriveData::Nodes::VariableInput* variable_input = dynamic_cast<deriveData::Nodes::VariableInput*>(node.get())){
                ImGui::PushItemWidth(100);
                if(i > variable_input->minNodes){
                    if(ImGui::InputText(("##ns" + std::to_string(i)).c_str(), &node->inputNames[i])){
                        if(deriveData::Nodes::DatasetOutput* ds_output = dynamic_cast<deriveData::Nodes::DatasetOutput*>(node.get()))
                            globals::datasets()[ds_output->datasetId]().attributes[i].display_name = node->inputNames[i]; 
                    }
                }
                else
                ImGui::TextUnformatted(node->inputNames[i].c_str());
                ImGui::PopItemWidth();
                ImGui::Spring(0);
                if(i > variable_input->minNodes && ImGui::Button(("X##p" + std::to_string(node_pins.inputIds[i])).c_str())){
                    if(deriveData::Nodes::DatasetOutput* ds_output = dynamic_cast<deriveData::Nodes::DatasetOutput*>(node.get())){
                        
                    }
                }
            }
            
        }
        
    }

    ImGui::End();
}
}