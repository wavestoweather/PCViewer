#include "data_derivation_workbench.hpp"
#include <logger.hpp>
#include "../imgui_nodes/utilities/builders.h"
#include <Nodes.hpp>
#include <datasets.hpp>
#include <imgui_stdlib.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>

namespace workbenches{
namespace nodes = ax::NodeEditor;

void show_label(std::string_view label, ImColor color){
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - ImGui::GetTextLineHeight());
    auto size = ImGui::CalcTextSize(label.data());
    auto padding = ImGui::GetStyle().FramePadding;
    auto spacing = ImGui::GetStyle().ItemSpacing;
    ImGui::SetCursorPos(ImGui::GetCursorPos() + ImVec2(spacing.x, -spacing.y));
    auto rectMin = ImGui::GetCursorScreenPos() - padding;
    auto rectMax = ImGui::GetCursorScreenPos() + size + padding;
    auto drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(rectMin, rectMax, color, size.y * 0.15f);
    ImGui::TextUnformatted(label.data());
}

data_derivation_workbench::data_derivation_workbench(std::string_view id):
    workbench(id)
{
    _execution_graphs.emplace("main", std::make_unique<ExecutionGraph>());
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

    auto& [nodes, pin_to_nodes, links, link_to_connection, pint_to_links] = *_execution_graphs[std::string(main_execution_graph_id)];

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
                        auto& ds = globals::datasets()[ds_output->datasetId]();
                        ds.attributes.erase(ds.attributes.begin() + i);
                        std::visit([i](auto&&  data){
                            data.columns.erase(data.columns.begin() + 1);
                            data.column_dimensions.erase(data.column_dimensions.begin() + 1);
                            }, ds.cpu_data());
                    }
                    _execution_graphs[std::string(main_execution_graph_id)]->removePin(node_pins.inputIds[i], true);
                }
                ImGui::Spring(0);
            }
            else if(node->inputNames[i].size()){
                ImGui::TextUnformatted(node->inputNames[i].c_str());
                ImGui::Spring(0);
            }

            if(!is_linked && dynamic_cast<deriveData::Nodes::Output*>(node.get()) == nullptr){
                auto memory_view = node->inputTypes[i]->data();
                if(memory_view.cols.size()){
                    switch(memory_view.cols.size()){
                        case 1:
                            ImGui::PushItemWidth(50);
                            ImGui::InputFloat("##test", memory_view.cols[0].data());
                            ImGui::PopItemWidth();
                            break;
                        case 2:
                            ImGui::PushItemWidth(100);
                            ImGui::InputFloat2("##inputv2", memory_view.cols[0].data());
                            ImGui::PopItemWidth();
                            break;
                        case 3:
                            ImGui::PushItemWidth(150);
                            ImGui::InputFloat3("##inputv3", memory_view.cols[0].data());
                            ImGui::PopItemWidth();
                            break;
                        case 4:
                            ImGui::PushItemWidth(200);
                            ImGui::InputFloat4("##inputv4", memory_view.cols[0].data());
                            ImGui::PopItemWidth();
                            break;
                    }
                }
                ImGui::Spring(0);
            }
            ImGui::PopStyleVar();
            builder.EndInput();
        }
        if(deriveData::Nodes::VariableInput* variable_input = dynamic_cast<deriveData::Nodes::VariableInput*>(node.get())){
            if(node_pins.inputIds.size() < variable_input->maxNodes && ImGui::Button("Add Pin")){
                std::string number = std::to_string(node_pins.inputIds.size());
                if(deriveData::Nodes::DatasetOutput* ds_output = dynamic_cast<deriveData::Nodes::DatasetOutput*>(node.get())){
                    auto& ds = globals::datasets()[ds_output->datasetId]();
                    ds.attributes.push_back(structures::attribute{number, number, structures::change_tracker<structures::min_max<float>>{structures::min_max<float>{-.1f, .1f}}, {}});
                    std::visit([](auto&& data){
                        data.columns.push_back({0});
                        data.column_dimensions.push_back({});
                    }, ds.cpu_data());
                }
                _execution_graphs[std::string(main_execution_graph_id)]->addPin(_cur_id, id, number, deriveData::FloatType::create(), true);
            }
        }

        // middle
        builder.Middle();
        if(deriveData::Nodes::DatasetInput* dataset_input = dynamic_cast<deriveData::Nodes::DatasetInput*>(node.get())){
            ImGui::Spring(1, 0);
            ImGui::TextUnformatted("Choose Dataset:");
            ImGui::PushItemWidth(150);
            if(nodes::BeginNodeCombo("##input", dataset_input->datasetId.data(), 0, 1.f / nodes::GetCurrentZoom())){
                for(const auto& [ds_id, ds]: globals::datasets.read()){
                    if(ImGui::MenuItem(ds_id.data())){
                        dataset_input->datasetId = ds_id;
                        std::string main_string(main_execution_graph_id);
                        while(node_pins.outputIds.size())
                            _execution_graphs[main_string]->removePin(node_pins.outputIds[0], false);
                        
                        for(const auto& a: ds.read().attributes)
                            _execution_graphs[main_string]->addPin(_cur_id, id, a.display_name, deriveData::FloatType::create(), false);
                    }
                }
                nodes::EndNodeCombo();
            }
            ImGui::PopItemWidth();
        }
        if(deriveData::Nodes::DatasetOutput* dataset_output = dynamic_cast<deriveData::Nodes::DatasetOutput*>(node.get())){
            ImGui::Spring(1, 0);
            ImGui::TextUnformatted("Choose Dataset:");
            ImGui::PushItemWidth(150);
            if(nodes::BeginNodeCombo("##output", dataset_output->datasetId.data(), 0, 1.f / nodes::GetCurrentZoom())){
                for(const auto& [ds_id, ds]: globals::datasets.read()){
                    if(ImGui::MenuItem(ds_id.data())){
                        dataset_output->datasetId = ds_id;
                        std::string main_string(main_execution_graph_id);
                        while(node_pins.inputIds.size())
                            _execution_graphs[main_string]->removePin(node_pins.inputIds[0], true);

                        for(const auto& a: ds.read().attributes)
                            _execution_graphs[main_string]->addPin(_cur_id, id, a.display_name, deriveData::FloatType::create(), true);

                        dynamic_cast<deriveData::Nodes::VariableInput*>(node.get())->minNodes = ds.read().original_attribute_size - 1;
                    }
                }
                nodes::EndNodeCombo();
            }
            ImGui::PopItemWidth();
        }
        ImGui::Spring(1, 0);
        ImGui::TextUnformatted(node->middleText.c_str());
        ImGui::Spring(1, 0);

        // outputs
        for(int i: util::size_range(node->outputTypes)){
            builder.Output(node_pins.outputIds[i]);
            auto alpha = ImGui::GetStyle().Alpha;
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
            bool is_linked = pint_to_links[node_pins.outputIds[i]].size() > 0;
            if(node->outputNames[i].size()){
                ImGui::TextUnformatted(node->outputNames[i].c_str());
                ImGui::Spring(0);
            }
            ax::Widgets::Icon({pin_icon_size, pin_icon_size}, node->outputTypes[i]->iconType(), is_linked, node->outputTypes[i]->color(), ImColor(32, 32, 32, int(alpha * 255)));
            ImGui::Spring(0);
            ImGui::PopStyleVar();
            builder.EndOutput();
        }
        builder.End();
    }

    // nodes drawing
    for(const auto& [connection, link]: links)
        nodes::Link(link.Id, link.pinAId, link.pinBId, link.color, 2);

    // handle creation action
    bool this_frame_create{false};
    if(!_create_new_node){
        if(nodes::BeginCreate()){
            nodes::PinId a, b;
            if(nodes::QueryNewNode(&a)){
                show_label("+ Create Node", {32, 45, 32, 180});

                if(nodes::AcceptNewItem()){
                    _create_new_node = true;
                    _new_link_pin_id = a.Get();
                    this_frame_create = true;
                    nodes::Suspend();
                    ImGui::OpenPopup("Create New Node");
                    nodes::Resume();
                }
            }

            a = {};
            if(nodes::QueryNewLink(&a, &b)){
                if(a && b){
                    auto& node_a_inputs = nodes[pin_to_nodes[a.Get()]].inputIds;
                    bool change = std::count_if(node_a_inputs.begin(), node_a_inputs.end(), [&a](int i){return i == a.Get();}) > 0;
                    if(change)
                        std::swap(a, b);
                    std::string main_string(main_execution_graph_id);
                    Link::Connection connection{};
                    connection.nodeAId = _execution_graphs[main_string]->pinToNodes[a.Get()];
                    connection.nodeBId = _execution_graphs[main_string]->pinToNodes[b.Get()];
                    auto& nodeAOutput = nodes[pin_to_nodes[a.Get()]].outputIds;
                    auto& nodeBInput = nodes[pin_to_nodes[b.Get()]].inputIds;
                    connection.nodeAAttribute = std::find(nodeAOutput.begin(), nodeAOutput.end(), a.Get()) - nodeAOutput.begin();
                    connection.nodeBAttribute = std::find(nodeBInput.begin(), nodeBInput.end(), b.Get()) - nodeBInput.begin();

                    bool wrongType = connection.nodeAAttribute < nodeAOutput.size() && connection.nodeBAttribute < nodeBInput.size() && typeid(*nodes[pin_to_nodes[a.Get()]].node->outputTypes[connection.nodeAAttribute]) != typeid(*nodes[pin_to_nodes[b.Get()]].node->inputTypes[connection.nodeBAttribute]);

                    bool inputToOutput = is_input_pin(a.Get()) ^ is_input_pin(b.Get());

                    if(wrongType)
                        show_label("Incompatible types", {32, 45, 32, 180});
                    if(!inputToOutput)
                        show_label("One pin has to be output, the other input", {32, 45, 32, 180});
                    if(a == b || pin_to_nodes[a.Get()] == pin_to_nodes[b.Get()] || wrongType || !inputToOutput)
                        nodes::RejectNewItem({255, 0, 0, 255}, 2.f);
                    else{
                        show_label("+ Create Link", {32, 45, 32, 180});
                        if(nodes::AcceptNewItem()){     // add check for validity
                            _execution_graphs[main_string]->addLink(_cur_id, a.Get(), b.Get(), nodes[pin_to_nodes[a.Get()]].node->outputTypes[connection.nodeAAttribute]->color());
                        }
                    } 
                }
            }
        }
    }

    ImGui::End();
}
}