#include "data_derivation_workbench.hpp"
#include <logger.hpp>
#include "../imgui_nodes/utilities/builders.h"
#include <Nodes.hpp>
#include <datasets.hpp>
#include <imgui_stdlib.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>
#include <load_colors_workbench.hpp>
#include <drawlists.hpp>
#include <drawlist_util.hpp>
#include <gpu_execution.hpp>
#include <vma_initializers.hpp>
#include <regex>
#include <stager.hpp>
#include <stopwatch.hpp>

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
    workbench(id),
    _editor_context(ax::NodeEditor::CreateEditor())
{
    _execution_graphs.emplace("main", std::make_unique<ExecutionGraph>());
    nodes::SetCurrentEditor(_editor_context);
    auto pool_info = util::vk::initializers::commandPoolCreateInfo(globals::vk_context.compute_queue_family_index);
    _compute_command_pool = util::vk::create_command_pool(pool_info);
    auto fence_info = util::vk::initializers::fenceCreateInfo();
    _compute_fence = util::vk::create_fence(fence_info);
}

data_derivation_workbench::~data_derivation_workbench(){
    ax::NodeEditor::DestroyEditor(_editor_context);
}

void data_derivation_workbench::show(){
    const std::string_view node_context_name{"Node Context Menu"};
    const std::string_view pin_context_name{"Pin Context Menu"};
    const std::string_view link_context_name{"Link Context Menu"};
    const std::string_view create_new_node_name{"Create New Node"};

    if(!active)
        return;
    const static std::string_view some_menu_id{};

    ImGui::SetNextWindowSize({800, 800}, ImGuiCond_Once);
    ImGui::Begin(id.c_str(), &active);

    if(ImGui::Button("Execute Graph")){
        try{
            _execute_graph(main_execution_graph_id);
        }
        catch(const std::runtime_error& e){
            logger << logging::error_prefix << " " << e.what() << logging::endl;
        }
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if(ImGui::BeginCombo("Execution backend", structures::data_derivation::execution_names[_settings.execution_backend].data())){
        for(auto e: structures::enum_iteration<structures::data_derivation::execution>())
            if(ImGui::MenuItem(structures::data_derivation::execution_names[e].data()))
                _settings.execution_backend = e;
        
        ImGui::EndCombo();
    }
    nodes::SetCurrentEditor(_editor_context);
    nodes::Begin("Derivation workbench");
    nodes::PushStyleVar(ax::NodeEditor::StyleVar::StyleVar_LinkStrength, _settings.spline_curviness);

    auto&           editor_style = nodes::GetStyle();
    const ImVec4    header_color{.1f,.1f,.1f,1.f};
    const float     pin_icon_size = 15;

    std::vector<std::string_view> dls, tls;
    for(const auto& [dl_id, dl]: globals::drawlists.read())
        dls.emplace_back(dl_id);
    for(const auto& [ds_id, ds]: globals::datasets.read())
        for(const auto& [tl_id, tl]: ds.read().templatelist_index)
            tls.emplace_back(tl_id);

    auto& [nodes, pin_to_nodes, links, link_to_connection, pin_to_links] = *_execution_graphs[std::string(main_execution_graph_id)];

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
            bool is_linked = pin_to_links[node_pins.inputIds[i]].size() > 0;
            ax::Widgets::Icon({pin_icon_size, pin_icon_size}, node->inputTypes[i]->iconType(), is_linked, node->inputTypes[i]->color(), ImColor(32, 32, 32, int(alpha * 255)));
            ImGui::Spring(0);
            if(deriveData::Nodes::VariableInput* variable_input = dynamic_cast<deriveData::Nodes::VariableInput*>(node.get())){
                if(variable_input->namedInput){
                    ImGui::PushItemWidth(100);
                    if(i > variable_input->minNodes){
                        bool empty = node->inputNames[i] == "##";
                        if(empty) node->inputNames[i] = "";
                        if(ImGui::InputText(("##ns" + std::to_string(i)).c_str(), &node->inputNames[i])){
                            if(deriveData::Nodes::DatasetOutput* ds_output = dynamic_cast<deriveData::Nodes::DatasetOutput*>(node.get())){
                                globals::datasets()[ds_output->datasetId]().attributes[i].display_name = node->inputNames[i]; 
                                globals::attributes()[globals::datasets()[ds_output->datasetId]().attributes[i].id]().display_name = node->inputNames[i];
                            }
                        }
                        if(node->inputNames[i].empty()) node->inputNames[i] = "##";
                    }
                    else
                        ImGui::TextUnformatted(node->inputNames[i].c_str());
                    ImGui::PopItemWidth();
                    ImGui::Spring(0);
                }
                if(i > variable_input->minNodes && ImGui::Button(("X##p" + std::to_string(node_pins.inputIds[i])).c_str())){
                    variable_input->pinRemoveAction(i);
                    if(deriveData::Nodes::DatasetOutput* ds_output = dynamic_cast<deriveData::Nodes::DatasetOutput*>(node.get())){
                        auto& ds = globals::datasets.ref_no_track()[ds_output->datasetId].ref_no_track();
                        std::visit([i](auto&&  data){
                            data.columns.erase(data.columns.begin() + i);
                            data.column_dimensions.erase(data.column_dimensions.begin() + i);
                            if(data.column_transforms.size())
                                data.column_transforms.erase(data.column_transforms.begin() + i);
                            }, ds.cpu_data());
                        globals::dataset_attribute_deletions.emplace_back(std::pair<std::string_view, std::string>{ds.id, ds.attributes[i].id});
                    }
                    _execution_graphs[std::string(main_execution_graph_id)]->removePin(node_pins.inputIds[i], true);
                    ImGui::PopStyleVar();
                    builder.EndInput();
                    break;
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
                variable_input->pinAddAction();
                std::string pin_name{};
                if(deriveData::Nodes::DatasetOutput* ds_output = dynamic_cast<deriveData::Nodes::DatasetOutput*>(node.get())){
                    pin_name = std::to_string(node_pins.inputIds.size());
                    // checking if the name is available
                    while(globals::attributes.read().count(pin_name) > 0)
                        pin_name += " ";
                    auto& ds = globals::datasets.ref_no_track()[ds_output->datasetId]();
                    std::visit([](auto&& data){
                        data.columns.emplace_back(1, 0.f);    // emplace vector with 1 element which is 0
                        data.column_dimensions.emplace_back();// emplace empty (signals constant)
                        if(data.column_transforms.size())
                            data.column_transforms.emplace_back(structures::scale_offset<float>{1.f, 0.f});
                    }, ds.cpu_data());
                    globals::dataset_attribute_creations.emplace_back(std::pair<std::string_view, std::string>{ds.id, pin_name});
                }
                _execution_graphs[std::string(main_execution_graph_id)]->addPin(_cur_id, id, pin_name, deriveData::FloatType::create(), true);
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
                        _cur_dimensions.clear();
                        for(const auto& d: std::visit([](auto&& data){return data.dimension_names;}, ds.read().cpu_data.read())) _cur_dimensions.emplace_back(d);
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
        node->imguiMiddleElements(std::vector<std::string_view>(_cur_dimensions.begin(), _cur_dimensions.end()), dls, tls);
        ImGui::Spring(1, 0);

        // outputs
        for(size_t i: util::size_range(node->outputTypes)){
            builder.Output(node_pins.outputIds[i]);
            auto alpha = ImGui::GetStyle().Alpha;
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
            bool is_linked = pin_to_links[node_pins.outputIds[i]].size() > 0;
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
                    bool change = std::count_if(node_a_inputs.begin(), node_a_inputs.end(), [&a](auto i){return i == a.Get();}) > 0;
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

                    bool wrongType = connection.nodeAAttribute < as<int64_t>(nodeAOutput.size()) && connection.nodeBAttribute < as<int64_t>(nodeBInput.size()) && typeid(*nodes[pin_to_nodes[a.Get()]].node->outputTypes[connection.nodeAAttribute]) != typeid(*nodes[pin_to_nodes[b.Get()]].node->inputTypes[connection.nodeBAttribute]);

                    bool inputToOutput = _is_input_pin(a.Get()) ^ _is_input_pin(b.Get());

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
        nodes::EndCreate();
    }

    // handle deletion action
    if(nodes::BeginDelete()){
        nodes::LinkId d{};
        std::string main_string(main_execution_graph_id);
        while(nodes::QueryDeletedLink(&d)){
            if(nodes::AcceptDeletedItem())
                _execution_graphs[main_string]->removeLink(d.Get());
        }
        nodes::NodeId n{};
        while(nodes::QueryDeletedNode(&n)){
            if(nodes::AcceptDeletedItem())
                _execution_graphs[main_string]->removeNode(n.Get());
        }
    }
    nodes::EndDelete();

    // node creation dialogue
    nodes::Suspend();
    nodes::NodeId n{};
    nodes::PinId p{};
    nodes::LinkId l{};
    bool focus_text_input{};
    if(nodes::ShowNodeContextMenu(&n)){
        ImGui::OpenPopup(node_context_name.data());
        _context_node_id = n.Get();
    }
    else if(nodes::ShowPinContextMenu(&p)){
        ImGui::OpenPopup(pin_context_name.data());
        _context_pin_id = p.Get();
    }
    else if(nodes::ShowLinkContextMenu(&l)){
        ImGui::OpenPopup(link_context_name.data());
        _context_link_id = l.Get();
    }
    else if(nodes::ShowBackgroundContextMenu()){
        ImGui::OpenPopup(create_new_node_name.data());
        _create_new_node = false;
        this_frame_create = true;
    }
    nodes::Resume();

    nodes::Suspend();
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {8, 8});
    if(ImGui::BeginPopup(node_context_name.data())){
        ImGui::TextUnformatted(node_context_name.data());
        ImGui::Separator();
        if(ImGui::MenuItem("Delete"))
            nodes::DeleteNode(_context_node_id);
        ImGui::EndPopup();
    }

    if(ImGui::BeginPopup(pin_context_name.data())){
        ImGui::TextUnformatted(pin_context_name.data());
        ImGui::Separator();
        ImGui::EndPopup();
    }

    if(ImGui::BeginPopup(link_context_name.data())){
        ImGui::TextUnformatted(link_context_name.data());
        ImGui::Separator();
        if(ImGui::MenuItem("Delete"))
            nodes::DeleteLink(_context_link_id);
        ImGui::DragFloat("Spline curviness", &_settings.spline_curviness, 0.f, 0.f, 1e6);
        ImGui::EndPopup();
    }

    if(ImGui::BeginPopup(create_new_node_name.data())){
        deriveData::Type* prev_type{};
        std::string main_string(main_execution_graph_id);
        if(_new_link_pin_id > 0){
            Link::Connection connection{};
            connection.nodeAId = _execution_graphs[main_string]->pinToNodes[_new_link_pin_id];
            auto& node_a_output = nodes[pin_to_nodes[_new_link_pin_id]].outputIds;
            connection.nodeAAttribute = std::find_if(node_a_output.begin(), node_a_output.end(), [&](auto i){return i == _new_link_pin_id;}) - node_a_output.begin();
            if(connection.nodeAAttribute < as<int64_t>(node_a_output.size()))
                prev_type = nodes[connection.nodeAId].node->outputTypes[connection.nodeAAttribute].get();
        }

        // regex search
        ImGui::SetNextItemWidth(100);
        if(_attribute_regex_error)
            ImGui::PushStyleColor(ImGuiCol_FrameBg, {1.f, 0.f, 0.f, .5f});
        if(!ImGui::IsAnyItemActive() && !ImGui::IsAnyItemFocused() && !ImGui::IsAnyItemHovered() && !ImGui::IsMouseClicked(0))
            ImGui::SetKeyboardFocusHere();
        ImGui::InputText("Search node", &_create_node_regex);
        if(_attribute_regex_error)
            ImGui::PopStyleColor();
        std::string lowercase; lowercase.resize(_create_node_regex.size());
        std::transform(_create_node_regex.begin(), _create_node_regex.end(), lowercase.begin(), ::tolower);
        std::regex reg;
        try{
            reg = std::regex(lowercase);
            _attribute_regex_error = false;
        }
        catch(std::regex_error e){
            _attribute_regex_error = true;
        }

        std::unique_ptr<deriveData::Nodes::Node> node{};
        for(const auto& [name, entry]: deriveData::Nodes::Registry::nodes){
            if(prev_type && (entry.prototype->inputTypes.empty() || typeid(*prev_type) != typeid(*entry.prototype->inputTypes[0])))
                continue;
            std::string lowercase_node; lowercase_node.resize(name.size());
            std::transform(name.begin(), name.end(), lowercase_node.begin(), ::tolower);
            if(!std::regex_search(lowercase_node, reg))
                continue;
            if(ImGui::MenuItem(name.c_str()))
                node = entry.create();
        }

        if(node){
            int64_t node_id = _cur_id++;
            _create_new_node = false;
            nodes::SetNodePosition(node_id, _popup_pos);
            nodes.insert({node_id, NodePins(std::move(node), &_cur_id)});
            for(auto input_id: nodes[node_id].inputIds)
                pin_to_nodes[input_id] = node_id;
            for(auto output_id: nodes[node_id].outputIds)
                pin_to_nodes[output_id] = node_id;
            
            if(_new_link_pin_id != 0){
                _execution_graphs[main_string]->addLink(_cur_id, _new_link_pin_id, nodes[node_id].inputIds[0], nodes[node_id].node->inputTypes[0]->color());
                _new_link_pin_id = 0;
            }
        }
        ImGui::EndPopup();
    }
    else{
        _create_new_node = false;
        _new_link_pin_id = {};
        _create_node_regex.clear();
    }
    ImGui::PopStyleVar();
    nodes::Resume();
    if(this_frame_create)
        _popup_pos = ImGui::GetMousePos();

    nodes::PopStyleVar();
    nodes::End();

    ImGui::End();
}

bool data_derivation_workbench::_is_input_pin(int64_t pin_id){
    auto& [nodes, pin_to_nodes, links, link_to_connection, pin_to_links] = *_execution_graphs[std::string(main_execution_graph_id)];
    auto& node = nodes[pin_to_nodes[pin_id]];
    return std::count(node.inputIds.begin(), node.inputIds.end(), pin_id) > 0;
}

std::set<int64_t> data_derivation_workbench::_get_active_links_recursive(int64_t node)
{
    std::set<int64_t> output;
    auto& [nodes, pin_to_nodes, links, link_to_connection, pin_to_links] = *_execution_graphs[std::string(main_execution_graph_id)];
    for(auto i: nodes[node].inputIds){
        if(pin_to_links.count(i) && pin_to_links[i].size()){
            output.insert(pin_to_links[i][0]);    // there can only be a single input linked
            auto prefNode = link_to_connection[pin_to_links[i][0]].nodeAId;
            auto prefSet = _get_active_links_recursive(prefNode);
            output.insert(prefSet.begin(), prefSet.end());
        }
    }
    return output;
}

void data_derivation_workbench::_build_cache_recursive(int64_t node, recursion_data& data){
    auto& [nodes, pin_to_nodes, links, link_to_connection, pin_to_links] = *_execution_graphs[std::string(main_execution_graph_id)];
    auto& [active_links, data_storage, node_infos, create_vector_sizes, op_codes_list, print_infos, buffer_init_values] = data;

    // check cache for previous nodes, if not generated, generate
    for(int i: irange(nodes[node].inputIds)){
        if(pin_to_links.count(nodes[node].inputIds[i]) == 0 || pin_to_links[nodes[node].inputIds[i]].empty())   // pin not connected, use inserted value
            continue;
        auto linkId = pin_to_links[nodes[node].inputIds[i]][0];
        auto prevNodeId = link_to_connection[linkId].nodeAId;
        if(node_infos.count(prevNodeId) == 0)
            _build_cache_recursive(prevNodeId, data);
    }

    // merging the data views for data processing
    // input data
    deriveData::float_column_views inputData;
    int64_t inputDataSize{-1};
    bool equalDataLayout = true;
    deriveData::column_memory_view<float> curDataLayout{};
    deriveData::column_memory_view<float> outputLayout{};
    std::vector<int> inplaceIndices;
    for(auto i: irange(nodes[node].inputIds)){
        if(pin_to_links.count(nodes[node].inputIds[i]) == 0 || pin_to_links[nodes[node].inputIds[i]].empty()){   // pin not connected, use inserted value
            inputData.push_back(nodes[node].node->inputTypes[i]->data());
        }
        else{   // pin connected
            auto linkId = pin_to_links[nodes[node].inputIds[i]][0];
            auto prevNodeId = link_to_connection[linkId].nodeAId;
            auto prevNodeOutInd = link_to_connection[linkId].nodeAAttribute;
            // add input data and check for integrity
            inputData.push_back(node_infos[prevNodeId].output_views[prevNodeOutInd]);
            if(inputDataSize > 1 && inputDataSize != inputData.back().size())
                throw std::runtime_error("DeriveWorkbench::buildCacheRecursive() Inputs to node (id = " + std::to_string(node) + ", type = \"" + nodes[node].node->middleText
                                        + "\") do not have the same size. Input at pin index " 
                                        + std::to_string(i) + " has size " + std::to_string(inputData.back().size()) + " while other inputs before have size " 
                                        + std::to_string(inputDataSize) + "!");
            // check for inplace
            bool inplace = --node_infos[prevNodeId].output_counts[prevNodeOutInd] == 0;
            if(inplace)
                inplaceIndices.push_back(i);
            // check data layout for inflation
            if(curDataLayout)
                equalDataLayout &= inputData.back().equalDataLayout(curDataLayout);
            else
                curDataLayout = inputData.back();
            // setting the output dimensions
            if(!outputLayout || inputData.back().dimensionSizes.size() > outputLayout.dimensionSizes.size())
                outputLayout = inputData.back();
        }
        inputDataSize = std::max(inputDataSize, as<int64_t>(inputData.back().size()));
    }
    assert(inputDataSize != -1 || nodes[node].inputIds.empty());
    // handling vector creation nodes
    if(dynamic_cast<deriveData::Nodes::DataCreation*>(nodes[node].node.get())){
        if(dynamic_cast<deriveData::Nodes::Active_Indices*>(nodes[node].node.get())){
            std::string& selected = nodes[node].node->input_elements[nodes[node].node->middle_input_id]["Drawlist/Templatelist"]["selected_dl_tl"].get<std::string>();
            bool dl_selected = globals::drawlists.read() | util::contains_if<decltype(*globals::drawlists.read().begin())>([&selected](const auto& dl){return dl.first == selected;});
            bool tl_selected{}; 
            for(const auto& [ds_id, ds]: globals::datasets.read())
                tl_selected |= ds.read().templatelists | util::contains_if<decltype(*ds.read().templatelists.begin())>([&selected](const auto& tl){return tl->name == selected;});
            if(!dl_selected && !tl_selected)
                throw std::runtime_error{"Active Indices node: Selection " + selected + " is not an available drawlist or templatelist"};
            static std::vector<uint32_t> iota_vec = util::i_range(uint32_t(40)) | util::to<std::vector<uint32_t>>();
            if(dl_selected){
                auto dl_en = globals::drawlists.ref_no_track() | util::try_find_if<std::remove_reference_t<decltype(*globals::drawlists.ref_no_track().begin())>>([&selected](const auto& dl){return dl.first == selected;});
                auto& dl = dl_en->get().second.ref_no_track();
                // downloading the activation into the activation bitset and adding the view to the bitset as well as the indices to the input views
                util::drawlist::download_activation(dl);
                create_vector_sizes.emplace_back(std::make_unique<uint32_t>(as<uint32_t>(dl.const_templatelist().data_size)));
                deriveData::column_memory_view<float> view;
                view.dimensionSizes = std::visit([](auto&& d){return deriveData::memory_view<uint32_t>(const_cast<std::vector<uint32_t>&>(d.dimension_sizes));}, dl.dataset_read().cpu_data.read());
                view.columnDimensionIndices = deriveData::memory_view<uint32_t>(iota_vec.data(), std::visit([](auto&& d){return d.dimension_sizes.size();}, dl.dataset_read().cpu_data.read()));
                view.cols.emplace_back(deriveData::memory_view<float>(reinterpret_cast<float*>(dl.active_indices_bitset.data()), dl.active_indices_bitset.num_blocks()));
                assert(!dl.const_templatelist().indices.empty() || dl.const_templatelist().flags.identity_indices && "If indices are empty the templatelist has to be the identity indexlist, otherwise it is empty which is forbidden!");
                view.cols.emplace_back(deriveData::memory_view<float>(reinterpret_cast<float*>(const_cast<uint32_t*>(dl.const_templatelist().indices.data())), dl.const_templatelist().indices.size()));
                inputData.emplace_back(std::move(view));
            }
            else{
                const structures::templatelist* tl;
                const structures::dataset*      ds;
                for(const auto& [ds_id, ds_]: globals::datasets.read()){
                    if(auto tl_pair = ds_.read().templatelist_index | util::try_find_if<std::remove_reference_t<decltype(*ds_.read().templatelist_index.begin())>>([&selected](const auto& t){return t.first == selected;})){
                        tl = tl_pair->get().second;
                        ds = &ds_.read();
                        break;
                    }
                }
                deriveData::column_memory_view<float> view;
                view.dimensionSizes = std::visit([](auto&& d){return deriveData::memory_view<uint32_t>(const_cast<std::vector<uint32_t>&>(d.dimension_sizes));}, ds->cpu_data.read());
                view.columnDimensionIndices = deriveData::memory_view<uint32_t>(iota_vec.data(), std::visit([](auto&& d){return d.dimension_sizes.size();}, ds->cpu_data.read()));
                view.cols.emplace_back(deriveData::memory_view<float>(reinterpret_cast<float*>(const_cast<uint32_t*>(tl->indices.data())), tl->indices.size()));
                inputData.emplace_back(std::move(view));
            }
            outputLayout.dimensionSizes = inputData.back().dimensionSizes;
            outputLayout.columnDimensionIndices = inputData.back().columnDimensionIndices;
        }
        else{
            inputDataSize = as<int64_t>(inputData[0](0, 0));
            static uint32_t dimsIndex{0};
            create_vector_sizes.emplace_back(std::make_unique<uint32_t>(as<uint32_t>(inputDataSize)));
            outputLayout.dimensionSizes = deriveData::memory_view<uint32_t>(*create_vector_sizes.back());
            outputLayout.columnDimensionIndices = deriveData::memory_view<uint32_t>(dimsIndex);
        }
    }

    // For the reduction nodes the output data layout equals the index layout which sits at index 0 of the inputData views
    bool reduction_node = dynamic_cast<deriveData::Nodes::Reduction*>(nodes[node].node.get());
    if(reduction_node)
        outputLayout = inputData[0];

    // removing inplace which can not be used due to data inflation (different data layouts)
    if(!equalDataLayout && !reduction_node){
        logger << logging::warning_prefix << " Data layouts for node " << node << " with title " << nodes[node].node->name << " and body " << nodes[node].node->middleText << " has to inflate data because the input data has not euqal data layout" << logging::endl;
        std::vector<int> keptInplaceIndices;
        for(int i: inplaceIndices){
            if(inputData[i].full())
                keptInplaceIndices.push_back(i);
        }
        inplaceIndices = std::move(keptInplaceIndices);
    }

    // clearing inplace buffers if node does not support inplace operation
    if(!nodes[node].node->inplace_possible)
        inplaceIndices.clear();

    // output data (merging the inplace buffers and adding new buffers. if the data layout does not fit to the outputLayout inplace can not be used)
    deriveData::float_column_views outputData(nodes[node].outputIds.size());
    std::vector<deriveData::memory_view<float>> memoryViewPool;
    uint32_t poolStart{};
    if(deriveData::Nodes::DatasetInput* inputNode = dynamic_cast<deriveData::Nodes::DatasetInput*>(nodes[node].node.get())){
        auto& ds = globals::datasets.ref_no_track()[inputNode->datasetId].ref_no_track();
        if(!std::holds_alternative<structures::data<float>>(ds.cpu_data.read()))
            throw std::runtime_error{"data_derivation_workbench::_build_cache_recursive() Data derivation only for float datasets possible!"};
        auto& ds_data = std::get<structures::data<float>>(ds.cpu_data.ref_no_track());
        for(int i: irange(outputData)){
            deriveData::column_memory_view<float> columnView;
            columnView.dimensionNames = std::vector<std::string_view>(ds_data.dimension_names.begin(), ds_data.dimension_names.end());
            columnView.dimensionSizes = deriveData::memory_view<uint32_t>(ds_data.dimension_sizes);
            columnView.columnDimensionIndices = deriveData::memory_view<uint32_t>(ds_data.column_dimensions[i]);
            columnView.cols = {deriveData::memory_view<float>(ds_data.columns[i])};
            outputData[i] = columnView;
        }
    }
    else{
        for(int i: inplaceIndices){
            for(auto& col: inputData[i].cols){
                if(col.size() >= outputLayout.size())
                    memoryViewPool.emplace_back(col);
            }
        }
        if(memoryViewPool.size() < nodes[node].node->outputChannels()){
            int storageSize = as<int>(data_storage.size());
            int missingBuffer = as<int>(nodes[node].node->outputChannels() - memoryViewPool.size());
            data_storage.insert(data_storage.end(), missingBuffer, std::vector<float>(outputLayout.columnSize()));
            for(auto i: irange(storageSize, as<unsigned long>(data_storage.size())))
                memoryViewPool.push_back(data_storage[i]);
        }
        for(int i: irange(nodes[node].outputIds)){
            deriveData::column_memory_view<float> columnMem;
            columnMem.dimensionSizes = outputLayout.dimensionSizes;
            columnMem.columnDimensionIndices = outputLayout.columnDimensionIndices;
            for(int j: irange(nodes[node].node->outputTypes[i]->data().cols))
                columnMem.cols.push_back(memoryViewPool[poolStart++]);
            outputData[i] = std::move(columnMem);
        }
    }

    if(deriveData::Nodes::Serialization* s = dynamic_cast<deriveData::Nodes::Serialization*>(nodes[node].node.get())){
        outputData = inputData;
        if(inplaceIndices.empty()){
            data_storage.emplace_back(inputData[0].cols[0].begin(), inputData[0].cols[0].end());
            outputData[0].cols[0] = deriveData::memory_view(data_storage.back());
        }
        print_infos.emplace_back(recursion_data::print_info{nodes[node].node.get(), outputData});
    }

    // executing the node
    switch(_settings.execution_backend){
    case structures::data_derivation::execution::Cpu:
        nodes[node].node->applyOperationCpu(inputData, outputData);
        break;
    case structures::data_derivation::execution::Gpu:
        nodes[node].node->applyOperationGpu(op_codes_list, inputData, outputData, buffer_init_values);
        break;
    }

    // saving the cache and setting up the counts for the current data
    node_infos[node].output_counts.resize(nodes[node].outputIds.size());
    for(int i: irange(nodes[node].outputIds)){
        for(const auto link: pin_to_links[nodes[node].outputIds[i]]){
            if(active_links.count(link) > 0)
                ++node_infos[node].output_counts[i];
        }

        if(deriveData::Nodes::DatasetInput* datasetInput = dynamic_cast<deriveData::Nodes::DatasetInput*>(nodes[node].node.get()))
            ++node_infos[node].output_counts[i];   // make the dataset input not movable
    }
    node_infos[node].output_views = outputData;

    // checking for dataset output and moving or copying the data to the output
    if(deriveData::Nodes::DatasetOutput* n = dynamic_cast<deriveData::Nodes::DatasetOutput*>(nodes[node].node.get())){
        // getting the dataset data layout
        auto& dataset = globals::datasets().at(n->datasetId)();
        if(!std::holds_alternative<structures::data<float>>(dataset.cpu_data()))
            throw std::runtime_error{"data_derivation_workbench::_build_cache_recursive() Data derivation only for float datasets possible!"};
        auto& ds_data = std::get<structures::data<float>>(dataset.cpu_data());
        deriveData::column_memory_view<float> dataset_layout;
        dataset_layout.dimensionSizes = deriveData::memory_view<uint32_t>(ds_data.dimension_sizes);
        bool anyChange = false;
        for(int i: irange(inputData)){
            if(pin_to_links.count(nodes[node].inputIds[i]) == 0 || pin_to_links[nodes[node].inputIds[i]].empty())  // not connected
                continue;
            anyChange = true;
            // checking for consistent dataset layout 
            if(inputData[i].dimensionSizes.size() && !inputData[i].equalDimensions(dataset_layout))
                throw std::runtime_error("DeriveWorkbench::buildCacheRecursive() Data layout at dataset output node for dataset " + 
                    std::string(n->datasetId) + " for attribute " + n->inputNames[i] + " does not match the output of the previous node. Aborting...");
            
            // searching the vector which contains the data and move to dataset data
            bool data_in_data_storage{};
            for(auto& d: data_storage){
                if(d.data() == inputData[i].cols[0].data()){
                    if(std::count(inplaceIndices.begin(), inplaceIndices.end(), i))
                        ds_data.columns[i] = std::move(d);
                    else
                        ds_data.columns[i] = d;
                    data_in_data_storage = true;
                }
            }
            if(!data_in_data_storage) // copy data from input view
                ds_data.columns[i] = std::vector(inputData[i].cols[0].begin(), inputData[i].cols[0].end());
            
            ds_data.column_dimensions[i] = std::vector<uint32_t>(inputData[i].columnDimensionIndices.begin(), inputData[i].columnDimensionIndices.end());
            

            // updating min and max of the dataset attributes
            dataset.attributes[i].bounds().min = std::numeric_limits<float>::max();
            dataset.attributes[i].bounds().max = -std::numeric_limits<float>::max();
            for(float f: ds_data.columns[i]){
                dataset.attributes[i].bounds().min = std::min(f, dataset.attributes[i].bounds.read().min);
                dataset.attributes[i].bounds().max = std::max(f, dataset.attributes[i].bounds.read().max);
            }

            // updating global attribute min max
            std::string_view att_id = dataset.attributes[i].id;
            auto& gb_att = ATTRIBUTE_WRITE(att_id); gb_att.bounds() = {};
            for(const auto& [ds_id, ds]: globals::datasets.read()){
                size_t att_index = ds.read().attributes | util::index_of_if<const structures::attribute>([att_id](const auto& a){return a.id == att_id;});
                if(att_index != util::n_pos){
                    gb_att.bounds().min = std::min(ds.read().attributes[att_index].bounds.read().min, gb_att.bounds.read().min);
                    gb_att.bounds().max = std::max(ds.read().attributes[att_index].bounds.read().max, gb_att.bounds.read().max);
                }
            }
            if(gb_att.bounds.read().min == gb_att.bounds.read().max){
                gb_att.bounds().max += .1f;
                gb_att.bounds().min -= .1f;
            }
        }
    }
}

void data_derivation_workbench::_execute_graph(std::string_view id){
    std::string id_string(id);
    auto& [nodes, pinToNodes, links, linkToConnection, pinToLinks] = *_execution_graphs[id_string];

    // check for output nodes
    std::set<int64_t> outputNodes{};
    for(auto& [id, nodePins]: nodes){
        if(dynamic_cast<deriveData::Nodes::Output*>(nodePins.node.get()))
            outputNodes.insert(id);
    }
    if(outputNodes.empty())
        throw std::runtime_error{"data_derivation_workbench::_execute_graph() No output nodes in graph. Nothing done, as calculations would be lost"};

    if(_execution_graphs[id_string]->hasCircularConnections())
        throw std::runtime_error{"data_derivation_workbench::_execute_graph() Recursion detected in the graph. This is not allowed! Fix before rerun"};
    

    recursion_data data{};
    for(auto node: outputNodes){
        auto active_links = _get_active_links_recursive(node);
        data.active_links.insert(active_links.begin(), active_links.end());
    }
    structures::stopwatch stopwatch; stopwatch.start();
    for(auto node: outputNodes)
        _build_cache_recursive(node, data);

    if(_settings.execution_backend == structures::data_derivation::execution::Gpu){
        // creating all gpu storage data buffer and exchanging the addresses in the gpu code
        auto memory_info = util::vma::initializers::allocationCreateInfo();
        std::vector<structures::buffer_info> gpu_buffers;           // these also have to be destroyed in order to not leak gpu memory
        std::vector<std::vector<VkBuffer>>   print_infos_buffers(data.print_infos.size());   // gpu buffers containing the data for the print nodes
        std::string code_list = data.op_codes_list.str();
        for(const auto& s: data.data_storage){
            auto buffer_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, s.size() * sizeof(s[0]));
            gpu_buffers.emplace_back(util::vk::create_buffer(buffer_info, memory_info));
            std::stringstream  gpu_address; gpu_address << 'g' << util::vk::get_buffer_address(gpu_buffers.back());
            std::stringstream  cpu_address; cpu_address << 'g' << s.data();
            code_list = std::regex_replace(code_list, std::regex(cpu_address.str()), gpu_address.str());

            size_t print_index = data.print_infos | util::index_of_if<recursion_data::print_info>([&s](auto& info){return info.data | util::contains_if<deriveData::column_memory_view<float>>([&s](auto& mem_view){return mem_view.cols[0].data() == s.data();});});
            if(print_index != util::n_pos)
                print_infos_buffers[print_index] = {gpu_buffers.back().buffer};
        }

        // getting the pipelines and executing them
        code_list = deriveData::optimize_operations(code_list);
        auto pipelines = deriveData::create_gpu_pipelines(code_list, _compute_command_pool);
        auto command_buffer = util::vk::create_begin_command_buffer(_compute_command_pool);
        // fill buffers with init values
        for(const auto& init_val: data.buffer_init_values){
            size_t buffer_index = data.data_storage | util::index_of_if<std::vector<float>>([&init_val](auto&& v){return v.data() == init_val.vector;});
            assert(buffer_index != util::n_pos);
            vkCmdFillBuffer(command_buffer, gpu_buffers[buffer_index].buffer, 0, gpu_buffers[buffer_index].size, 0);
        }
        vkResetFences(globals::vk_context.device, 1, &_compute_fence);
        for(const auto& pipeline: pipelines.pipelines){
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
            for(auto&&  [amt_of_threads, i]: util::enumerate(pipeline.amt_of_threads)){
                if(pipeline.push_constants_data.size() >= i && pipeline.push_constants_data.size())
                    vkCmdPushConstants(command_buffer, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, as<uint32_t>(pipeline.push_constants_data[i].size()), pipeline.push_constants_data[i].data()); 
                vkCmdDispatch(command_buffer, (uint32_t(amt_of_threads) + workgroup_size - 1) / workgroup_size, 1, 1);
            }
            vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, {}, 0, {}, 0, {}, 0, {});
        }
        structures::stopwatch gpu_watch; gpu_watch.start();
        util::vk::end_commit_command_buffer(command_buffer, globals::vk_context.compute_queue.const_access().get(), {}, {}, {}, _compute_fence);
        auto res = vkWaitForFences(globals::vk_context.device, 1, &_compute_fence, VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);
        if(logger.logging_level >= logging::level::l_5)
            logger << logging::info_prefix << " Gpu execution without download took " << gpu_watch.lap() << "ms." << logging::endl;
        vkFreeCommandBuffers(globals::vk_context.device, _compute_command_pool, 1, &command_buffer);

        // downloading print data
        for(auto&& [print_info, i]: util::enumerate(data.print_infos)){
            structures::stager::staging_buffer_info buffer_staging{};
            buffer_staging.transfer_dir = structures::stager::transfer_direction::download;
            buffer_staging.dst_buffer = print_infos_buffers[i][0];
            buffer_staging.data_download = util::memory_view(print_info.data[0].cols[0].data(), print_info.data[0].cols[0].size());
            globals::stager.add_staging_task(buffer_staging);
        }
        globals::stager.wait_for_completion();
        for(const auto& [node, d]: data.print_infos)
            logger << logging::info_prefix << " " << dynamic_cast<deriveData::Nodes::Serialization*>(node)->serialize(d) << logging::endl;

        // cleanup vulkan resources
        for(auto& gpu_buffer: gpu_buffers)
            util::vk::destroy_buffer(gpu_buffer);
        for(auto& pipeline: pipelines.pipelines){
            util::vk::destroy_pipeline_layout(pipeline.layout);
            util::vk::destroy_pipeline(pipeline.pipeline);
        }
    }
    else
        for(const auto& [node, d]: data.print_infos)
            logger << logging::info_prefix << " " << dynamic_cast<deriveData::Nodes::Serialization*>(node)->serialize(d) << logging::endl;

    // with the execution of all output nodes all changes are already applied
    logger << logging::info_prefix << " data_derivation_workbench::_execute_graph() " << stopwatch.lap() <<"ms passed. Amount of data allocations: " << data.data_storage.size() << logging::endl;
}
}