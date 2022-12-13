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
    workbench(id),
    _editor_context(ax::NodeEditor::CreateEditor())
{
    _execution_graphs.emplace("main", std::make_unique<ExecutionGraph>());
    nodes::SetCurrentEditor(_editor_context);
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
    nodes::SetCurrentEditor(_editor_context);
    nodes::Begin("Derivation workbench");

    auto&           editor_style = nodes::GetStyle();
    const ImVec4    header_color{.1,.1,.1,1};
    const float     pin_icon_size = 15;

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
                            data.columns.erase(data.columns.begin() + i);
                            data.column_dimensions.erase(data.column_dimensions.begin() + i);
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
                    ds.attributes.back().bounds.changed = true;
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
        ImGui::EndPopup();
    }

    if(ImGui::BeginPopup(create_new_node_name.data())){
        deriveData::Type* prev_type{};
        std::string main_string(main_execution_graph_id);
        if(_new_link_pin_id > 0){
            Link::Connection connection{};
            connection.nodeAId = _execution_graphs[main_string]->pinToNodes[_new_link_pin_id];
            auto& node_a_output = nodes[pin_to_nodes[_new_link_pin_id]].outputIds;
            connection.nodeAAttribute = std::find_if(node_a_output.begin(), node_a_output.end(), [&](int i){return i == _new_link_pin_id;}) - node_a_output.begin();
            if(connection.nodeAAttribute < node_a_output.size())
                prev_type = nodes[connection.nodeAId].node->outputTypes[connection.nodeAAttribute].get();
        }

        std::unique_ptr<deriveData::Nodes::Node> node{};
        for(const auto& [name, entry]: deriveData::Nodes::Registry::nodes){
            if(prev_type && (entry.prototype->inputTypes.empty() || typeid(*prev_type) != typeid(*entry.prototype->inputTypes[0])))
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
    }
    ImGui::PopStyleVar();
    nodes::Resume();
    if(this_frame_create)
        _popup_pos = ImGui::GetMousePos();

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
    std::set<long> output;
    auto& [nodes, pin_to_nodes, links, link_to_connection, pin_to_links] = *_execution_graphs[std::string(main_execution_graph_id)];
    for(int i: nodes[node].inputIds){
        if(pin_to_links.count(i) && pin_to_links[i].size()){
            output.insert(pin_to_links[i][0]);    // there can only be a single input linked
            long prefNode = link_to_connection[pin_to_links[i][0]].nodeAId;
            auto prefSet = _get_active_links_recursive(prefNode);
            output.insert(prefSet.begin(), prefSet.end());
        }
    }
    return output;
}

void data_derivation_workbench::_build_cache_recursive(int64_t node, recursion_data& data){
    auto& [nodes, pin_to_nodes, links, link_to_connection, pin_to_links] = *_execution_graphs[std::string(main_execution_graph_id)];
    auto& [active_links, data_storage, node_infos, create_vector_sizes] = data;

    // check cache for previous nodes, if not generated, generate
    for(int i: irange(nodes[node].inputIds)){
        if(pin_to_links.count(nodes[node].inputIds[i]) == 0 || pin_to_links[nodes[node].inputIds[i]].empty())   // pin not connected, use inserted value
            continue;
        long linkId = pin_to_links[nodes[node].inputIds[i]][0];
        long prevNodeId = link_to_connection[linkId].nodeAId;
        if(node_infos.count(prevNodeId) == 0)
            _build_cache_recursive(prevNodeId, data);
    }

    // merging the data views for data processing
    // input data
    deriveData::float_column_views inputData;
    long inputDataSize{-1};
    bool equalDataLayout = true;
    deriveData::column_memory_view<float> curDataLayout{};
    deriveData::column_memory_view<float> outputLayout{};
    std::vector<int> inplaceIndices;
    for(long i: irange(nodes[node].inputIds)){
        if(pin_to_links.count(nodes[node].inputIds[i]) == 0 || pin_to_links[nodes[node].inputIds[i]].empty()){   // pin not connected, use inserted value
            inputData.push_back(nodes[node].node->inputTypes[i]->data());
        }
        else{   // pin connected
            long linkId = pin_to_links[nodes[node].inputIds[i]][0];
            long prevNodeId = link_to_connection[linkId].nodeAId;
            long prevNodeOutInd = link_to_connection[linkId].nodeAAttribute;
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
            if(outputLayout || inputData.back().dimensionSizes.size() > outputLayout.dimensionSizes.size())
                outputLayout = inputData.back();
        }
        inputDataSize = std::max<long>(inputDataSize, inputData.back().size());
    }
    assert(inputDataSize != -1 || nodes[node].inputIds.empty());
    // handling vector cration nodes
    if(dynamic_cast<deriveData::Nodes::DataCreation*>(nodes[node].node.get())){
        inputDataSize = inputData[0](0, 0);
        static uint32_t dimsIndex{0};
        create_vector_sizes.emplace_back(std::make_unique<uint32_t>(inputDataSize));
        outputLayout.dimensionSizes = deriveData::memory_view<uint32_t>(*create_vector_sizes.back());
        outputLayout.columnDimensionIndices = deriveData::memory_view<uint32_t>(dimsIndex);
    }

    // removing inplace which can not be used due to data inflation (different data layouts)
    if(!equalDataLayout){
        std::cout << "[Warning] Data layouts for node " << node << " with title " << nodes[node].node->name << " and body " << nodes[node].node->middleText << " has to inflate data because the input data has not euqal data layout" << std::endl;
        std::vector<int> keptInplaceIndices;
        for(int i: inplaceIndices){
            if(inputData[i].full())
                keptInplaceIndices.push_back(i);
        }
        inplaceIndices = std::move(keptInplaceIndices);
    }

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
            columnView.dimensionSizes = deriveData::memory_view<uint32_t>(ds_data.dimension_sizes);
            columnView.columnDimensionIndices = deriveData::memory_view<uint32_t>(ds_data.column_dimensions[i]);
            columnView.cols = {deriveData::memory_view<float>(ds_data.columns[i])};
            outputData[i] = columnView;
        }
    }
    else{
        for(int i: inplaceIndices){
            if(inputData[i].equalDataLayout(outputLayout))
                memoryViewPool.insert(memoryViewPool.end(), inputData[i].cols.begin(), inputData[i].cols.end());
        }
        if(memoryViewPool.size() < nodes[node].node->outputChannels()){
            int storageSize = data_storage.size();
            int missingBuffer = nodes[node].node->outputChannels() - memoryViewPool.size();
            data_storage.insert(data_storage.end(), missingBuffer, std::vector<float>(inputDataSize));
            for(int i: irange(storageSize, data_storage.size()))
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

    // executing the node
    nodes[node].node->applyOperationCpu(inputData, outputData);

    // safing the cache and setting up the counts for the current data
    node_infos[node].output_counts.resize(nodes[node].outputIds.size());
    for(int i: irange(nodes[node].outputIds)){
        for(const long link: pin_to_links[nodes[node].outputIds[i]]){
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
            for(auto& d: data_storage){
                if(d.data() == inputData[i].cols[0].data()){
                    if(std::count(inplaceIndices.begin(), inplaceIndices.end(), i))
                        ds_data.columns[i] = std::move(d);
                    else
                        ds_data.columns[i] = d;
                    ds_data.column_dimensions[i] = std::vector<uint32_t>(inputData[i].columnDimensionIndices.begin(), inputData[i].columnDimensionIndices.end());
                }
            }

            // updating min and max of the attributes
            dataset.attributes[i].bounds().min = std::numeric_limits<float>::max();
            dataset.attributes[i].bounds().max = -std::numeric_limits<float>::max();
            for(float f: ds_data.columns[i]){
                dataset.attributes[i].bounds().min = std::min(f, dataset.attributes[i].bounds.read().min);
                dataset.attributes[i].bounds().max = std::max(f, dataset.attributes[i].bounds.read().max);
            }
            if(dataset.attributes[i].bounds.read().min == dataset.attributes[i].bounds.read().max){
                dataset.attributes[i].bounds().max += .1f;
                dataset.attributes[i].bounds().min -= .1f;
            }
        }
    }
}

void data_derivation_workbench::_execute_graph(std::string_view id){
    std::string id_string(id);
    auto& [nodes, pinToNodes, links, linkToConnection, pinToLinks] = *_execution_graphs[id_string];

    // checkfor output nodes
    std::set<long> outputNodes{};
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
    for(auto node: outputNodes)
        _build_cache_recursive(node, data);

    // wiht the execution of all output nodes all changes are already applied
    logger << logging::info_prefix << " data_derivation_workbench::_execute_graph() Amount of data allocations: " << data.data_storage.size() << logging::endl;
}
}