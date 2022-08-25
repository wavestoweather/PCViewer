#include "DeriveWorkbench.hpp"
#include "../imgui/imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "../imgui/imgui_internal.h"
#include "../imgui_nodes/imgui_node_editor.h"
#include "../imgui_nodes/imgui_node_editor_internal.h"
#include "../imgui_nodes/utilities/builders.h"
#include "Nodes.hpp"
#include "ExecutionGraph.hpp"

namespace nodes = ax::NodeEditor;

void showLabel(const char* label, ImColor color)
{
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - ImGui::GetTextLineHeight());
    auto size = ImGui::CalcTextSize(label);
    auto padding = ImGui::GetStyle().FramePadding;
    auto spacing = ImGui::GetStyle().ItemSpacing;
    ImGui::SetCursorPos(ImGui::GetCursorPos() + ImVec2(spacing.x, -spacing.y));
    auto rectMin = ImGui::GetCursorScreenPos() - padding;
    auto rectMax = ImGui::GetCursorScreenPos() + size + padding;
    auto drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(rectMin, rectMax, color, size.y * 0.15f);
    ImGui::TextUnformatted(label);
};

void DeriveWorkbench::show() 
{
    if(!active)
        return;
    ImGui::SetNextWindowSize({800, 800}, ImGuiCond_Once);
    ImGui::Begin("DeriveWorkbench", &active);
    ImGui::Dummy({});ImGui::SameLine(ImGui::GetWindowSize().x / 2);
    if(ImGui::Button("Execute Graph")){
        try{
            executeGraph();
        }
        catch(const std::runtime_error& e){
            std::cout << "[Error] " << e.what() << std::endl;
        }
    }
    nodes::SetCurrentEditor(_editorContext);
    nodes::Begin("DeriveWorkbench");


    auto& editorStyle = nodes::GetStyle();
    const ImVec4 headerColor{.1,.1,.1,1};
    const float pinIconSize = 15;

    auto& [nodes, pinToNodes, links, linkToConnection, pinToLinks] = _executionGraphs[0];
    
    auto cursorTopLeft = ImGui::GetCursorStartPos();

    nodes::Utilities::BlueprintNodeBuilder builder; // created without a header texture as not needed
    if(nodes.count(0))
        nodes.erase(0);
    for(auto& [id, nodePins]: nodes){
        assert(id > 0);
        auto& node = nodePins.node;
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
            builder.Input(nodePins.inputIds[i]);
            auto alpha = ImGui::GetStyle().Alpha;
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
            bool isLinked = pinToLinks[nodePins.inputIds[i]].size() > 0;
            ax::Widgets::Icon({pinIconSize, pinIconSize}, node->inputTypes[i]->iconType(), isLinked, node->inputTypes[i]->color(), ImColor(32, 32, 32, int(alpha * 255)));
            ImGui::Spring(0);
            if(deriveData::VariableInput* variableInput = dynamic_cast<deriveData::VariableInput*>(node.get())){
                if(variableInput->minNodes > nodePins.inputIds.size() && ImGui::Button(("X##p" + std::to_string(nodePins.inputIds[i])).c_str())){
                    _executionGraphs[0].removePin(nodePins.inputIds[i], true);
                }
            }
            else if(node->inputNames[i].size()){
                ImGui::TextUnformatted(node->inputNames[i].c_str());
                ImGui::Spring(0);
            }

            if(!isLinked && dynamic_cast<deriveData::OutputNode*>(node.get()) == nullptr){
                auto memoryView = node->inputTypes[i]->data();
                if(memoryView.cols.size()){
                    switch(memoryView.cols.size()){
                        case 1:
                            ImGui::PushItemWidth(50);
                            ImGui::InputFloat("##test", memoryView.cols[0].data());
                            ImGui::PopItemWidth();
                            break;
                        case 2:
                            ImGui::PushItemWidth(100);
                            ImGui::InputFloat2("##inputv2", memoryView.cols[0].data());
                            ImGui::PopItemWidth();
                            break;
                        case 3:
                            ImGui::PushItemWidth(150);
                            ImGui::InputFloat3("##inputv3", memoryView.cols[0].data());
                            ImGui::PopItemWidth();
                            break;
                        case 4:
                            ImGui::PushItemWidth(200);
                            ImGui::InputFloat4("##inputv4", memoryView.cols[0].data());
                            ImGui::PopItemWidth();
                            break;
                    }
                }
                ImGui::Spring(0);
            }
            ImGui::PopStyleVar();
            builder.EndInput();
        }
        if(deriveData::VariableInput* variableInput = dynamic_cast<deriveData::VariableInput*>(node.get())) 
            if(nodePins.inputIds.size() < variableInput->maxNodes && ImGui::Button("Add Pin")){
                _executionGraphs[0].addPin(_curId, id, std::to_string(nodePins.inputIds.size()), deriveData::FloatType::create(), true);
            }


        // middle
        builder.Middle();
        if(deriveData::DatasetInputNode* datasetInput = dynamic_cast<deriveData::DatasetInputNode*>(node.get())){
            ImGui::Spring(1, 0);
            ImGui::PushItemWidth(150);
            if(nodes::BeginNodeCombo("", datasetInput->datasetId.data())){
                for(const auto& ds: *_datasets){
                    if(ImGui::MenuItem(ds.name.c_str())){
                        datasetInput->datasetId = ds.name;

                        while(nodePins.outputIds.size())
                            _executionGraphs[0].removePin(nodePins.outputIds[0], false);
                        
                        for(const auto& a: ds.attributes)
                            _executionGraphs[0].addPin(_curId, id, a.name, deriveData::FloatType::create(), false);
                    }
                }
                nodes::EndNodeCombo();
            }
            ImGui::PopItemWidth();
        }
        if(deriveData::DatasetOutputNode* datasetOutput = dynamic_cast<deriveData::DatasetOutputNode*>(node.get())){
            ImGui::Spring(1, 0);
            ImGui::PushItemWidth(150);
            if(nodes::BeginNodeCombo("", datasetOutput->datasetId.data())){
                for(const auto& ds: *_datasets){
                    if(ImGui::MenuItem(ds.name.c_str())){
                        datasetOutput->datasetId = ds.name;
                        
                        while(nodePins.inputIds.size())
                            _executionGraphs[0].removePin(nodePins.inputIds[0], true);

                        for(const auto& a: ds.attributes)
                            _executionGraphs[0].addPin(_curId, id, a.name, deriveData::FloatType::create(), true);

                        dynamic_cast<deriveData::VariableInput*>(node.get())->minNodes = nodePins.inputIds.size() - 1;
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
        for(int i: irange(node->outputTypes)){
            builder.Output(nodePins.outputIds[i]);
            auto alpha = ImGui::GetStyle().Alpha;
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
            bool isLinked = pinToLinks[nodePins.outputIds[i]].size() > 0;
            if(node->outputNames[i].size()){
                ImGui::TextUnformatted(node->outputNames[i].c_str());
                ImGui::Spring(0);
            }
            ax::Widgets::Icon({pinIconSize, pinIconSize}, node->outputTypes[i]->iconType(), isLinked, node->outputTypes[i]->color(), ImColor(32, 32, 32, int(alpha * 255)));
            ImGui::Spring(0);
            ImGui::PopStyleVar();
            builder.EndOutput();
        }

        builder.End();
    }

    // nodes drawing
    for(auto& [connection, link]: links)
        nodes::Link(link.Id, link.pinAId, link.pinBId, link.color, 2);

    // handle creation action
    bool thisFrameCreate{false};
    if(!_createNewNode){
    if(nodes::BeginCreate()){
        nodes::PinId a, b;
        if(nodes::QueryNewNode(&a)){
            showLabel("+ Create Node", {32, 45, 32, 180});

            if(nodes::AcceptNewItem()){
                _createNewNode = true;
                _newLinkPinId = a.Get();
                thisFrameCreate = true;
                nodes::Suspend();
                ImGui::OpenPopup("Create New Node");
                nodes::Resume();
            }
        }

        a = {};
        if (nodes::QueryNewLink(&a, &b)){
            if(a && b){ // if link was created
                auto& nodeAInputs = nodes[pinToNodes[a.Get()]].inputIds;
                bool change = std::count_if(nodeAInputs.begin(), nodeAInputs.end(), [&](int i){return i == a.Get();}) > 0;
                if(change)
                    std::swap(a, b);
                Link::Connection connection{};
                connection.nodeAId = _executionGraphs[0].pinToNodes[a.Get()];
                connection.nodeBId = _executionGraphs[0].pinToNodes[b.Get()];
                auto& nodeAOutput = nodes[pinToNodes[a.Get()]].outputIds;
                auto& nodeBInput = nodes[pinToNodes[b.Get()]].inputIds;
                connection.nodeAAttribute = std::find(nodeAOutput.begin(), nodeAOutput.end(), a.Get()) - nodeAOutput.begin();
                connection.nodeBAttribute = std::find(nodeBInput.begin(), nodeBInput.end(), b.Get()) - nodeBInput.begin();
                
                bool wrongType = connection.nodeAAttribute < nodeAOutput.size() && connection.nodeBAttribute < nodeBInput.size() && typeid(*nodes[pinToNodes[a.Get()]].node->outputTypes[connection.nodeAAttribute]) != typeid(*nodes[pinToNodes[b.Get()]].node->inputTypes[connection.nodeBAttribute]);
                
                bool inputToOutput = isInputPin(a.Get()) ^ isInputPin(b.Get());

                if(wrongType)
                    showLabel("Incompatible types", {32, 45, 32, 180});
                if(!inputToOutput)
                    showLabel("One pin has to be output, the other input", {32, 45, 32, 180});
                if(a == b || pinToNodes[a.Get()] == pinToNodes[b.Get()] || wrongType || !inputToOutput)
                    nodes::RejectNewItem({255, 0, 0, 255}, 2.f);
                else{
                    showLabel("+ Create Link", {32, 45, 32, 180});
                    if(nodes::AcceptNewItem()){     // add check for validity
                        _executionGraphs[0].addLink(_curId, a.Get(), b.Get(), nodes[pinToNodes[a.Get()]].node->outputTypes[connection.nodeAAttribute]->color());
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
        while(nodes::QueryDeletedLink(&d)){
            if(nodes::AcceptDeletedItem()){
                _executionGraphs[0].removeLink(d.Get());
            }
        }
        nodes::NodeId n{};
        while(nodes::QueryDeletedNode(&n)){
            if(nodes::AcceptDeletedItem()){
                _executionGraphs[0].removeNode(n.Get());
            }
        }
    }
    nodes::EndDelete();

    // dialogue for node creation
    //if(thisFrameCreate)
    _popupPos = ImGui::GetMousePos();
    nodes::Suspend();
    nodes::NodeId n{};
    nodes::PinId p{};
    nodes::LinkId l{};
    if(nodes::ShowNodeContextMenu(&n)){
        ImGui::OpenPopup("Node Context Menu");
        _contextNodeId = n.Get();
    }
    else if(nodes::ShowPinContextMenu(&p)){
        ImGui::OpenPopup("Pin Context Menu");
        _contextPinId = p.Get();
    }
    else if(nodes::ShowLinkContextMenu(&l)){
        ImGui::OpenPopup("Link Context Menu");
        _contextLinkId = l.Get();
    }
    else if(nodes::ShowBackgroundContextMenu()){
        ImGui::OpenPopup("Create New Node");
        _createNewNode = false;
    }
    nodes::Resume();

    nodes::Suspend();
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {8,8});
    if(ImGui::BeginPopup("Node Context Menu")){
        ImGui::TextUnformatted("Node Context Menu");
        ImGui::Separator();
        if (ImGui::MenuItem("Delete"))
            nodes::DeleteNode(_contextNodeId);
        ImGui::EndPopup();
    }

    if(ImGui::BeginPopup("Pin Context Menu")){
        ImGui::TextUnformatted("Pin Context Menu");
        ImGui::Separator();
        ImGui::EndPopup();
    }

    if(ImGui::BeginPopup("Link Context Menu")){
        ImGui::TextUnformatted("Link Context Menu");
        ImGui::Separator();
        if(ImGui::MenuItem("Delete"))
            nodes::DeleteLink(_contextLinkId);
        ImGui::EndPopup();
    }

    if(ImGui::BeginPopup("Create New Node")){
        auto openPopupPosition = _popupPos;
        //ImGui::TextUnformatted("Create New Node");

        deriveData::Type* prevType{};
        if(_newLinkPinId > 0){
            Link::Connection connection{};
            connection.nodeAId = _executionGraphs[0].pinToNodes[_newLinkPinId];
            auto& nodeAOutput = nodes[pinToNodes[_newLinkPinId]].outputIds;
            connection.nodeAAttribute = std::find_if(nodeAOutput.begin(), nodeAOutput.end(), [&](int i){return i == _newLinkPinId;}) - nodeAOutput.begin();
            if(connection.nodeAAttribute < nodeAOutput.size())
                prevType = nodes[connection.nodeAId].node->outputTypes[connection.nodeAAttribute].get();
        }

        std::unique_ptr<deriveData::Node> node{};
        for(const auto& [name, entry]: deriveData::NodesRegistry::nodes){
            if(prevType && (entry.prototype->inputTypes.empty() || typeid(*prevType) != typeid(*entry.prototype->inputTypes[0])))
                continue;
            if(ImGui::MenuItem(name.c_str())){
                node = entry.create();
                //std::cout << "Creating " << name << std::endl;
            }
        }

        if(node){
            int nId = _curId++;
            _createNewNode = false;            
            nodes::SetNodePosition(nId, openPopupPosition);
            nodes.insert({nId, NodePins(std::move(node), &_curId)});
            for(auto inputId: nodes[nId].inputIds)
                pinToNodes[inputId] = nId;
            for(auto outputId: nodes[nId].outputIds)
                pinToNodes[outputId] = nId;

            if(_newLinkPinId != 0){
                // adding the link to the links list
                _executionGraphs[0].addLink(_curId, _newLinkPinId, nodes[nId].inputIds[0], nodes[nId].node->inputTypes[0]->color());
                _newLinkPinId = 0;
            }
        }

        ImGui::EndPopup();
    }
    else{
        _createNewNode = false;
        _newLinkPinId = {};
    }
    ImGui::PopStyleVar();
    nodes::Resume();

    nodes::End();

    ImGui::End();
}

void DeriveWorkbench::addDataset(std::string_view datasetId) 
{
    
}

void DeriveWorkbench::removeDataset(std::string_view datasetId) 
{
    
}

bool DeriveWorkbench::isInputPin(long pinId) 
{
    auto& [nodes, pinToNodes, links, linkToConnection, pinToLinks] = _executionGraphs[0];
    auto& node = nodes[pinToNodes[pinId]];
    return std::count(node.inputIds.begin(), node.inputIds.end(), pinId) > 0;
}

std::set<long> DeriveWorkbench::getActiveLinksRecursive(long node) 
{
    std::set<long> output;
    auto& [nodes, pinToNodes, links, linkToConnection, pinToLinks] = _executionGraphs[0];
    for(int i: nodes[node].inputIds){
        if(pinToLinks.count(i) && pinToLinks[i].size()){
            output.insert(i);
            long prefNode = linkToConnection[pinToLinks[i][0]].nodeAId;
            auto prefSet = getActiveLinksRecursive(prefNode);
            output.insert(prefSet.begin(), prefSet.end());
        }
    }
    return output;
}

void DeriveWorkbench::buildCacheRecursive(long node, RecursionData& data){
    auto& [nodes, pinToNodes, links, linkToConnection, pinToLinks] = _executionGraphs[0];
    auto& [activeLinks ,dataStorage, nodeInfos] = data;
    // check cache for previous nodes, if not generated, generate
    for(int i: irange(nodes[node].inputIds)){
        if(pinToLinks.count(nodes[node].inputIds[i]) == 0 || pinToLinks[nodes[node].inputIds[i]].empty())   // pin not connected, use inserted value
            continue;
        long linkId = pinToLinks[nodes[node].inputIds[i]][0];
        long prevNodeId = linkToConnection[linkId].nodeAId;
        if(nodeInfos.count(prevNodeId) == 0)
            buildCacheRecursive(prevNodeId, data);
    }

    // merging the data views for data processing
    // input data
    deriveData::float_column_views inputData;
    long inputDataSize{-1};
    bool equalDataLayout = true;
    const deriveData::column_memory_view<float>* curDataLayout{};
    const deriveData::column_memory_view<float>* outputLayout{};
    std::vector<int> inplaceIndices;
    for(long i: irange(nodes[node].inputIds)){
        if(pinToLinks.count(nodes[node].inputIds[i]) == 0 || pinToLinks[nodes[node].inputIds[i]].empty()){   // pin not connected, use inserted value
            inputData.push_back(nodes[node].node->inputTypes[i]->data());
        }
        else{   // pin connected
            long linkId = pinToLinks[nodes[node].inputIds[i]][0];
            long prevNodeId = linkToConnection[linkId].nodeAId;
            long prevNodeOutInd = linkToConnection[linkId].nodeAAttribute;
            // add input data and check for integrity
            inputData.push_back(data.nodeInfos[prevNodeId].outputViews[prevNodeOutInd]);
            if(inputDataSize > 1 && inputDataSize != inputData.back().size())
                throw std::runtime_error("DeriveWorkbench::buildCacheRecursive() Inputs to node (id=" + std::to_string(node) + ", type = \"" + nodes[node].node->middleText
                                        + "\") do not have the same size. Input at pin index " 
                                        + std::to_string(i) + " has size " + std::to_string(inputData.back().size()) + " while other inputs before have size " 
                                        + std::to_string(inputDataSize) + "!");
            // check for inplace
            bool inplace = --data.nodeInfos[prevNodeId].outputCounts[prevNodeOutInd] == 0;
            if(inplace)
                inplaceIndices.push_back(i);
            // check data layout for inflation
            if(curDataLayout)
                equalDataLayout &= inputData.back().equalDataLayout(*curDataLayout);
            else
                curDataLayout = &inputData.back();
            // setting the output dimensions
            if(outputLayout == nullptr || inputData.back().dimensionSizes.size() > outputLayout->dimensionSizes.size())
                outputLayout = &inputData.back();
        }
        inputDataSize = std::max<long>(inputDataSize, inputData.back().size());
    }
    assert(inputDataSize != -1);
    // handling vector cration nodes
    if(dynamic_cast<deriveData::DataCreationNode*>(nodes[node].node.get())){
        inputDataSize = inputData[0](0, 0);
        static deriveData::column_memory_view<float> layout;
        static uint32_t size;
        static uint32_t dimsIndex{0};
        size = inputDataSize;
        layout.dimensionSizes = deriveData::memory_view<uint32_t>(size);
        layout.columnDimensionIndices = deriveData::memory_view<uint32_t>(dimsIndex);
        outputLayout = &layout;
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
    for(int i: inplaceIndices){
        if(inputData[i].equalDataLayout(*outputLayout))
            memoryViewPool.insert(memoryViewPool.end(), inputData[i].cols.begin(), inputData[i].cols.end());
    }
    if(memoryViewPool.size() < nodes[node].node->outputChannels()){
        int storageSize = data.dataStorage.size();
        int missingBuffer = nodes[node].node->outputChannels() - memoryViewPool.size();
        data.dataStorage.insert(data.dataStorage.end(), missingBuffer, std::vector<float>(inputDataSize));
        for(int i: irange(storageSize, data.dataStorage.size()))
            memoryViewPool.push_back(data.dataStorage[i]);
    }
    for(int i: irange(nodes[node].outputIds)){
        deriveData::column_memory_view<float> columnMem;
        columnMem.dimensionSizes = outputLayout->dimensionSizes;
        columnMem.columnDimensionIndices = outputLayout->columnDimensionIndices;
        for(int j: irange(nodes[node].node->outputTypes[i]->data().cols))
            columnMem.cols.push_back(memoryViewPool[poolStart++]);
        outputData.push_back(std::move(columnMem));
    }
    assert(poolStart == memoryViewPool.size());

    // executing the node
    nodes[node].node->applyOperationCpu(inputData, outputData);

    // safing the cache and setting up the counts for the current data
    data.nodeInfos[node].outputCounts.resize(nodes[node].outputIds.size());
    for(int i: irange(nodes[node].outputIds)){
        for(const long link: pinToLinks[nodes[node].outputIds[i]]){
            if(data.activeLinks.count(link) > 1)
                ++data.nodeInfos[node].outputCounts[i];
        }

        if(deriveData::DatasetInputNode* datasetInput = dynamic_cast<deriveData::DatasetInputNode*>(nodes[node].node.get()))
            ++data.nodeInfos[node].outputCounts[i];   // make the dataset input not movable
    }
    data.nodeInfos[node].outputViews = outputData;
}

void DeriveWorkbench::executeGraph() 
{
    auto& [nodes, pinToNodes, links, linkToConnection, pinToLinks] = _executionGraphs[0];

    // checkfor output nodes
    std::set<long> outputNodes{};
    for(auto& [id, nodePins]: nodes){
        if(dynamic_cast<deriveData::OutputNode*>(nodePins.node.get()))
            outputNodes.insert(id);
    }
    if(outputNodes.empty()){
        std::cout << "DeriveWorkbench::executeGraph() No output nodes in graph. Nothing done, as calculations would be lost" << std::endl;
        return;
    }

    if(_executionGraphs[0].hasCircularConnections()){
        std::cout << "Recursion detected in the graph. This is not allowed! Fix before rerun" << std::endl;
        return;
    }

    RecursionData data{};
    for(auto node: outputNodes){
        auto activeLinks = getActiveLinksRecursive(node);
        data.activeLinks.insert(activeLinks.begin(), activeLinks.end());
    }
    for(auto node: outputNodes)
        buildCacheRecursive(node, data);

    // wiht the execution of all output nodes all changes are already applied
    std::cout << "Amount of data allocations: " << data.dataStorage.size() << std::endl;
}

DeriveWorkbench::DeriveWorkbench(std::list<DataSet>* datasets):
    _datasets(datasets)
{
    _editorContext = ax::NodeEditor::CreateEditor();
    _executionGraphs.resize(1);
    auto node = 
    _executionGraphs[0].nodes.insert({10000,NodePins(deriveData::CreateVec2Node::create(), &_curId)});
    for(int pin: _executionGraphs[0].nodes[10000].inputIds)
        _executionGraphs[0].pinToNodes[pin] = 10000;
    for(int pin: _executionGraphs[0].nodes[10000].outputIds)
        _executionGraphs[0].pinToNodes[pin] = 10000;
}

DeriveWorkbench::~DeriveWorkbench() 
{
    ax::NodeEditor::DestroyEditor(_editorContext);
}
