#include "DeriveWorkbench.hpp"
#include "../imgui/imgui.h"
#include "../imgui/imgui_internal.h"
#include "../imgui_nodes/imgui_node_editor.h"
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
    nodes::SetCurrentEditor(_editorContext);
    nodes::Begin("DeriveWorkbench");


    auto& editorStyle = nodes::GetStyle();
    const ImVec4 headerColor{.1,.1,.1,1};
    const float pinIconSize = 15;

    auto& [nodes, pinToNodes, links, linkToConnection, pinToLinks] = _executionGraphs[0];
    
    auto cursorTopLeft = ImGui::GetCursorStartPos();

    nodes::Utilities::BlueprintNodeBuilder builder; // created without a header texture as not needed
    for(auto& [id, nodePins]: nodes){
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
            if(node->inputNames[i].size()){
                ImGui::TextUnformatted(node->inputNames[i].c_str());
                ImGui::Spring(0);
            }
            if(!isLinked){
                auto memoryView = node->inputTypes[i]->data();
                if(memoryView.size()){
                    switch(memoryView.size()){
                        case 1:
                            ImGui::PushItemWidth(50);
                            ImGui::InputFloat("##test", memoryView.data());
                            ImGui::PopItemWidth;
                        break;
                    }
                }
                ImGui::Spring(0);
            }
            ImGui::PopStyleVar();
            builder.EndInput();
        }

        // middle
        builder.Middle();
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
    if(!_createNewNode){
    if(nodes::BeginCreate()){
        nodes::PinId a, b;
        if(nodes::QueryNewNode(&a)){
            showLabel("+ Create Node", {32, 45, 32, 180});

            if(nodes::AcceptNewItem()){
                _createNewNode = true;
                _newLinkPinId = a.Get();
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
                connection.nodeAAttribute = std::find_if(nodeAOutput.begin(), nodeAOutput.end(), [&](int i){return i == a.Get();}) - nodeAOutput.begin();
                connection.nodeBAttribute = std::find_if(nodeBInput.begin(), nodeBInput.end(), [&](int i){return i == b.Get();}) - nodeBInput.begin();
                
                bool wrongType = connection.nodeAAttribute < nodeAOutput.size() && connection.nodeBAttribute < nodeBInput.size() && typeid(*nodes[pinToNodes[a.Get()]].node->outputTypes[connection.nodeAAttribute]) != typeid(*nodes[pinToNodes[b.Get()]].node->inputTypes[connection.nodeBAttribute]);
                if(wrongType)
                    showLabel("Incompatible types", {32, 45, 32, 180});
                if(a == b || pinToNodes[a.Get()] == pinToNodes[b.Get()] || wrongType)
                    nodes::RejectNewItem({255, 0, 0, 255}, 2.f);
                else{
                    showLabel("+ Create Link", {32, 45, 32, 180});
                    if(nodes::AcceptNewItem()){     // add check for validity
                        // adding the link to the links list
                        pinToLinks[a.Get()].push_back(_curId);
                        // unlinking links to the new input
                        if(pinToLinks[b.Get()].size()){
                            long id = pinToLinks[b.Get()][0];
                            auto& c = linkToConnection[id];
                            auto& aLinks = pinToLinks[links[c].pinAId.Get()];
                            aLinks.erase(std::find(aLinks.begin(), aLinks.end(), id));
                            auto& bLinks = pinToLinks[links[c].pinBId.Get()];
                            bLinks.erase(std::find(bLinks.begin(), bLinks.end(), id));
                            links.erase(c);
                            linkToConnection.erase(id);
                        }  
                        pinToLinks[b.Get()] = {_curId};
                        linkToConnection[_curId] = connection;
                        links[connection] = {_curId++,a,b, nodes[pinToNodes[a.Get()]].node->outputTypes[connection.nodeAAttribute]->color()};
                    }
                } 
            }
        }
    }
    nodes::EndCreate();
    }
    //else
    //    _newLinkPinId = 0;

    // handle deletion action
    if(nodes::BeginDelete()){
        nodes::LinkId d{};
        while(nodes::QueryDeletedLink(&d)){
            if(nodes::AcceptDeletedItem()){
                Link::Connection c = linkToConnection[d.Get()];
                linkToConnection.erase(d.Get());
                auto& l = links[c];
                auto& aLinks = pinToLinks[l.pinAId.Get()];
                aLinks.erase(std::find(aLinks.begin(), aLinks.end(), l.Id.Get()));
                auto& bLinks = pinToLinks[l.pinBId.Get()];
                bLinks.erase(std::find(bLinks.begin(), bLinks.end(), l.Id.Get()));
                links.erase(c);
            }
        }
        nodes::NodeId n{};
        while(nodes::QueryDeletedNode(&n)){
            if(nodes::AcceptDeletedItem()){
                for(auto& pin: nodes[n.Get()].inputIds)
                    pinToNodes.erase(pin);
                for(auto & pin: nodes[n.Get()].outputIds)
                    pinToNodes.erase(pin);
                nodes.erase(n.Get());
            }
        }
    }
    nodes::EndDelete();

    // dialogue for node creation
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
            if(prevType && typeid(*prevType) != typeid(*entry.prototype->inputTypes[0]))
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
                Link::Connection connection{};
                connection.nodeAId = _executionGraphs[0].pinToNodes[_newLinkPinId];
                connection.nodeBId = nId;
                auto& nodeAOutput = nodes[pinToNodes[_newLinkPinId]].outputIds;
                connection.nodeAAttribute = std::find_if(nodeAOutput.begin(), nodeAOutput.end(), [&](int i){return i == _newLinkPinId;}) - nodeAOutput.begin();
                connection.nodeBAttribute = 0;
                pinToLinks[_newLinkPinId].push_back(_curId);
                pinToLinks[nodes[nId].inputIds[0]] = {_curId};
                linkToConnection[_curId] = connection;
                links[connection] = {_curId++, _newLinkPinId, nodes[nId].inputIds[0], nodes[connection.nodeAId].node->outputTypes[connection.nodeAAttribute]->color()};
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

    // drawing as overlay the execute button
    ImGui::SetCursorScreenPos(ImGui::GetWindowPos() + ImVec2{ImGui::GetWindowSize().x / 2, 30});
    if(ImGui::Button("Execute Graph")){
        std::cout << "Gotcha" << std::endl;
    }

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
