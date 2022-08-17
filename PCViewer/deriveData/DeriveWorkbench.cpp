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
    nodes::Begin("DeriveWorkbench", {800, 800});


    auto& editorStyle = nodes::GetStyle();
    static int nodeId = 0;
    static int linkId = 0;
    const ImVec4 headerColor{.1,.1,.1,1};
    const float pinIconSize = 8;

    auto& [nodes, pinToNodes, links] = _executionGraphs[0];
    
    auto cursorTopLeft = ImGui::GetCursorStartPos();
    for(auto& [id, nodePins]: nodes){
        auto& node = nodePins.node;
        nodes::BeginNode(id);
        // drawing header
        //ImGui::Text(node->name.c_str());
        //ImGui::Separator();
        
        // drawing middle
        ImGui::BeginChild("left pins", {80, 80}, true);
        for(int i: irange(node->inputTypes)){
            auto alpha = ImGui::GetStyle().Alpha;

            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
            ax::Widgets::IconType iconType = node->inputTypes[i]->iconType();
            auto c = node->inputTypes[i]->color();
            auto color = ImColor(c[0], c[1], c[2], c[3]);
            nodes::BeginPin(nodePins.inputIds[i], nodes::PinKind::Input);
            ax::Widgets::Icon({pinIconSize, pinIconSize}, iconType, true, color, ImColor(32, 32, 32, int(alpha * 255)));

            ImGui::PopStyleVar();

            nodes::EndPin();
        }
        ImGui::EndChild();
        ImGui::SameLine();
        //ImGui::BeginChild("right pins");
        for(int i: irange(node->outputTypes)){
            auto alpha = ImGui::GetStyle().Alpha;

            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
            ax::Widgets::IconType iconType = node->inputTypes[i]->iconType();
            auto c = node->outputTypes[i]->color();
            auto color = ImColor(c[0], c[1], c[2], c[3]);
            nodes::BeginPin(nodePins.outputIds[i], nodes::PinKind::Output);
            ax::Widgets::Icon({pinIconSize, pinIconSize}, iconType, true, color, ImColor(32, 32, 32, int(alpha * 255)));
            ImGui::PopStyleVar();

            nodes::EndPin();
        }
        //ImGui::EndChild();

        // drawing end

        nodes::EndNode();
    }

    // handle creation action
    if(nodes::BeginCreate()){
        nodes::PinId a, b;
        if(nodes::QueryNewNode(&a)){
            if(nodes::AcceptNewItem()){
                // TODO: 
            }
        }

        a = {};
        if (nodes::QueryNewLink(&a, &b)){
            if(a && b){ // if link was created
                if(nodes::AcceptNewItem()){     // add check for validity
                    // adding the link to the links list
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
                    links[connection] = {linkId++,a,b};

                    // adding the link to the nodes editor
                    nodes::Link(links[connection].Id, a, b);

                    //if(not valid link)
                    //    nodes::RejectNewItem();
                }
            }
        }
    }
    nodes::EndCreate();

    // handle deletion action
    if(nodes::BeginDelete()){
        nodes::LinkId d;
        while(nodes::QueryDeletedLink(&d)){
            if(nodes::AcceptDeletedItem()){
                for(auto& [connection, link]: links){
                    if(link.Id == d){
                        links.erase(connection);
                        break;
                    }
                }
            }
        }
    }
    nodes::EndDelete();

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
    auto node = 
    _executionGraphs[0].nodes.insert({10000,NodePins(deriveData::MultiplicationInverseNode::create())});
    for(int pin: _executionGraphs[0].nodes[10000].inputIds)
        _executionGraphs[0].pinToNodes[pin] = 10000;
    for(int pin: _executionGraphs[0].nodes[10000].outputIds)
        _executionGraphs[0].pinToNodes[pin] = 10000;
}

DeriveWorkbench::~DeriveWorkbench() 
{
    ax::NodeEditor::DestroyEditor(_editorContext);
}
