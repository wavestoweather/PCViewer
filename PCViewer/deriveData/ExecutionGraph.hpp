#pragma once
#include "Nodes.hpp"
#include "../imgui_nodes/imgui_node_editor.h"
#include <map>
#include <set>

struct Link{
    ax::NodeEditor::LinkId Id{};
    ax::NodeEditor::PinId pinAId{};
    ax::NodeEditor::PinId pinBId{};

    struct Connection{
        int nodeAId;
        int nodeBId;
        int nodeAAttribute;
        int nodeBAttribute;

        bool operator<(const Connection& c) const{
            return nodeAId < c.nodeAId || (nodeAId == c.nodeAId && nodeBId < c.nodeBId) || 
            (nodeBId == c.nodeBId && nodeAAttribute < c.nodeAAttribute) || (nodeAAttribute == c.nodeAAttribute && nodeBAttribute < c.nodeBAttribute); 
        }
    };
};

struct NodePins{
    std::unique_ptr<deriveData::Node> node;
    std::vector<int> inputIds;
    std::vector<int> outputIds;

    NodePins(std::unique_ptr<deriveData::Node> node = {}): node(std::move(node)){
        if(!node)
            return;
        static int id{};
        inputIds.resize(node->inputTypes.size());
        outputIds.resize(node->outputTypes.size());
        for(int i: irange(inputIds))
            inputIds[i] = id++;
        for(int i: irange(outputIds))
            outputIds[i] = id++;
    }
};

// handles the data and logic to edit and execute teh execution graph
struct ExecutionGraph{
    std::map<int, NodePins> nodes;                  // maps ids to nodes
    std::map<int, int> pinToNodes;    // maps pin ids to node ids
    std::map<Link::Connection, Link> links;         // maps which map

    bool hasCircularConnections() const{
        std::map<int, std::set<int>> connectedNodes;     // stores for each node id to which node id it connects
        for(const auto& [connection, link]: links)
            connectedNodes[connection.nodeAId].insert(connection.nodeBId);

        std::set<int> visited;
        for(const auto& [id, node]: nodes){
            if(visited.count(id) > 0)
                continue;
            visited.insert(id);
            std::set<int> follower = connectedNodes[id];
            std::set<int> curVisited = follower;
            while(follower.size()){
                int id = *follower.begin();
                follower.erase(id);
                curVisited.insert(id);
                follower.insert(connectedNodes[id].begin(), connectedNodes[id].end());
                for(int i: curVisited){
                    if(follower.count(i) > 0)
                        return true;
                }
            }
        }
        return false;
    }
};