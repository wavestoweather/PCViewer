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
        long nodeAId;
        long nodeBId;
        long nodeAAttribute;
        long nodeBAttribute;

        bool operator<(const Connection& c) const{
            return nodeAId < c.nodeAId || (nodeAId == c.nodeAId && nodeBId < c.nodeBId) || 
            (nodeBId == c.nodeBId && nodeAAttribute < c.nodeAAttribute) || (nodeAAttribute == c.nodeAAttribute && nodeBAttribute < c.nodeBAttribute); 
        }
    };
};

struct NodePins{
    std::unique_ptr<deriveData::Node> node;
    std::vector<long> inputIds;
    std::vector<long> outputIds;

    NodePins(std::unique_ptr<deriveData::Node> n = {}, long* curId = {}): node(std::move(n)){
        if(!node)
            return;
        assert(curId);
        inputIds.resize(node->inputTypes.size());
        outputIds.resize(node->outputTypes.size());
        for(int i: irange(inputIds))
            inputIds[i] = (*curId)++;
        for(int i: irange(outputIds))
            outputIds[i] = (*curId)++;
    }
};

// handles the data and logic to edit and execute teh execution graph
struct ExecutionGraph{
    std::map<long, NodePins> nodes;                     // maps ids to nodes
    std::map<long, long> pinToNodes;                    // maps pin ids to node ids
    std::map<Link::Connection, Link> links;             // maps which map
    std::map<long, Link::Connection> linkToConnection;  // maps a link id to the connection
    std::map<long, std::vector<long>> pinToLinks;       // maps pin ids to a vector of all link ids that are connected

    bool hasCircularConnections() const{
        std::map<long, std::set<long>> connectedNodes;     // stores for each node id to which node id it connects
        for(const auto& [connection, link]: links)
            connectedNodes[connection.nodeAId].insert(connection.nodeBId);

        std::set<long> visited;
        for(const auto& [id, node]: nodes){
            if(visited.count(id) > 0)
                continue;
            visited.insert(id);
            std::set<long> follower = connectedNodes[id];
            std::set<long> curVisited = follower;
            while(follower.size()){
                long id = *follower.begin();
                follower.erase(id);
                curVisited.insert(id);
                follower.insert(connectedNodes[id].begin(), connectedNodes[id].end());
                for(long i: curVisited){
                    if(follower.count(i) > 0)
                        return true;
                }
            }
        }
        return false;
    }
};