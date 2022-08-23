#pragma once
#include "Nodes.hpp"
#include "../imgui_nodes/imgui_node_editor.h"
#include <map>
#include <set>

struct Link{
    ax::NodeEditor::LinkId Id{};
    ax::NodeEditor::PinId pinAId{};
    ax::NodeEditor::PinId pinBId{};
    ImVec4 color{1.f, 1.f, 1.f, 1.f};

    struct Connection{
        long nodeAId;
        long nodeBId;
        long nodeAAttribute;
        long nodeBAttribute;

        bool operator==(const Connection& c) const{
            return nodeAId == c.nodeAId && nodeBId == c.nodeBId && nodeAAttribute == c.nodeAAttribute && nodeBAttribute == c.nodeBAttribute;
        }
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

    void addNode(long& curId, std::unique_ptr<deriveData::Node>&& node){
        long newId = curId++;
        nodes.insert({newId++, NodePins(std::move(node), &curId)});
        const auto& curNode = nodes[newId];
        for(long i: curNode.inputIds)
            pinToNodes[i] = newId;
        for(long i: curNode.outputIds)
            pinToNodes[i] = newId;
    }
    void removeNode(long nodeId){
        const auto& curNode = nodes[nodeId];
        for(long i: curNode.inputIds){
            while(pinToLinks.count(i) > 0 && pinToLinks[i].size()){
                removeLink(pinToLinks[i][0]);
            }
            pinToLinks.erase(i);
            pinToNodes.erase(i);
        }
        for(long i: curNode.outputIds){
            while(pinToLinks.count(i) > 0 && pinToLinks[i].size()){
                removeLink(pinToLinks[i][0]);
            }
            pinToLinks.erase(i);
            pinToNodes.erase(i);
        }
        nodes.erase(nodeId);
    }
    void addLink(long& curId, long pinAId, long pinBId, const ImVec4& color = {1.f, 1.f, 1.f, 1.f}){
        Link::Connection c{};
        c.nodeAId = pinToNodes[pinAId];
        c.nodeBId = pinToNodes[pinBId];
        c.nodeAAttribute = std::find(nodes[c.nodeAId].outputIds.begin(), nodes[c.nodeAId].outputIds.end(), pinAId) - nodes[c.nodeAId].outputIds.begin();
        c.nodeBAttribute = std::find(nodes[c.nodeBId].inputIds.begin(), nodes[c.nodeBId].inputIds.end(), pinBId) - nodes[c.nodeBId].inputIds.begin();
        if(links.count(c))  // link already exists, nothing to do
            return;
        long linkId = curId++;
        links[c] = {linkId, pinAId, pinBId, color};
        linkToConnection[linkId] = c;
        if(pinToLinks.count(pinBId) && pinToLinks[pinBId].size())
            removeLink(pinToLinks[pinBId][0]);
        pinToLinks[pinAId].push_back(linkId);
        pinToLinks[pinBId] = {linkId};
    }
    void removeLink(long link){
        const auto& connection = linkToConnection[link];
        const auto& linkRef = links[connection];
        if(pinToLinks.count(linkRef.pinAId.Get()) > 0 && pinToLinks[linkRef.pinAId.Get()].size()){
            auto& mappedLinks = pinToLinks[linkRef.pinAId.Get()];
            mappedLinks.erase(std::find(mappedLinks.begin(), mappedLinks.end(), link));
            if(mappedLinks.empty())
                pinToLinks.erase(linkRef.pinAId.Get());
        }
        if(pinToLinks.count(linkRef.pinBId.Get()) > 0 && pinToLinks[linkRef.pinBId.Get()].size()){
            auto& mappedLinks = pinToLinks[linkRef.pinBId.Get()];
            mappedLinks.erase(std::find(mappedLinks.begin(), mappedLinks.end(), link));
            if(mappedLinks.empty())
                pinToLinks.erase(linkRef.pinBId.Get());
        }
        links.erase(connection);
        linkToConnection.erase(link);
    }

    bool hasCircularConnections() const{
        std::map<long, std::set<long>> connectedNodes;     // stores for each node id to which node id it connects
        for(const auto& [connection, link]: links)
            connectedNodes[connection.nodeAId].insert(connection.nodeBId);

        std::set<long> visited;
        for(const auto& [nodeId, node]: nodes){
            if(visited.count(nodeId) > 0)
                continue;
            visited.insert(nodeId);
            std::set<long> follower = connectedNodes[nodeId];
            std::set<long> curVisited = follower;
            while(follower.size()){
                long id = *follower.begin();
                follower.erase(id);
                curVisited.insert(id);
                follower.insert(connectedNodes[id].begin(), connectedNodes[id].end());
                if(follower.count(nodeId) > 0)
                    return true;
            }
        }
        return false;
    }
};