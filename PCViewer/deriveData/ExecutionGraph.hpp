#pragma once
#include "Nodes.hpp"
#include <map>
#include <set>

struct Link{
    std::string name{};

    struct Connection{
        int nodeAId;
        int nodeBId;
        int nodeAAttribute;
        int nodeBAttribute;
    };
};

// handles the data and logic to edit and execute teh execution graph
struct ExecutionGraph{
    std::map<int, std::unique_ptr<deriveData::Node>> nodes;     // maps ids to nodes
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
        return false
    }
};