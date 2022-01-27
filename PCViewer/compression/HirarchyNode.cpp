#include "HirarchyNode.hpp"

HirarchyNode::HirarchyNode(const std::vector<float>& pos, float inEps, float inEpsMul, uint32_t inDepth, uint32_t inMaxDepth):
rTree(pos.size()),
eps(inEps),
epsMul(inEpsMul),
depth(inDepth),
maxDepth(inMaxDepth)
{
    addDataPoint(pos);  //automatically adds itself to the follower data
}

void HirarchyNode::addDataPoint(const std::vector<float>& d){
    std::vector<uint32_t> closest;
    auto backInsert = [&](const uint32_t& id){closest.push_back(id); return false;}; // return false to quit when first was found
    rTree.Search(d.data(), d.data(), backInsert);
    if(closest.size()){         //found leader
        if(depth < maxDepth)    //only push down the hirarchy if not at leaf nodes
            leaders[closest.front()].addDataPoint(d);
    }
    else{                       //new leader/child has to be created
        std::vector<float> mins(d), maxs(d);
        for(int i = 0; i < d.size(); ++i){
            mins[i] -= eps;
            maxs[i] += eps;
        }
        rTree.Insert(mins.data(), maxs.data(), leaders.size());
        followerData.insert(followerData.end(), d.begin(), d.end());
        
        if(depth < maxDepth){   //only create new child and propagate data if not at leaf nodes
            leaders.emplace_back(d, eps * epsMul, epsMul, depth + 1, maxDepth);
        }
    }
}