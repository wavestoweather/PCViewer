#pragma once

#include "../rTree/RTreeDynamic.h"
#include <string>

// class which represents a single node in the compression hirarchy.
class HirarchyNode{
public:
    HirarchyNode(const std::vector<float>& pos, float inEps, float inEpsMul, uint32_t inDepth, uint32_t inMaxDepth);

    RTreeDynamic<uint32_t, float> rTree;
    std::vector<HirarchyNode> leaders;      //these are the children of the current node
    std::vector<float> followerData;        //stored as extra array, as for leaf nodes not leaders exist
    const float eps, epsMul;
    const uint32_t depth, maxDepth;

    // adds the data point to the hirarchy, forwards to the correct child
    void addDataPoint(const std::vector<float>& d);
private:

};