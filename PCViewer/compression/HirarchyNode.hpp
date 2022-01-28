#pragma once

#include "../rTree/RTreeDynamic.h"
#include <string>
#include <map>
#include <mutex>
#include <shared_mutex>

// class which represents a single node in the compression hirarchy.
class HirarchyNode{
public:
    HirarchyNode():rTree(0), eps(0), epsMul(0), depth(0), maxDepth(0){};
    HirarchyNode(const std::vector<float>& pos, float inEps, float inEpsMul, uint32_t inDepth, uint32_t inMaxDepth);

    RTreeDynamic<uint32_t, float> rTree;
    uint32_t rTreeCount;                    //is maintained here, as rTree does not provide functionality for this
    std::map<uint32_t, HirarchyNode> leaders;//these are the children of the current node
    std::vector<float> followerData;        //stored as extra array, as for leaf nodes not leaders exist
    const float eps, epsMul;
    const uint32_t depth, maxDepth;


    // adds the data point to the hirarchy, forwards to the correct child
    void addDataPoint(const std::vector<float>& d);
    int calcCacheScore();
    HirarchyNode* getCacheNode(int& cacheScore);                    //returns the node of the whole tree with the highest cache score which is stored in cacheScore
    void cacheNode(const std::string_view& cachePath, const std::string& parentId, float* parentCenter, float parentEps, HirarchyNode* chacheNode);
    size_t getByteSize();
    std::shared_mutex& getMutex(){return _insertLock;};
private:
    static uint32_t _globalUpdateStamp;
    uint32_t _updateStamp{};
    uint32_t _leaderId{};
    std::shared_mutex _insertLock;
};