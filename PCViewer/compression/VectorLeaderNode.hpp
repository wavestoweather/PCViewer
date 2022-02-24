#pragma once

#include "../rTree/RTreeDynamic.h"
#include "HierarchyCreateNode.hpp"
#include <string>
#include <map>
#include <mutex>
#include <shared_mutex>
#include "../rTree/RTreeUtil.h"
#include <memory>
#include <atomic>
#include "CacheManagerInterface.hpp"

class VectorLeaderNode;
// class which represents a single node in the compression hirarchy.
class VectorLeaderNode: public HierarchyCreateNode{
public:
    VectorLeaderNode():rTree(nullptr), eps(0), epsMul(0), depth(0), maxDepth(0){};
    VectorLeaderNode(const std::vector<float>& pos, float inEps, float inEpsMul, uint32_t inDepth, uint32_t inMaxDepth);

    //Vector leader node stores all follower data in vectors to provide fast access
    //Further we are using a structure of arrays to provide faster access for random accesses
    //None of the below vectors deletes data when a child is being cached, except that in follower the pointer is set to a nullptr
    std::shared_ptr<RTreeUtil::RTreeAPI<uint32_t, float>> rTree;    //the rTree is the spatial index for the follower
    std::vector<std::shared_ptr<VectorLeaderNode>> follower;        //these are the children of the current node. If a child is being cached and should be destroyed, the pionter is simply set to nullptr
    std::vector<float> followerData;                                //stores all centers of the children nodes. The center is appended only once, then only the counts are increased
    std::vector<uint32_t> followerCounts;                           //stores for each child the count. This is also kept track fo for leaf nodes
    const float eps, epsMul;
    const uint32_t depth, maxDepth;


    // adds the data point to the hirarchy, forwards to the correct child
    void addDataPoint(const std::vector<float>& d);
    long calcCacheScore();      //returns teh score for the current node. The lower the number, the more likely it is this node is being cached
    HierarchyCreateNode* getCacheNode(long& cacheScore);                    //returns the node of the whole tree with the highest cache score which is stored in cacheScore
    void getCacheNodes(HierarchyCacheInfo& info);
    void cacheNode(const std::string_view& cachePath, const std::string& parentId, float* parentCenter, float parentEps, HierarchyCreateNode* chacheNode){throw std::runtime_error{"Standard caching not supported"};};
    void cacheNodes(CacheManagerInterface& cacheManager, const std::string& parentId, float* parentCenter, float parentEps, const std::set<HierarchyCreateNode*>& cacheNodes);
    size_t getByteSize();
    std::shared_mutex& getMutex(){return _insertLock;};

    static uint32_t getChildIndex(uint32_t dimensionality, float* parentCenter, float* childCenter, float parentEps, float childEps);
private:
    static std::atomic<size_t> _globalUpdateStamp;
    uint32_t _updateStamp{};
    uint32_t _leaderId{};
    std::shared_mutex _insertLock;
};