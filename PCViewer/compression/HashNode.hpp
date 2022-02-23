#pragma once

#include "HierarchyCreateNode.hpp"
#include <string>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <memory>
#include "../robin_hood_map/robin_hood.h"

// class which represents a single node in the compression hirarchy.
class HashNode: public HierarchyCreateNode{
public:
    struct Follower{
        std::shared_ptr<HashNode> fNode;
        std::vector<float> fInfo;
    };

    HashNode():eps(0), epsMul(0), depth(0), maxDepth(0){};
    HashNode(const std::vector<float>& pos, float inEps, float inEpsMul, uint32_t inDepth, uint32_t inMaxDepth);

    robin_hood::unordered_map<uint32_t, Follower> followers; //these are the children of the current node
    const float eps, epsMul;
    const uint32_t depth, maxDepth;


    // adds the data point to the hirarchy, forwards to the correct child
    void addDataPoint(const std::vector<float>& d);
    long calcCacheScore();
    HierarchyCreateNode* getCacheNode(long& cacheScore);                    //returns the node of the whole tree with the highest cache score which is stored in cacheScore
    void cacheNode(const std::string_view& cachePath, const std::string& parentId, float* parentCenter, float parentEps, HierarchyCreateNode* chacheNode);
    void cacheNode(CacheManagerInterface& cacheManager, const std::string& parentId, float* parentCenter, float parentEps, HierarchyCreateNode* chacheNode){throw std::runtime_error{"Cache manager not supported for hash node"};};
    size_t getByteSize();
    std::shared_mutex& getMutex(){return _insertLock;};

    static uint32_t getChildIndex(uint32_t dimensionality, const float* parentCenter, const float* childCenter, float parentEps, float childEps);
private:
    static uint32_t _globalUpdateStamp;
    uint32_t _updateStamp{};
    uint32_t _leaderId{};
    std::shared_mutex _insertLock;
    std::vector<float> _pos;
};