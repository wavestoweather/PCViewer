#pragma once
#include <vector>
#include <cstring>
#include <shared_mutex>
#include <queue>
#include <set>
#include "CacheManagerInterface.hpp"



// interface for the hierarchy create node. To cluster the dataset an instance from a class implementing this interface is being
// created and all data points are inserted via the
// -'addDataPoint()' method
// To avoid ram overflow the
// -'getCacheNode()' method should return a node which should be cached, which is then initiated by calling
// -'cacheNode()'
// The caching is implemented in this way to be able to pass information from the root node to all siblings
// -'cacheNode()' is automatically called when problematic ram usage was detected
// -'getByteSize()' returns the byte size of the whole compression hierarchy and is used to instantiate caching
// -'getMutex' should return a mutex which can exclusively lock the root node 

// The final writeout is done via a cacheNode call on the root node.
// The output data format is as follows:
//  Section CacheInstance:
//      [number]dimensions [number]fullDataSize [number]epsilon \n      //note: The newline character has to be taken into account when loading the data!
//      [vector<bytes>] tableData                                       //note: The table data is column major linearized
//      \n                                                              //note: The newline character has to be taken into account when loading the data!
// --------
// [vector<CacheInstance]cacheInstances                                 //note:The final file is simply a vector of cache instances
class HierarchyCreateNode{
public:
    struct NodeInfo{
        HierarchyCreateNode* node;
        long score;
        size_t byteSize;
        bool operator<(const NodeInfo& other) const{
            return score < other.score;
        }
        bool operator>(const NodeInfo& other) const{
            return score > other.score;
        }
    };

    struct HierarchyCacheInfo{
        std::priority_queue<NodeInfo, std::vector<NodeInfo>, std::greater<NodeInfo>> queue;
        uint32_t curByteSize;
        uint32_t cachingSize;
    };

    virtual void addDataPoint(const std::vector<float>& d) = 0;
    virtual HierarchyCreateNode* getCacheNode(long& cacheScore) = 0;
    virtual void getCacheNodes(HierarchyCacheInfo& info) = 0;
    virtual void cacheNode(const std::string_view& cachePath, const std::string& parentId, float* parentCenter, float parentEps, HierarchyCreateNode* chacheNode) = 0;
    virtual void cacheNodes(CacheManagerInterface& cacheInterfaceManager, const std::string& parentId, float* parentCenter, float parentEps, const std::set<HierarchyCreateNode*>& cacheNodes) = 0; //caches all nodes in the cacheNodes set. If cacheNodes set is empty, everything should be cached
    virtual size_t getByteSize() = 0;
    virtual std::shared_mutex& getMutex() = 0;
};