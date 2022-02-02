#pragma once
#include <vector>
#include <cstring>
#include <shared_mutex>

class HierarchyCreateNode{
public:
    virtual void addDataPoint(const std::vector<float>& d) = 0;
    virtual HierarchyCreateNode* getCacheNode(int& cacheScore) = 0;
    virtual void cacheNode(const std::string_view& cachePath, const std::string& parentId, float* parentCenter, float parentEps, HierarchyCreateNode* chacheNode) = 0;
    virtual size_t getByteSize() = 0;
    virtual std::shared_mutex& getMutex() = 0;
};