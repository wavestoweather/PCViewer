#include "HashNode.hpp"
#include <iostream>
#include <limits>
#include <fstream>
#include <cmath>

HashNode::HashNode(const std::vector<float> &pos, float inEps, float inEpsMul, uint32_t inDepth, uint32_t inMaxDepth) : eps(inEps),
                                                                                                                        epsMul(inEpsMul),
                                                                                                                        depth(inDepth),
                                                                                                                        maxDepth(inMaxDepth),
                                                                                                                        _pos(pos)
{
    addDataPoint(pos); //automatically adds itself to the follower data
}

void HashNode::addDataPoint(const std::vector<float> &d)
{
    std::unique_lock<std::shared_mutex> lock(_insertLock);
    uint32_t dataIndex = getChildIndex(d.size(), _pos.data(), d.data(), eps, eps * epsMul);

    auto &fd = followers[dataIndex];
    if (fd.fInfo.empty())
        fd.fInfo.resize(d.size() + 1, 0);
    float a = fd.fInfo.back() / (fd.fInfo.back() + 1);
    for (int i = 0; i < d.size(); ++i)
    {
        fd.fInfo[i] = a * fd.fInfo[i] + (1.f - a) * d[i]; //averaging the folower positions
    }
    fd.fInfo.back() += 1.0f; //incrementing follower count
    //forwarding to follower, if not the last
    if (depth < maxDepth)
    {
        if (!fd.fNode)
            fd.fNode = std::make_shared<HashNode>(d, eps * epsMul, epsMul, depth + 1, maxDepth);
        fd.fNode->addDataPoint(d);
        lock.unlock();
    }
    else
        lock.unlock();

    _updateStamp = ++_globalUpdateStamp;
}

long HashNode::calcCacheScore()
{
    return long(_updateStamp);
}

HierarchyCreateNode *HashNode::getCacheNode(long &cacheScore)
{
    long bestCache{std::numeric_limits<long>::max()};
    HierarchyCreateNode *bestNode{};
    for (auto &f : followers)
    {
        long tmpCache;
        f.second.fNode->getCacheNode(tmpCache);
        if (tmpCache < bestCache)
        {
            bestCache = tmpCache;
            bestNode = f.second.fNode.get();
        }
    }
    if (long c = calcCacheScore(); c < bestCache)
    {
        bestCache = c;
        bestNode = this;
    }
    cacheScore = bestCache;
    return bestNode;
}

void HashNode::cacheNode(const std::string_view &cachePath, const std::string &parentId, float *parentCenter, float parentEps, HierarchyCreateNode *chacheNode)
{
    size_t curInd = getChildIndex(_pos.size(), parentCenter, _pos.data(), parentEps, eps);
    std::string curId = parentId + "_" + std::to_string(curInd);
    if (this == chacheNode)
    {
        for (auto &f : followers)
        {
            if(f.second.fNode)
                f.second.fNode->cacheNode(cachePath, curId, _pos.data(), eps, f.second.fNode.get());
        }
        std::ofstream f(std::string(cachePath) + "/" + curId, std::ios_base::app | std::ios_base::binary); //opening an append filestream
        // adding fixed size information header
        f << _pos.size() + 1 << " " << followers.size() * (_pos.size() + 1) << " " << eps << "\n"; //space needed to easily be able to parse the file again
        std::vector<float> data(followers.size() * (_pos.size() + 1));
        uint32_t c = 0;
        for (auto &f : followers)
            for (float d : f.second.fInfo)
                data[c++] = d;
        f.write(reinterpret_cast<char *>(data.data()), data.size() * sizeof(data[0]));
        f << "\n";
        followers.clear();                                                                                 //deleting all leader nodes
    }
    int del = -1;
    for (auto &f : followers)
    {
        if(f.second.fNode)
            f.second.fNode->cacheNode(cachePath, curId, _pos.data(), eps, chacheNode);
        if (f.second.fNode.get() == chacheNode)
            del = f.first;
    }
    if (del >= 0) //removing child if cached
    {
        followers[del].fNode.reset();
    }
}

size_t HashNode::getByteSize()
{
    size_t size = followers.size() * sizeof(*followers.begin());
    size += size;
    for (auto &f : followers)
    {
        if (f.second.fNode)
            size += f.second.fNode->getByteSize();
    }
    return size;
}

uint32_t HashNode::getChildIndex(uint32_t dimensionality, const float *parentCenter, const float *childCenter, float parentEps, float childEps)
{
    size_t curInd{};
    uint32_t maxLeadersPerAx = static_cast<uint32_t>(ceil(parentEps / childEps));
    uint32_t multiplier = 1;
    for (int i = 0; i < dimensionality; ++i)
    {
        curInd += static_cast<size_t>((childCenter[i] - parentCenter[i] + parentEps) / (2 * parentEps) * (maxLeadersPerAx - 1) + .5) * multiplier;
        multiplier *= maxLeadersPerAx;
    }
    return curInd;
}

uint32_t HashNode::_globalUpdateStamp = 0;