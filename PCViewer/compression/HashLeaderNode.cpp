#include "HashVectorLeaderNode.hpp"
#include <iostream>
#include <limits>
#include <fstream>
#include <sstream>
#include <cmath>
#include "../PCUtil.h"

HashVectorLeaderNode::HashVectorLeaderNode(const std::vector<float>& pos, float inEps, float inEpsMul, uint32_t inDepth, uint32_t inMaxDepth):
eps(inEps),
epsMul(inEpsMul),
depth(inDepth),
maxDepth(inMaxDepth)
{
    _id = getChildIndex(pos.size(), nullptr, pos.data(), inEps / inEpsMul, inEps);
    addDataPoint(pos);  //automatically adds itself to the follower data
}

void HashVectorLeaderNode::addDataPoint(const std::vector<float>& d){
    //tmp varialbes for stopwatches
    static uint32_t searchC{}, followerIncC{}, followerRetC{}, addC{}, insertC{}, insertDataCountC{}, insertTreeC{};
    static float searchT{}, followerIncT{}, followerRetT{}, addT{}, insertT{}, insertDataCountT{}, insertTreeT{};
    uint64_t childIndex = getChildIndex(d.size(), nullptr, d.data(), eps, eps * epsMul);
    //end tmp variables
    PCUtil::AverageWatch addWatch(addT, addC);

    std::unique_lock<std::shared_mutex> lock(_insertLock);
    bool found;
    {
    PCUtil::AverageWatch searchWatch(searchT, searchC);
    found = map.contains(childIndex);
    }
    if(!found){
        PCUtil::AverageWatch addWatch(insertT, insertC);
        {
        PCUtil::AverageWatch insTreeWatch(insertTreeT, insertTreeC);
        map[childIndex] = followerCounts.size();
        }
        {
        PCUtil::AverageWatch insertDataWatch(insertDataCountT, insertDataCountC);
        for(float f: d)
            followerData.push_back(f);
        followerCounts.push_back(1);
        }
        if(depth < maxDepth)
            follower.push_back(std::make_shared<HashVectorLeaderNode>(d, eps * epsMul, epsMul, depth + 1, maxDepth));
        lock.unlock();
    }
    else{
        uint32_t ind = map[childIndex];
        {
        PCUtil::AverageWatch followerIncWatch(followerIncT, followerIncC);
        float a = float(followerCounts[ind]) / float(++followerCounts[ind]);  //automatically increments the counter
        for(int i = 0; i < d.size(); ++i){
            followerData[ind * d.size() + i] = a * followerData[ind * d.size() + i] + (1.f - a) * d[i];
        }
        }
        if(depth < maxDepth){
            std::shared_ptr<HashVectorLeaderNode> fol;
            {
            PCUtil::AverageWatch folloerRecWatch(followerRetT, followerRetC);
            fol = follower[ind];
            }
            if(fol){
                lock.unlock();
                fol->addDataPoint(d);
            }
            else{
                follower[ind] = std::make_shared<HashVectorLeaderNode>(d, eps * epsMul, epsMul, depth + 1, maxDepth);
                lock.unlock();
            }
        }
        else
            lock.unlock();
    }
    _updateStamp = ++_globalUpdateStamp;
}

long HashVectorLeaderNode::calcCacheScore(){
    return long(_updateStamp) - long(followerCounts.size());
}

HierarchyCreateNode* HashVectorLeaderNode::getCacheNode(long& cacheScore){
    long bestCache{std::numeric_limits<long>::max()};
    HierarchyCreateNode* bestNode{};
    for(auto& f: follower){
        long tmpCache;
        if(f)
            f->getCacheNode(tmpCache);
        if(tmpCache < bestCache){
            bestCache = tmpCache;
            bestNode = f.get();
        }
    }
    if(long c = calcCacheScore(); c < bestCache){
        bestCache = c;
        bestNode = this;
    }
    cacheScore = bestCache;
    return bestNode;
}

void HashVectorLeaderNode::getCacheNodes(HierarchyCacheInfo& info){
    for(auto& f: follower){
        if(f)
            f->getCacheNodes(info);
    }
    if(long c = calcCacheScore(); info.queue.empty() || info.curByteSize < info.cachingSize || c < info.queue.top().score){    //push the current node
        size_t size = followerData.size() * sizeof(followerData[0]) + follower.size() * sizeof(follower[0]) + followerCounts.size() * sizeof(followerCounts[0]);
        size +=  map.size() * sizeof(map.begin().operator*());           //size from rTree (is being tracked inside the rTree)
        info.curByteSize += size;
        //deleting the topmost NodeInfo if enough nodes have been gathered to have more than cachingSize bytes
        if(info.queue.size() && info.curByteSize - info.queue.top().byteSize > info.cachingSize){
            info.curByteSize -= info.queue.top().byteSize;
            info.queue.pop();
        }
        info.queue.push({this, c, size});
    }
};

void HashVectorLeaderNode::cacheNodes(CacheManagerInterface& cacheManager, const std::string& parentId, float* parentCenter, float parentEps, const  std::set<HierarchyCreateNode*>& cacheNodes){
    uint32_t rowSize = followerData.size() / followerCounts.size();
    uint32_t finalRowSize = rowSize + 1;
    size_t curInd = _id;
    std::string curId = parentId + "_" + std::to_string(curInd);
    bool found = cacheNodes.empty() || cacheNodes.find(this) != cacheNodes.end();
    if(found){
        map = robin_hood::unordered_map<uint64_t, uint32_t>();
        for(auto& f: follower){
            if(f)
                f->cacheNodes(cacheManager, curId, followerData.data(), eps, {});
        }
        follower = std::vector<std::shared_ptr<HashVectorLeaderNode>>();    //deleting all leader nodes
        
        std::vector<float> outData(followerCounts.size() * finalRowSize);
        for(int i = 0; i < followerCounts.size(); ++i){
            for(int j = 0; j < rowSize; ++j)
                outData[i * finalRowSize + j] = followerData[i * rowSize + j];
            outData[i * finalRowSize + rowSize] = followerCounts[i];
        }
        followerData = std::vector<float>();
        followerCounts = std::vector<uint32_t>();
        std::stringstream f;    //creating the data stream
        // adding fixed size information header
        f << finalRowSize << " " << outData.size() << " " << eps << "\n";   //space needed to easily be able to parse the file again
        f.write(reinterpret_cast<char*>(outData.data()), outData.size() * sizeof(outData[0]));
        f << "\n";
        cacheManager.addData(curId, f);
    }
    for(auto& f: follower){
        if(f)
            f->cacheNodes(cacheManager, curId, followerData.data(), eps, cacheNodes);
        if(cacheNodes.empty() || cacheNodes.find(f.get()) != cacheNodes.end())
            f = {};         //setting the follower to a null ptr
    }
}

size_t HashVectorLeaderNode::getByteSize(){
    //size from arrays in the class
    size_t size = followerData.size() * sizeof(followerData[0]) + follower.size() * sizeof(follower[0]) + followerCounts.size() * sizeof(followerCounts[0]);
    size += map.size() * sizeof(map.begin().operator*()); ;           //size from map
    for(auto& f: follower){
        if(f)
            size += f->getByteSize();
    }
    return size;
}

uint64_t HashVectorLeaderNode::getChildIndex(uint32_t dimensionality, const float* parentCenter, const float* childCenter, float parentEps, float childEps){
    uint64_t curInd{};
    uint32_t maxLeadersPerAx = static_cast<uint32_t>(ceil(parentEps / childEps));
    uint64_t multiplier = 1;
    for(int i = 0; i < dimensionality; ++i){
        uint32_t parentInd = childCenter[i] / (parentEps * 2);
        float bottom = childCenter[i] * parentEps * 2;          //bottom border of 
        float nor = childCenter[i] - bottom;
        uint32_t ind = nor / (childEps * 2);
        curInd += ind * multiplier;
        multiplier *= maxLeadersPerAx;
    }
    return curInd;
}

std::atomic<size_t> HashVectorLeaderNode::_globalUpdateStamp{0};