#include "HashLeaderNode.hpp"
#include <iostream>
#include <limits>
#include <fstream>
#include <sstream>
#include <cmath>
#include "../PCUtil.h"

HashLeaderNode::HashLeaderNode(const std::vector<float>& pos, float inEps, float inEpsMul, uint32_t inDepth, uint32_t inMaxDepth):
eps(inEps),
epsMul(inEpsMul),
depth(inDepth),
maxDepth(inMaxDepth)
{
    _id = getChildIndex(pos.size(), nullptr, pos.data(), inEps / inEpsMul, inEps);
    addDataPoint(pos);  //automatically adds itself to the follower data
}

void HashLeaderNode::addDataPoint(const std::vector<float>& d){
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
        auto fol =  std::make_shared<FollowerInfo>();
        {
        PCUtil::AverageWatch insTreeWatch(insertTreeT, insertTreeC);
        map[childIndex] = fol;
        }
        {
        PCUtil::AverageWatch insertDataWatch(insertDataCountT, insertDataCountC);
        for(float f: d)
            fol->pos.push_back(f);
        fol->count = 1;
        }
        if(depth < maxDepth)
            fol->follower = std::make_shared<HashLeaderNode>(d, eps * epsMul, epsMul, depth + 1, maxDepth);
        lock.unlock();
    }
    else{
        auto child = map[childIndex];
        {
        PCUtil::AverageWatch followerIncWatch(followerIncT, followerIncC);
        float a = float(child->count) / float(++child->count);  //automatically increments the counter
        for(int i = 0; i < d.size(); ++i){
            child->pos[i] = a * child->pos[i] + (1.f - a) * d[i];
        }
        }
        if(depth < maxDepth){
            std::shared_ptr<HashLeaderNode> fol;
            {
            PCUtil::AverageWatch folloerRecWatch(followerRetT, followerRetC);
            fol = child->follower;
            }
            if(fol){
                lock.unlock();
                fol->addDataPoint(d);
            }
            else{
                child->follower= std::make_shared<HashLeaderNode>(d, eps * epsMul, epsMul, depth + 1, maxDepth);
                lock.unlock();
            }
        }
        else
            lock.unlock();
    }
    _updateStamp = ++_globalUpdateStamp;
}

long HashLeaderNode::calcCacheScore(){
    return long(_updateStamp) - long(map.size());
}

HierarchyCreateNode* HashLeaderNode::getCacheNode(long& cacheScore){
    long bestCache{std::numeric_limits<long>::max()};
    HierarchyCreateNode* bestNode{};
    for(auto& f: map){
        long tmpCache;
        if(f.second->follower)
            f.second->follower->getCacheNode(tmpCache);
        if(tmpCache < bestCache){
            bestCache = tmpCache;
            bestNode = f.second->follower.get();
        }
    }
    if(long c = calcCacheScore(); c < bestCache){
        bestCache = c;
        bestNode = this;
    }
    cacheScore = bestCache;
    return bestNode;
}

void HashLeaderNode::getCacheNodes(HierarchyCacheInfo& info){
    for(auto& f: map){
        if(f.second->follower)
            f.second->follower->getCacheNodes(info);
    }
    if(long c = calcCacheScore(); info.queue.empty() || info.curByteSize < info.cachingSize || c < info.queue.top().score){    //push the current node
        size_t size =  map.size() * (sizeof(map.begin().operator*()) + sizeof(FollowerInfo) + map.begin()->second->pos.size() * sizeof(float));           //size from rTree (is being tracked inside the rTree)
        info.curByteSize += size;
        //deleting the topmost NodeInfo if enough nodes have been gathered to have more than cachingSize bytes
        if(info.queue.size() && info.curByteSize - info.queue.top().byteSize > info.cachingSize){
            info.curByteSize -= info.queue.top().byteSize;
            info.queue.pop();
        }
        info.queue.push({this, c, size});
    }
};

void HashLeaderNode::cacheNodes(CacheManagerInterface& cacheManager, const std::string& parentId, float* parentCenter, float parentEps, const  std::set<HierarchyCreateNode*>& cacheNodes){
    uint32_t rowSize = map.begin()->second->pos.size();
    uint32_t finalRowSize = rowSize + 1;
    size_t curInd = _id;
    std::string curId = parentId + "_" + std::to_string(curInd);
    bool found = cacheNodes.empty() || cacheNodes.find(this) != cacheNodes.end();
    if(found){
        for(auto f: map){
            if(f.second->follower)
                f.second->follower->cacheNodes(cacheManager, curId, nullptr, eps, {});
        }
        
        std::vector<float> outData(map.size() * finalRowSize);
        int i = 0;
        for(const auto& fol: map){
            for(int j = 0; j < rowSize; ++j)
                outData[i * finalRowSize + j] = fol.second->pos[j];
            outData[i * finalRowSize + rowSize] = fol.second->count;
            i++;
        }
        map = robin_hood::unordered_map<uint64_t, std::shared_ptr<FollowerInfo>>();
        std::stringstream f;    //creating the data stream
        // adding fixed size information header
        f << finalRowSize << " " << outData.size() << " " << eps << "\n";   //space needed to easily be able to parse the file again
        f.write(reinterpret_cast<char*>(outData.data()), outData.size() * sizeof(outData[0]));
        f << "\n";
        cacheManager.addData(curId, f);
    }
    for(auto& f: map){
        if(f.second->follower)
            f.second->follower->cacheNodes(cacheManager, curId, nullptr, eps, cacheNodes);
        if(cacheNodes.empty() || cacheNodes.find(f.second->follower.get()) != cacheNodes.end())
            f = {};         //setting the follower to a null ptr
    }
}

size_t HashLeaderNode::getByteSize(){
    //size from arrays in the class
    size_t size =  map.size() * (sizeof(map.begin().operator*()) + sizeof(FollowerInfo) + map.begin()->second->pos.size() * sizeof(float));           //size from rTree (is being tracked inside the rTree)
    for(auto& f: map){
        if(f.second->follower)
            size += f.second->follower->getByteSize();
    }
    return size;
}

uint64_t HashLeaderNode::getChildIndex(uint32_t dimensionality, const float* parentCenter, const float* childCenter, float parentEps, float childEps){
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

std::atomic<size_t> HashLeaderNode::_globalUpdateStamp{0};