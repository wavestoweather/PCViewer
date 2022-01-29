#include "HirarchyNode.hpp"
#include <iostream>
#include <limits>
#include <fstream>

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
    std::unique_lock<std::shared_mutex> lock(_insertLock);
    rTree.Search(d.data(), d.data(), backInsert);
    if(closest.size()){         //found leader
        lock.unlock();         //can release lock already
        followerData[closest.front() * (rTree.NUMDIMS + 1) + rTree.NUMDIMS] += 1;
        if(depth < maxDepth)    //only push down the hirarchy if not at leaf nodes
            leaders[closest.front()].addDataPoint(d);
    }
    else{                       //new leader/child has to be created
        std::vector<float> mins(d), maxs(d);
        for(int i = 0; i < d.size(); ++i){
            mins[i] -= eps;
            maxs[i] += eps;
        }
        uint32_t newId = _leaderId++;
        rTree.Insert(mins.data(), maxs.data(), newId);
        followerData.insert(followerData.end(), d.begin(), d.end());
        followerData.push_back(1);  //adding the counter for each row
        if(int x  = followerData.size() / d.size(); x > 2e6){
            std::cout << "\rNote: large HirarchyNode with " << x << "elements";
        }
        
        if(depth < maxDepth){   //only create new child and propagate data if not at leaf nodes
            leaders.try_emplace(newId, d, eps * epsMul, epsMul, depth + 1, maxDepth);
        }
        lock.unlock();     //can release the lock, all critical things already done
    }
    _updateStamp = ++_globalUpdateStamp;
}

int HirarchyNode::calcCacheScore(){
    return followerData.size() / rTree.NUMDIMS - _updateStamp;
}

HirarchyNode* HirarchyNode::getCacheNode(int& cacheScore){
    int bestCache{std::numeric_limits<int>::max()};
    HirarchyNode* bestNode{};
    for(auto& f: leaders){
        int tmpCache;
        f.second.getCacheNode(tmpCache);
        if(tmpCache < bestCache){
            bestCache = tmpCache;
            bestNode = &f.second;
        }
    }
    if(int c = calcCacheScore(); c < bestCache){
        bestCache = c;
        bestNode = this;
    }
    cacheScore = bestCache;
    return bestNode;
}

void HirarchyNode::cacheNode(const std::string_view& cachePath, const std::string& parentId, float* parentCenter, float parentEps, HirarchyNode* chacheNode){
    uint32_t curInd{};
    uint32_t maxLeadersPerAx = static_cast<uint32_t>(ceil(parentEps / eps));
    uint32_t multiplier = 1;
    for(int i = 0; i < rTree.NUMDIMS; ++i){
        curInd += static_cast<uint32_t>((followerData[i] - parentCenter[i] + parentEps)  / (2 * parentEps) * (maxLeadersPerAx - 1)+ .5) * multiplier;
        multiplier *= maxLeadersPerAx;
    }
    std::string curId = parentId + "_" + std::to_string(curInd);
    if(this == chacheNode){
        for(auto& f: leaders){
            f.second.cacheNode(cachePath, curId, followerData.data(), eps, &f.second);
        }
        leaders.clear();    //deleting all leader nodes
        std::ofstream f(std::string(cachePath) + "/" + curId, std::ios_base::app | std::ios_base::binary);    //opening an append filestream
        // adding fixed size information header
        f << rTree.NUMDIMS << " " << followerData.size() << "\n";   //space needed to easily be able to parse the file again
        f.write(reinterpret_cast<char*>(followerData.data()), followerData.size() * sizeof(followerData[0]));
        f << "\n";
    }
    int del = -1;
    for(auto& f: leaders){
        f.second.cacheNode(cachePath, curId, followerData.data(), eps, chacheNode);
        if(&f.second == chacheNode)
            del = f.first;
    }
    if(del >= 0) //removing child if cached
    {
        rTree.Remove(leaders[del].followerData.data(), leaders[del].followerData.data(), del);
        leaders.erase(del);
    }
}

size_t HirarchyNode::getByteSize(){
    size_t size = followerData.size() * sizeof(followerData[0]);
    size += 2 * size;   //byte size of r tree is around double the size of the follower data
    for(auto& f: leaders){
        size += f.second.getByteSize();
    }
    return size;
}

uint32_t HirarchyNode::_globalUpdateStamp = 0;