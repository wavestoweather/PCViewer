#include "LeaderNode.hpp"
#include <iostream>
#include <limits>
#include <fstream>

LeaderNode::LeaderNode(const std::vector<float>& pos, float inEps, float inEpsMul, uint32_t inDepth, uint32_t inMaxDepth):
eps(inEps),
epsMul(inEpsMul),
depth(inDepth),
maxDepth(inMaxDepth)
{
    switch(pos.size()){
        case 1: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 1>>(); break;
        case 2: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 2>>(); break;
        case 3: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 3>>(); break;
        case 4: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 4>>(); break;
        case 5: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 5>>(); break;
        case 6: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 6>>(); break;
        case 7: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 7>>(); break;
        case 8: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 8>>(); break;
        case 9: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 9>>(); break;
        case 10: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 10>>(); break;
        case 11: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 11>>(); break;
        case 12: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 12>>(); break;
        case 13: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 13>>(); break;
        case 14: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 14>>(); break;
        case 15: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 15>>(); break;
        case 16: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 16>>(); break;
        case 17: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 17>>(); break;
        case 18: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 18>>(); break;
        case 19: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 19>>(); break;
        case 20: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 20>>(); break;
        case 21: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 21>>(); break;
        case 22: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 22>>(); break;
        case 23: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 23>>(); break;
        case 24: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 24>>(); break;
        case 25: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 25>>(); break;
        case 26: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 26>>(); break;
        case 27: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 27>>(); break;
        case 28: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 28>>(); break;
        case 29: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 29>>(); break;
        case 30: rTree = std::make_unique<RTreeUtil::RTreeS<uint32_t, float, 30>>(); break;
        default: std::cout << "Using dynamic rtree for leaders clustering";
            rTree = std::make_unique<RTreeUtil::RTreeD<uint32_t, float>>(0);
    }
    addDataPoint(pos);  //automatically adds itself to the follower data
}

void LeaderNode::addDataPoint(const std::vector<float>& d){
    std::vector<uint32_t> closest;
    auto backInsert = [&](const uint32_t& id){closest.push_back(id); return false;}; // return false to quit when first was found
    std::unique_lock<std::shared_mutex> lock(_insertLock);
    rTree->Search(d.data(), d.data(), backInsert);
    if(closest.size()){         //found leader
        lock.unlock();         //can release lock already
        followerData[closest.front() * (rTree->NumDims() + 1) + rTree->NumDims()] += 1;
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
        rTree->Insert(mins.data(), maxs.data(), newId);
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

int LeaderNode::calcCacheScore(){
    return followerData.size() / rTree->NumDims() - _updateStamp;
}

HierarchyCreateNode* LeaderNode::getCacheNode(int& cacheScore){
    int bestCache{std::numeric_limits<int>::max()};
    HierarchyCreateNode* bestNode{};
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

void LeaderNode::cacheNode(const std::string_view& cachePath, const std::string& parentId, float* parentCenter, float parentEps, HierarchyCreateNode* chacheNode){
    size_t curInd{};
    uint32_t maxLeadersPerAx = static_cast<uint32_t>(ceil(parentEps / eps));
    uint32_t multiplier = 1;
    for(int i = 0; i < rTree->NumDims(); ++i){
        curInd += static_cast<size_t>((followerData[i] - parentCenter[i] + parentEps)  / (2 * parentEps) * (maxLeadersPerAx - 1)+ .5) * multiplier;
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
        f << rTree->NumDims() + 1 << " " << followerData.size() << " " << eps << "\n";   //space needed to easily be able to parse the file again
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
        rTree->Remove(leaders[del].followerData.data(), leaders[del].followerData.data(), del);
        leaders.erase(del);
    }
}

size_t LeaderNode::getByteSize(){
    size_t size = followerData.size() * sizeof(followerData[0]);
    size += 2 * size;   //byte size of r tree is around double the size of the follower data
    for(auto& f: leaders){
        size += f.second.getByteSize();
    }
    return size;
}

uint32_t LeaderNode::_globalUpdateStamp = 0;