#include "VectorLeaderNode.hpp"
#include <iostream>
#include <limits>
#include <fstream>
#include <sstream>

VectorLeaderNode::VectorLeaderNode(const std::vector<float>& pos, float inEps, float inEpsMul, uint32_t inDepth, uint32_t inMaxDepth):
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

void VectorLeaderNode::addDataPoint(const std::vector<float>& d){
    uint32_t closest; bool leaderFound{false};
    auto searchCallback = [&](const uint32_t& id){closest = id; leaderFound = true; return false;}; // return false to quit when first was found
    std::unique_lock<std::shared_mutex> lock(_insertLock);
    assert(d.size() == rTree->NumDims());                                                           //safety check
    rTree->Search(d.data(), d.data(), searchCallback);
    if(leaderFound){    
        followerCounts[closest]++;                                                                  //found leader
        if(depth < maxDepth){                                                                       //only push down the hirarchy if not at leaf nodes
            auto f = follower[closest];
            if(f){
                lock.unlock();
                f->addDataPoint(d);
            }
            else{
                f = std::make_shared<VectorLeaderNode>(d, eps * epsMul, epsMul, depth + 1, maxDepth);
                lock.unlock();
            }
        }
        else
            lock.unlock();
    }
    else{                                                                                           //new leader/child has to be created
        std::vector<float> minsMaxs(2 * d.size());
        for(int i = 0; i < d.size(); ++i){
            minsMaxs[i] = d[i] - eps;
            minsMaxs[i + d.size()] = d[i] + eps;
        }
        uint32_t newId = followerCounts.size();
        rTree->Insert(minsMaxs.data(), minsMaxs.data() + d.size(), newId);
        followerData.insert(followerData.end(), d.begin(), d.end());
        followerCounts.push_back(1);                                                                //adding the counter for each row
        if(depth < maxDepth)
            follower.push_back(std::make_shared<VectorLeaderNode>(d, eps * epsMul, epsMul, depth + 1, maxDepth));
        
        lock.unlock();     //can release the lock, all critical things already done

        if(int x  = followerData.size() / d.size(); x > 2e6){
            std::cout << "\rNote: large HirarchyNode with " << x << "elements";
        }
    }
    _updateStamp = ++_globalUpdateStamp;
}

long VectorLeaderNode::calcCacheScore(){
    return long(_updateStamp) - long(followerCounts.size());
}

HierarchyCreateNode* VectorLeaderNode::getCacheNode(long& cacheScore){
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

void VectorLeaderNode::getCacheNodes(HierarchyCacheInfo& info){
    for(auto& f: follower){
        if(f)
            f->getCacheNodes(info);
    }
    if(long c = calcCacheScore(); info.queue.empty() || info.curByteSize < info.cachingSize || c < info.queue.top().score){    //push the current node
        size_t size = followerData.size() * sizeof(followerData[0]) + follower.size() * sizeof(follower[0]) + followerCounts.size() * sizeof(followerCounts[0]);
        size +=  rTree->ByteSize();           //size from rTree (is being tracked inside the rTree)
        info.curByteSize += size;
        //deleting the topmost NodeInfo if enough nodes have been gathered to have more than cachingSize bytes
        if(info.queue.size() && info.curByteSize - info.queue.top().byteSize > info.cachingSize){
            info.curByteSize -= info.queue.top().byteSize;
            info.queue.pop();
        }
        info.queue.push({this, c, size});
    }
};

void VectorLeaderNode::cacheNodes(CacheManagerInterface& cacheManager, const std::string& parentId, float* parentCenter, float parentEps, const  std::set<HierarchyCreateNode*>& cacheNodes){
    uint32_t rowSize = followerData.size() / followerCounts.size();
    uint32_t finalRowSize = rowSize + 1;
    size_t curInd = getChildIndex(rowSize, parentCenter, followerData.data(), parentEps, eps);
    std::string curId = parentId + "_" + std::to_string(curInd);
    bool found = cacheNodes.empty() || cacheNodes.find(this) != cacheNodes.end();
    if(found){
        rTree->RemoveAll();
        for(auto& f: follower){
            if(f)
                f->cacheNodes(cacheManager, curId, followerData.data(), eps, {});
        }
        follower = std::vector<std::shared_ptr<VectorLeaderNode>>();    //deleting all leader nodes
        
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

size_t VectorLeaderNode::getByteSize(){
    //size from arrays in the class
    size_t size = followerData.size() * sizeof(followerData[0]) + follower.size() * sizeof(follower[0]) + followerCounts.size() * sizeof(followerCounts[0]);
    size +=  rTree->ByteSize();           //size from rTree (is being tracked inside the rTree)
    for(auto& f: follower){
        if(f)
            size += f->getByteSize();
    }
    return size;
}

uint32_t VectorLeaderNode::getChildIndex(uint32_t dimensionality, float* parentCenter, float* childCenter, float parentEps, float childEps){
    size_t curInd{};
    uint32_t maxLeadersPerAx = static_cast<uint32_t>(ceil(parentEps / childEps));
    uint32_t multiplier = 1;
    for(int i = 0; i < dimensionality; ++i){
        curInd += static_cast<size_t>((childCenter[i] - parentCenter[i] + parentEps)  / (2 * parentEps) * (maxLeadersPerAx - 1)+ .5) * multiplier;
        multiplier *= maxLeadersPerAx;
    }
    return curInd;
}

std::atomic<size_t> VectorLeaderNode::_globalUpdateStamp{0};