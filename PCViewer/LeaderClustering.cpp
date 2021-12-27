#include "LeaderClustering.h"
#include "rTree/RTreeDynamic.h"
#include <map>

void LeaderClustering::cluster(const Data& in, Data& out, const std::vector<float>& epsilon){
    const int numDims = static_cast<int>(in.columns.size());
    RTreeDynamic<uint32_t, float> tree(numDims);
    std::vector<float> mins(numDims), maxs(numDims);
    struct LeaderData{
        std::vector<float> data;
        std::vector<uint32_t> follower;
    };
    std::map<uint32_t, LeaderData> leaders;

    std::vector<uint32_t> closest;
    auto backInsert = [&](const uint32_t& id){closest.push_back(id); return false;}; // return false to quit when first was found
    for(uint32_t i = 0; i < in.size(); ++i){
        closest.clear();
        for(int a = 0; a < numDims; ++a){
            mins[a] = in(i, a);
            maxs[a] = mins[a];
        }
        tree.Search(mins.data(), maxs.data(), backInsert);
        if(closest.size()){  //found leader
            leaders[closest.front()].follower.push_back(i);
        }
        else{                //new leader
            leaders[i].data = mins;
            for(int a = 0; a < numDims; ++a){
                mins[a] -= epsilon[a] / 2;
                maxs[a] += epsilon[a] / 2;
            }
            tree.Insert(mins.data(), maxs.data(), i);
        }
    }
}