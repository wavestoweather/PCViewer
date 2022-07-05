#pragma once
#include<vector>
#include<inttypes.h>

struct UVecHash{
    std::size_t operator()(std::vector<uint32_t> const& vec) const{
        std::size_t seed = vec.size();
        for(const auto& i : vec){
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};