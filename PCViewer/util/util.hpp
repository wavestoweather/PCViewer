#pragma once
#include <ranges.hpp>

namespace util{
template<typename T>
inline T align(T size, T alignment){
    return (size + alignment - 1) / alignment * alignment;
}

inline std::vector<uint32_t> bool_vector_to_uint(const std::vector<bool>& v){
    std::vector<uint32_t> res((v.size() + 31) / 32);
    for(size_t i: size_range(res)){
        const size_t start = i * 32;
        const size_t end = (i + 1) * 32;
        uint32_t bits{};
        for(size_t j: i_range(start, end))
            bits |= v[j] << (i % 32);
        res[i] = bits;
    }
    return res;
}
}