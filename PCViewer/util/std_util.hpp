#pragma once
#include <cstddef>

namespace std{
template <class T>
inline size_t hash_combine(std::size_t seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    return seed;
}
template<typename T>
inline int sign(const T& val){
    return (T(0) < val) - (val < T(0));
}
}