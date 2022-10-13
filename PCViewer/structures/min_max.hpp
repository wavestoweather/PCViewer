#pragma once
#include <limits>
#include <algorithm>

namespace structures{
template<class T>
struct min_max{
    T min{std::numeric_limits<T>::max()};
    T max{std::numeric_limits<T>::lowest()};
    bool operator==(const min_max<T>& o) const {return min == o.min && max == o.max;};
    T* data() { return &min; }
};
}

namespace std{
inline structures::min_max<float> min_max(const structures::min_max<float>& a, const structures::min_max<float>& b){
    return {min(a.min, b.min), max(a.max, b.max)};
}
}