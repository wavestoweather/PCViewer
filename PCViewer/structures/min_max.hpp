#pragma once
#include <limits>

namespace structures{
template<class T>
struct min_max{
    T min{std::numeric_limits<T>::max()};
    T max{std::numeric_limits<T>::lowest()};
    bool operator==(const min_max<T>& o) const {return min == o.min && max == o.max;};
    T* data() { return &min; }
};
}