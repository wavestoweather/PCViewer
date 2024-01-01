#pragma once

namespace structures{
template<typename T = float>
struct scale_offset{
    T scale{1.f};
    T offset{0};
};
}