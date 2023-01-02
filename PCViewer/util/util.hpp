#pragma once
#include <ranges.hpp>

namespace util{
template<typename T>
inline T align(T size, T alignment){
    return (size + alignment - 1) / alignment * alignment;
}

template<typename T>
inline T normalize_val_for_range(const T& val, const T& min, const T& max){
    return (val - min) / (max - min);
}

template<typename T>
inline T unnormalize_val_for_range(const T& normalized, const T& min, const T& max){
    return min + normalized * (max - min);
}

inline bool point_in_box(const ImVec2& point, const ImVec2& a, const ImVec2& b){
    return point.x > a.x && point.x < b.x && point.y > a.y && point.y < b.y;
}

inline ImU32 vec4_to_imu32(const ImVec4& v){
    return IM_COL32(v.x * 255, v.y * 255, v.z * 255, v.w * 255);
}

inline ImVec2 vec2_add_vec2(const ImVec2& a, const ImVec2& b){
    return {a.x + b.x, a.y + b.y};
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

inline double distance(const ImVec2& a, const ImVec2& b){
    double d{};
    for(int i: util::i_range(2)){
        double diff = a[i] - b[i];
        d += diff * diff;
    }
    return std::sqrt(d);
}

inline double distance(const ImVec4& a, const ImVec4& b){
    double d{};
    for(int i: util::i_range(4)){
        double diff = a[i] - b[i];
        d += diff * diff;
    }
    return std::sqrt(d);
}
}