#pragma once
#include <string_view>
#include <array>

namespace structures{

template<class T>
struct enum_names{
    //enum_names(std::initializer_list<std::string_view> l): names(l){}
    typename std::array<std::string_view, static_cast<size_t>(T::COUNT)> names;
    std::string_view operator[](T place) const {return names[static_cast<size_t>(place)];}
};

}