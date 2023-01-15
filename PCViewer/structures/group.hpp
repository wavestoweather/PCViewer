#pragma once
#include <vector>
#include <string_view>

namespace structures{
template<typename T>
struct group{
    std::string             name;
    std::vector<T>          child_elements;
    std::vector<group<T>>   child_groups;
};

using names_group = group<std::string_view>;
}