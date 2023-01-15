#pragma once
#include <vector>
#include <string>
#include <limits>
#include <map>
#include <array>
#include <change_tracker.hpp>
#include <min_max.hpp>
#include <group.hpp>
#include <imgui.h>

namespace structures{

// attribute that is stored in datasets
struct attribute{
    std::string                     id{};
    std::string                     display_name{};
    change_tracker<min_max<float>>  bounds{};                              // global shown bounds
    std::map<std::string_view, float> categories{};
    bool operator==(const attribute& o) const {return id == o.id && bounds.read() == o.bounds.read();}
};

struct query_attribute{
    bool            is_dimension: 1;
    bool            is_string_length_dimension: 1;
    bool            is_active: 1;
    bool            linearize: 1;
    std::string     id;
    size_t          dimension_size;
    std::vector<int> dependant_dimensions;
    int             dimension_subsample;
    int             dimension_slice;
    min_max<size_t> trim_indices;

    bool operator==(const query_attribute& o) const {return id == o.id && dimension_size == o.dimension_size && dependant_dimensions == o.dependant_dimensions;};
};

struct global_attribute: public attribute{
    // has all members from the attribute struct, needs a view additional infos
    change_tracker<bool>    active;
    change_tracker<ImVec4>  color{ImVec4{1.f, 1.f, 1.f, 1.f}};
};
using tracked_global_attribute_t = unique_tracker<global_attribute>;
using attributes_t = change_tracker<std::map<std::string_view, tracked_global_attribute_t>>;
}

namespace globals{
extern std::vector<std::string_view> selected_attributes;
extern structures::attributes_t     attributes;
extern structures::names_group      attribute_groups;
}