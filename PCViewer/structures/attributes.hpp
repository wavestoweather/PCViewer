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
#include <memory_view.hpp>

#define DECL_ATTRIBUTE_READ(att_id)   const structures::attribute&  attribute_read() const  {return globals::attributes.read().at(att_id).read();}
#define DECL_ATTRIBUTE_WRITE(att_id)        structures::attribute&  attribute_write()       {return globals::attributes()[att_id]();}

namespace structures{
// attribute that is stored in datasets
struct attribute{
    std::string                     id{};
    std::string                     display_name{};
    change_tracker<min_max<float>>  bounds{};                               // global shown bounds
    std::map<std::string, float>    categories{};
    std::vector<std::string_view>   ordered_categories{};                   // ordered by mapped value
    bool operator==(const attribute& o) const {return id == o.id && bounds.read() == o.bounds.read();}
    attribute() = default;
    attribute(const std::string& id, const std::string& display_name): id(id), display_name(display_name) {}
    attribute(const std::string& id, const std::string& display_name, const change_tracker<min_max<float>>& bounds): id(id), display_name(display_name), bounds(bounds) {}
    attribute(const attribute& o): id(o.id), display_name(o.display_name), bounds(o.bounds), categories(o.categories){ for(std::string_view cat: o.ordered_categories) ordered_categories.emplace_back(categories.find(std::string(cat))->first);}
    attribute& operator=(const attribute& o) {id = o.id; display_name = o.display_name; bounds = o.bounds; categories = o.categories; for(std::string_view cat: o.ordered_categories) ordered_categories.emplace_back(categories.find(std::string(cat))->first); return *this;}
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
    change_tracker<bool>    active{true};
    change_tracker<ImVec4>  color{ImVec4{1.f, 1.f, 1.f, 1.f}};

    uint32_t                usage_count{}; // usage count is increased when a datasets is loaded and contains this atttribute and is decreased when a dataset is destroyed. When the usage_count of an attribute reaches 0, it will be destroyed.

    global_attribute() = default;
    global_attribute(const attribute& a, bool active, ImVec4 color): attribute(a), active(active), color(color) {}
};
using tracked_global_attribute_t = unique_tracker<global_attribute>;
using attributes_t = change_tracker<std::map<std::string_view, tracked_global_attribute_t>>;
}

namespace globals{
extern std::vector<std::string_view> selected_attributes;
extern structures::attributes_t     attributes;
extern structures::names_group      attribute_groups;
}

namespace structures{
using activation_tracker = change_tracker<bool>;
using bounds_tracker = change_tracker<min_max<float>>;
using color_tracker = change_tracker<ImVec4>;
struct attribute_info{
    std::string_view                        attribute_id{};
    bool                                    linked_with_attribute{true};
    util::memory_view<activation_tracker>   active{};
    util::memory_view<bounds_tracker>       bounds{};
    util::memory_view<color_tracker>        color{};
    bool any_change() const {return active && active->changed || bounds && bounds->changed || color && color->changed;}
    void clear_change()     {if(active) active->changed = false; if(bounds) bounds->changed = false; if(color) color->changed = false;}
    DECL_ATTRIBUTE_READ(attribute_id);
    DECL_ATTRIBUTE_WRITE(attribute_id);

    bool operator==(const attribute_info& o) const {return attribute_id == o.attribute_id && linked_with_attribute == o.linked_with_attribute && active == o.active && bounds == o.bounds;}
};
using const_attribute_info_ref = std::reference_wrapper<const attribute_info>;
}
