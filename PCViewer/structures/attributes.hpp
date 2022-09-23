#pragma once
#include <vector>
#include <string>
#include <limits>
#include <map>
#include <array>
#include <change_tracker.hpp>
#include <min_max.hpp>

namespace structures{

struct attribute{
    std::string                     id{};
    std::string                     display_name{};
    change_tracker<min_max<float>>  bounds{};                              // global shown bounds
    min_max<float>                  data_bounds{};                         // for normalized data the normalization bounds
    std::map<std::string_view, float> categories{};
    bool operator==(const attribute& o) const {return id == o.id && bounds.read() == o.bounds.read();}
};

struct query_attribute{
    bool            is_dimension: 1;
    bool            is_string_length_dimension: 1;
    bool            is_dim_active: 1;
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
}

namespace globals{
// no global attributes as attributes are only stored in datasets (eases the change of attributes)
}