#pragma once
#include <vector>
#include <string>
#include <limits>
#include <map>
#include <array>
#include <change_tracker.hpp>

namespace structures{
template<class T>
struct min_max{
    T min{std::numeric_limits<T>::max()};
    T max{std::numeric_limits<T>::lowest()};
    bool operator==(const min_max<T>& o) const {return min == o.min && max == o.max;};
};

struct attribute{
    const std::string               id;
    std::string                     display_name;
    change_tracker<min_max<float>>  bounds;                              // global shown bounds
    min_max<float>                  data_bounds;                         // for normalized data the normalization bounds
    std::map<std::string_view, float> categories;
    std::vector<std::pair<std::string_view, float>> ctegories_ordered;
    bool operator==(const attribute& o) const {return id == o.id && bounds() == o.bounds();}
};

struct query_attribute{
    bool            is_dimension: 1;
    bool            is_string_length_dimension: 1;
    bool            is_enabled: 1;
    bool            is_active: 1;
    bool            linearize: 1;
    std::string     id;
    int             dimension_size;
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