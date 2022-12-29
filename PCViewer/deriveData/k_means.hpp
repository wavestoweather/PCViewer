#pragma once
#include "MemoryView.hpp"
#include <enum_names.hpp>

namespace k_means{
enum class distance_method_t{
    norm,
    squared_norm,
    l1_norm,
    COUNT
};

const structures::enum_names<distance_method_t> distance_method_names{
    "Norm",
    "Squared Norm",
    "L1 Norm"
};

enum class init_method_t{
    forgy,
    uniform_random,
    normal_random,
    plus_plus,
    COUNT
};

const structures::enum_names<init_method_t> init_method_names{
    "Gorgy",
    "Uniform Random",
    "Normal Random",
    "Plus Plus"
};

enum class mean_method_t{
    mean,
    median,
    medoid,
    COUNT
};

const structures::enum_names<mean_method_t> mean_method_names{
    "Mean",
    "Median",
    "Medoid"
};

struct k_means_settings_t{
    distance_method_t distance_method;
    int             cluster_count;
    init_method_t   init_method;
    mean_method_t   mean_method;
    int             max_iteration;
};

using float_column_views = std::vector<deriveData::column_memory_view<float>>;
void run(const float_column_views& input, float_column_views& output, const k_means_settings_t& settings);
}