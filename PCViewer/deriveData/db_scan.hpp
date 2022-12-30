#pragma once
#include "MemoryView.hpp"
#include <enum_names.hpp>

namespace db_scan{
struct db_scans_settings_t{
    float   epsilon;
    int     min_points;
};

using float_column_view = std::vector<deriveData::column_memory_view<float>>;
// output has to be initialized to 0 to indicate not done clustering
void run(const float_column_view& input, float_column_view& output, const db_scans_settings_t& settings);
}