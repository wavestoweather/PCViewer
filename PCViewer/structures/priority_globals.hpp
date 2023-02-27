#pragma once
#include <atomic>
#include <string_view>

namespace globals{
extern std::atomic<float>     priority_center_value;
extern std::atomic<float>     priority_center_distance;
extern std::string_view       priority_center_attribute_id;
extern const std::string_view priority_drawlist_standard_order;
}