#pragma once
#include <string>
#include <min_max.hpp>
#include <variant>
#include <vector>
#include <cinttypes>

namespace structures{
struct templatelist_convert_data{
    enum class destination: uint32_t{
        drawlist,
        templatelist,
        COUNT
    };

    std::string_view ds_id{};
    std::string_view tl_id{};
    std::string     dst_name{"drawlist name"};
    bool            random_subsampling{};
    int             subsampling{1};
    min_max<size_t> trim{};
    destination     dst{};

    const std::vector<uint32_t>* indices{};
};

struct templatelist_split_data{
    struct uniform_value_split{
        int                 split_count{5};
    };
    struct value_split{
        std::vector<float>  values{};
    };
    struct quantile_split{
        std::vector<float>  quantiles{};
    };
    struct automatic_split{};

    std::string_view ds_id{};
    std::string_view tl_id{};
    std::string     dst_name_format{"attr_x_%d"};
    bool            create_drawlists{true};
    int             attribute{};
    std::variant<uniform_value_split, value_split, quantile_split, automatic_split> additional_info{uniform_value_split{}};
};
}