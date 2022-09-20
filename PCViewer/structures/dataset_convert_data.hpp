#pragma once
#include <string>
#include <min_max.hpp>

namespace structures{
struct dataset_convert_data{
    enum class destination: uint32_t{
        drawlist,
        templatelist,
        COUNT
    };

    std::string_view ds_id{};
    std::string_view tl_id{};
    std::string     dst_name{};
    bool            random_subsampling{};
    int             subsampling{};
    min_max<size_t> trim{};
    destination     dst{};
};
}