#pragma once
#include <vector>
#include <string_view>
#include <min_max.hpp>

namespace structures{
struct load_behaviour{
    struct dl_from_loaded_ds{
        bool                            random_subsampling{};
        int                             subsampling{};
        min_max<size_t>                 trim{};

        std::vector<std::string_view>   coupled_workbenches{};
    };
    std::vector<dl_from_loaded_ds>  on_load;
};
}

namespace globals{
extern structures::load_behaviour load_behaviour;
}