#pragma once
#include <string>
#include <memory_view.hpp>

namespace util{
namespace histogram_registry{

inline std::string get_id_string(util::memory_view<const uint32_t> attribute_indices, util::memory_view<const int> bin_sizes){
    std::string id;
    for(int i: util::size_range(attribute_indices)){
        id += std::to_string(attribute_indices[i]);
        if(i < attribute_indices.size() - 1)
            id += '_';
    }
    id += '|';
    for(int i: util::size_range(bin_sizes)){
        id += std::to_string(bin_sizes[i]);
        if(i < bin_sizes.size() - 1)
            id += '_';
    }
    return id;
}

}
}