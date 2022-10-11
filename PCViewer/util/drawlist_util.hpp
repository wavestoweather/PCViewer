#pragma once
#include <string_view>
#include <drawlists.hpp>

namespace util{
namespace drawlist{
inline size_t drawlist_index(const std::string_view& dl_id){
    size_t index{};
    for(const auto& [id, dl]: globals::drawlists.read()){
        if(id == dl_id)
            return index;
        ++index;
    }
    return index;
}
}
}