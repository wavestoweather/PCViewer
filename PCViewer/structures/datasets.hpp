#pragma once
#include <attributes.hpp>
#include <change_tracker.hpp>
#include <data.hpp>
#include <memory>
#include <half.hpp>
#include <vk_mem_alloc.h>

namespace structures{

class dataset{
    std::string             id;
    std::string             display_name;
    std::string             backing_data;   // filename or folder of the data on the hdd
    std::vector<attribute>  attributes;
    uint32_t                original_attribute_size;
    data<float>             float_data;
    data<half>              half_data;
    std::vector<VkBuffer>   gpu_data;
    bool operator==(const dataset& o) const {return id == o.id;}
};

}

namespace globals{

template<class T>
using changing = structures::change_tracker<T>;
using datasets_t = changing<std::map<std::string_view, std::unique_ptr<changing<structures::dataset>>>>;
extern datasets_t datasets;

}